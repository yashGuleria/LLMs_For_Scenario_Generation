import os
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_xml_scenario(file_path: str) -> ET.Element:
    """Load and parse XML scenario file."""
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except Exception as e:
        logger.error(f"Error parsing XML file {file_path}: {str(e)}")
        return None


def validate_aircraft_count(root: ET.Element) -> Tuple[bool, int]:
    """Validate if scenario has exactly 12 aircraft."""
    flightplans = root.findall(".//initial-flightplans")
    count = len(flightplans)
    is_valid = count == 12
    if not is_valid:
        logger.error(f"Expected exactly 12 aircraft, found {count}")
    return is_valid, count


def validate_aircraft_type(aircraft: ET.Element) -> bool:
    """Validate if aircraft is A388."""
    try:
        aircraft_type = aircraft.find("type").text
        # Modified to handle "Airbus A388"
        is_valid = aircraft_type == 'Airbus A388' or aircraft_type == 'A388'
        if not is_valid:
            logger.error(
                f"Invalid aircraft type: {aircraft_type}, expected Airbus A388 or A388")
        return is_valid
    except AttributeError as e:
        logger.error(f"Error finding aircraft type: {str(e)}")
        return False


def find_matching_airway(route_waypoints: List[str], airways: Dict) -> str:
    """Find the airway that matches the route waypoints."""
    for airway_name, airway_data in airways.items():
        if route_waypoints == airway_data['air_route']:
            return airway_name
    return None


def validate_route_and_airports(aircraft: ET.Element, airways: Dict) -> Dict:
    """Validate aircraft's route and airports match an airway."""
    validation_info = {
        "is_valid": False,
        "matching_airway": None,
        "route_waypoints": [],
        "departure": None,
        "destination": None,
        "errors": []
    }

    # Extract route waypoints
    route_waypoints = [
        waypoint.text for waypoint in aircraft.findall("air_route")]
    validation_info["route_waypoints"] = route_waypoints

    # Find matching airway
    matching_airway = find_matching_airway(route_waypoints, airways)
    if matching_airway is None:
        validation_info["errors"].append(
            f"No matching airway found for route: {route_waypoints}")
        return validation_info

    validation_info["matching_airway"] = matching_airway

    # Validate airports
    scenario_dep = aircraft.find(".//dep/af").text
    scenario_des = aircraft.find(".//des/af").text
    validation_info["departure"] = scenario_dep
    validation_info["destination"] = scenario_des

    expected_dep = airways[matching_airway]['departure']['af']
    expected_des = airways[matching_airway]['destination']['af']

    if scenario_dep != expected_dep:
        validation_info["errors"].append(
            f"Invalid departure airport. Expected: {expected_dep}, Found: {scenario_dep}")

    if scenario_des != expected_des:
        validation_info["errors"].append(
            f"Invalid destination airport. Expected: {expected_des}, Found: {scenario_des}")

    validation_info["is_valid"] = len(validation_info["errors"]) == 0
    return validation_info


def validate_time_separation(aircraft_list: List[ET.Element]) -> Tuple[bool, List[Dict]]:
    """Validate that aircraft are separated by 150 seconds, starting at time 0."""
    validation_details = []
    is_valid = True

    # Sort aircraft by time string value
    aircraft_times = []
    for i, aircraft in enumerate(aircraft_list):
        try:
            time_element = aircraft.find("time")
            if time_element is None:
                logger.error(f"No time element found for aircraft {i}")
                return False, []

            start_time = float(time_element.text)
            aircraft_times.append((i, start_time))
        except (AttributeError, ValueError) as e:
            logger.error(f"Error reading time for aircraft {i}: {str(e)}")
            return False, []

    aircraft_times.sort(key=lambda x: x[1])

    # Validate time separation
    expected_time = 0
    for i, (aircraft_index, actual_time) in enumerate(aircraft_times):
        detail = {
            "aircraft_number": aircraft_index + 1,
            "expected_time": expected_time,
            "actual_time": actual_time,
            # 1 second tolerance
            "is_valid": abs(actual_time - expected_time) < 1
        }

        if not detail["is_valid"]:
            is_valid = False
            detail["error"] = f"Invalid time. Expected: {expected_time}, Found: {actual_time}"

        validation_details.append(detail)
        expected_time += 150  # Next expected time

    return is_valid, validation_details


def validate_scenario(file_path: str, airways: Dict) -> Dict:
    """Validate a single scenario file."""
    logger.info(f"Validating scenario: {file_path}")

    results = {
        "filename": os.path.basename(file_path),
        "aircraft_count_valid": False,
        "aircraft_count": 0,
        "types_valid": False,
        "routes_valid": False,
        "time_separation_valid": False,
        "all_valid": False,
        "route_details": [],
        "time_details": [],
        "errors": []
    }

    root = load_xml_scenario(file_path)
    if root is None:
        results["errors"].append("Failed to load XML file")
        return results

    # Validate aircraft count
    results["aircraft_count_valid"], results["aircraft_count"] = validate_aircraft_count(
        root)

    aircraft_list = root.findall(".//initial-flightplans")

    # Validate aircraft types
    types_valid = True
    for i, aircraft in enumerate(aircraft_list, 1):
        if not validate_aircraft_type(aircraft):
            types_valid = False
            results["errors"].append(f"Aircraft {i}: Invalid type (not A388)")
    results["types_valid"] = types_valid

    # Validate routes and airports
    routes_valid = True
    for i, aircraft in enumerate(aircraft_list, 1):
        route_info = validate_route_and_airports(aircraft, airways)
        route_info["aircraft_number"] = i
        results["route_details"].append(route_info)
        if not route_info["is_valid"]:
            routes_valid = False

    results["routes_valid"] = routes_valid

    # Validate time separation
    results["time_separation_valid"], results["time_details"] = validate_time_separation(
        aircraft_list)

    # Overall validation
    results["all_valid"] = all([
        results["aircraft_count_valid"],
        results["types_valid"],
        results["routes_valid"],
        results["time_separation_valid"]
    ])

    return results


def generate_validation_report(all_results: List[Dict], save_path: str = None) -> None:
    """Generate and save validation report."""
    report_content = [
        "12 Aircraft Scenario Validation Summary",
        "=" * 50,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 50,
        f"Total scenarios processed: {len(all_results)}",
        f"Completely valid scenarios: {sum(1 for r in all_results if r['all_valid'])} ({(sum(1 for r in all_results if r['all_valid'])/len(all_results))*100:.1f}%)",
        "",
        "Requirements Check Summary:",
        f"- Correct aircraft count (12): {sum(1 for r in all_results if r['aircraft_count_valid'])}",
        f"- All A388 aircraft: {sum(1 for r in all_results if r['types_valid'])}",
        f"- Valid routes and airports: {sum(1 for r in all_results if r['routes_valid'])}",
        f"- Correct time separation: {sum(1 for r in all_results if r['time_separation_valid'])}",
        "",
        "Detailed Results",
        "-" * 20
    ]

    for result in all_results:
        report_content.extend([
            f"\nScenario: {result['filename']}",
            f"Aircraft Count: {'✓' if result['aircraft_count_valid'] else '✗'} ({result['aircraft_count']}/12 aircraft)",
            f"Aircraft Types: {'✓' if result['types_valid'] else '✗'} (All A388)",
            f"Routes and Airports: {'✓' if result['routes_valid'] else '✗'}",
            f"Time Separation: {'✓' if result['time_separation_valid'] else '✗'} (150s intervals)"
        ])

        report_content.append("\nRoute Details:")
        for detail in result["route_details"]:
            report_content.extend([
                f"\nAircraft {detail['aircraft_number']}:",
                f"  Waypoints: {' -> '.join(detail['route_waypoints'])}",
                f"  Matching Airway: {detail['matching_airway'] if detail['matching_airway'] else 'None'}",
                f"  Departure: {detail['departure']}",
                f"  Destination: {detail['destination']}",
                f"  Valid: {'✓' if detail['is_valid'] else '✗'}"
            ])

            if detail["errors"]:
                report_content.append("  Errors:")
                for error in detail["errors"]:
                    report_content.append(f"    - {error}")

        report_content.append("\nTime Separation Details:")
        for detail in result["time_details"]:
            report_content.append(
                f"  Aircraft {detail['aircraft_number']}: "
                f"Time = {detail['actual_time']}s "
                f"({'✓' if detail['is_valid'] else '✗'})"
            )
            if not detail['is_valid']:
                report_content.append(
                    f"    Error: {detail.get('error', 'Invalid time')}")

        if result["errors"]:
            report_content.append("\nOther Errors:")
            for error in result["errors"]:
                report_content.append(f"  - {error}")

        report_content.append(
            f"\nOverall Status: {'VALID' if result['all_valid'] else 'INVALID'}")
        report_content.append("-" * 50)

    # Print to console and save to file
    for line in report_content:
        logger.info(line)

    if save_path:
        try:
            with open(save_path, 'w') as f:
                f.write('\n'.join(report_content))
            logger.info(f"\nReport saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving report to file: {str(e)}")


def main():
    # Import airways dictionary from testing_metrics.py
    from testing_metrics import airways

    # Directory containing scenario files
    # Updated directory name
    scenarios_dir = "generated_scenarios_prompt13_3_2_6jan_temp0.5"

    # Create reports directory if it doesn't exist
    reports_dir = "validation_reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # Generate timestamp for the report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"validation_report_prompt13_3_2_6jan_temp0.5_{timestamp}.txt"
    report_path = os.path.join(reports_dir, report_filename)

    # Collect and validate all scenarios
    all_results = []
    scenario_files = [f for f in os.listdir(
        scenarios_dir) if f.endswith('.xdat')]

    for scenario_file in scenario_files:
        file_path = os.path.join(scenarios_dir, scenario_file)
        results = validate_scenario(file_path, airways)
        all_results.append(results)

    # Generate validation report
    generate_validation_report(all_results, report_path)


if __name__ == "__main__":
    main()
