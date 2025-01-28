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
    """Validate if scenario has exactly 4 aircraft."""
    flightplans = root.findall(".//initial-flightplans")
    count = len(flightplans)
    is_valid = count == 4
    if not is_valid:
        logger.error(f"Expected exactly 4 aircraft, found {count}")
    return is_valid, count


def validate_aircraft_types(aircraft_list: List[ET.Element]) -> Tuple[bool, Dict]:
    """Validate if there are exactly 2 A320s and 2 B737s."""
    type_counts = {'A320': 0, 'B737': 0}
    details = {'is_valid': False, 'counts': type_counts, 'errors': []}

    for i, aircraft in enumerate(aircraft_list, 1):
        try:
            aircraft_type = aircraft.find("type").text
            if aircraft_type in type_counts:
                type_counts[aircraft_type] += 1
            else:
                details['errors'].append(
                    f"Aircraft {i}: Invalid type: {aircraft_type}")
        except AttributeError as e:
            details['errors'].append(
                f"Aircraft {i}: Error reading type: {str(e)}")
            continue
        except Exception as e:
            details['errors'].append(
                f"Aircraft {i}: Unexpected error: {str(e)}")
            continue

    is_valid = type_counts['A320'] == 2 and type_counts['B737'] == 2
    if not is_valid:
        details['errors'].append(
            f"Invalid aircraft type distribution. Expected: 2 A320s and 2 B737s. "
            f"Found: {type_counts['A320']} A320s and {type_counts['B737']} B737s")

    details['is_valid'] = is_valid
    details['counts'] = type_counts
    return is_valid, details


def validate_airway_n892(aircraft: ET.Element, airways: Dict) -> Dict:
    """Validate if aircraft is on airway N892."""
    validation_info = {
        "is_valid": False,
        "route_waypoints": [],
        "departure": None,
        "destination": None,
        "errors": []
    }

    # Extract route waypoints
    route_waypoints = [
        waypoint.text for waypoint in aircraft.findall("air_route")]
    validation_info["route_waypoints"] = route_waypoints

    # Check if waypoints match N892
    expected_waypoints = airways['N892']['air_route']
    if route_waypoints != expected_waypoints:
        validation_info["errors"].append(
            f"Invalid waypoints for N892. Expected: {expected_waypoints}, Found: {route_waypoints}")

    # Validate airports
    scenario_dep = aircraft.find(".//dep/af").text
    scenario_des = aircraft.find(".//des/af").text
    validation_info["departure"] = scenario_dep
    validation_info["destination"] = scenario_des

    expected_dep = airways['N892']['departure']['af']
    expected_des = airways['N892']['destination']['af']

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

    # Sort aircraft by time
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
            "is_valid": abs(actual_time - expected_time) <= 1
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
        "type_details": {},
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
    results["types_valid"], type_details = validate_aircraft_types(
        aircraft_list)
    results["type_details"] = type_details

    # Validate routes (all should be N892)
    routes_valid = True
    for i, aircraft in enumerate(aircraft_list, 1):
        route_info = validate_airway_n892(aircraft, airways)
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
        "N892 Scenario Validation Summary",
        "=" * 50,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 50,
        f"Total scenarios processed: {len(all_results)}",
        f"Completely valid scenarios: {sum(1 for r in all_results if r['all_valid'])} ({(sum(1 for r in all_results if r['all_valid'])/len(all_results))*100:.1f}%)",
        "",
        "Requirements Check Summary:",
        f"- Correct aircraft count (4): {sum(1 for r in all_results if r['aircraft_count_valid'])}",
        f"- Correct aircraft types (2 A320s, 2 B737s): {sum(1 for r in all_results if r['types_valid'])}",
        f"- Valid routes (N892): {sum(1 for r in all_results if r['routes_valid'])}",
        f"- Correct time separation (150s): {sum(1 for r in all_results if r['time_separation_valid'])}",
        "",
        "Detailed Results",
        "-" * 20
    ]

    for result in all_results:
        report_content.extend([
            f"\nScenario: {result['filename']}",
            f"Aircraft Count: {'✓' if result['aircraft_count_valid'] else '✗'} ({result['aircraft_count']}/4 aircraft)",
            f"Aircraft Types: {'✓' if result['types_valid'] else '✗'}"
        ])

        # Add type distribution details
        if 'type_details' in result and 'counts' in result['type_details']:
            counts = result['type_details']['counts']
            report_content.append(f"  A320s: {counts.get('A320', 0)}/2")
            report_content.append(f"  B737s: {counts.get('B737', 0)}/2")

        report_content.extend([
            f"Routes (N892): {'✓' if result['routes_valid'] else '✗'}",
            f"Time Separation: {'✓' if result['time_separation_valid'] else '✗'} (150s intervals)"
        ])

        report_content.append("\nRoute Details:")
        for detail in result["route_details"]:
            report_content.extend([
                f"\nAircraft {detail['aircraft_number']}:",
                f"  Waypoints: {' -> '.join(detail['route_waypoints'])}",
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
    # Import airways dictionary
    from testing_metrics import airways

    # Directory containing scenario files
    scenarios_dir = "generated_scenarios_Prompt8_temp0"  # Update this to your directory

    # Create reports directory if it doesn't exist
    reports_dir = "validation_reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # Generate timestamp for the report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"validation_report_prompt8_{timestamp}.txt"
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
