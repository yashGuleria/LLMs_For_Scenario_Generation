import os
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Tuple
from datetime import datetime

# Logging configuration
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
    """Validate if scenario has 12 aircraft."""
    flightplans = root.findall(".//initial-flightplans")
    count = len(flightplans)
    is_valid = count == 12
    if not is_valid:
        logger.error(f"Expected less than 4 aircraft, found {count}")
    return is_valid, count


def validate_aircraft_type(aircraft: ET.Element) -> bool:
    """Validate if aircraft type is in the allowed set."""
    aircraft_type = aircraft.find("type").text
    valid_types = {'A320', 'B737', 'B738', 'B734', 'B744', 'A388', 'A333'}

    if aircraft_type not in valid_types:
        logger.error(f"Invalid aircraft type: {aircraft_type}")
        return False
    return True


def find_matching_airway(route_waypoints: List[str], airways: Dict) -> str:
    """
    Find the airway that matches the route waypoints.
    Returns the airway name if found, None otherwise.
    """
    for airway_name, airway_data in airways.items():
        if route_waypoints == airway_data['air_route']:
            return airway_name
    return None


def validate_route_and_airports(aircraft: ET.Element, airways: Dict) -> bool:
    """
    Validate that the aircraft's route matches an airway and has correct airports.
    Returns True if valid, False otherwise.
    """
    # Extract route waypoints from aircraft
    route_waypoints = [
        waypoint.text for waypoint in aircraft.findall("air_route")]

    # Find matching airway
    matching_airway = find_matching_airway(route_waypoints, airways)
    if matching_airway is None:
        logger.error(f"No matching airway found for route: {route_waypoints}")
        return False

    # Get airports from scenario
    scenario_dep = aircraft.find(".//dep/af").text
    scenario_des = aircraft.find(".//des/af").text

    # Get expected airports from matching airway
    expected_dep = airways[matching_airway]['departure']['af']
    expected_des = airways[matching_airway]['destination']['af']

    # Validate airports
    is_valid = True
    if scenario_dep != expected_dep:
        logger.error(f"Invalid departure airport for airway {matching_airway}. "
                     f"Expected: {expected_dep}, Found: {scenario_dep}")
        is_valid = False

    if scenario_des != expected_des:
        logger.error(f"Invalid destination airport for airway {matching_airway}. "
                     f"Expected: {expected_des}, Found: {scenario_des}")
        is_valid = False

    return is_valid


def validate_scenario(file_path: str, airways: Dict) -> Dict:
    """Validate a single scenario file."""
    logger.info(f"Validating scenario: {file_path}")

    results = {
        "filename": os.path.basename(file_path),
        "aircraft_count_valid": False,
        "aircraft_count": 0,
        "types_valid": False,
        "routes_valid": False,
        "all_valid": False,
        "route_details": []  # Add this to store route validation details
    }

    root = load_xml_scenario(file_path)
    if root is None:
        return results

    # Validate aircraft count
    results["aircraft_count_valid"], results["aircraft_count"] = validate_aircraft_count(
        root)

    # Validate each aircraft
    aircraft_list = root.findall(".//initial-flightplans")

    # Check aircraft types
    types_valid = True
    for aircraft in aircraft_list:
        if not validate_aircraft_type(aircraft):
            types_valid = False
            break

    # Check routes and airports
    routes_valid = True
    for i, aircraft in enumerate(aircraft_list, 1):
        route_waypoints = [
            waypoint.text for waypoint in aircraft.findall("air_route")]
        matching_airway = find_matching_airway(route_waypoints, airways)
        scenario_dep = aircraft.find(".//dep/af").text
        scenario_des = aircraft.find(".//des/af").text

        route_info = {
            "aircraft_number": i,
            "route_waypoints": route_waypoints,
            "matching_airway": matching_airway,
            "departure": scenario_dep,
            "destination": scenario_des,
            "is_valid": True,
            "errors": []
        }

        if matching_airway is None:
            route_info["is_valid"] = False
            route_info["errors"].append(
                f"No matching airway found for route: {route_waypoints}")
            routes_valid = False
        else:
            expected_dep = airways[matching_airway]['departure']['af']
            expected_des = airways[matching_airway]['destination']['af']

            if scenario_dep != expected_dep:
                route_info["is_valid"] = False
                route_info["errors"].append(
                    f"Invalid departure airport. Expected: {expected_dep}, Found: {scenario_dep}")
                routes_valid = False

            if scenario_des != expected_des:
                route_info["is_valid"] = False
                route_info["errors"].append(
                    f"Invalid destination airport. Expected: {expected_des}, Found: {scenario_des}")
                routes_valid = False

        results["route_details"].append(route_info)

    results["types_valid"] = types_valid
    results["routes_valid"] = routes_valid
    results["all_valid"] = all([
        results["aircraft_count_valid"],
        types_valid,
        routes_valid
    ])

    return results


def generate_validation_report(all_results: List[Dict], save_path: str = None) -> None:
    """Generate and save validation report."""
    report_content = [  # Define report_content as a list
        "Validation Summary",
        "=" * 50,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 50,
        f"Total scenarios processed: {len(all_results)}",
        f"Completely valid scenarios: {sum(1 for r in all_results if r['all_valid'])} ({(sum(1 for r in all_results if r['all_valid'])/len(all_results))*100:.1f}%)",
        f"Scenarios with correct aircraft count: {sum(1 for r in all_results if r['aircraft_count_valid'])} ({(sum(1 for r in all_results if r['aircraft_count_valid'])/len(all_results))*100:.1f}%)",
        "",
        "Detailed Results",
        "-" * 20
    ]

    for result in all_results:
        report_content.extend([
            f"\nScenario: {result['filename']}",
            f"Aircraft Count: {'✓' if result['aircraft_count_valid'] else '✗'} ({result['aircraft_count']} aircraft)",
            f"Routes Valid: {'✓' if result['routes_valid'] else '✗'}"
        ])

        # Add detailed route validation information
        report_content.append("\nRoute Details:")
        for route_info in result["route_details"]:
            report_content.extend([
                f"\nAircraft {route_info['aircraft_number']}:",
                f"  Waypoints: {' -> '.join(route_info['route_waypoints'])}",
                f"  Matching Airway: {route_info['matching_airway'] if route_info['matching_airway'] else 'None'}",
                f"  Departure: {route_info['departure']}",
                f"  Destination: {route_info['destination']}",
                f"  Valid: {'✓' if route_info['is_valid'] else '✗'}"
            ])

            if route_info["errors"]:
                report_content.append("  Errors:")
                for error in route_info["errors"]:
                    report_content.append(f"    - {error}")

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
    scenarios_dir = "generated_scenarios_prompt11_2_2_6jan_temp0.8_attempt3"

    # Create reports directory if it doesn't exist
    reports_dir = "validation_reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # Generate timestamp for the report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"validation_report_prompt11_2_2_6jan_temp0.8_attempt3_{timestamp}.txt"
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
