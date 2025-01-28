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
    """Validate if scenario has exactly 7 aircraft."""
    flightplans = root.findall(".//initial-flightplans")
    count = len(flightplans)
    is_valid = count == 7
    if not is_valid:
        logger.error(f"Expected 7 aircraft, found {count}")
    return is_valid, count


def validate_aircraft_types(aircraft_list: List[ET.Element]) -> bool:
    """Validate if all aircraft are of type A320."""
    for aircraft in aircraft_list:
        # Debug logging to see what we're getting
        # Changed from "type" to ".//type"
        aircraft_type = aircraft.find(".//type")
        if aircraft_type is None or aircraft_type.text != "A320":
            logger.error(
                f"Invalid aircraft type: {aircraft_type.text if aircraft_type is not None else 'None'}. Expected: A320")
            return False
        # Added debug logging
        logger.debug(f"Found aircraft type: {aircraft_type.text}")
    return True


def validate_time_separation(aircraft_list: List[ET.Element]) -> Tuple[bool, List[str]]:
    """
    Validate if:
    1. First aircraft starts at time 100
    2. Aircraft are separated by 300 seconds
    """
    errors = []

    # Extract and sort times
    aircraft_times = []
    for aircraft in aircraft_list:
        # Changed from "time" to ".//time"
        time_elem = aircraft.find(".//time")
        if time_elem is not None:
            aircraft_times.append((aircraft, int(time_elem.text)))

    # Sort by time
    aircraft_times.sort(key=lambda x: x[1])

    # Validate times
    for idx, (aircraft, time) in enumerate(aircraft_times):
        logger.debug(f"Aircraft {idx+1} time: {time}")  # Debug logging

        # Check first aircraft time
        if idx == 0 and time != 100:
            errors.append(
                f"First aircraft should start at time 100, found: {time}")

        # Check time separation
        if idx > 0:
            prev_time = aircraft_times[idx-1][1]
            time_diff = time - prev_time
            if time_diff != 300:
                errors.append(f"Invalid time separation between aircraft {idx} and {idx+1}: "
                              f"{time_diff} seconds (expected 300)")

    return len(errors) == 0, errors


def validate_route_and_airports(aircraft: ET.Element, airways: Dict) -> bool:
    """Validate that the aircraft's route matches an airway and has correct airports."""
    # Extract route waypoints
    route_waypoints = [
        waypoint.text for waypoint in aircraft.findall("air_route")]

    # Find matching airway
    matching_airway = None
    for airway_name, airway_data in airways.items():
        if route_waypoints == airway_data['air_route']:
            matching_airway = airway_name
            break

    if matching_airway is None:
        logger.error(f"No matching airway found for route: {route_waypoints}")
        return False

    # Validate airports
    scenario_dep = aircraft.find(".//dep/af").text
    scenario_des = aircraft.find(".//des/af").text

    expected_dep = airways[matching_airway]['departure']['af']
    expected_des = airways[matching_airway]['destination']['af']

    if scenario_dep != expected_dep or scenario_des != expected_des:
        logger.error(f"Airport mismatch for airway {matching_airway}")
        logger.error(f"Expected: {expected_dep} -> {expected_des}")
        logger.error(f"Found: {scenario_dep} -> {scenario_des}")
        return False

    return True


def validate_scenario(file_path: str, airways: Dict) -> Dict:
    """Validate a single scenario file."""
    logger.info(f"Validating scenario: {file_path}")

    results = {
        "filename": os.path.basename(file_path),
        "aircraft_count_valid": False,
        "aircraft_count": 0,
        "all_a320": False,
        "time_separation_valid": False,
        "routes_valid": False,
        "time_errors": [],
        "all_valid": False
    }

    root = load_xml_scenario(file_path)
    if root is None:
        return results

    # Get all aircraft
    aircraft_list = root.findall(".//initial-flightplans")

    # Validate aircraft count
    results["aircraft_count_valid"], results["aircraft_count"] = validate_aircraft_count(
        root)

    # Validate aircraft types
    results["all_a320"] = validate_aircraft_types(aircraft_list)

    # Validate time separation
    results["time_separation_valid"], results["time_errors"] = validate_time_separation(
        aircraft_list)

    # Validate routes
    routes_valid = True
    for aircraft in aircraft_list:
        if not validate_route_and_airports(aircraft, airways):
            routes_valid = False
            break
    results["routes_valid"] = routes_valid

    # Overall validation
    results["all_valid"] = all([
        results["aircraft_count_valid"],
        results["all_a320"],
        results["time_separation_valid"],
        results["routes_valid"]
    ])

    return results


def generate_validation_report(all_results: List[Dict], save_path: str = None) -> None:
    """Generate and save validation report."""
    total_scenarios = len(all_results)
    valid_scenarios = sum(1 for r in all_results if r["all_valid"])
    valid_count = sum(1 for r in all_results if r["aircraft_count_valid"])
    valid_types = sum(1 for r in all_results if r["all_a320"])
    valid_times = sum(1 for r in all_results if r["time_separation_valid"])
    valid_routes = sum(1 for r in all_results if r["routes_valid"])

    report_content = [
        "Validation Summary",
        "=" * 50,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 50,
        f"Total scenarios processed: {total_scenarios}",
        f"Completely valid scenarios: {valid_scenarios} ({(valid_scenarios/total_scenarios)*100:.1f}%)",
        f"Scenarios with correct aircraft count: {valid_count} ({(valid_count/total_scenarios)*100:.1f}%)",
        f"Scenarios with all A320 aircraft: {valid_types} ({(valid_types/total_scenarios)*100:.1f}%)",
        f"Scenarios with correct time separation: {valid_times} ({(valid_times/total_scenarios)*100:.1f}%)",
        f"Scenarios with valid routes: {valid_routes} ({(valid_routes/total_scenarios)*100:.1f}%)",
        "",
        "Detailed Results",
        "-" * 20
    ]

    for result in all_results:
        report_content.extend([
            f"\nScenario: {result['filename']}",
            f"Aircraft Count: {'✓' if result['aircraft_count_valid'] else '✗'} ({result['aircraft_count']} aircraft)",
            f"All A320: {'✓' if result['all_a320'] else '✗'}",
            f"Time Separation: {'✓' if result['time_separation_valid'] else '✗'}"
        ])

        if not result["time_separation_valid"]:
            report_content.extend(
                [f"  - {error}" for error in result["time_errors"]])

        report_content.append(
            f"Routes Valid: {'✓' if result['routes_valid'] else '✗'}")
        report_content.append(
            f"Overall Status: {'VALID' if result['all_valid'] else 'INVALID'}")

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
    scenarios_dir = "generated_scenarios_level2"

    # Create reports directory if it doesn't exist
    reports_dir = "validation_reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # Generate timestamp for the report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"validation_report_level2{timestamp}.txt"
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
