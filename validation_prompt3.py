import os
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Tuple
from datetime import datetime
# Import airways dictionary from testing_metrics
from testing_metrics import airways

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
    """Validate if scenario has exactly 3 aircraft."""
    flightplans = root.findall(".//initial-flightplans")
    count = len(flightplans)
    is_valid = count == 3
    return is_valid, count


def find_matching_airway(route_waypoints: List[str]) -> str:
    """Find the airway that matches the route waypoints."""
    for airway_name, airway_data in airways.items():
        if route_waypoints == airway_data['air_route']:
            return airway_name
    return None


def validate_airways_and_timing(aircraft_list: List[ET.Element]) -> Tuple[bool, List[Dict]]:
    """
    Validate that aircraft are using the correct airways and proper timing.
    Returns:
    - bool: True if airways and timing are valid
    - List[Dict]: List of validation details for each aircraft
    """
    validation_details = []
    found_airways = set()
    required_airways = {'M758', 'N884', 'M761'}
    aircraft_times = []

    for i, aircraft in enumerate(aircraft_list, 1):
        route_waypoints = [
            waypoint.text for waypoint in aircraft.findall("air_route")]
        time = int(aircraft.find("time").text)
        aircraft_times.append(time)
        matching_airway = find_matching_airway(route_waypoints)

        details = {
            "aircraft_number": i,
            "route_waypoints": route_waypoints,
            "matching_airway": matching_airway,
            "time": time,
            "is_valid": True,
            "errors": []
        }

        if matching_airway:
            found_airways.add(matching_airway)
            # Validate airports
            dep = aircraft.find(".//dep/af").text
            des = aircraft.find(".//des/af").text
            expected_dep = airways[matching_airway]['departure']['af']
            expected_des = airways[matching_airway]['destination']['af']

            if dep != expected_dep:
                details["errors"].append(
                    f"Invalid departure airport. Expected: {expected_dep}, Found: {dep}")
            if des != expected_des:
                details["errors"].append(
                    f"Invalid destination airport. Expected: {expected_des}, Found: {des}")
        else:
            details["errors"].append(f"No matching airway found for waypoints")
            details["is_valid"] = False

        validation_details.append(details)

    # Validate airways set
    if found_airways != required_airways:
        for detail in validation_details:
            detail["errors"].append(
                f"Incorrect airway combination. Required: {required_airways}, Found: {found_airways}")
            detail["is_valid"] = False

    # Validate timing
    aircraft_times.sort()
    if aircraft_times[0] != 100:
        for detail in validation_details:
            detail["errors"].append(
                f"First aircraft should start at 100s, found {aircraft_times[0]}s")
            detail["is_valid"] = False

    for i in range(len(aircraft_times)-1):
        if aircraft_times[i+1] - aircraft_times[i] != 200:
            for detail in validation_details:
                detail["errors"].append(
                    f"Invalid time separation between aircraft {i+1} and {i+2}")
                detail["is_valid"] = False

    return all(detail["is_valid"] for detail in validation_details), validation_details


def validate_scenario(file_path: str) -> Dict:
    """Validate a single scenario file."""
    logger.info(f"Validating scenario: {file_path}")

    results = {
        "filename": os.path.basename(file_path),
        "aircraft_count_valid": False,
        "aircraft_count": 0,
        "airways_timing_valid": False,
        "all_valid": False,
        "validation_details": []
    }

    root = load_xml_scenario(file_path)
    if root is None:
        return results

    # Validate aircraft count
    results["aircraft_count_valid"], results["aircraft_count"] = validate_aircraft_count(
        root)

    # Validate airways and timing
    aircraft_list = root.findall(".//initial-flightplans")
    results["airways_timing_valid"], results["validation_details"] = validate_airways_and_timing(
        aircraft_list)

    # Overall validation
    results["all_valid"] = all([
        results["aircraft_count_valid"],
        results["airways_timing_valid"]
    ])

    return results


def generate_validation_report(all_results: List[Dict], save_path: str = None) -> None:
    """Generate and save validation report."""
    report_content = [
        "Scenario Validation Summary",
        "=" * 50,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 50,
        f"Total scenarios processed: {len(all_results)}",
        f"Completely valid scenarios: {sum(1 for r in all_results if r['all_valid'])} ({(sum(1 for r in all_results if r['all_valid'])/len(all_results))*100:.1f}%)"
    ]

    for result in all_results:
        report_content.extend([
            "",
            f"Scenario: {result['filename']}",
            f"Aircraft Count: {'✓' if result['aircraft_count_valid'] else '✗'} ({result['aircraft_count']} aircraft)",
            f"Airways and Timing: {'✓' if result['airways_timing_valid'] else '✗'}",
            "",
            "Validation Details:"
        ])

        for detail in result["validation_details"]:
            report_content.extend([
                "",
                f"Aircraft {detail['aircraft_number']}:",
                f"  Waypoints: {' -> '.join(detail['route_waypoints'])}",
                f"  Matching Airway: {detail['matching_airway']}",
                f"  Time: {detail['time']}s",
                f"  Valid: {'✓' if detail['is_valid'] else '✗'}"
            ])

            if detail["errors"]:
                report_content.append("  Errors:")
                for error in detail["errors"]:
                    report_content.append(f"    - {error}")

        report_content.extend([
            "",
            f"Overall Status: {'VALID' if result['all_valid'] else 'INVALID'}",
            "-" * 50
        ])

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
    # Directory containing scenario files
    # Update this path as needed
    scenarios_dir = "generated_scenarios_prompt3_7jan_temp0.5"

    # Create reports directory if it doesn't exist
    reports_dir = "validation_reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # Generate timestamp for the report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"validation_report_prompt3_7jan_temp0.5_{timestamp}.txt"
    report_path = os.path.join(reports_dir, report_filename)

    # Collect and validate all scenarios
    all_results = []
    scenario_files = [f for f in os.listdir(
        scenarios_dir) if f.endswith('.xdat')]

    for scenario_file in scenario_files:
        file_path = os.path.join(scenarios_dir, scenario_file)
        results = validate_scenario(file_path)
        all_results.append(results)

    # Generate validation report
    generate_validation_report(all_results, report_path)


if __name__ == "__main__":
    main()
