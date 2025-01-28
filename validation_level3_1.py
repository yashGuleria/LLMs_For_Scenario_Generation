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
    """Validate if scenario has Type 1 aircraft count (1-4 aircraft)."""
    flightplans = root.findall(".//initial-flightplans")
    count = len(flightplans)
    is_valid = 1 <= count <= 4  # Modified for Type 1 requirement
    if not is_valid:
        logger.error(f"Type 1 scenario requires 1-4 aircraft, found {count}")
    return is_valid, count


def validate_aircraft_type(aircraft: ET.Element) -> bool:
    """Validate if aircraft type is in the allowed set."""
    aircraft_type = aircraft.find("type").text
    valid_types = {'A320', 'B737', 'B738', 'B734', 'B744', 'A388', 'A333'}

    if aircraft_type not in valid_types:
        logger.error(f"Invalid aircraft type: {aircraft_type}")
        return False
    return True


def validate_same_airway(aircraft_list: List[ET.Element], airways: Dict) -> Tuple[bool, str, List[Dict]]:
    """
    Validate that all aircraft are on the same airway.
    Returns:
    - bool: True if all aircraft are on the same valid airway
    - str: The common airway name if found, None otherwise
    - List[Dict]: List of validation details for each aircraft
    """
    validation_details = []
    common_airway = None
    is_valid = True

    for i, aircraft in enumerate(aircraft_list, 1):
        route_waypoints = [
            waypoint.text for waypoint in aircraft.findall("air_route")]
        matching_airway = find_matching_airway(route_waypoints, airways)

        details = {
            "aircraft_number": i,
            "route_waypoints": route_waypoints,
            "matching_airway": matching_airway,
            "is_valid": True,
            "errors": []
        }

        if matching_airway is None:
            details["is_valid"] = False
            details["errors"].append(
                f"No matching airway found for route: {route_waypoints}")
            is_valid = False
        elif common_airway is None:
            common_airway = matching_airway
        elif matching_airway != common_airway:
            details["is_valid"] = False
            details["errors"].append(
                f"Aircraft not on common airway. Expected: {common_airway}, Found: {matching_airway}")
            is_valid = False

        validation_details.append(details)

    return is_valid, common_airway, validation_details


def find_matching_airway(route_waypoints: List[str], airways: Dict) -> str:
    """Find the airway that matches the route waypoints."""
    for airway_name, airway_data in airways.items():
        if route_waypoints == airway_data['air_route']:
            return airway_name
    return None


def validate_airports(aircraft: ET.Element, airway_name: str, airways: Dict) -> Tuple[bool, Dict]:
    """Validate departure and destination airports match the airway."""
    scenario_dep = aircraft.find(".//dep/af").text
    scenario_des = aircraft.find(".//des/af").text

    validation_info = {
        "departure": scenario_dep,
        "destination": scenario_des,
        "is_valid": True,
        "errors": []
    }

    expected_dep = airways[airway_name]['departure']['af']
    expected_des = airways[airway_name]['destination']['af']

    if scenario_dep != expected_dep:
        validation_info["is_valid"] = False
        validation_info["errors"].append(
            f"Invalid departure airport. Expected: {expected_dep}, Found: {scenario_dep}")

    if scenario_des != expected_des:
        validation_info["is_valid"] = False
        validation_info["errors"].append(
            f"Invalid destination airport. Expected: {expected_des}, Found: {scenario_des}")

    return validation_info["is_valid"], validation_info


def validate_scenario(file_path: str, airways: Dict) -> Dict:
    """Validate a single scenario file for Type 1 requirements."""
    logger.info(f"Validating scenario: {file_path}")

    results = {
        "filename": os.path.basename(file_path),
        "aircraft_count_valid": False,
        "aircraft_count": 0,
        "types_valid": False,
        "same_airway_valid": False,
        "common_airway": None,
        "airports_valid": False,
        "all_valid": False,
        "validation_details": []
    }

    root = load_xml_scenario(file_path)
    if root is None:
        return results

    # Validate aircraft count
    results["aircraft_count_valid"], results["aircraft_count"] = validate_aircraft_count(
        root)

    aircraft_list = root.findall(".//initial-flightplans")

    # Validate aircraft types
    results["types_valid"] = all(validate_aircraft_type(
        aircraft) for aircraft in aircraft_list)

    # Validate same airway requirement
    results["same_airway_valid"], results["common_airway"], airway_validation = validate_same_airway(
        aircraft_list, airways)

    # Validate airports for each aircraft
    all_airports_valid = True
    for i, aircraft in enumerate(aircraft_list):
        validation_detail = airway_validation[i]

        if validation_detail["matching_airway"]:
            airports_valid, airports_info = validate_airports(
                aircraft, validation_detail["matching_airway"], airways)
            validation_detail.update(airports_info)

            if not airports_valid:
                all_airports_valid = False

    results["airports_valid"] = all_airports_valid
    results["validation_details"] = airway_validation

    # Overall validation
    results["all_valid"] = all([
        results["aircraft_count_valid"],
        results["types_valid"],
        results["same_airway_valid"],
        results["airports_valid"]
    ])

    return results


def generate_validation_report(all_results: List[Dict], save_path: str = None) -> None:
    """Generate and save validation report."""
    report_content = [
        "Type 1 Scenario Validation Summary",
        "=" * 50,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 50,
        f"Total scenarios processed: {len(all_results)}",
        f"Completely valid scenarios: {sum(1 for r in all_results if r['all_valid'])} ({(sum(1 for r in all_results if r['all_valid'])/len(all_results))*100:.1f}%)",
        f"Scenarios with correct aircraft count (1-4): {sum(1 for r in all_results if r['aircraft_count_valid'])} ({(sum(1 for r in all_results if r['aircraft_count_valid'])/len(all_results))*100:.1f}%)",
        f"Scenarios with all aircraft on same airway: {sum(1 for r in all_results if r['same_airway_valid'])} ({(sum(1 for r in all_results if r['same_airway_valid'])/len(all_results))*100:.1f}%)",
        "",
        "Detailed Results",
        "-" * 20
    ]

    for result in all_results:
        report_content.extend([
            f"\nScenario: {result['filename']}",
            f"Aircraft Count: {'✓' if result['aircraft_count_valid'] else '✗'} ({result['aircraft_count']} aircraft)",
            f"Common Airway: {'✓' if result['same_airway_valid'] else '✗'} ({result['common_airway'] if result['common_airway'] else 'None'})",
            f"Valid Aircraft Types: {'✓' if result['types_valid'] else '✗'}",
            f"Valid Airports: {'✓' if result['airports_valid'] else '✗'}"
        ])

        report_content.append("\nValidation Details:")
        for detail in result["validation_details"]:
            report_content.extend([
                f"\nAircraft {detail['aircraft_number']}:",
                f"  Waypoints: {' -> '.join(detail['route_waypoints'])}",
                f"  Matching Airway: {detail['matching_airway'] if detail['matching_airway'] else 'None'}",
                f"  Departure: {detail.get('departure', 'N/A')}",
                f"  Destination: {detail.get('destination', 'N/A')}",
                f"  Valid: {'✓' if detail['is_valid'] else '✗'}"
            ])

            if detail["errors"]:
                report_content.append("  Errors:")
                for error in detail["errors"]:
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
    # Updated directory name
    scenarios_dir = "generated_scenarios_prompt7_3_1_7jan_temp0.8_modeified_system_2ndAPI_2"

    # Create reports directory if it doesn't exist
    reports_dir = "validation_reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # Generate timestamp for the report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Updated filename
    report_filename = f"validation_report_prompt7_3_1_7jan_temp0.8_modeified_system_2ndAPI_2_{timestamp}.txt"
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
