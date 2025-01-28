import os
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Tuple
from datetime import datetime

"""Generate 7 aircraft in sector 6. Out of these, 4 aircraft should be on airway M758 and 3 aircraft should be on airway N892. Ensure to use correct airways and their corresponding departure and destinations."""

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


def validate_airway_distribution(aircraft_list: List[ET.Element], airways: Dict) -> Tuple[bool, Dict, List[str]]:
    """
    Validate if:
    1. 4 aircraft are on M758
    2. 3 aircraft are on N892
    3. Airways match with correct departure/destination pairs
    """
    errors = []
    airway_count = {"M758": 0, "N884": 0, "other": 0}

    for aircraft in aircraft_list:
        # Extract route waypoints
        route_waypoints = [
            waypoint.text for waypoint in aircraft.findall(".//air_route")]

        # Find matching airway
        matching_airway = None
        for airway_name, airway_data in airways.items():
            if route_waypoints == airway_data['air_route']:
                matching_airway = airway_name
                break

        if matching_airway is None:
            errors.append(
                f"No matching airway found for route: {route_waypoints}")
            airway_count["other"] += 1
            continue

        # Count the airway usage
        if matching_airway in ["M758", "N884"]:
            airway_count[matching_airway] += 1
        else:
            airway_count["other"] += 1
            errors.append(f"Invalid airway used: {matching_airway}")

        # Validate airports for the airway
        scenario_dep = aircraft.find(".//dep/af").text
        scenario_des = aircraft.find(".//des/af").text

        expected_dep = airways[matching_airway]['departure']['af']
        expected_des = airways[matching_airway]['destination']['af']

        if scenario_dep != expected_dep or scenario_des != expected_des:
            errors.append(f"Airport mismatch for airway {matching_airway}")
            errors.append(f"Expected: {expected_dep} -> {expected_des}")
            errors.append(f"Found: {scenario_dep} -> {scenario_des}")

    # Validate airway distribution
    is_valid = (airway_count["M758"] == 4 and
                airway_count["N884"] == 3 and
                airway_count["other"] == 0)

    if not is_valid:
        if airway_count["M758"] != 4:
            errors.append(
                f"Expected 4 aircraft on M758, found {airway_count['M758']}")
        if airway_count["N884"] != 3:
            errors.append(
                f"Expected 3 aircraft on N884, found {airway_count['N884']}")
        if airway_count["other"] > 0:
            errors.append(
                f"Found {airway_count['other']} aircraft on invalid airways")

    return is_valid, airway_count, errors


def validate_scenario(file_path: str, airways: Dict) -> Dict:
    """Validate a single scenario file."""
    logger.info(f"Validating scenario: {file_path}")

    results = {
        "filename": os.path.basename(file_path),
        "aircraft_count_valid": False,
        "aircraft_count": 0,
        "airway_distribution_valid": False,
        "airway_counts": {},
        "distribution_errors": [],
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

    # Validate airway distribution and routes
    results["airway_distribution_valid"], results["airway_counts"], results["distribution_errors"] = (
        validate_airway_distribution(aircraft_list, airways)
    )

    # Overall validation
    results["all_valid"] = all([
        results["aircraft_count_valid"],
        results["airway_distribution_valid"]
    ])

    return results


def generate_validation_report(all_results: List[Dict], save_path: str = None) -> None:
    """Generate and save validation report."""
    total_scenarios = len(all_results)
    valid_scenarios = sum(1 for r in all_results if r["all_valid"])
    valid_count = sum(1 for r in all_results if r["aircraft_count_valid"])
    valid_distribution = sum(
        1 for r in all_results if r["airway_distribution_valid"])

    report_content = [
        "Validation Summary",
        "=" * 50,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 50,
        f"Total scenarios processed: {total_scenarios}",
        f"Completely valid scenarios: {valid_scenarios} ({(valid_scenarios/total_scenarios)*100:.1f}%)",
        f"Scenarios with correct aircraft count: {valid_count} ({(valid_count/total_scenarios)*100:.1f}%)",
        f"Scenarios with correct airway distribution: {valid_distribution} ({(valid_distribution/total_scenarios)*100:.1f}%)",
        "",
        "Detailed Results",
        "-" * 20
    ]

    for result in all_results:
        report_content.extend([
            f"\nScenario: {result['filename']}",
            f"Aircraft Count: {'✓' if result['aircraft_count_valid'] else '✗'} ({result['aircraft_count']} aircraft)",
            f"Airway Distribution: {'✓' if result['airway_distribution_valid'] else '✗'}"
        ])

        if result["airway_counts"]:
            report_content.append("Airway Counts:")
            for airway, count in result["airway_counts"].items():
                report_content.append(f"  - {airway}: {count}")

        if not result["airway_distribution_valid"]:
            report_content.extend(
                [f"  - {error}" for error in result["distribution_errors"]])

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
    report_filename = f"validation_report_{timestamp}.txt"
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
