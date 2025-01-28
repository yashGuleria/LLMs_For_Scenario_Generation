import os
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class AircraftError:
    aircraft_index: int
    aircraft_type: Optional[str] = None
    route_waypoints: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class ScenarioValidationResult:
    filename: str
    aircraft_count_valid: bool
    aircraft_count: int
    types_valid: bool
    routes_valid: bool
    all_valid: bool
    aircraft_errors: List[AircraftError] = field(default_factory=list)
    general_errors: List[str] = field(default_factory=list)


def load_xml_scenario(file_path: str) -> Optional[ET.Element]:
    """Load and parse XML scenario file."""
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except Exception as e:
        logger.error(f"Error parsing XML file {file_path}: {str(e)}")
        return None


def validate_aircraft_count(root: ET.Element) -> Tuple[bool, int, List[str]]:
    """Validate if scenario has exactly 3 aircraft."""
    errors = []
    flightplans = root.findall(".//initial-flightplans")
    count = len(flightplans)
    is_valid = count == 3

    if not is_valid:
        errors.append(
            f"Invalid aircraft count: Expected 3 aircraft, found {count}")

    return is_valid, count, errors


def validate_aircraft_type(aircraft: ET.Element, aircraft_index: int) -> Tuple[bool, str, List[str]]:
    """Validate if aircraft type is in the allowed set."""
    errors = []
    aircraft_type = aircraft.find("type").text
    valid_types = {'A320', 'B737', 'B738', 'B734', 'B744', 'A388', 'A333'}

    if aircraft_type not in valid_types:
        errors.append(
            f"Aircraft {aircraft_index + 1}: Invalid aircraft type '{aircraft_type}'")
        return False, aircraft_type, errors

    return True, aircraft_type, errors


def validate_route_and_airports(
    aircraft: ET.Element,
    airways: Dict,
    aircraft_index: int
) -> Tuple[bool, List[str], List[str]]:
    """Validate aircraft's route, matching airway, and airports."""
    errors = []
    route_waypoints = [
        waypoint.text for waypoint in aircraft.findall("air_route")]

    # Find matching airway
    matching_airway = None
    for airway_name, airway_data in airways.items():
        if route_waypoints == airway_data['air_route']:
            matching_airway = airway_name
            break

    if matching_airway is None:
        errors.append(
            f"Aircraft {aircraft_index + 1}: No matching airway found for route: {route_waypoints}")
        return False, route_waypoints, errors

    # Validate airports
    scenario_dep = aircraft.find(".//dep/af").text
    scenario_des = aircraft.find(".//des/af").text
    expected_dep = airways[matching_airway]['departure']['af']
    expected_des = airways[matching_airway]['destination']['af']

    if scenario_dep != expected_dep:
        errors.append(
            f"Aircraft {aircraft_index + 1}: Invalid departure airport for airway {matching_airway}. "
            f"Expected: {expected_dep}, Found: {scenario_dep}"
        )

    if scenario_des != expected_des:
        errors.append(
            f"Aircraft {aircraft_index + 1}: Invalid destination airport for airway {matching_airway}. "
            f"Expected: {expected_des}, Found: {scenario_des}"
        )

    is_valid = len(errors) == 0
    return is_valid, route_waypoints, errors


def validate_scenario(file_path: str, airways: Dict) -> ScenarioValidationResult:
    """Validate a single scenario file with detailed error reporting."""
    logger.info(f"Validating scenario: {file_path}")

    result = ScenarioValidationResult(
        filename=os.path.basename(file_path),
        aircraft_count_valid=False,
        aircraft_count=0,
        types_valid=True,
        routes_valid=True,
        all_valid=False,
        aircraft_errors=[],
        general_errors=[]
    )

    root = load_xml_scenario(file_path)
    if root is None:
        result.general_errors.append("Failed to parse XML file")
        return result

    # Validate aircraft count
    result.aircraft_count_valid, result.aircraft_count, count_errors = validate_aircraft_count(
        root)
    result.general_errors.extend(count_errors)

    # Validate each aircraft
    aircraft_list = root.findall(".//initial-flightplans")

    for idx, aircraft in enumerate(aircraft_list):
        aircraft_error = AircraftError(aircraft_index=idx + 1)

        # Validate aircraft type
        type_valid, aircraft_type, type_errors = validate_aircraft_type(
            aircraft, idx)
        if not type_valid:
            result.types_valid = False
        aircraft_error.aircraft_type = aircraft_type
        aircraft_error.errors.extend(type_errors)

        # Validate route and airports
        route_valid, route_waypoints, route_errors = validate_route_and_airports(
            aircraft, airways, idx)
        if not route_valid:
            result.routes_valid = False
        aircraft_error.route_waypoints = route_waypoints
        aircraft_error.errors.extend(route_errors)

        if len(aircraft_error.errors) > 0:
            result.aircraft_errors.append(aircraft_error)

    result.all_valid = all([
        result.aircraft_count_valid,
        result.types_valid,
        result.routes_valid
    ])

    return result


def generate_enhanced_validation_report(all_results: List[ScenarioValidationResult], save_path: str = None) -> None:
    """Generate and save detailed validation report."""
    total_scenarios = len(all_results)
    valid_scenarios = sum(1 for r in all_results if r.all_valid)

    report_content = [
        "Enhanced Validation Summary",
        "=" * 70,
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 70,
        f"Total scenarios processed: {total_scenarios}",
        f"Completely valid scenarios: {valid_scenarios} ({(valid_scenarios/total_scenarios)*100:.1f}%)",
        "",
        "Detailed Results by Scenario",
        "=" * 70,
        ""
    ]

    # Add detailed results for each scenario
    for result in all_results:
        scenario_header = f"Scenario: {result.filename}"
        report_content.extend([
            scenario_header,
            "-" * len(scenario_header),
            f"Status: {'Valid' if result.all_valid else 'Invalid'}",
            f"Aircraft Count: {result.aircraft_count} {'(Valid)' if result.aircraft_count_valid else '(Invalid)'}",
            ""
        ])

        # Add general errors if any
        if result.general_errors:
            report_content.extend([
                "General Errors:",
                *[f"  - {error}" for error in result.general_errors],
                ""
            ])

        # Add aircraft-specific errors if any
        if result.aircraft_errors:
            report_content.append("Aircraft-specific Errors:")
            for aircraft_error in result.aircraft_errors:
                report_content.extend([
                    f"  Aircraft {aircraft_error.aircraft_index}:",
                    f"    Type: {aircraft_error.aircraft_type}",
                    f"    Route: {' -> '.join(aircraft_error.route_waypoints)}",
                    *[f"    - {error}" for error in aircraft_error.errors],
                    ""
                ])

        report_content.append("-" * 70 + "\n")

    # Print to console and save to file
    for line in report_content:
        logger.info(line)

    if save_path:
        try:
            with open(save_path, 'w') as f:
                f.write('\n'.join(report_content))
            logger.info(f"\nDetailed report saved to: {save_path}")
        except Exception as e:
            logger.error(f"Error saving report to file: {str(e)}")


def main():
    # Import airways dictionary from testing_metrics.py
    from testing_metrics import airways

    # Directory containing scenario files
    scenarios_dir = "generated_scenarios_level1_2_temp_0.8"

    # Create reports directory if it doesn't exist
    reports_dir = "validation_reports"
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # Generate timestamp for the report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"detailed_validation_report_level1_2_temp_0.8_{timestamp}.txt"
    report_path = os.path.join(reports_dir, report_filename)

    # Collect and validate all scenarios
    all_results = []
    scenario_files = [f for f in os.listdir(
        scenarios_dir) if f.endswith('.xdat')]

    for scenario_file in scenario_files:
        file_path = os.path.join(scenarios_dir, scenario_file)
        results = validate_scenario(file_path, airways)
        all_results.append(results)

    # Generate enhanced validation report
    generate_enhanced_validation_report(all_results, report_path)


if __name__ == "__main__":
    main()
