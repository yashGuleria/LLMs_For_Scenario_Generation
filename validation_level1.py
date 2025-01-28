import os
import xml.etree.ElementTree as ET
import logging
from typing import Dict, List, Tuple

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
        "all_valid": False
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
    for aircraft in aircraft_list:
        if not validate_route_and_airports(aircraft, airways):
            routes_valid = False
            break

    results["types_valid"] = types_valid
    results["routes_valid"] = routes_valid
    results["all_valid"] = all([
        results["aircraft_count_valid"],
        types_valid,
        routes_valid
    ])

    return results


def generate_validation_report(all_results: List[Dict]) -> None:
    """Generate and print validation report."""
    total_scenarios = len(all_results)
    valid_scenarios = sum(1 for r in all_results if r["all_valid"])
    valid_count = sum(1 for r in all_results if r["aircraft_count_valid"])
    valid_types = sum(1 for r in all_results if r["types_valid"])
    valid_routes = sum(1 for r in all_results if r["routes_valid"])

    logger.info("\nValidation Summary:")
    logger.info(f"Total scenarios processed: {total_scenarios}")
    logger.info(
        f"Completely valid scenarios: {valid_scenarios} ({(valid_scenarios/total_scenarios)*100:.1f}%)")
    logger.info(
        f"Scenarios with correct aircraft count: {valid_count} ({(valid_count/total_scenarios)*100:.1f}%)")
    logger.info(
        f"Scenarios with valid aircraft types: {valid_types} ({(valid_types/total_scenarios)*100:.1f}%)")
    logger.info(
        f"Scenarios with valid routes and airports: {valid_routes} ({(valid_routes/total_scenarios)*100:.1f}%)")

    # List invalid scenarios
    invalid_scenarios = [r["filename"]
                         for r in all_results if not r["all_valid"]]
    if invalid_scenarios:
        logger.info("\nInvalid scenarios:")
        for scenario in invalid_scenarios:
            logger.info(f"- {scenario}")


def main():
    # Import airways dictionary from testing_metrics.py
    from testing_metrics import airways

    # Directory containing scenario files
    scenarios_dir = "generated_scenarios_level1"

    # Collect and validate all scenarios
    all_results = []
    scenario_files = [f for f in os.listdir(
        scenarios_dir) if f.endswith('.xdat')]

    for scenario_file in scenario_files:
        file_path = os.path.join(scenarios_dir, scenario_file)
        results = validate_scenario(file_path, airways)
        all_results.append(results)

    # Generate validation report
    generate_validation_report(all_results)


if __name__ == "__main__":
    main()
