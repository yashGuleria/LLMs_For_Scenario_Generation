import ast  # For safely evaluating string representations of lists
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

# Load Data


def load_sector_data(filepath):
    data = pd.read_csv(filepath, header=None)
    return data.values[:, 0], data.values[:, 1]


def load_airways_data(filepath):
    return pd.read_csv(filepath)


def parse_xdat(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    aircraft_data = []

    for flightplan in root.findall(".//initial-flightplans//flightplan"):
        aircraft_id = flightplan.find("aircraft-id").text
        route = flightplan.find("route").text.strip(
            "[]").replace("'", "").split(", ")  # Waypoints
        lat = float(flightplan.find("latitude").text)
        lon = float(flightplan.find("longitude").text)
        altitude = float(flightplan.find("altitude").text)
        aircraft_data.append({
            'id': aircraft_id,
            'route': route,
            'lat': lat,
            'lon': lon,
            'altitude': altitude
        })
    return aircraft_data

# Haversine formula for distance calculation


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Parse waypoints safely


def parse_air_route(route_str):
    try:
        waypoints = ast.literal_eval(route_str)
        if isinstance(waypoints, list):
            return waypoints
        else:
            print(f"Unexpected format in Air Route: {route_str}")
            return []
    except Exception as e:
        print(f"Error parsing Air Route: {e}")
        return []

# Convert coordinate strings with directional indicators to decimal degrees


def parse_coordinates(coord):
    try:
        # Check if the coordinate ends with a directional indicator
        if coord[-1] in ['N', 'S', 'E', 'W']:
            value = float(coord[:-1])  # Remove the directional indicator
            if coord[-1] == 'S' or coord[-1] == 'W':  # South and West are negative
                value = -value
            return value
        return float(coord)  # If no directional indicator, parse as float
    except ValueError:
        print(f"Error parsing coordinate: {coord}")
        return None


# Generate route coordinates

# Generate route coordinates
def generate_route_coordinates(waypoints, airways_data):
    coordinates = []
    for wp in waypoints:
        wp = wp.strip()  # Remove extra whitespace
        match = airways_data[airways_data['Waypoint 1'] == wp]
        if not match.empty:
            lat = parse_coordinates(match.iloc[0]['Waypoint 1 Latitude'])
            lon = parse_coordinates(match.iloc[0]['Waypoint 1 Longitude'])
            if lat is not None and lon is not None:
                coordinates.append((lat, lon))
        else:
            print(f"Waypoint '{wp}' not found in airways_data.")
    return coordinates


# Visualization Setup
sector_lat, sector_lon = load_sector_data('sector6coords.csv')
airways_data = load_airways_data('airways_data.csv')
traffic_data = parse_xdat('complexScenario2.xdat')

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(sector_lon, sector_lat, label="Sector Boundary", color='blue')

# Plot Airways
for _, row in airways_data.iterrows():
    waypoints = parse_air_route(row['Air Route'])
    coordinates = generate_route_coordinates(waypoints, airways_data)
    if coordinates:
        lats, lons = zip(*coordinates)
        ax.plot(lons, lats, linestyle='--', label=f"Airway {row['Airway']}")

# Initialize Aircraft Plots
aircraft_plots = {}
for aircraft in traffic_data:
    plot, = ax.plot([], [], 'ro', label=aircraft['id'])
    aircraft_plots[aircraft['id']] = plot

# Update Function for Animation


def update(frame):
    for aircraft in traffic_data:
        current_lat = aircraft['lat'] + frame * \
            0.01  # Mock update for movement
        current_lon = aircraft['lon'] + frame * 0.01
        aircraft_plots[aircraft['id']].set_data(current_lon, current_lat)
    return aircraft_plots.values()


# Animate
ani = FuncAnimation(fig, update, frames=100, interval=1000, blit=True)
plt.legend()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Aircraft Movement Simulation")
plt.show()
