from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from typing import List, Tuple
import xml.etree.ElementTree as ET


@dataclass  # Now this will work
class Aircraft:
    callsign: str
    initial_lat: float
    initial_lon: float
    initial_alt: float
    initial_hdg: float
    final_alt: float
    time: int
    speed: float = 350


def parse_sector_coordinates(filepath: str) -> np.ndarray:
    """Parse sector coordinates from CSV file"""
    df = pd.read_csv(filepath)
    # Reshape the coordinates into pairs of lat/lon
    coords = df.values.reshape(-1, 2)
    return coords


def create_grid(bounds: dict, cell_size_nm: float = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Create a grid based on sector boundaries"""
    # Convert cell size from nautical miles to degrees (approximate)
    cell_size_deg = cell_size_nm / 60  # 1 degree ≈ 60 nm

    # Create grid lines
    lat_grid = np.arange(
        bounds['min_lat'], bounds['max_lat'] + cell_size_deg, cell_size_deg)
    lon_grid = np.arange(
        bounds['min_lon'], bounds['max_lon'] + cell_size_deg, cell_size_deg)

    return np.meshgrid(lon_grid, lat_grid)


def plot_2d_grid(sector_coords: np.ndarray, aircraft_list: List[Aircraft], save_path: str = None):
    """Create 2D visualization of the grid with sector and aircraft"""
    # Calculate bounds with some padding
    bounds = {
        'min_lat': np.min(sector_coords[:, 0]) - 1,
        'max_lat': np.max(sector_coords[:, 0]) + 1,
        'min_lon': np.min(sector_coords[:, 1]) - 1,
        'max_lon': np.max(sector_coords[:, 1]) + 1
    }

    # Create grid
    lon_grid, lat_grid = create_grid(bounds)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot grid
    plt.plot(lon_grid, lat_grid, 'k-', alpha=0.2)
    plt.plot(lon_grid.T, lat_grid.T, 'k-', alpha=0.2)

    # Plot sector boundary
    sector_coords_closed = np.vstack(
        [sector_coords, sector_coords[0]])  # Close the polygon
    plt.plot(sector_coords_closed[:, 1], sector_coords_closed[:,
             0], 'r-', linewidth=2, label='Sector Boundary')

    # Plot aircraft positions
    if aircraft_list:
        ac_lats = [ac.initial_lat for ac in aircraft_list]
        ac_lons = [ac.initial_lon for ac in aircraft_list]
        plt.scatter(ac_lons, ac_lats, c='blue',
                    marker='^', s=100, label='Aircraft')

        # Add aircraft callsigns as labels
        for ac in aircraft_list:
            plt.annotate(ac.callsign,
                         (ac.initial_lon, ac.initial_lat),
                         xytext=(5, 5), textcoords='offset points')

    # Customize plot
    plt.grid(True, alpha=0.3)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Airspace Grid with Sector Boundary and Aircraft Positions')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_3d_grid(sector_coords: np.ndarray, aircraft_list: List[Aircraft], save_path: str = None):
    """Create 3D visualization of the grid with sector and aircraft"""
    # Calculate bounds with some padding
    bounds = {
        'min_lat': np.min(sector_coords[:, 0]) - 1,
        'max_lat': np.max(sector_coords[:, 0]) + 1,
        'min_lon': np.min(sector_coords[:, 1]) - 1,
        'max_lon': np.max(sector_coords[:, 1]) + 1
    }

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create grid
    lon_grid, lat_grid = create_grid(bounds)

    # Create altitude levels (FL85 to FL415)
    alt_levels = np.arange(85, 415, 30)  # 3000ft steps

    # Plot vertical grid lines
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            ax.plot([lon_grid[i, j], lon_grid[i, j]],
                    [lat_grid[i, j], lat_grid[i, j]],
                    [85, 415], 'k-', alpha=0.1)

    # Plot horizontal grid at each flight level
    for alt in alt_levels:
        ax.plot_surface(lon_grid, lat_grid,
                        np.ones_like(lon_grid) * alt,
                        alpha=0.1, shade=False)

    # Plot sector boundary at bottom and top
    sector_coords_closed = np.vstack([sector_coords, sector_coords[0]])
    ax.plot(sector_coords_closed[:, 1], sector_coords_closed[:, 0],
            [85] * len(sector_coords_closed), 'r-', linewidth=2)
    ax.plot(sector_coords_closed[:, 1], sector_coords_closed[:, 0],
            [415] * len(sector_coords_closed), 'r-', linewidth=2)

    # Plot aircraft positions
    if aircraft_list:
        ac_lats = [ac.initial_lat for ac in aircraft_list]
        ac_lons = [ac.initial_lon for ac in aircraft_list]
        # Convert to flight levels
        ac_alts = [ac.initial_alt/100 for ac in aircraft_list]

        ax.scatter(ac_lons, ac_lats, ac_alts,
                   c='blue', marker='^', s=100, label='Aircraft')

        # Add aircraft callsigns as labels
        for ac in aircraft_list:
            ax.text(ac.initial_lon, ac.initial_lat, ac.initial_alt/100,
                    ac.callsign, size=8)

    # Customize plot
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Flight Level')
    ax.set_title(
        '3D Airspace Grid with Sector Boundary and Aircraft Positions')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def parse_coordinates(lat_str: str, lon_str: str) -> Tuple[float, float]:
    """Convert coordinates from DDMMSS.SS format to decimal degrees"""
    lat_deg = float(lat_str[:2])
    lat_min = float(lat_str[2:4])
    lat_sec = float(lat_str[4:6] + "." + lat_str[7:9])

    lon_deg = float(lon_str[:3])
    lon_min = float(lon_str[3:5])
    lon_sec = float(lon_str[5:7] + "." + lon_str[8:10])

    lat = lat_deg + lat_min/60 + lat_sec/3600
    lon = lon_deg + lon_min/60 + lon_sec/3600

    if 'S' in lat_str:
        lat = -lat
    if 'W' in lon_str:
        lon = -lon

    return lat, lon


def visualize_airspace(xml_content: str, sector_coords_file: str):
    """Main function to create both 2D and 3D visualizations"""
    # Parse aircraft data
    root = ET.fromstring(xml_content)
    aircraft_list = []

    for fp in root.findall('.//initial-flightplans'):
        time = int(fp.find('time').text)
        callsign = fp.find('callsign').text
        init = fp.find('init')

        lat_str = init.find('pos/lat').text
        lon_str = init.find('pos/lon').text
        lat, lon = parse_coordinates(lat_str, lon_str)

        initial_alt = float(init.find('alt').text)
        initial_hdg = float(init.find('hdg').text)
        final_alt = float(fp.find('rfl').text)

        aircraft = Aircraft(
            callsign=callsign,
            initial_lat=lat,
            initial_lon=lon,
            initial_alt=initial_alt,
            initial_hdg=initial_hdg,
            final_alt=final_alt,
            time=time
        )
        aircraft_list.append(aircraft)

    # Parse sector coordinates
    sector_coords = parse_sector_coordinates(sector_coords_file)

    # Create visualizations
    plot_2d_grid(sector_coords, aircraft_list, 'airspace_2d.png')
    plot_3d_grid(sector_coords, aircraft_list, 'airspace_3d.png')


# Example usage:
if __name__ == "__main__":
    # Read the XML content
    with open('complexScenario2.xdat', 'r') as f:
        xml_content = f.read()

    # Create visualizations
    visualize_airspace(xml_content, 'sector6coords.csv')
