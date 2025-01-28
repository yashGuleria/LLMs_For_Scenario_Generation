import pandas as pd
import numpy as np
import json
import os
from math import radians, degrees, cos, sin, asin, sqrt, atan2
from datetime import datetime


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in nautical miles"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    nm = 3440.065  # Earth's radius in nautical miles

    return c * nm


def interpolate_position(lat1, lon1, lat2, lon2, fraction):
    """Interpolate position between two points using great circle path"""
    # If points are the same, return the same position
    if abs(lat1 - lat2) < 1e-10 and abs(lon1 - lon2) < 1e-10:
        return lat1, lon1

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    d = 2 * asin(sqrt(sin((lat2-lat1)/2)**2 +
                      cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2))

    # If distance is very small, return direct linear interpolation
    if abs(d) < 1e-10:
        lat_new = lat1 + fraction * (lat2 - lat1)
        lon_new = lon1 + fraction * (lon2 - lon1)
        return degrees(lat_new), degrees(lon_new)

    A = sin((1-fraction)*d)/sin(d)
    B = sin(fraction*d)/sin(d)

    x = A * cos(lat1) * cos(lon1) + B * cos(lat2) * cos(lon2)
    y = A * cos(lat1) * sin(lon1) + B * cos(lat2) * sin(lon2)
    z = A * sin(lat1) + B * sin(lat2)

    lat_new = atan2(z, sqrt(x**2 + y**2))
    lon_new = atan2(y, x)

    return degrees(lat_new), degrees(lon_new)


def get_climb_rate(aircraft_type):
    """Return climb/descent rate based on aircraft type"""
    climb_rates = {
        'B738': 2178,
        'B744': 2153
    }
    return climb_rates.get(aircraft_type, 2000)


def generate_trajectory(row):
    """Generate trajectory for a single aircraft with detailed distance metrics"""
    # Parse waypoints
    waypoints = json.loads(row['waypoints'].replace("'", '"'))

    # Aircraft parameters
    speed_knots = 350
    speed_nm_sec = speed_knots / 3600  # Convert to NM/sec
    climb_rate = get_climb_rate(row['aircraft_type'])
    climb_rate_per_sec = climb_rate / 60  # Convert from ft/min to ft/sec

    initial_fl = float(row['initial_flight_level']) * 100  # Convert FL to feet
    final_fl = float(row['final_flight_level']) * 100

    # Calculate total route distance and segment distances
    total_distance = 0
    segment_distances = []
    for i in range(len(waypoints) - 1):
        dist = haversine_distance(
            waypoints[i]['latitude'], waypoints[i]['longitude'],
            waypoints[i+1]['latitude'], waypoints[i+1]['longitude']
        )
        segment_distances.append(dist)
        total_distance += dist

    trajectories = []
    current_time = int(row['start_time'])
    current_alt = initial_fl
    cumulative_distance = 0

    # Process each waypoint pair
    for i in range(len(waypoints) - 1):
        wp1 = waypoints[i]
        wp2 = waypoints[i + 1]
        segment_name = f"{wp1['name']}-{wp2['name']}"
        segment_distance = segment_distances[i]

        # Calculate time needed for this segment
        segment_time = segment_distance / speed_nm_sec
        num_intervals = int(segment_time / 10)  # 10-second intervals

        for j in range(num_intervals + 1):
            fraction = j / num_intervals if num_intervals > 0 else 1

            # Interpolate position
            lat, lon = interpolate_position(
                wp1['latitude'], wp1['longitude'],
                wp2['latitude'], wp2['longitude'],
                fraction
            )

            # Calculate distance metrics
            distance_in_segment = segment_distance * fraction
            current_cumulative_distance = cumulative_distance + distance_in_segment

            # Handle altitude changes
            if current_alt != final_fl:
                alt_change = climb_rate_per_sec * 10  # Change for 10-second interval
                if current_alt < final_fl:
                    current_alt = min(current_alt + alt_change, final_fl)
                else:
                    current_alt = max(current_alt - alt_change, final_fl)

            # Store trajectory point with detailed metrics
            trajectories.append({
                'timestamp': current_time,
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'altitude': round(current_alt, 2),
                'waypoint_segment': segment_name,
                'segment_distance_nm': round(segment_distance, 2),
                'distance_in_segment_nm': round(distance_in_segment, 2),
                'cumulative_distance_nm': round(current_cumulative_distance, 2),
                'total_route_distance_nm': round(total_distance, 2)
            })

            current_time += 10

        cumulative_distance += segment_distance

    # Add final waypoint
    trajectories.append({
        'timestamp': current_time,
        'latitude': wp2['latitude'],
        'longitude': wp2['longitude'],
        'altitude': final_fl,
        'waypoint_segment': segment_name,
        'segment_distance_nm': round(segment_distances[-1], 2),
        'distance_in_segment_nm': round(segment_distances[-1], 2),
        'cumulative_distance_nm': round(total_distance, 2),
        'total_route_distance_nm': round(total_distance, 2)
    })

    return pd.DataFrame(trajectories)


def main():
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'aircraft_trajectories_24jan'
    os.makedirs(output_dir, exist_ok=True)

    # Read input data
    df = pd.read_csv('merged_aircraft_airways_24jan.csv')

    # Store trajectory information
    trajectory_info = {}

    # Generate trajectories for each aircraft
    for _, row in df.iterrows():
        trajectory_df = generate_trajectory(row)

        # Save trajectory to CSV
        output_file = os.path.join(output_dir,
                                   f"{row['callsign']}_{row['aircraft_type']}_trajectory.csv")
        trajectory_df.to_csv(output_file, index=False)

        # Store information for summary
        trajectory_info[row['callsign']] = {
            'aircraft_type': row['aircraft_type'],
            'points': len(trajectory_df),
            'total_distance': trajectory_df.iloc[0]['total_route_distance_nm'],
            'output_file': output_file
        }

        print(
            f"\nGenerated trajectory for {row['callsign']} ({row['aircraft_type']})")
        print(
            f"Total distance: {trajectory_info[row['callsign']]['total_distance']} nm")
        print(
            f"Number of points: {trajectory_info[row['callsign']]['points']}")
        print(f"Saved to: {output_file}")

    # Print summary
    print("\nTrajectory Generation Summary:")
    print("=" * 50)
    for callsign, info in trajectory_info.items():
        print(f"Aircraft: {callsign} ({info['aircraft_type']})")
        print(f"Total distance: {info['total_distance']} nm")
        print(f"Trajectory points: {info['points']}")
        print(f"Output file: {info['output_file']}")
        print("-" * 30)


if __name__ == "__main__":
    main()
