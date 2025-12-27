#!/usr/bin/env python3
"""
Lake Champlain Bathymetry Visualizer

Creates visual maps of lake depth and potential shipwreck locations
using ASCII art and generates coordinate-based analysis.
"""

import math
import os
from typing import List, Tuple, Dict


# Known depth points from nautical charts (lat, lon, depth_ft)
DEPTH_POINTS = [
    # Deep main channel
    (44.55, -73.33, 400),  # Deepest point - Split Rock area
    (44.50, -73.32, 350),
    (44.45, -73.30, 300),
    (44.40, -73.32, 280),
    (44.35, -73.33, 250),
    (44.30, -73.34, 200),
    (44.25, -73.35, 180),

    # Burlington area
    (44.48, -73.22, 100),
    (44.47, -73.23, 80),
    (44.46, -73.24, 60),
    (44.45, -73.25, 40),

    # Valcour Island area
    (44.62, -73.42, 60),
    (44.63, -73.43, 80),
    (44.61, -73.44, 50),

    # Shallow areas
    (44.70, -73.35, 30),
    (44.75, -73.32, 25),
    (44.80, -73.30, 20),

    # Missisquoi Bay (shallow)
    (44.95, -73.15, 15),
    (44.97, -73.12, 10),
    (44.98, -73.10, 8),
]

# Known shipwrecks
WRECKS = [
    ("Champlain II", 44.206, -73.3763, 25, "⚓"),
    ("Phoenix", 44.5497, -73.3352, 90, "🚢"),
    ("General Butler", 44.4705, -73.2283, 40, "⛵"),
    ("Spitfire", 44.6167, -73.4333, 60, "💣"),
    ("O.J. Walker", 44.4787, -73.2407, 65, "⛵"),
    ("Water Witch", 44.2333, -73.3347, 90, "🚢"),
]

# Potential undiscovered targets
TARGETS = [
    ("Target A", 44.2499, -73.3447, 82, "high"),
    ("Target B", 44.2532, -73.3353, 59, "high"),
    ("Target C", 44.2566, -73.3353, 92, "high"),
    ("Target D", 44.2600, -73.3447, 33, "high"),
    ("Target E", 44.4615, -73.2074, 72, "medium"),
    ("Target F", 44.4705, -73.2389, 43, "medium"),
    ("Target G", 44.6200, -73.4280, 57, "medium"),  # Near Valcour battle
]


def interpolate_depth(lat: float, lon: float, points: List[Tuple]) -> float:
    """Interpolate depth at a location using inverse distance weighting"""
    total_weight = 0
    weighted_depth = 0

    for plat, plon, pdepth in points:
        dist = math.sqrt((lat - plat)**2 + (lon - plon)**2)
        if dist < 0.001:
            return pdepth
        weight = 1 / (dist ** 2)
        weighted_depth += weight * pdepth
        total_weight += weight

    return weighted_depth / total_weight if total_weight > 0 else 0


def depth_to_char(depth: float) -> str:
    """Convert depth to ASCII character"""
    if depth < 20:
        return '░'  # Very shallow
    elif depth < 50:
        return '▒'  # Shallow
    elif depth < 100:
        return '▓'  # Medium
    elif depth < 200:
        return '█'  # Deep
    else:
        return '▀'  # Very deep


def depth_to_color_code(depth: float) -> str:
    """Return ANSI color code for depth"""
    if depth < 20:
        return '\033[96m'  # Cyan (shallow)
    elif depth < 50:
        return '\033[94m'  # Blue
    elif depth < 100:
        return '\033[34m'  # Dark blue
    elif depth < 200:
        return '\033[35m'  # Magenta (deep)
    else:
        return '\033[31m'  # Red (very deep)


def generate_ascii_map(lat_min: float, lat_max: float,
                       lon_min: float, lon_max: float,
                       width: int = 60, height: int = 30) -> str:
    """Generate ASCII bathymetry map"""
    reset = '\033[0m'
    bold = '\033[1m'
    red = '\033[91m'
    green = '\033[92m'
    yellow = '\033[93m'

    lines = []
    lines.append(f"\n{bold}LAKE CHAMPLAIN BATHYMETRY MAP{reset}")
    lines.append(f"Lat: {lat_min:.2f}° to {lat_max:.2f}°N | Lon: {lon_min:.2f}° to {lon_max:.2f}°W")
    lines.append("=" * (width + 10))

    lat_step = (lat_max - lat_min) / height
    lon_step = (lon_max - lon_min) / width

    # Create grid
    for row in range(height, -1, -1):
        lat = lat_min + row * lat_step
        line = f"{lat:5.2f}° |"

        for col in range(width + 1):
            lon = lon_min + col * lon_step

            # Check for wrecks
            is_wreck = False
            is_target = False

            for name, wlat, wlon, wdepth, symbol in WRECKS:
                if abs(lat - wlat) < lat_step and abs(lon - wlon) < lon_step:
                    line += f"{green}⚓{reset}"
                    is_wreck = True
                    break

            if not is_wreck:
                for name, tlat, tlon, tdepth, priority in TARGETS:
                    if abs(lat - tlat) < lat_step and abs(lon - tlon) < lon_step:
                        if priority == "high":
                            line += f"{red}✦{reset}"
                        else:
                            line += f"{yellow}✧{reset}"
                        is_target = True
                        break

            if not is_wreck and not is_target:
                depth = interpolate_depth(lat, lon, DEPTH_POINTS)
                color = depth_to_color_code(depth)
                char = depth_to_char(depth)
                line += f"{color}{char}{reset}"

        lines.append(line)

    # X-axis labels
    lines.append("      +" + "-" * (width + 1))
    lon_labels = f"       {lon_min:.1f}°"
    lon_labels += " " * (width // 2 - 5) + f"{(lon_min + lon_max)/2:.1f}°"
    lon_labels += " " * (width // 2 - 5) + f"{lon_max:.1f}°W"
    lines.append(lon_labels)

    # Legend
    lines.append("")
    lines.append(f"{bold}LEGEND:{reset}")
    lines.append(f"  {green}⚓{reset} = Known shipwreck")
    lines.append(f"  {red}✦{reset} = High-priority target (potential undiscovered wreck)")
    lines.append(f"  {yellow}✧{reset} = Medium-priority target")
    lines.append(f"  ░ = Shallow (<20ft)  ▒ = Medium (<50ft)  ▓ = Deep (<100ft)  █ = Very deep")

    return "\n".join(lines)


def generate_target_detail_map(target_lat: float, target_lon: float,
                               name: str, radius_km: float = 2.0) -> str:
    """Generate detailed map around a specific target"""
    reset = '\033[0m'
    bold = '\033[1m'
    red = '\033[91m'

    # Convert radius to degrees
    radius_deg = radius_km / 111

    lat_min = target_lat - radius_deg
    lat_max = target_lat + radius_deg
    lon_min = target_lon - radius_deg / math.cos(math.radians(target_lat))
    lon_max = target_lon + radius_deg / math.cos(math.radians(target_lat))

    width = 40
    height = 20

    lines = []
    lines.append(f"\n{bold}DETAIL MAP: {name}{reset}")
    lines.append(f"Center: {target_lat:.5f}°N, {target_lon:.5f}°W")
    lines.append(f"Radius: {radius_km}km")
    lines.append("-" * 50)

    lat_step = (lat_max - lat_min) / height
    lon_step = (lon_max - lon_min) / width

    for row in range(height, -1, -1):
        lat = lat_min + row * lat_step
        line = ""

        for col in range(width + 1):
            lon = lon_min + col * lon_step

            # Check if this is the target center
            if abs(lat - target_lat) < lat_step/2 and abs(lon - target_lon) < lon_step/2:
                line += f"{red}✦{reset}"
            else:
                # Check for nearby wrecks
                is_wreck = False
                for wname, wlat, wlon, wdepth, symbol in WRECKS:
                    if abs(lat - wlat) < lat_step and abs(lon - wlon) < lon_step:
                        line += "⚓"
                        is_wreck = True
                        break

                if not is_wreck:
                    depth = interpolate_depth(lat, lon, DEPTH_POINTS)
                    line += depth_to_char(depth)

        lines.append(line)

    lines.append("-" * 50)
    lines.append(f"Google Maps: https://www.google.com/maps?q={target_lat},{target_lon}")

    return "\n".join(lines)


def create_depth_profile(lat: float, lon_start: float, lon_end: float, name: str) -> str:
    """Create a cross-sectional depth profile"""
    reset = '\033[0m'
    bold = '\033[1m'
    blue = '\033[94m'

    lines = []
    lines.append(f"\n{bold}DEPTH PROFILE: {name}{reset}")
    lines.append(f"At latitude {lat:.4f}°N, from {lon_start:.3f}°W to {lon_end:.3f}°W")
    lines.append("")

    width = 60
    max_depth = 400  # Max depth in feet
    height = 20

    lon_step = (lon_end - lon_start) / width

    # Get depths
    depths = []
    for i in range(width + 1):
        lon = lon_start + i * lon_step
        depth = interpolate_depth(lat, lon, DEPTH_POINTS)
        depths.append(depth)

    # Draw profile (inverted - surface at top)
    lines.append("Surface")
    lines.append("~" * (width + 5))

    for row in range(height):
        depth_threshold = (row / height) * max_depth
        line = f"{int(depth_threshold):4d}ft|"

        for d in depths:
            if d >= depth_threshold:
                line += f"{blue}█{reset}"
            else:
                line += " "

        lines.append(line)

    lines.append("-" * (width + 5))
    lines.append(f"     {lon_start:.2f}°W" + " " * (width - 15) + f"{lon_end:.2f}°W")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("   LAKE CHAMPLAIN BATHYMETRY & TREASURE VISUALIZATION")
    print("=" * 70)

    # Generate main overview map
    print(generate_ascii_map(44.15, 44.75, -73.45, -73.15, width=50, height=25))

    # Generate depth profile through deep channel
    print(create_depth_profile(44.50, -73.40, -73.20, "Main Channel Cross-Section"))

    # Generate detail maps for top targets
    print("\n" + "=" * 70)
    print("   DETAILED TARGET ANALYSIS")
    print("=" * 70)

    for name, lat, lon, depth, priority in TARGETS[:3]:
        print(generate_target_detail_map(lat, lon, f"{name} ({priority} priority)", radius_km=1.5))
        print()

    # Summary with coordinates
    print("\n" + "=" * 70)
    print("   📍 ALL POTENTIAL DISCOVERY LOCATIONS")
    print("=" * 70)

    print("\n   HIGH-PRIORITY TARGETS (Split Rock Point area):\n")
    for name, lat, lon, depth, priority in TARGETS:
        if priority == "high":
            print(f"   {name}:")
            print(f"      Coordinates: {lat:.5f}°N, {lon:.5f}°W")
            print(f"      Est. depth: ~{depth}ft")
            print(f"      Google Maps: https://www.google.com/maps?q={lat},{lon}")
            print()

    print("\n   MEDIUM-PRIORITY TARGETS:\n")
    for name, lat, lon, depth, priority in TARGETS:
        if priority == "medium":
            print(f"   {name}:")
            print(f"      Coordinates: {lat:.5f}°N, {lon:.5f}°W")
            print(f"      Est. depth: ~{depth}ft")
            print(f"      Google Maps: https://www.google.com/maps?q={lat},{lon}")
            print()

    # Save maps to file
    output_dir = os.path.dirname(__file__)
    map_file = os.path.join(output_dir, "lake_champlain_map.txt")

    with open(map_file, 'w') as f:
        # Write plain text version (no ANSI codes)
        f.write("LAKE CHAMPLAIN BATHYMETRY MAP\n")
        f.write("=" * 60 + "\n\n")

        for name, lat, lon, depth, priority in TARGETS:
            f.write(f"{name} ({priority})\n")
            f.write(f"  Lat: {lat:.5f}°N, Lon: {lon:.5f}°W\n")
            f.write(f"  Depth: ~{depth}ft\n")
            f.write(f"  Maps: https://www.google.com/maps?q={lat},{lon}\n\n")

    print(f"\n   Map saved to: {map_file}")

    return TARGETS


if __name__ == "__main__":
    targets = main()
