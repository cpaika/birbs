#!/usr/bin/env python3
"""
Generate actual image files of Lake Champlain bathymetry and targets.
Uses PPM format (no external libraries required).
"""

import os
import math


def create_ppm_image(width: int, height: int, pixels: list, filename: str):
    """Create a PPM image file."""
    with open(filename, 'wb') as f:
        # PPM header
        f.write(f"P6\n{width} {height}\n255\n".encode())
        # Pixel data
        for row in pixels:
            for r, g, b in row:
                f.write(bytes([r, g, b]))
    print(f"  Created: {filename}")


def depth_to_color(depth: float) -> tuple:
    """Convert depth to RGB color."""
    if depth < 20:
        return (135, 206, 235)  # Light blue (shallow)
    elif depth < 50:
        return (70, 130, 180)   # Steel blue
    elif depth < 100:
        return (30, 80, 150)    # Medium blue
    elif depth < 200:
        return (20, 50, 120)    # Dark blue
    elif depth < 300:
        return (15, 30, 90)     # Very dark blue
    else:
        return (10, 20, 60)     # Deep navy


# Known depth data
DEPTH_POINTS = [
    (44.55, -73.33, 400),
    (44.50, -73.32, 350),
    (44.45, -73.30, 300),
    (44.40, -73.32, 280),
    (44.35, -73.33, 250),
    (44.30, -73.34, 200),
    (44.25, -73.35, 180),
    (44.48, -73.22, 100),
    (44.47, -73.23, 80),
    (44.62, -73.42, 60),
    (44.70, -73.35, 30),
]

WRECKS = [
    ("Champlain II", 44.206, -73.3763),
    ("Phoenix", 44.5497, -73.3352),
    ("General Butler", 44.4705, -73.2283),
    ("Spitfire", 44.6167, -73.4333),
    ("O.J. Walker", 44.4787, -73.2407),
    ("Water Witch", 44.2333, -73.3347),
]

TARGETS = [
    ("Target A", 44.2499, -73.3447, "high"),
    ("Target B", 44.2532, -73.3353, "high"),
    ("Target C", 44.2566, -73.3353, "high"),
    ("Target D", 44.2600, -73.3447, "high"),
    ("Target E", 44.4615, -73.2074, "medium"),
    ("Target F", 44.4705, -73.2389, "medium"),
]


def interpolate_depth(lat: float, lon: float) -> float:
    """Interpolate depth using inverse distance weighting."""
    total_weight = 0
    weighted_depth = 0

    for plat, plon, pdepth in DEPTH_POINTS:
        dist = math.sqrt((lat - plat)**2 + (lon - plon)**2)
        if dist < 0.001:
            return pdepth
        weight = 1 / (dist ** 2)
        weighted_depth += weight * pdepth
        total_weight += weight

    return weighted_depth / total_weight if total_weight > 0 else 100


def generate_main_map(output_dir: str):
    """Generate main bathymetry map."""
    width = 200
    height = 250

    lat_min, lat_max = 44.15, 44.75
    lon_min, lon_max = -73.50, -73.15

    pixels = []

    for row in range(height):
        lat = lat_max - (row / height) * (lat_max - lat_min)
        pixel_row = []

        for col in range(width):
            lon = lon_min + (col / width) * (lon_max - lon_min)

            # Check for targets (high priority = red)
            is_target = False
            is_high = False
            for name, tlat, tlon, priority in TARGETS:
                if abs(lat - tlat) < 0.008 and abs(lon - tlon) < 0.008:
                    is_target = True
                    is_high = priority == "high"
                    break

            # Check for wrecks (green)
            is_wreck = False
            for name, wlat, wlon in WRECKS:
                if abs(lat - wlat) < 0.008 and abs(lon - wlon) < 0.008:
                    is_wreck = True
                    break

            if is_target:
                if is_high:
                    pixel_row.append((255, 50, 50))  # Red for high priority
                else:
                    pixel_row.append((255, 200, 50))  # Yellow for medium
            elif is_wreck:
                pixel_row.append((50, 255, 50))  # Green for wrecks
            else:
                depth = interpolate_depth(lat, lon)
                pixel_row.append(depth_to_color(depth))

        pixels.append(pixel_row)

    create_ppm_image(width, height, pixels, os.path.join(output_dir, "lake_champlain_bathymetry.ppm"))


def generate_target_detail(target_name: str, target_lat: float, target_lon: float,
                           output_dir: str, radius_km: float = 2.0):
    """Generate detailed view of a target location."""
    width = 150
    height = 150

    radius_deg = radius_km / 111
    lat_min = target_lat - radius_deg
    lat_max = target_lat + radius_deg
    lon_min = target_lon - radius_deg / math.cos(math.radians(target_lat))
    lon_max = target_lon + radius_deg / math.cos(math.radians(target_lat))

    pixels = []

    for row in range(height):
        lat = lat_max - (row / height) * (lat_max - lat_min)
        pixel_row = []

        for col in range(width):
            lon = lon_min + (col / width) * (lon_max - lon_min)

            # Distance from target center
            dist = math.sqrt((lat - target_lat)**2 + (lon - target_lon)**2)

            # Target marker (crosshair)
            is_center = dist < 0.002
            is_crosshair = (abs(lat - target_lat) < 0.0005 or abs(lon - target_lon) < 0.0005) and dist < 0.01

            if is_center:
                pixel_row.append((255, 0, 0))  # Red center
            elif is_crosshair:
                pixel_row.append((255, 100, 100))  # Light red crosshair
            else:
                depth = interpolate_depth(lat, lon)
                color = depth_to_color(depth)

                # Add subtle grid
                grid_lat = int(lat * 1000) % 5 == 0
                grid_lon = int(lon * 1000) % 5 == 0
                if grid_lat or grid_lon:
                    color = tuple(min(255, c + 20) for c in color)

                pixel_row.append(color)

        pixels.append(pixel_row)

    filename = f"target_{target_name.lower().replace(' ', '_')}.ppm"
    create_ppm_image(width, height, pixels, os.path.join(output_dir, filename))


def generate_depth_profile(output_dir: str):
    """Generate depth profile cross-section."""
    width = 200
    height = 100

    lat = 44.50
    lon_min, lon_max = -73.40, -73.20
    max_depth = 400

    pixels = []

    # Get depths along profile
    depths = []
    for col in range(width):
        lon = lon_min + (col / width) * (lon_max - lon_min)
        depth = interpolate_depth(lat, lon)
        depths.append(depth)

    for row in range(height):
        depth_line = (row / height) * max_depth
        pixel_row = []

        for col in range(width):
            if row < 10:
                # Water surface
                pixel_row.append((100, 180, 255))
            elif depths[col] >= depth_line:
                # Below lake floor (brown/tan)
                pixel_row.append((139, 119, 101))
            else:
                # Water
                blue_intensity = int(255 - (depth_line / max_depth) * 150)
                pixel_row.append((30, 60, blue_intensity))

        pixels.append(pixel_row)

    create_ppm_image(width, height, pixels, os.path.join(output_dir, "depth_profile.ppm"))


def main():
    output_dir = os.path.dirname(__file__)

    print("=" * 60)
    print("  GENERATING LAKE CHAMPLAIN BATHYMETRY IMAGES")
    print("=" * 60)

    print("\n📊 Creating main bathymetry map...")
    generate_main_map(output_dir)

    print("\n🎯 Creating target detail maps...")
    for name, lat, lon, priority in TARGETS[:4]:
        generate_target_detail(name, lat, lon, output_dir)

    print("\n📈 Creating depth profile...")
    generate_depth_profile(output_dir)

    print("\n" + "=" * 60)
    print("  IMAGE FILES CREATED:")
    print("=" * 60)
    print(f"""
  Main map:     {output_dir}/lake_champlain_bathymetry.ppm
  Target A:     {output_dir}/target_target_a.ppm
  Target B:     {output_dir}/target_target_b.ppm
  Target C:     {output_dir}/target_target_c.ppm
  Target D:     {output_dir}/target_target_d.ppm
  Profile:      {output_dir}/depth_profile.ppm

  LEGEND:
  🔴 Red = High-priority targets (potential wrecks)
  🟡 Yellow = Medium-priority targets
  🟢 Green = Known shipwrecks
  🔵 Blue gradient = Water depth (darker = deeper)
""")

    return output_dir


if __name__ == "__main__":
    main()
