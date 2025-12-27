#!/usr/bin/env python3
"""
Generate PNG images of Lake Champlain bathymetry and targets.
Uses pure Python with zlib for PNG compression.
"""

import os
import math
import zlib
import struct


def create_png_image(width: int, height: int, pixels: list, filename: str):
    """Create a PNG image file using pure Python."""

    def make_chunk(chunk_type, data):
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xffffffff
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc)

    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b'IHDR', ihdr_data)

    # IDAT chunk (image data)
    raw_data = b''
    for row in pixels:
        raw_data += b'\x00'  # Filter type: None
        for r, g, b in row:
            raw_data += bytes([r, g, b])

    compressed = zlib.compress(raw_data, 9)
    idat = make_chunk(b'IDAT', compressed)

    # IEND chunk
    iend = make_chunk(b'IEND', b'')

    with open(filename, 'wb') as f:
        f.write(signature + ihdr + idat + iend)

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
    width = 160
    height = 200

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
                if abs(lat - tlat) < 0.012 and abs(lon - tlon) < 0.012:
                    is_target = True
                    is_high = priority == "high"
                    break

            # Check for wrecks (green)
            is_wreck = False
            for name, wlat, wlon in WRECKS:
                if abs(lat - wlat) < 0.012 and abs(lon - wlon) < 0.012:
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

    create_png_image(width, height, pixels, os.path.join(output_dir, "bathymetry_map.png"))


def generate_target_detail(target_name: str, target_lat: float, target_lon: float,
                           output_dir: str, radius_km: float = 2.0):
    """Generate detailed view of a target location."""
    width = 120
    height = 120

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
            is_center = dist < 0.003
            is_crosshair = (abs(lat - target_lat) < 0.001 or abs(lon - target_lon) < 0.001) and dist < 0.015

            if is_center:
                pixel_row.append((255, 0, 0))  # Red center
            elif is_crosshair:
                pixel_row.append((255, 100, 100))  # Light red crosshair
            else:
                depth = interpolate_depth(lat, lon)
                pixel_row.append(depth_to_color(depth))

        pixels.append(pixel_row)

    filename = f"target_{target_name.lower().replace(' ', '_')}.png"
    create_png_image(width, height, pixels, os.path.join(output_dir, filename))


def generate_depth_profile(output_dir: str):
    """Generate depth profile cross-section."""
    width = 160
    height = 80

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
            if row < 5:
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

    create_png_image(width, height, pixels, os.path.join(output_dir, "depth_profile.png"))


def main():
    output_dir = os.path.dirname(__file__)

    print("=" * 60)
    print("  GENERATING LAKE CHAMPLAIN PNG IMAGES")
    print("=" * 60)

    print("\nCreating main bathymetry map...")
    generate_main_map(output_dir)

    print("\nCreating target detail maps...")
    for name, lat, lon, priority in TARGETS[:4]:
        generate_target_detail(name, lat, lon, output_dir)

    print("\nCreating depth profile...")
    generate_depth_profile(output_dir)

    print("\n" + "=" * 60)
    print("  PNG FILES CREATED")
    print("=" * 60)
    print(f"""
  Main map:     bathymetry_map.png
  Target A:     target_target_a.png
  Target B:     target_target_b.png
  Target C:     target_target_c.png
  Target D:     target_target_d.png
  Profile:      depth_profile.png

  LEGEND:
  RED = High-priority targets (potential wrecks)
  YELLOW = Medium-priority targets
  GREEN = Known shipwrecks
  BLUE gradient = Water depth (darker = deeper)
""")

    return output_dir


if __name__ == "__main__":
    main()
