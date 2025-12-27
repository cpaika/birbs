#!/usr/bin/env python3
"""
Analyze Real CORONA Hollow Ways Data

Parses the actual shapefile from Jason Ur's Harvard dataset
to find potential undiscovered archaeological sites based on
hollow way convergence patterns.

Key insight: Where multiple ancient roads converge = settlement
"""

import struct
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict
import math


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Polyline:
    """A hollow way (ancient road) feature"""
    record_id: int
    points: List[Point]
    length_m: float = 0.0


def read_shapefile_header(shp_path: str) -> Dict:
    """Read shapefile header"""
    with open(shp_path, 'rb') as f:
        file_code = struct.unpack('>i', f.read(4))[0]
        f.seek(24)
        file_length = struct.unpack('>i', f.read(4))[0] * 2
        version = struct.unpack('<i', f.read(4))[0]
        shape_type = struct.unpack('<i', f.read(4))[0]

        xmin = struct.unpack('<d', f.read(8))[0]
        ymin = struct.unpack('<d', f.read(8))[0]
        xmax = struct.unpack('<d', f.read(8))[0]
        ymax = struct.unpack('<d', f.read(8))[0]

        return {
            'shape_type': shape_type,
            'xmin': xmin, 'ymin': ymin,
            'xmax': xmax, 'ymax': ymax,
            'file_length': file_length
        }


def read_polylines(shp_path: str) -> List[Polyline]:
    """Read all polyline features from shapefile"""
    polylines = []

    with open(shp_path, 'rb') as f:
        # Skip header (100 bytes)
        f.seek(100)

        record_num = 0
        while True:
            # Read record header
            header = f.read(8)
            if len(header) < 8:
                break

            record_id = struct.unpack('>i', header[0:4])[0]
            content_length = struct.unpack('>i', header[4:8])[0] * 2

            # Read shape type
            shape_type = struct.unpack('<i', f.read(4))[0]

            if shape_type == 3:  # PolyLine
                # Bounding box
                xmin = struct.unpack('<d', f.read(8))[0]
                ymin = struct.unpack('<d', f.read(8))[0]
                xmax = struct.unpack('<d', f.read(8))[0]
                ymax = struct.unpack('<d', f.read(8))[0]

                # Number of parts and points
                num_parts = struct.unpack('<i', f.read(4))[0]
                num_points = struct.unpack('<i', f.read(4))[0]

                # Part indices
                parts = [struct.unpack('<i', f.read(4))[0] for _ in range(num_parts)]

                # Points
                points = []
                for _ in range(num_points):
                    x = struct.unpack('<d', f.read(8))[0]
                    y = struct.unpack('<d', f.read(8))[0]
                    points.append(Point(x, y))

                # Calculate length
                length = 0
                for i in range(len(points) - 1):
                    dx = points[i+1].x - points[i].x
                    dy = points[i+1].y - points[i].y
                    length += math.sqrt(dx*dx + dy*dy)

                polylines.append(Polyline(
                    record_id=record_id,
                    points=points,
                    length_m=length
                ))

            elif shape_type == 0:  # Null shape
                pass
            else:
                # Skip unknown shape
                f.read(content_length - 4)

            record_num += 1

    return polylines


def find_endpoints(polylines: List[Polyline]) -> List[Tuple[float, float]]:
    """Extract all endpoints of hollow ways"""
    endpoints = []
    for pl in polylines:
        if pl.points:
            endpoints.append((pl.points[0].x, pl.points[0].y))
            endpoints.append((pl.points[-1].x, pl.points[-1].y))
    return endpoints


def cluster_endpoints(endpoints: List[Tuple[float, float]], radius: float = 500) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """
    Cluster endpoints into grid cells.
    Points within 'radius' meters are considered convergent.
    """
    # Use grid cells of size 'radius'
    clusters = defaultdict(list)

    for x, y in endpoints:
        cell_x = int(x / radius)
        cell_y = int(y / radius)
        clusters[(cell_x, cell_y)].append((x, y))

    return clusters


def find_convergence_points(clusters: Dict, min_roads: int = 4) -> List[Dict]:
    """
    Find points where multiple hollow ways converge.
    These are likely settlement locations.
    """
    convergence_points = []

    for (cell_x, cell_y), points in clusters.items():
        if len(points) >= min_roads:
            # Calculate centroid
            avg_x = sum(p[0] for p in points) / len(points)
            avg_y = sum(p[1] for p in points) / len(points)

            convergence_points.append({
                'x': avg_x,
                'y': avg_y,
                'road_count': len(points),
                'cell': (cell_x, cell_y)
            })

    return sorted(convergence_points, key=lambda p: p['road_count'], reverse=True)


def utm_to_latlon(x: float, y: float, zone: int = 37) -> Tuple[float, float]:
    """
    Convert UTM Zone 37N to lat/lon.
    The dataset is in UTM Zone 37N, WGS84.
    """
    import math

    # WGS84 parameters
    a = 6378137.0  # semi-major axis
    f = 1 / 298.257223563  # flattening
    e2 = 2*f - f*f  # eccentricity squared
    e_prime2 = e2 / (1 - e2)

    # UTM parameters
    k0 = 0.9996
    central_meridian = 39.0  # Zone 37

    # Remove false easting
    x = x - 500000

    # Footpoint latitude
    M = y / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))

    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    phi1 = mu + (3*e1/2 - 27*e1**3/32) * math.sin(2*mu)
    phi1 += (21*e1**2/16 - 55*e1**4/32) * math.sin(4*mu)
    phi1 += (151*e1**3/96) * math.sin(6*mu)
    phi1 += (1097*e1**4/512) * math.sin(8*mu)

    # Calculate latitude
    N1 = a / math.sqrt(1 - e2 * math.sin(phi1)**2)
    T1 = math.tan(phi1)**2
    C1 = e_prime2 * math.cos(phi1)**2
    R1 = a * (1 - e2) / ((1 - e2 * math.sin(phi1)**2)**1.5)
    D = x / (N1 * k0)

    lat = phi1 - (N1 * math.tan(phi1) / R1) * (
        D**2/2 -
        (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e_prime2) * D**4/24 +
        (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e_prime2 - 3*C1**2) * D**6/720
    )

    lon = central_meridian + math.degrees(
        (D - (1 + 2*T1 + C1) * D**3/6 +
         (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e_prime2 + 24*T1**2) * D**5/120)
        / math.cos(phi1)
    )

    lat = math.degrees(lat)

    return lat, lon


def read_projection(prj_path: str) -> str:
    """Read projection file"""
    try:
        with open(prj_path, 'r') as f:
            return f.read().strip()
    except:
        return "Unknown"


def main():
    print("=" * 70)
    print("   ANALYZING REAL CORONA HOLLOW WAYS DATA")
    print("   Finding potential undiscovered archaeological sites")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(__file__), "corona_data")
    shp_path = os.path.join(data_dir, "hollow_ways_oip137.shp")
    prj_path = os.path.join(data_dir, "hollow_ways_oip137.prj")

    if not os.path.exists(shp_path):
        print(f"Error: Shapefile not found at {shp_path}")
        print("Run: curl -L -o hollow_ways.zip 'https://dataverse.harvard.edu/api/access/datafile/2971729'")
        return

    # Read projection
    print(f"\n📐 PROJECTION:")
    proj = read_projection(prj_path)
    print(f"   {proj[:80]}...")

    # Read header
    print(f"\n📊 READING SHAPEFILE HEADER...")
    header = read_shapefile_header(shp_path)
    print(f"   Shape type: {'PolyLine' if header['shape_type'] == 3 else header['shape_type']}")
    print(f"   Bounds (UTM 37N):")
    print(f"      X: {header['xmin']:.0f} to {header['xmax']:.0f}")
    print(f"      Y: {header['ymin']:.0f} to {header['ymax']:.0f}")

    # Convert bounds to lat/lon
    min_lat, min_lon = utm_to_latlon(header['xmin'], header['ymin'])
    max_lat, max_lon = utm_to_latlon(header['xmax'], header['ymax'])
    print(f"   Bounds (WGS84):")
    print(f"      Lat: {min_lat:.4f}° to {max_lat:.4f}°N")
    print(f"      Lon: {min_lon:.4f}° to {max_lon:.4f}°E")

    # Read polylines
    print(f"\n📏 READING HOLLOW WAYS...")
    polylines = read_polylines(shp_path)
    print(f"   Total hollow way segments: {len(polylines)}")

    total_length = sum(pl.length_m for pl in polylines)
    print(f"   Total length: {total_length/1000:.1f} km")

    # Find endpoints
    print(f"\n🔍 ANALYZING ROAD NETWORK...")
    endpoints = find_endpoints(polylines)
    print(f"   Total endpoints: {len(endpoints)}")

    # Cluster endpoints
    clusters = cluster_endpoints(endpoints, radius=300)  # 300m radius
    print(f"   Endpoint clusters: {len(clusters)}")

    # Find convergence points
    convergence = find_convergence_points(clusters, min_roads=4)
    print(f"   High-convergence points (4+ roads): {len(convergence)}")

    # Report top convergence points
    print("\n" + "=" * 70)
    print("   🎯 POTENTIAL UNDISCOVERED SETTLEMENT SITES")
    print("   (Locations where multiple hollow ways converge)")
    print("=" * 70)

    print(f"\n   Found {len(convergence)} potential sites based on road convergence.\n")

    # Top 15 convergence points
    for i, point in enumerate(convergence[:15], 1):
        lat, lon = utm_to_latlon(point['x'], point['y'])
        print(f"   {i:2d}. CONVERGENCE POINT")
        print(f"       📍 Coordinates: {lat:.5f}°N, {lon:.5f}°E")
        print(f"       🛤️  Roads converging: {point['road_count']}")
        print(f"       📊 UTM (Zone 37N): {point['x']:.0f}E, {point['y']:.0f}N")
        print()

    # Summary
    print("=" * 70)
    print("   DISCOVERY METHODOLOGY")
    print("=" * 70)
    print(f"""
   This analysis identified {len(convergence)} locations where ancient roads
   (hollow ways) converge - a strong indicator of settlement sites.

   VALIDATION STEPS:
   1. Cross-reference top coordinates with Pleiades gazetteer
   2. Check EAMENA database for documented sites
   3. If NOT in any database → POTENTIAL NEW DISCOVERY

   KEY INSIGHT:
   Hollow ways are ancient trackways created by millennia of foot
   and animal traffic. They radiate from settlements like spokes
   from a wheel hub. Where 4+ hollow ways meet, there was almost
   certainly a settlement - documented or not.

   DATA SOURCE: Jason Ur, Harvard University
   "Landscapes of Settlement and Movement in Northeastern Syria"
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NYWQO2
""")

    # Output coordinates for further investigation
    print("=" * 70)
    print("   TOP 5 SITES FOR IMMEDIATE INVESTIGATION")
    print("=" * 70)

    for i, point in enumerate(convergence[:5], 1):
        lat, lon = utm_to_latlon(point['x'], point['y'])
        print(f"\n   SITE {i}: {point['road_count']} CONVERGING ROADS")
        print(f"   Google Maps: https://www.google.com/maps?q={lat:.5f},{lon:.5f}")
        print(f"   Pleiades search: https://pleiades.stoa.org/search?SearchableText={lat:.2f}+{lon:.2f}")

    return convergence


if __name__ == "__main__":
    convergence_points = main()
