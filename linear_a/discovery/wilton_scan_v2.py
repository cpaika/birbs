#!/usr/bin/env python3
"""
Wilton Maine Formation Scanner v2

Uses Open-Elevation API and USGS data to find unusual formations.
"""

import urllib.request
import urllib.error
import json
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# Wilton, Maine center
WILTON = (44.5925, -70.2286)

# Open-Elevation API
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"


@dataclass
class Formation:
    lat: float
    lon: float
    name: str
    elevation_m: float
    anomaly_m: float
    formation_type: str
    google_maps_url: str


def get_elevations_batch(points: List[Tuple[float, float]]) -> List[Optional[float]]:
    """Get elevations for multiple points in one request."""
    try:
        locations = [{"latitude": lat, "longitude": lon} for lat, lon in points]
        data = json.dumps({"locations": locations}).encode('utf-8')

        req = urllib.request.Request(
            OPEN_ELEVATION_URL,
            data=data,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'Formation-Scanner/1.0'
            }
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return [r.get('elevation') for r in result.get('results', [])]

    except Exception as e:
        print(f"  API error: {e}")
        return [None] * len(points)


def scan_for_anomalies(center_lat: float, center_lon: float,
                       radius_km: float = 3.0,
                       grid_size: int = 15) -> List[Dict]:
    """
    Scan a grid and identify elevation anomalies.
    """
    # Generate grid points
    points = []
    lat_step = (radius_km * 2) / grid_size / 111
    lon_step = (radius_km * 2) / grid_size / (111 * math.cos(math.radians(center_lat)))

    for i in range(grid_size):
        for j in range(grid_size):
            lat = center_lat - radius_km/111 + i * lat_step
            lon = center_lon - radius_km/(111 * math.cos(math.radians(center_lat))) + j * lon_step
            points.append((lat, lon))

    print(f"  Querying {len(points)} elevation points...")

    # Get elevations in batches
    elevations = get_elevations_batch(points)

    # Find anomalies (local maxima/minima)
    anomalies = []

    # Calculate mean and std
    valid_elevs = [e for e in elevations if e is not None]
    if not valid_elevs:
        return []

    mean_elev = sum(valid_elevs) / len(valid_elevs)
    variance = sum((e - mean_elev)**2 for e in valid_elevs) / len(valid_elevs)
    std_elev = math.sqrt(variance) if variance > 0 else 1

    print(f"  Mean elevation: {mean_elev:.1f}m, Std dev: {std_elev:.1f}m")

    for idx, (point, elev) in enumerate(zip(points, elevations)):
        if elev is None:
            continue

        # Check if this is an anomaly (>1.5 std from mean)
        z_score = abs(elev - mean_elev) / std_elev if std_elev > 0 else 0

        if z_score > 1.5:
            anomaly_type = "peak" if elev > mean_elev else "depression"
            anomalies.append({
                "lat": point[0],
                "lon": point[1],
                "elevation": elev,
                "anomaly": elev - mean_elev,
                "z_score": z_score,
                "type": anomaly_type
            })

    return anomalies


def check_specific_locations():
    """Check specific interesting locations around Wilton."""

    # Interesting locations in/around Wilton to check
    LOCATIONS = [
        # Town features
        ("Wilton Town Center", 44.5925, -70.2286),
        ("Wilson Lake - North Shore", 44.6200, -70.2350),
        ("Wilson Lake - South Shore", 44.6050, -70.2380),
        ("Kineowatha Park", 44.5890, -70.2150),

        # Stream confluences (often have Native American sites)
        ("Wilson Stream - Bridge", 44.5850, -70.2180),
        ("Wilson Stream - Upper", 44.5700, -70.2050),

        # Ridges and hills
        ("Bald Mountain Area", 44.6100, -70.1800),
        ("East Wilton Ridge", 44.5950, -70.1850),

        # Historical areas
        ("Old Cemetery Area", 44.5900, -70.2300),
        ("Bass Hill", 44.5780, -70.2400),

        # Nearby interesting terrain
        ("Temple Hill", 44.6700, -70.2100),
        ("Farmington Area", 44.6700, -70.1500),
        ("Mount Blue Foothills", 44.6400, -70.2800),

        # Potential archaeological areas
        ("Sandy River Confluence", 44.6650, -70.1480),
        ("Clearwater Lake Area", 44.6300, -70.1900),
    ]

    points = [(lat, lon) for name, lat, lon in LOCATIONS]
    elevations = get_elevations_batch(points)

    results = []
    for (name, lat, lon), elev in zip(LOCATIONS, elevations):
        if elev is not None:
            results.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "elevation": elev,
                "google_maps": f"https://www.google.com/maps?q={lat},{lon}"
            })

    return results


def main():
    print("=" * 70)
    print("   WILTON, MAINE - TERRAIN ANOMALY SCANNER")
    print("   Finding unusual geological/archaeological formations")
    print("=" * 70)

    print(f"\n📍 Center: Wilton, Maine ({WILTON[0]:.4f}°N, {WILTON[1]:.4f}°W)")

    # Check specific locations
    print("\n🔍 Checking key locations...")
    locations = check_specific_locations()

    if locations:
        print(f"\n   Retrieved elevation for {len(locations)} locations")

        # Sort by elevation
        locations.sort(key=lambda x: x['elevation'], reverse=True)

        print("\n" + "=" * 70)
        print("   ELEVATION PROFILE - KEY LOCATIONS")
        print("=" * 70)

        print("\n   HIGHEST POINTS (potential lookouts/peaks):\n")
        for loc in locations[:5]:
            print(f"   📍 {loc['name']}")
            print(f"      Elevation: {loc['elevation']:.0f}m ({loc['elevation']*3.28:.0f}ft)")
            print(f"      🗺️  {loc['google_maps']}")
            print()

        print("\n   LOWEST POINTS (valleys/water features):\n")
        for loc in locations[-3:]:
            print(f"   📍 {loc['name']}")
            print(f"      Elevation: {loc['elevation']:.0f}m ({loc['elevation']*3.28:.0f}ft)")
            print(f"      🗺️  {loc['google_maps']}")
            print()

    # Grid scan for anomalies
    print("\n🔍 Running grid scan for terrain anomalies...")
    time.sleep(1)  # Rate limiting

    anomalies = scan_for_anomalies(WILTON[0], WILTON[1], radius_km=5, grid_size=12)

    if anomalies:
        # Sort by z-score
        anomalies.sort(key=lambda x: x['z_score'], reverse=True)

        peaks = [a for a in anomalies if a['type'] == 'peak']
        depressions = [a for a in anomalies if a['type'] == 'depression']

        print(f"\n   Found {len(anomalies)} terrain anomalies:")
        print(f"   • High points (peaks): {len(peaks)}")
        print(f"   • Low points (depressions): {len(depressions)}")

        print("\n" + "=" * 70)
        print("   🔴 UNUSUAL FORMATIONS DETECTED")
        print("=" * 70)

        print("\n   TOP PEAKS/MOUNDS:\n")
        for i, a in enumerate(peaks[:5], 1):
            print(f"   {i}. ELEVATED FORMATION")
            print(f"      📍 {a['lat']:.5f}°N, {a['lon']:.5f}°W")
            print(f"      📊 Elevation: {a['elevation']:.0f}m (+{a['anomaly']:.0f}m above mean)")
            print(f"      🎯 Anomaly score: {a['z_score']:.2f}σ")
            print(f"      🗺️  https://www.google.com/maps?q={a['lat']},{a['lon']}")
            print()

        if depressions:
            print("\n   TOP DEPRESSIONS/BASINS:\n")
            for i, a in enumerate(depressions[:3], 1):
                print(f"   {i}. DEPRESSION")
                print(f"      📍 {a['lat']:.5f}°N, {a['lon']:.5f}°W")
                print(f"      📊 Elevation: {a['elevation']:.0f}m ({a['anomaly']:.0f}m below mean)")
                print(f"      🗺️  https://www.google.com/maps?q={a['lat']},{a['lon']}")
                print()

    # Generate summary with Google Maps links
    print("\n" + "=" * 70)
    print("   📍 GOOGLE MAPS LINKS - ALL FINDINGS")
    print("=" * 70)

    all_findings = []

    if locations:
        for loc in locations[:5]:
            all_findings.append({
                "name": loc['name'],
                "url": loc['google_maps'],
                "note": f"Elevation: {loc['elevation']:.0f}m"
            })

    if anomalies:
        for i, a in enumerate(peaks[:5], 1):
            all_findings.append({
                "name": f"Anomaly Peak #{i}",
                "url": f"https://www.google.com/maps?q={a['lat']},{a['lon']}",
                "note": f"+{a['anomaly']:.0f}m above mean"
            })

    print()
    for f in all_findings:
        print(f"   {f['name']}: {f['note']}")
        print(f"   {f['url']}")
        print()

    return anomalies, locations


if __name__ == "__main__":
    anomalies, locations = main()
