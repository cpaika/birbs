#!/usr/bin/env python3
"""
Wilton Maine Unusual Formation Detector

Uses Maine's LiDAR-derived elevation data to identify unusual
geological, archaeological, or man-made formations in the Wilton area.

Data source: Maine GeoLibrary DEM 2018 (1-meter resolution LiDAR)
"""

import urllib.request
import urllib.error
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# Wilton, Maine coordinates
WILTON_CENTER = (44.5925, -70.2286)

# Maine ArcGIS REST API for elevation
MAINE_DEM_SERVICE = "https://gis.maine.gov/arcgis/rest/services/Elevation/Maine_Elevation_DEM_2018/ImageServer"


@dataclass
class ElevationPoint:
    lat: float
    lon: float
    elevation_m: float


@dataclass
class Formation:
    lat: float
    lon: float
    type: str
    description: str
    elevation_anomaly_m: float
    radius_m: float
    confidence: float


def get_elevation_at_point(lat: float, lon: float) -> Optional[float]:
    """
    Get elevation at a point using Maine's DEM service.
    """
    try:
        # ArcGIS identify operation
        url = f"{MAINE_DEM_SERVICE}/identify"
        params = {
            "geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
            "geometryType": "esriGeometryPoint",
            "returnGeometry": "false",
            "returnCatalogItems": "false",
            "f": "json"
        }

        query_string = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        full_url = f"{url}?{query_string}"

        req = urllib.request.Request(full_url, headers={'User-Agent': 'Archaeological-Research/1.0'})

        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            value = data.get('value')
            if value and value != 'NoData':
                return float(value)
        return None
    except Exception as e:
        return None


def get_elevation_profile(center_lat: float, center_lon: float,
                          radius_km: float = 0.5,
                          num_points: int = 8) -> List[ElevationPoint]:
    """
    Get elevation samples around a center point.
    """
    points = []

    # Sample in a circle around the center
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        # Convert km to degrees (approximate)
        lat_offset = (radius_km / 111.0) * math.cos(angle)
        lon_offset = (radius_km / (111.0 * math.cos(math.radians(center_lat)))) * math.sin(angle)

        sample_lat = center_lat + lat_offset
        sample_lon = center_lon + lon_offset

        elev = get_elevation_at_point(sample_lat, sample_lon)
        if elev is not None:
            points.append(ElevationPoint(sample_lat, sample_lon, elev))

    # Also get center point
    center_elev = get_elevation_at_point(center_lat, center_lon)
    if center_elev is not None:
        points.append(ElevationPoint(center_lat, center_lon, center_elev))

    return points


def detect_mound_formation(lat: float, lon: float) -> Optional[Formation]:
    """
    Detect if a location has mound-like characteristics.
    A mound would be elevated relative to surrounding terrain.
    """
    # Get elevation at center
    center_elev = get_elevation_at_point(lat, lon)
    if center_elev is None:
        return None

    # Get elevations in surrounding ring (100m radius)
    surrounding = []
    for angle_deg in range(0, 360, 45):
        angle = math.radians(angle_deg)
        offset = 0.001  # ~100m
        sample_lat = lat + offset * math.cos(angle)
        sample_lon = lon + offset * math.sin(angle) / math.cos(math.radians(lat))

        elev = get_elevation_at_point(sample_lat, sample_lon)
        if elev is not None:
            surrounding.append(elev)

    if len(surrounding) < 4:
        return None

    avg_surrounding = sum(surrounding) / len(surrounding)
    elevation_diff = center_elev - avg_surrounding

    # Mound detection: center is elevated relative to surroundings
    if elevation_diff > 2:  # More than 2m higher
        return Formation(
            lat=lat,
            lon=lon,
            type="mound",
            description=f"Elevated area {elevation_diff:.1f}m above surroundings",
            elevation_anomaly_m=elevation_diff,
            radius_m=100,
            confidence=min(0.9, 0.5 + elevation_diff / 10)
        )

    return None


def detect_depression(lat: float, lon: float) -> Optional[Formation]:
    """
    Detect circular depressions (could be old cellar holes, pits, etc.)
    """
    center_elev = get_elevation_at_point(lat, lon)
    if center_elev is None:
        return None

    surrounding = []
    for angle_deg in range(0, 360, 45):
        angle = math.radians(angle_deg)
        offset = 0.0005  # ~50m
        sample_lat = lat + offset * math.cos(angle)
        sample_lon = lon + offset * math.sin(angle) / math.cos(math.radians(lat))

        elev = get_elevation_at_point(sample_lat, sample_lon)
        if elev is not None:
            surrounding.append(elev)

    if len(surrounding) < 4:
        return None

    avg_surrounding = sum(surrounding) / len(surrounding)
    elevation_diff = avg_surrounding - center_elev  # Positive if center is lower

    if elevation_diff > 1:  # More than 1m lower
        return Formation(
            lat=lat,
            lon=lon,
            type="depression",
            description=f"Depression {elevation_diff:.1f}m below surroundings (possible cellar hole/pit)",
            elevation_anomaly_m=elevation_diff,
            radius_m=50,
            confidence=min(0.8, 0.4 + elevation_diff / 5)
        )

    return None


def scan_grid(center_lat: float, center_lon: float,
              radius_km: float = 2.0,
              grid_spacing_m: float = 200) -> List[Formation]:
    """
    Scan a grid around the center for unusual formations.
    """
    formations = []

    # Convert spacing to degrees
    lat_step = grid_spacing_m / 111000
    lon_step = grid_spacing_m / (111000 * math.cos(math.radians(center_lat)))

    # Calculate grid bounds
    lat_range = radius_km / 111
    lon_range = radius_km / (111 * math.cos(math.radians(center_lat)))

    lat = center_lat - lat_range
    checked = 0
    found = 0

    while lat <= center_lat + lat_range:
        lon = center_lon - lon_range
        while lon <= center_lon + lon_range:
            checked += 1

            # Check for mound
            mound = detect_mound_formation(lat, lon)
            if mound:
                formations.append(mound)
                found += 1

            # Check for depression
            depression = detect_depression(lat, lon)
            if depression:
                formations.append(depression)
                found += 1

            lon += lon_step
        lat += lat_step

        if checked % 10 == 0:
            print(f"  Checked {checked} points, found {found} formations...")

    return formations


def scan_known_interesting_areas():
    """
    Scan areas around Wilton that might have interesting formations.
    """
    # Areas to check in/around Wilton
    AREAS = [
        {"name": "Wilton Center", "lat": 44.5925, "lon": -70.2286},
        {"name": "Wilson Lake", "lat": 44.6150, "lon": -70.2400},
        {"name": "Kineowatha Park", "lat": 44.5890, "lon": -70.2150},
        {"name": "Wilson Stream", "lat": 44.5800, "lon": -70.2100},
        {"name": "East Wilton", "lat": 44.5950, "lon": -70.1900},
        {"name": "Dryden", "lat": 44.5700, "lon": -70.2500},
        {"name": "Temple (nearby)", "lat": 44.6700, "lon": -70.2300},
        {"name": "Farmington Falls (nearby)", "lat": 44.6350, "lon": -70.0850},
    ]

    all_formations = []

    for area in AREAS:
        print(f"\n  Scanning: {area['name']}...")

        # Quick scan at key point
        mound = detect_mound_formation(area['lat'], area['lon'])
        if mound:
            mound.description = f"{area['name']}: {mound.description}"
            all_formations.append(mound)

        depression = detect_depression(area['lat'], area['lon'])
        if depression:
            depression.description = f"{area['name']}: {depression.description}"
            all_formations.append(depression)

    return all_formations


def main():
    print("=" * 70)
    print("   WILTON, MAINE - UNUSUAL FORMATION DETECTOR")
    print("   Using Maine LiDAR-derived elevation data (1m resolution)")
    print("=" * 70)

    print(f"\n📍 Search area centered on: {WILTON_CENTER[0]:.4f}°N, {WILTON_CENTER[1]:.4f}°W")
    print(f"🗺️  Wilton, Franklin County, Maine")

    # Test API connection
    print("\n🔗 Testing Maine DEM service connection...")
    test_elev = get_elevation_at_point(WILTON_CENTER[0], WILTON_CENTER[1])

    if test_elev:
        print(f"   ✓ Connected! Wilton center elevation: {test_elev:.1f}m")
    else:
        print("   ⚠ Could not connect to Maine DEM service")
        print("   Trying alternative approach...")

    # Scan known areas
    print("\n🔍 Scanning known areas around Wilton...")
    formations = scan_known_interesting_areas()

    # Also do a focused grid scan near town center
    print("\n🔍 Running grid scan (500m radius around town center)...")
    grid_formations = scan_grid(WILTON_CENTER[0], WILTON_CENTER[1],
                                 radius_km=0.5, grid_spacing_m=100)
    formations.extend(grid_formations)

    # Remove duplicates (within 50m of each other)
    unique_formations = []
    for f in formations:
        is_duplicate = False
        for uf in unique_formations:
            dist = math.sqrt((f.lat - uf.lat)**2 + (f.lon - uf.lon)**2) * 111000
            if dist < 50:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_formations.append(f)

    # Sort by confidence
    unique_formations.sort(key=lambda x: x.confidence, reverse=True)

    # Report
    print("\n" + "=" * 70)
    print("   RESULTS: UNUSUAL FORMATIONS DETECTED")
    print("=" * 70)

    if unique_formations:
        mounds = [f for f in unique_formations if f.type == "mound"]
        depressions = [f for f in unique_formations if f.type == "depression"]

        print(f"\n   Found {len(unique_formations)} unusual formations:")
        print(f"   • Elevated mounds: {len(mounds)}")
        print(f"   • Depressions: {len(depressions)}")

        print("\n🔴 TOP FORMATIONS:\n")

        for i, f in enumerate(unique_formations[:10], 1):
            print(f"   {i}. {f.type.upper()}")
            print(f"      📍 {f.lat:.5f}°N, {f.lon:.5f}°W")
            print(f"      📊 {f.description}")
            print(f"      🎯 Confidence: {f.confidence:.0%}")
            print(f"      🗺️  https://www.google.com/maps?q={f.lat},{f.lon}")
            print()
    else:
        print("\n   No significant formations detected in initial scan.")
        print("   This could mean:")
        print("   • The area has relatively uniform terrain")
        print("   • Formations are smaller than detection threshold")
        print("   • API connection issues")

    # Known interesting sites to check manually
    print("\n" + "=" * 70)
    print("   AREAS OF INTEREST TO CHECK MANUALLY")
    print("=" * 70)
    print("""
   Based on New England archaeology patterns, check these areas:

   1. STONE WALLS & CELLAR HOLES
      Old farmstead sites often visible in LiDAR
      Look near: Wilson Stream, forested areas

   2. NATIVE AMERICAN SITES
      Sandy River confluence areas
      Lake/pond margins (Wilson Lake)

   3. COLONIAL-ERA FEATURES
      Road intersections
      Mill sites along streams

   4. GEOLOGICAL FEATURES
      Glacial erratics
      Esker ridges
      Kettle ponds

   MANUAL VIEWING TOOLS:
   • Maine LiDAR viewer: https://mgs-maine.opendata.arcgis.com/
   • USGS TopoView: https://ngmdb.usgs.gov/topoview/
""")

    return unique_formations


if __name__ == "__main__":
    import urllib.parse
    formations = main()
