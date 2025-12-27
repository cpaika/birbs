#!/usr/bin/env python3
"""
Real Satellite and Bathymetric Analysis for Linear A Discovery

Uses actual data sources:
1. Open-Elevation API for terrain analysis
2. GEBCO bathymetry data via WMS
3. OpenStreetMap for coastline data
4. NASA SRTM elevation data
"""

import requests
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import time
import math
from PIL import Image
from io import BytesIO
import os


# =============================================================================
# DATA SOURCE CLIENTS
# =============================================================================

class OpenElevationClient:
    """Fetch real elevation data from Open-Elevation API."""

    BASE_URL = "https://api.open-elevation.com/api/v1/lookup"

    def get_elevation(self, lat: float, lon: float) -> Optional[float]:
        """Get elevation for a single point."""
        try:
            response = requests.get(
                self.BASE_URL,
                params={"locations": f"{lat},{lon}"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    return data['results'][0]['elevation']
        except Exception as e:
            print(f"    Elevation API error: {e}")
        return None

    def get_elevation_profile(self, points: List[Tuple[float, float]]) -> List[Optional[float]]:
        """Get elevations for multiple points."""
        locations = "|".join(f"{lat},{lon}" for lat, lon in points)
        try:
            response = requests.post(
                self.BASE_URL,
                json={"locations": [{"latitude": lat, "longitude": lon} for lat, lon in points]},
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return [r['elevation'] for r in data.get('results', [])]
        except Exception as e:
            print(f"    Elevation API error: {e}")
        return [None] * len(points)


class GEBCOBathymetryClient:
    """Fetch bathymetry data from GEBCO WMS service."""

    WMS_URL = "https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/mapserv"

    def get_bathymetry_image(self, min_lat: float, min_lon: float,
                             max_lat: float, max_lon: float,
                             width: int = 256, height: int = 256) -> Optional[Image.Image]:
        """Get bathymetry image for a region."""
        params = {
            'service': 'WMS',
            'version': '1.1.1',
            'request': 'GetMap',
            'layers': 'GEBCO_LATEST',
            'styles': '',
            'format': 'image/png',
            'transparent': 'true',
            'srs': 'EPSG:4326',
            'bbox': f"{min_lon},{min_lat},{max_lon},{max_lat}",
            'width': width,
            'height': height,
        }

        try:
            response = requests.get(self.WMS_URL, params=params, timeout=30)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
        except Exception as e:
            print(f"    GEBCO WMS error: {e}")
        return None


class OpenTopoClient:
    """Fetch elevation data from OpenTopography."""

    API_URL = "https://portal.opentopography.org/API/globaldem"

    def get_srtm_elevation(self, min_lat: float, min_lon: float,
                           max_lat: float, max_lon: float) -> Optional[dict]:
        """Get SRTM elevation data for a region."""
        params = {
            'demtype': 'SRTMGL1',
            'south': min_lat,
            'north': max_lat,
            'west': min_lon,
            'east': max_lon,
            'outputFormat': 'JSON',
        }

        try:
            response = requests.get(self.API_URL, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"    OpenTopo error: {e}")
        return None


class OverpassOSMClient:
    """Fetch coastline and archaeological site data from OpenStreetMap."""

    API_URL = "https://overpass-api.de/api/interpreter"

    def get_coastline(self, min_lat: float, min_lon: float,
                      max_lat: float, max_lon: float) -> List[Dict]:
        """Get coastline data for a region."""
        query = f"""
        [out:json][timeout:25];
        (
          way["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """

        try:
            response = requests.post(self.API_URL, data={'data': query}, timeout=30)
            if response.status_code == 200:
                return response.json().get('elements', [])
        except Exception as e:
            print(f"    OSM API error: {e}")
        return []

    def get_archaeological_sites(self, min_lat: float, min_lon: float,
                                  max_lat: float, max_lon: float) -> List[Dict]:
        """Get known archaeological sites from OSM."""
        query = f"""
        [out:json][timeout:25];
        (
          node["historic"="archaeological_site"]({min_lat},{min_lon},{max_lat},{max_lon});
          way["historic"="archaeological_site"]({min_lat},{min_lon},{max_lat},{max_lon});
          node["historic"="ruins"]({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        """

        try:
            response = requests.post(self.API_URL, data={'data': query}, timeout=30)
            if response.status_code == 200:
                return response.json().get('elements', [])
        except Exception as e:
            print(f"    OSM API error: {e}")
        return []


# =============================================================================
# ANALYSIS CLASSES
# =============================================================================

@dataclass
class RealAnalysisResult:
    """Results from real data analysis."""
    location: str
    lat: float
    lon: float
    elevation_m: Optional[float]
    distance_to_coast_m: Optional[float]
    nearby_archaeological_sites: List[str]
    bathymetry_analyzed: bool
    anomalies_detected: List[str]
    priority_score: float
    data_sources_used: List[str]
    raw_data: Dict


class RealSatelliteAnalyzer:
    """Analyze real satellite and elevation data."""

    def __init__(self):
        self.elevation_client = OpenElevationClient()
        self.bathymetry_client = GEBCOBathymetryClient()
        self.osm_client = OverpassOSMClient()

    def analyze_site(self, name: str, lat: float, lon: float,
                     radius_km: float = 2.0) -> RealAnalysisResult:
        """Perform real analysis of a potential site."""

        print(f"\n  📍 Analyzing: {name}")
        print(f"     Coordinates: {lat:.4f}°N, {lon:.4f}°E")

        data_sources = []
        anomalies = []
        raw_data = {}

        # 1. Get real elevation data
        print("     Fetching elevation data...")
        elevation = self.elevation_client.get_elevation(lat, lon)
        if elevation is not None:
            data_sources.append("Open-Elevation API")
            raw_data['elevation'] = elevation
            print(f"     ✓ Elevation: {elevation}m")

            # Check for coastal/submerged status
            if elevation < 0:
                anomalies.append(f"SUBMERGED: Currently {abs(elevation):.1f}m below sea level")
            elif elevation < 5:
                anomalies.append(f"LOW COASTAL: Only {elevation:.1f}m above sea level")
        else:
            print("     ✗ Elevation data unavailable")

        # 2. Get elevation profile (look for anomalies)
        print("     Analyzing elevation profile...")
        profile_points = [
            (lat + 0.005, lon),
            (lat - 0.005, lon),
            (lat, lon + 0.005),
            (lat, lon - 0.005),
            (lat + 0.003, lon + 0.003),
            (lat - 0.003, lon - 0.003),
        ]

        time.sleep(1)  # Rate limiting
        elevations = self.elevation_client.get_elevation_profile(profile_points)
        valid_elevations = [e for e in elevations if e is not None]

        if valid_elevations:
            elev_std = np.std(valid_elevations)
            elev_range = max(valid_elevations) - min(valid_elevations)
            raw_data['elevation_profile'] = valid_elevations
            raw_data['elevation_std'] = elev_std
            raw_data['elevation_range'] = elev_range

            print(f"     ✓ Elevation range: {elev_range:.1f}m (std: {elev_std:.1f}m)")

            # Flat areas near coast often indicate harbors
            if elev_range < 5 and elevation is not None and elevation < 20:
                anomalies.append("FLAT COASTAL PLAIN: Potential ancient harbor area")

        # 3. Get nearby archaeological sites from OSM
        print("     Searching for known archaeological sites...")
        time.sleep(1)

        # Define search box
        delta = radius_km / 111  # Rough conversion km to degrees
        sites = self.osm_client.get_archaeological_sites(
            lat - delta, lon - delta,
            lat + delta, lon + delta
        )

        nearby_sites = []
        for site in sites:
            site_name = site.get('tags', {}).get('name', 'Unnamed site')
            site_type = site.get('tags', {}).get('historic', 'unknown')
            nearby_sites.append(f"{site_name} ({site_type})")
            data_sources.append("OpenStreetMap")

        if nearby_sites:
            print(f"     ✓ Found {len(nearby_sites)} nearby archaeological sites")
            for s in nearby_sites[:3]:
                print(f"       • {s}")
            if len(nearby_sites) > 3:
                print(f"       ... and {len(nearby_sites) - 3} more")
        else:
            print("     ○ No registered archaeological sites nearby")
            # This could be interesting - unexplored area!
            if elevation is not None and elevation > 0:
                anomalies.append("NO REGISTERED SITES: Potentially unexplored area")

        raw_data['nearby_sites'] = nearby_sites

        # 4. Analyze bathymetry for coastal sites
        bathymetry_analyzed = False
        if elevation is not None and elevation < 50:
            print("     Fetching bathymetry data...")
            time.sleep(1)

            bath_img = self.bathymetry_client.get_bathymetry_image(
                lat - 0.05, lon - 0.05,
                lat + 0.05, lon + 0.05
            )

            if bath_img:
                bathymetry_analyzed = True
                data_sources.append("GEBCO Bathymetry")

                # Analyze the bathymetry image
                img_array = np.array(bath_img.convert('L'))
                bath_mean = np.mean(img_array)
                bath_std = np.std(img_array)

                raw_data['bathymetry_mean'] = float(bath_mean)
                raw_data['bathymetry_std'] = float(bath_std)

                print(f"     ✓ Bathymetry analyzed (mean: {bath_mean:.1f}, std: {bath_std:.1f})")

                # High variance in shallow areas could indicate submerged structures
                if bath_std > 30 and elevation is not None and elevation < 10:
                    anomalies.append("HIGH BATHYMETRIC VARIANCE: Possible submerged structures")

        # 5. Calculate priority score
        priority_score = self._calculate_priority(
            elevation, anomalies, nearby_sites, bathymetry_analyzed
        )

        # Estimate distance to coast (simplified)
        distance_to_coast = None
        if elevation is not None:
            if elevation < 0:
                distance_to_coast = 0  # Underwater
            elif elevation < 10:
                distance_to_coast = elevation * 50  # Rough estimate
            else:
                distance_to_coast = elevation * 100

        return RealAnalysisResult(
            location=name,
            lat=lat,
            lon=lon,
            elevation_m=elevation,
            distance_to_coast_m=distance_to_coast,
            nearby_archaeological_sites=nearby_sites,
            bathymetry_analyzed=bathymetry_analyzed,
            anomalies_detected=anomalies,
            priority_score=priority_score,
            data_sources_used=list(set(data_sources)),
            raw_data=raw_data
        )

    def _calculate_priority(self, elevation: Optional[float],
                           anomalies: List[str],
                           nearby_sites: List[str],
                           bathymetry: bool) -> float:
        """Calculate priority score based on real data."""
        score = 0.0

        # Elevation factors
        if elevation is not None:
            if elevation < 0:
                score += 0.4  # Submerged = high priority
            elif elevation < 10:
                score += 0.3  # Coastal
            elif elevation < 50:
                score += 0.2  # Near coast
            elif elevation > 500:
                score += 0.15  # Peak sanctuary potential

        # Anomalies increase priority
        score += len(anomalies) * 0.1

        # No nearby sites = potentially unexplored
        if not nearby_sites:
            score += 0.2

        # Bathymetry data available = can analyze underwater
        if bathymetry:
            score += 0.1

        return min(1.0, score)


# =============================================================================
# PRIORITY SITE LIST
# =============================================================================

PRIORITY_SITES = [
    # Underwater sites (highest priority)
    ("Submerged Olous", 35.2758, 25.7286),
    ("Mochlos offshore", 35.1833, 25.9167),
    ("Zakros harbor", 35.0900, 26.2700),

    # Known sites with potential
    ("Gournia unexcavated", 35.1089, 25.7761),
    ("Kommos harbor", 34.9667, 24.7500),
    ("Phaistos periphery", 35.0500, 24.8100),

    # Peak sanctuaries
    ("Mount Juktas", 35.2167, 25.1500),
    ("Kato Syme sanctuary", 35.0667, 25.4333),

    # Trade route colonies
    ("Miletos area", 37.5306, 27.2778),
    ("Trianda Rhodes", 36.4167, 28.1833),

    # Unexplored areas
    ("Gavdos island", 34.8500, 24.0833),
    ("Dia island", 35.4667, 25.2167),
]


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("   REAL SATELLITE & BATHYMETRIC ANALYSIS FOR LINEAR A DISCOVERY")
    print("=" * 70)
    print("\nUsing actual data from:")
    print("  • Open-Elevation API (SRTM elevation data)")
    print("  • GEBCO Web Map Service (ocean bathymetry)")
    print("  • OpenStreetMap Overpass API (archaeological sites)")
    print("\n" + "=" * 70)

    analyzer = RealSatelliteAnalyzer()
    results = []

    for name, lat, lon in PRIORITY_SITES:
        try:
            result = analyzer.analyze_site(name, lat, lon)
            results.append(result)

            # Rate limiting to avoid API throttling
            time.sleep(2)

        except Exception as e:
            print(f"\n  ❌ Error analyzing {name}: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("   ANALYSIS SUMMARY")
    print("=" * 70)

    # Sort by priority
    results.sort(key=lambda x: -x.priority_score)

    print("\n📊 SITES RANKED BY DISCOVERY POTENTIAL:\n")
    print(f"{'Rank':<5} {'Site':<25} {'Score':<8} {'Elev':<10} {'Anomalies'}")
    print("-" * 70)

    for i, r in enumerate(results, 1):
        elev_str = f"{r.elevation_m:.0f}m" if r.elevation_m is not None else "N/A"
        anomaly_count = len(r.anomalies_detected)
        print(f"{i:<5} {r.location:<25} {r.priority_score:.2f}    {elev_str:<10} {anomaly_count} detected")

    # Detailed results for top sites
    print("\n" + "=" * 70)
    print("   TOP PRIORITY SITES - DETAILED ANALYSIS")
    print("=" * 70)

    for r in results[:5]:
        print(f"\n🎯 {r.location}")
        print(f"   Coordinates: {r.lat:.4f}°N, {r.lon:.4f}°E")
        print(f"   Priority Score: {r.priority_score:.2f}")
        print(f"   Elevation: {r.elevation_m}m" if r.elevation_m else "   Elevation: Unknown")
        print(f"   Data Sources: {', '.join(r.data_sources_used)}")

        if r.anomalies_detected:
            print(f"   Anomalies:")
            for a in r.anomalies_detected:
                print(f"      ⚠️  {a}")

        if r.nearby_archaeological_sites:
            print(f"   Nearby Sites: {len(r.nearby_archaeological_sites)}")

    # Final recommendations
    print("\n" + "=" * 70)
    print("   RECOMMENDATIONS BASED ON REAL DATA")
    print("=" * 70)

    submerged = [r for r in results if r.elevation_m is not None and r.elevation_m < 0]
    coastal = [r for r in results if r.elevation_m is not None and 0 <= r.elevation_m < 10]
    unexplored = [r for r in results if not r.nearby_archaeological_sites]

    print(f"""
   FINDINGS FROM REAL DATA ANALYSIS:

   📍 Submerged Sites: {len(submerged)}
      These are currently underwater and require diving/ROV survey
""")
    for r in submerged[:3]:
        print(f"      • {r.location}: {r.elevation_m:.1f}m below sea level")

    print(f"""
   📍 Coastal Sites: {len(coastal)}
      Low elevation, high potential for harbor archaeology
""")
    for r in coastal[:3]:
        print(f"      • {r.location}: {r.elevation_m:.1f}m elevation")

    print(f"""
   📍 Potentially Unexplored: {len(unexplored)}
      No registered archaeological sites in OSM - may be unexplored
""")
    for r in unexplored[:3]:
        print(f"      • {r.location}")

    print("""
   ═══════════════════════════════════════════════════════════════════

   NOTE: This analysis uses REAL DATA from public APIs.
   However, for professional archaeological prospecting you would need:

   1. Commercial satellite imagery (WorldView, Pleiades)
   2. LiDAR surveys for vegetation penetration
   3. Magnetometry/resistivity surveys
   4. Professional bathymetric multibeam sonar
   5. Archaeological permits from Greek Ministry of Culture
    """)

    return results


if __name__ == '__main__':
    results = main()
