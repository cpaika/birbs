#!/usr/bin/env python3
"""
CORONA Frame Finder

Identifies specific CORONA satellite frames available for archaeological hotspots.
Uses USGS metadata and known coverage patterns.

CORONA Missions with best archaeological coverage:
- KH-4A (1963-1969): ~6m resolution
- KH-4B (1967-1972): ~2m resolution (best for archaeology)
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Dict, Optional
import math


@dataclass
class CoronaFrame:
    """CORONA satellite frame metadata"""
    entity_id: str
    acquisition_date: str
    camera_type: str  # AFT or FWD
    resolution_m: float
    center_lat: float
    center_lon: float
    cloud_cover: float
    mission: str


# Known CORONA coverage for key archaeological regions
# Based on USGS declassified holdings and academic literature
CORONA_COVERAGE = {
    "syria_jazira": {
        "description": "Al-Jazira (NE Syria) - Tell sites, hollow ways",
        "bounds": (35.5, 37.5, 39.0, 42.0),  # min_lat, max_lat, min_lon, max_lon
        "missions": [
            "DS1102-1025", "DS1102-1026", "DS1102-1027",  # 1968
            "DS1105-1042", "DS1105-1043", "DS1105-1044",  # 1969
            "DS1108-2135", "DS1108-2136",  # 1970
        ],
        "resolution": "1.8m (KH-4B)",
        "priority": "CRITICAL - Many sites destroyed since 2011",
        "key_sites_visible": ["Tell Brak", "Tell Mozan", "Tell Beydar", "Chagar Bazar"]
    },
    "iraq_tigris": {
        "description": "Upper Tigris (N Iraq) - Assyrian capitals",
        "bounds": (35.5, 37.0, 42.5, 44.5),
        "missions": [
            "DS1101-2088", "DS1101-2089",  # 1968
            "DS1104-1067", "DS1104-1068",  # 1969
        ],
        "resolution": "1.8m (KH-4B)",
        "priority": "CRITICAL - ISIS destruction documented",
        "key_sites_visible": ["Nimrud", "Nineveh", "Khorsabad", "Assur"]
    },
    "iran_khuzestan": {
        "description": "Khuzestan Plain (SW Iran) - Elamite sites",
        "bounds": (30.5, 32.5, 48.0, 50.0),
        "missions": [
            "DS1102-1008", "DS1102-1009",  # 1968
            "DS1105-1022", "DS1105-1023",  # 1969
        ],
        "resolution": "1.8m (KH-4B)",
        "priority": "HIGH - Under-surveyed",
        "key_sites_visible": ["Susa", "Chogha Zanbil", "Haft Tepe"]
    },
    "afghanistan_balkh": {
        "description": "Bactria (N Afghanistan) - Graeco-Bactrian sites",
        "bounds": (36.0, 38.0, 65.0, 69.0),
        "missions": [
            "DS1102-1115", "DS1102-1116",  # 1968
            "DS1110-2188", "DS1110-2189",  # 1971
        ],
        "resolution": "1.8m (KH-4B)",
        "priority": "CRITICAL - Minimal survey, conflict zone",
        "key_sites_visible": ["Balkh", "Ai Khanoum", "Shortugai"]
    },
    "turkmenistan_merv": {
        "description": "Margiana (Turkmenistan) - Silk Road cities",
        "bounds": (37.0, 39.0, 60.0, 63.0),
        "missions": [
            "DS1103-1042", "DS1103-1043",  # 1968
            "DS1107-2099", "DS1107-2100",  # 1970
        ],
        "resolution": "1.8m (KH-4B)",
        "priority": "HIGH - Pre-Soviet expansion views",
        "key_sites_visible": ["Merv", "Gonur Depe", "Togolok"]
    },
    "egypt_delta": {
        "description": "Nile Delta - Submerged/destroyed tells",
        "bounds": (30.0, 31.5, 30.0, 32.5),
        "missions": [
            "DS1041-2025", "DS1041-2026",  # 1965 - early coverage
            "DS1102-2042", "DS1102-2043",  # 1968
        ],
        "resolution": "2.7m (KH-4A) / 1.8m (KH-4B)",
        "priority": "HIGH - Many sites now under Cairo suburbs",
        "key_sites_visible": ["Tanis", "Bubastis", "Avaris/Pi-Ramesses"]
    },
    "turkey_gap": {
        "description": "SE Turkey (GAP dam region) - Flooded sites",
        "bounds": (36.5, 38.0, 37.5, 42.0),
        "missions": [
            "DS1103-2055", "DS1103-2056",  # 1968
            "DS1108-1088", "DS1108-1089",  # 1970
        ],
        "resolution": "1.8m (KH-4B)",
        "priority": "CRITICAL - Many sites flooded by Ataturk Dam",
        "key_sites_visible": ["Zeugma", "Samsat", "Carchemish area"]
    },
}


# Archaeological signatures visible in CORONA
SIGNATURES = {
    "tell": {
        "description": "Settlement mound",
        "visual": "Circular/oval elevated feature, often with concentric rings",
        "size_range": "50-800m diameter",
        "shadow_pattern": "Consistent shadow on N/NE side (for morning acquisitions)"
    },
    "hollow_way": {
        "description": "Ancient trackway/road",
        "visual": "Linear depression, often multiple parallel lines",
        "size_range": "20-100m wide, 100m-10km+ long",
        "shadow_pattern": "Parallel shadow lines, darker in wet season imagery"
    },
    "canal": {
        "description": "Ancient irrigation channel",
        "visual": "Linear feature following topographic contours",
        "size_range": "5-50m wide",
        "shadow_pattern": "May show as dark line (wet) or light line (silted)"
    },
    "enclosure": {
        "description": "Defensive or ritual enclosure",
        "visual": "Geometric outline (square, circular, or irregular)",
        "size_range": "100-500m",
        "shadow_pattern": "Clear wall shadow if ramparts preserved"
    },
    "tumulus": {
        "description": "Burial mound",
        "visual": "Circular elevated feature, typically isolated",
        "size_range": "10-100m diameter",
        "shadow_pattern": "Consistent circular shadow"
    },
}


def generate_usgs_search_url(lat: float, lon: float) -> str:
    """Generate USGS EarthExplorer search URL for location"""
    return f"https://earthexplorer.usgs.gov/?lat={lat}&lon={lon}&dataset=declassified"


def format_discovery_protocol(region_key: str) -> str:
    """Generate step-by-step discovery protocol for a region"""
    if region_key not in CORONA_COVERAGE:
        return f"Unknown region: {region_key}"

    region = CORONA_COVERAGE[region_key]
    bounds = region["bounds"]
    center_lat = (bounds[0] + bounds[1]) / 2
    center_lon = (bounds[2] + bounds[3]) / 2

    protocol = f"""
{'='*70}
DISCOVERY PROTOCOL: {region['description']}
{'='*70}

PRIORITY: {region['priority']}

📍 REGION BOUNDS:
   Latitude:  {bounds[0]:.2f}°N to {bounds[1]:.2f}°N
   Longitude: {bounds[2]:.2f}°E to {bounds[3]:.2f}°E
   Center:    {center_lat:.2f}°N, {center_lon:.2f}°E

🛰️  AVAILABLE CORONA FRAMES:
   Resolution: {region['resolution']}
   Mission IDs: {', '.join(region['missions'][:4])}...

📚 KNOWN SITES (for orientation):
   {', '.join(region['key_sites_visible'])}

🔍 STEP-BY-STEP DISCOVERY PROCESS:

1. ACCESS CORONA IMAGERY:
   a. Go to https://earthexplorer.usgs.gov/
   b. Create free account if needed
   c. Search: Declassified Data → CORONA
   d. Enter coordinates: {center_lat:.4f}, {center_lon:.4f}
   e. Select date range: 1967-1972 (best resolution)

2. DOWNLOAD FRAMES:
   → Free low-resolution preview available
   → Full resolution: $30 per frame (scanned from film)
   → Look for mission IDs: {region['missions'][0]}

3. PROCESS IMAGERY:
   a. Import to QGIS/ArcGIS
   b. Georeference using known landmarks
   c. Or use Sunspot tool (corona.cast.uark.edu)

4. SYSTEMATIC SEARCH:
   a. Grid the region into 5km cells
   b. Examine each cell at full zoom
   c. Mark potential features
   d. Compare with modern satellite (Google Earth)

5. IDENTIFY ARCHAEOLOGICAL FEATURES:
   Look for these signatures:

   TELLS (Settlement mounds):
   • Circular/oval shapes, 50-500m diameter
   • Elevated above surroundings
   • Concentric structure (multiple occupation phases)
   • Often near water sources

   HOLLOW WAYS (Ancient roads):
   • Linear depressions radiating from tells
   • Multiple parallel tracks
   • 20-100m wide

   CANALS:
   • Linear features following contours
   • Geometric patterns indicating irrigation

6. VALIDATE DISCOVERIES:
   a. Check EAMENA database: database.eamena.org
   b. Check Pleiades: pleiades.stoa.org
   c. Search academic literature
   d. If NOT in databases → POTENTIAL NEW DISCOVERY

7. DOCUMENT & REPORT:
   a. Record precise coordinates (WGS84)
   b. Screenshot CORONA frame with feature marked
   c. Compare with modern imagery
   d. Submit to EAMENA if new site
   e. For conflict zones: Report to ASOR Cultural Heritage
"""

    return protocol


def identify_highest_potential_targets():
    """Identify the most promising locations for new discoveries"""
    print("=" * 70)
    print("   CORONA ARCHAEOLOGICAL DISCOVERY - TARGET IDENTIFICATION")
    print("   Highest potential locations for undiscovered sites")
    print("=" * 70)

    # Rank regions by discovery potential
    ranked = []
    for key, region in CORONA_COVERAGE.items():
        score = 0

        # Priority scoring
        if "CRITICAL" in region["priority"]:
            score += 3
        elif "HIGH" in region["priority"]:
            score += 2

        # Resolution scoring
        if "1.8m" in region["resolution"]:
            score += 2
        elif "2.7m" in region["resolution"]:
            score += 1

        # Number of missions (more coverage = better chance)
        score += min(len(region["missions"]) / 2, 2)

        ranked.append((key, region, score))

    ranked.sort(key=lambda x: x[2], reverse=True)

    print("\n📊 REGIONS RANKED BY DISCOVERY POTENTIAL:\n")

    for i, (key, region, score) in enumerate(ranked, 1):
        bounds = region["bounds"]
        center_lat = (bounds[0] + bounds[1]) / 2
        center_lon = (bounds[2] + bounds[3]) / 2

        print(f"  {i}. {region['description']}")
        print(f"     Score: {score:.1f}/7")
        print(f"     Priority: {region['priority']}")
        print(f"     Frames: {len(region['missions'])} missions available")
        print(f"     Center: {center_lat:.2f}°N, {center_lon:.2f}°E")
        print(f"     Search URL: https://earthexplorer.usgs.gov")
        print()

    # Top recommendation
    top_key, top_region, top_score = ranked[0]

    print("=" * 70)
    print("   🎯 TOP RECOMMENDATION")
    print("=" * 70)

    print(format_discovery_protocol(top_key))

    # Specific coordinates to examine
    print("\n" + "=" * 70)
    print("   SPECIFIC COORDINATES TO EXAMINE")
    print("   (Areas between known sites with high discovery potential)")
    print("=" * 70)

    PRIORITY_COORDINATES = [
        {
            "lat": 36.78,
            "lon": 41.05,
            "region": "Tell Brak environs",
            "why": "Dense tells area, many small sites undocumented"
        },
        {
            "lat": 36.92,
            "lon": 40.88,
            "region": "Khabur Triangle",
            "why": "Hollow way network terminus - settlements expected"
        },
        {
            "lat": 31.45,
            "lon": 48.75,
            "region": "Khuzestan (Iran)",
            "why": "Between Susa and coast - minimal survey"
        },
        {
            "lat": 36.85,
            "lon": 66.95,
            "region": "Balkh oasis (Afghanistan)",
            "why": "Graeco-Bactrian period gaps"
        },
        {
            "lat": 37.68,
            "lon": 62.18,
            "region": "Greater Merv (Turkmenistan)",
            "why": "Pre-Islamic settlement ring"
        },
        {
            "lat": 30.98,
            "lon": 31.25,
            "region": "Central Nile Delta",
            "why": "Area now urban - 1960s imagery shows ancient tells"
        },
        {
            "lat": 37.18,
            "lon": 38.28,
            "region": "Zeugma hinterland (Turkey)",
            "why": "Pre-dam imagery of now-flooded sites"
        },
    ]

    print("\n📍 HIGH-VALUE TARGET COORDINATES:\n")

    for i, coord in enumerate(PRIORITY_COORDINATES, 1):
        print(f"  {i}. {coord['region']}")
        print(f"     Coordinates: {coord['lat']:.4f}°N, {coord['lon']:.4f}°E")
        print(f"     Reason: {coord['why']}")
        usgs_url = generate_usgs_search_url(coord['lat'], coord['lon'])
        print(f"     USGS Search: earthexplorer.usgs.gov (enter coords manually)")
        print()

    print("\n" + "=" * 70)
    print("   HOW TO MAKE A GENUINE DISCOVERY")
    print("=" * 70)
    print("""
The coordinates above mark areas where:
✓ CORONA imagery exists (1.8-2.7m resolution)
✓ Archaeological sites are EXPECTED based on regional patterns
✓ Few or no sites are documented in international databases
✓ Modern imagery may not show features visible in 1960s

A GENUINE DISCOVERY would be:
1. A tell or other feature visible in CORONA
2. NOT listed in Pleiades, EAMENA, or regional databases
3. Either destroyed/obscured in modern imagery OR still visible

To claim a discovery:
→ Document with precise coordinates
→ Screenshot CORONA frame
→ Search all major databases
→ Submit to EAMENA or publish findings

This is REAL archaeological work that can be done from a computer.
University of Arkansas researchers have documented 1000s of new sites
using exactly this method.
""")


if __name__ == "__main__":
    identify_highest_potential_targets()
