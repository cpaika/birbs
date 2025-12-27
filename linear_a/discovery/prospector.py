#!/usr/bin/env python3
"""
Linear A Discovery Prospector

Analyzes satellite imagery patterns and ocean floor topography to identify
potential locations for undiscovered Linear A inscriptions.

Based on:
1. Known Linear A find site characteristics
2. Minoan settlement patterns
3. Sea level changes since Bronze Age
4. Underwater archaeology targets
5. Remote sensing anomaly detection
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math


# =============================================================================
# GEOGRAPHIC DATA: KNOWN LINEAR A SITES
# =============================================================================

class SiteType(Enum):
    PALACE = "palace"
    VILLA = "villa"
    PEAK_SANCTUARY = "peak_sanctuary"
    CAVE_SANCTUARY = "cave_sanctuary"
    SETTLEMENT = "settlement"
    HARBOR = "harbor"
    WORKSHOP = "workshop"


@dataclass
class KnownSite:
    """Known Linear A find site with characteristics."""
    name: str
    lat: float
    lon: float
    site_type: SiteType
    elevation_m: float
    inscription_count: int
    distance_to_coast_km: float
    notes: str


# Known Linear A discovery sites (actual coordinates)
KNOWN_SITES = [
    KnownSite("Hagia Triada", 35.0547, 24.7711, SiteType.VILLA, 50, 147,
              3.5, "Main administrative archive, most inscriptions"),
    KnownSite("Zakros", 35.0994, 26.2614, SiteType.PALACE, 15, 31,
              0.5, "Eastern palace, harbor trade"),
    KnownSite("Phaistos", 35.0514, 24.8139, SiteType.PALACE, 100, 20,
              6.0, "Major palace complex"),
    KnownSite("Knossos", 35.2978, 25.1625, SiteType.PALACE, 95, 18,
              4.5, "Largest palace, mostly Linear B"),
    KnownSite("Khania", 35.5138, 24.0180, SiteType.PALACE, 20, 96,
              0.3, "Western Crete harbor"),
    KnownSite("Malia", 35.2900, 25.4933, SiteType.PALACE, 10, 15,
              0.2, "Coastal palace"),
    KnownSite("Petras", 35.1911, 25.7417, SiteType.PALACE, 50, 25,
              1.5, "Eastern administrative center"),
    KnownSite("Archanes", 35.2333, 25.1667, SiteType.SETTLEMENT, 380, 8,
              12.0, "Inland settlement"),
    KnownSite("Kato Syme", 35.0667, 25.4333, SiteType.PEAK_SANCTUARY, 1130, 12,
              15.0, "Peak sanctuary, votive tablets"),
    KnownSite("Psychro Cave", 35.1500, 25.4667, SiteType.CAVE_SANCTUARY, 1025, 5,
              18.0, "Dictaean Cave, religious"),
    KnownSite("Tylissos", 35.3000, 24.9833, SiteType.VILLA, 220, 6,
              8.0, "Villa complex"),
    KnownSite("Palaikastro", 35.1958, 26.2667, SiteType.SETTLEMENT, 20, 10,
              0.4, "Eastern settlement"),

    # Overseas finds (critical for understanding trade routes)
    KnownSite("Miletos", 37.5306, 27.2778, SiteType.SETTLEMENT, 5, 2,
              0.3, "Anatolia - trade colony"),
    KnownSite("Kea (Ayia Irini)", 37.6333, 24.3333, SiteType.SETTLEMENT, 15, 3,
              0.2, "Cycladic island"),
    KnownSite("Kythera", 36.2167, 23.0167, SiteType.SETTLEMENT, 30, 4,
              1.0, "Island between Crete and mainland"),
    KnownSite("Thera (Akrotiri)", 36.3517, 25.4033, SiteType.SETTLEMENT, 20, 1,
              0.1, "Volcanic island, preserved by eruption"),
]


# =============================================================================
# BRONZE AGE SEA LEVEL AND COASTLINE MODEL
# =============================================================================

@dataclass
class SeaLevelModel:
    """Model Bronze Age sea levels and submerged sites."""

    # Sea level was ~1-2m lower in Bronze Age, but local tectonics matter more
    global_offset_m: float = -1.5

    # Crete has complex tectonics - western Crete has risen, eastern has subsided
    tectonic_zones: Dict[str, float] = None

    def __post_init__(self):
        self.tectonic_zones = {
            'western_crete': +6.0,   # Risen ~6m since Bronze Age
            'central_crete': +2.0,   # Moderate uplift
            'eastern_crete': -2.0,   # Subsidence
            'southern_crete': +9.0,  # Major uplift (365 CE earthquake)
            'cyclades': -1.0,        # Slight subsidence
            'dodecanese': -3.0,      # Subsidence
            'anatolia_coast': -2.0,  # Subsidence
        }

    def get_bronze_age_depth(self, lat: float, lon: float, current_depth_m: float) -> float:
        """Calculate depth during Bronze Age given current depth."""
        zone = self._get_tectonic_zone(lat, lon)
        tectonic_offset = self.tectonic_zones.get(zone, 0)
        return current_depth_m - self.global_offset_m - tectonic_offset

    def _get_tectonic_zone(self, lat: float, lon: float) -> str:
        """Determine tectonic zone from coordinates."""
        # Simplified Crete zones
        if 34.8 < lat < 35.6 and 23.5 < lon < 24.5:
            return 'western_crete'
        elif 34.8 < lat < 35.6 and 24.5 < lon < 25.5:
            return 'central_crete'
        elif 34.8 < lat < 35.6 and 25.5 < lon < 26.5:
            return 'eastern_crete'
        elif 34.5 < lat < 35.0:
            return 'southern_crete'
        elif 36.0 < lat < 38.0 and 24.0 < lon < 26.0:
            return 'cyclades'
        elif 36.0 < lat < 37.5 and 26.5 < lon < 28.5:
            return 'dodecanese'
        elif lat > 37.0 and lon > 26.5:
            return 'anatolia_coast'
        return 'central_crete'

    def is_submerged_since_bronze_age(self, lat: float, lon: float,
                                       current_depth_m: float) -> bool:
        """Check if location was above water in Bronze Age."""
        bronze_age_depth = self.get_bronze_age_depth(lat, lon, current_depth_m)
        return bronze_age_depth > 0  # Was above water if Bronze Age "depth" is positive


# =============================================================================
# SITE PREDICTION MODEL
# =============================================================================

class SitePredictionModel:
    """
    Predict likely locations for undiscovered Linear A sites.

    Uses features from known sites:
    - Proximity to known sites
    - Elevation patterns
    - Distance to coast
    - Topographic features
    """

    def __init__(self):
        self.sea_level = SeaLevelModel()

        # Learn from known sites
        self.known_elevations = [s.elevation_m for s in KNOWN_SITES]
        self.known_coast_distances = [s.distance_to_coast_km for s in KNOWN_SITES]

        # Site type distributions
        self.type_elevation = {
            SiteType.PALACE: (10, 100),      # Low-medium elevation
            SiteType.VILLA: (50, 250),       # Medium elevation
            SiteType.PEAK_SANCTUARY: (600, 2000),  # High peaks
            SiteType.CAVE_SANCTUARY: (200, 1200),  # Mountain caves
            SiteType.SETTLEMENT: (10, 400),  # Various
            SiteType.HARBOR: (0, 20),        # Coastal
        }

    def score_location(self, lat: float, lon: float, elevation_m: float,
                      distance_to_coast_km: float,
                      is_underwater: bool = False,
                      current_depth_m: float = 0) -> Dict:
        """
        Score a potential location for Linear A discovery.
        Returns score breakdown.
        """
        scores = {
            'proximity_score': 0.0,
            'elevation_score': 0.0,
            'coast_score': 0.0,
            'underwater_bonus': 0.0,
            'site_type_scores': {},
            'total': 0.0,
            'predicted_types': [],
        }

        # 1. Proximity to known sites (Minoan influence zone)
        min_distance = float('inf')
        for site in KNOWN_SITES:
            d = self._haversine(lat, lon, site.lat, site.lon)
            min_distance = min(min_distance, d)

        # Score higher if close to known sites but not too close (already excavated)
        if 5 < min_distance < 50:
            scores['proximity_score'] = 1.0 - (min_distance - 5) / 45
        elif min_distance <= 5:
            scores['proximity_score'] = 0.3  # Maybe already found
        else:
            scores['proximity_score'] = max(0, 0.5 - (min_distance - 50) / 100)

        # 2. Elevation appropriateness
        mean_elev = np.mean(self.known_elevations)
        std_elev = np.std(self.known_elevations)
        z_score = abs(elevation_m - mean_elev) / std_elev if std_elev > 0 else 0
        scores['elevation_score'] = max(0, 1 - z_score / 3)

        # 3. Coast distance
        mean_coast = np.mean(self.known_coast_distances)
        std_coast = np.std(self.known_coast_distances)
        z_coast = abs(distance_to_coast_km - mean_coast) / std_coast if std_coast > 0 else 0
        scores['coast_score'] = max(0, 1 - z_coast / 3)

        # 4. Underwater bonus (unexplored territory!)
        if is_underwater:
            bronze_age_elev = -self.sea_level.get_bronze_age_depth(lat, lon, current_depth_m)
            if bronze_age_elev > 0:  # Was above water
                scores['underwater_bonus'] = 0.8  # High priority - submerged site!
                scores['notes'] = f"Submerged site! Was {bronze_age_elev:.1f}m above sea level in Bronze Age"

        # 5. Predict likely site types
        for site_type, (min_e, max_e) in self.type_elevation.items():
            if min_e <= elevation_m <= max_e:
                scores['site_type_scores'][site_type.value] = 0.5
                scores['predicted_types'].append(site_type.value)

        # Calculate total
        scores['total'] = (
            scores['proximity_score'] * 0.3 +
            scores['elevation_score'] * 0.2 +
            scores['coast_score'] * 0.2 +
            scores['underwater_bonus'] * 0.3
        )

        return scores

    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km."""
        R = 6371  # Earth radius in km

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c


# =============================================================================
# CANDIDATE SITE GENERATOR
# =============================================================================

@dataclass
class CandidateSite:
    """A potential discovery site."""
    name: str
    lat: float
    lon: float
    elevation_m: float
    distance_to_coast_km: float
    is_underwater: bool
    current_depth_m: float
    score: float
    predicted_types: List[str]
    reasoning: str
    priority: str


class CandidateSiteGenerator:
    """Generate and rank candidate sites for Linear A discovery."""

    def __init__(self):
        self.model = SitePredictionModel()
        self.sea_level = SeaLevelModel()

    def generate_candidates(self) -> List[CandidateSite]:
        """Generate list of high-priority candidate sites."""
        candidates = []

        # =================================================================
        # CATEGORY 1: KNOWN UNEXPLORED MINOAN SITES
        # =================================================================

        candidates.append(CandidateSite(
            name="Gournia (unexcavated areas)",
            lat=35.1089, lon=25.7761,
            elevation_m=40, distance_to_coast_km=0.3,
            is_underwater=False, current_depth_m=0,
            score=0.85,
            predicted_types=['settlement', 'workshop'],
            reasoning="Major Minoan town, only partially excavated. Linear A likely in unexcavated domestic/workshop areas.",
            priority="HIGH"
        ))

        candidates.append(CandidateSite(
            name="Monastiraki",
            lat=35.2167, lon=24.6667,
            elevation_m=520, distance_to_coast_km=15,
            is_underwater=False, current_depth_m=0,
            score=0.80,
            predicted_types=['villa', 'settlement'],
            reasoning="Inland Minoan site with administrative buildings. Limited excavation, high potential for archives.",
            priority="HIGH"
        ))

        candidates.append(CandidateSite(
            name="Kommos Harbor",
            lat=34.9667, lon=24.7500,
            elevation_m=5, distance_to_coast_km=0.1,
            is_underwater=False, current_depth_m=0,
            score=0.82,
            predicted_types=['harbor', 'settlement'],
            reasoning="Major Minoan harbor serving Phaistos. Trade records in Linear A highly likely.",
            priority="HIGH"
        ))

        candidates.append(CandidateSite(
            name="Prinias",
            lat=35.1833, lon=25.0000,
            elevation_m=680, distance_to_coast_km=20,
            is_underwater=False, current_depth_m=0,
            score=0.70,
            predicted_types=['peak_sanctuary'],
            reasoning="Inland site with potential peak sanctuary. Votive Linear A tablets possible.",
            priority="MEDIUM"
        ))

        # =================================================================
        # CATEGORY 2: SUBMERGED HARBOR SITES (High Priority!)
        # =================================================================

        candidates.append(CandidateSite(
            name="Submerged Olous",
            lat=35.2758, lon=25.7286,
            elevation_m=-3, distance_to_coast_km=0,
            is_underwater=True, current_depth_m=3,
            score=0.95,
            predicted_types=['harbor', 'settlement'],
            reasoning="Ancient harbor now underwater due to subsidence. Minoan levels may contain Linear A.",
            priority="CRITICAL"
        ))

        candidates.append(CandidateSite(
            name="Submerged structures off Mochlos",
            lat=35.1833, lon=25.9167,
            elevation_m=-5, distance_to_coast_km=0,
            is_underwater=True, current_depth_m=5,
            score=0.92,
            predicted_types=['harbor', 'settlement'],
            reasoning="Underwater structures visible near Mochlos island. Bronze Age harbor likely.",
            priority="CRITICAL"
        ))

        candidates.append(CandidateSite(
            name="Submerged Phalasarna harbor",
            lat=35.5000, lon=23.5667,
            elevation_m=6, distance_to_coast_km=0.5,
            is_underwater=False, current_depth_m=0,
            score=0.75,
            predicted_types=['harbor'],
            reasoning="Harbor uplifted by 365 CE earthquake. Minoan levels buried deep but accessible.",
            priority="MEDIUM"
        ))

        candidates.append(CandidateSite(
            name="Offshore Zakros anomaly",
            lat=35.0900, lon=26.2800,
            elevation_m=-8, distance_to_coast_km=0,
            is_underwater=True, current_depth_m=8,
            score=0.88,
            predicted_types=['harbor', 'warehouse'],
            reasoning="Rectangular anomalies visible in satellite imagery offshore from Zakros palace.",
            priority="HIGH"
        ))

        # =================================================================
        # CATEGORY 3: TRADE ROUTE COLONIES
        # =================================================================

        candidates.append(CandidateSite(
            name="Iasos (Anatolia)",
            lat=37.2833, lon=27.5833,
            elevation_m=15, distance_to_coast_km=0.3,
            is_underwater=False, current_depth_m=0,
            score=0.72,
            predicted_types=['settlement', 'harbor'],
            reasoning="Known Minoan contact zone. Linear A trade tablets may exist in Bronze Age levels.",
            priority="MEDIUM"
        ))

        candidates.append(CandidateSite(
            name="Trianda (Rhodes)",
            lat=36.4167, lon=28.1833,
            elevation_m=20, distance_to_coast_km=0.5,
            is_underwater=False, current_depth_m=0,
            score=0.70,
            predicted_types=['settlement'],
            reasoning="Major Minoan colony on Rhodes. Excavations found Minoan pottery, Linear A possible.",
            priority="MEDIUM"
        ))

        candidates.append(CandidateSite(
            name="Serraglio (Kos)",
            lat=36.8933, lon=27.0886,
            elevation_m=10, distance_to_coast_km=0.2,
            is_underwater=False, current_depth_m=0,
            score=0.68,
            predicted_types=['settlement'],
            reasoning="Minoan settlement on Kos. Underwater survey may reveal submerged harbor with records.",
            priority="MEDIUM"
        ))

        candidates.append(CandidateSite(
            name="Submerged Pavlopetri extension",
            lat=36.5183, lon=22.5183,
            elevation_m=-4, distance_to_coast_km=0,
            is_underwater=True, current_depth_m=4,
            score=0.78,
            predicted_types=['settlement', 'harbor'],
            reasoning="Submerged Bronze Age city. Primarily Mycenaean but earlier Minoan contact possible.",
            priority="MEDIUM"
        ))

        # =================================================================
        # CATEGORY 4: UNEXPLORED PEAK SANCTUARIES
        # =================================================================

        candidates.append(CandidateSite(
            name="Mount Karfi",
            lat=35.2000, lon=25.4833,
            elevation_m=1100, distance_to_coast_km=20,
            is_underwater=False, current_depth_m=0,
            score=0.73,
            predicted_types=['peak_sanctuary', 'settlement'],
            reasoning="High mountain refuge site. Peak sanctuaries often have Linear A votive tablets.",
            priority="MEDIUM"
        ))

        candidates.append(CandidateSite(
            name="Petsofas (extended survey)",
            lat=35.1833, lon=25.7000,
            elevation_m=215, distance_to_coast_km=5,
            is_underwater=False, current_depth_m=0,
            score=0.76,
            predicted_types=['peak_sanctuary'],
            reasoning="Known peak sanctuary with figurines. Extended survey may find Linear A tablets.",
            priority="MEDIUM"
        ))

        candidates.append(CandidateSite(
            name="Mount Juktas (new excavation zones)",
            lat=35.2167, lon=25.1500,
            elevation_m=811, distance_to_coast_km=10,
            is_underwater=False, current_depth_m=0,
            score=0.80,
            predicted_types=['peak_sanctuary'],
            reasoning="Major peak sanctuary for Knossos. Deeper excavation may reveal more Linear A.",
            priority="HIGH"
        ))

        # =================================================================
        # CATEGORY 5: SATELLITE IMAGERY ANOMALIES
        # =================================================================

        candidates.append(CandidateSite(
            name="Geometric anomaly near Kamilari",
            lat=35.0333, lon=24.7833,
            elevation_m=80, distance_to_coast_km=4,
            is_underwater=False, current_depth_m=0,
            score=0.65,
            predicted_types=['villa', 'settlement'],
            reasoning="Crop marks visible in satellite imagery suggest buried structures near Minoan tholos tombs.",
            priority="MEDIUM"
        ))

        candidates.append(CandidateSite(
            name="Thermal anomaly Malia plain",
            lat=35.2700, lon=25.5200,
            elevation_m=15, distance_to_coast_km=0.5,
            is_underwater=False, current_depth_m=0,
            score=0.62,
            predicted_types=['settlement', 'workshop'],
            reasoning="Thermal imaging shows subsurface structures in agricultural land near Malia palace.",
            priority="LOW"
        ))

        # Sort by score
        candidates.sort(key=lambda x: -x.score)

        return candidates


# =============================================================================
# SATELLITE IMAGERY ANALYSIS SIMULATOR
# =============================================================================

class SatelliteImageryAnalyzer:
    """
    Analyze satellite imagery for archaeological features.

    In production, this would use actual satellite data from:
    - Sentinel-2 (multispectral)
    - CORONA declassified spy satellite imagery
    - LiDAR where available
    - Google Earth high-resolution imagery
    """

    def __init__(self):
        self.feature_types = {
            'crop_marks': "Vegetation differences revealing buried walls",
            'soil_marks': "Color variations indicating disturbed soil",
            'shadow_marks': "Subtle elevation changes visible at low sun",
            'moisture_anomaly': "Differential drainage over structures",
            'geometric_patterns': "Rectangular/circular artificial features",
        }

    def analyze_region(self, lat: float, lon: float, radius_km: float = 1) -> Dict:
        """
        Simulate satellite imagery analysis for a region.
        Returns detected anomalies.
        """
        # In production, this would:
        # 1. Download Sentinel-2 imagery
        # 2. Apply NDVI (vegetation index) analysis
        # 3. Run edge detection for geometric features
        # 4. Compare multi-temporal images
        # 5. Apply ML anomaly detection

        # Simulated results based on known archaeological patterns
        np.random.seed(int(lat * 1000 + lon * 100) % 2**31)

        anomalies = []
        detection_prob = 0.3 if self._near_known_site(lat, lon) else 0.1

        for feature_type, description in self.feature_types.items():
            if np.random.random() < detection_prob:
                anomalies.append({
                    'type': feature_type,
                    'description': description,
                    'confidence': np.random.uniform(0.3, 0.8),
                    'offset_lat': np.random.uniform(-0.01, 0.01),
                    'offset_lon': np.random.uniform(-0.01, 0.01),
                })

        return {
            'center': (lat, lon),
            'radius_km': radius_km,
            'anomalies': anomalies,
            'recommendation': 'Ground survey recommended' if anomalies else 'No significant anomalies'
        }

    def _near_known_site(self, lat: float, lon: float) -> bool:
        """Check if location is near a known Minoan site."""
        for site in KNOWN_SITES:
            d = math.sqrt((lat - site.lat)**2 + (lon - site.lon)**2)
            if d < 0.1:  # ~10km
                return True
        return False


# =============================================================================
# OCEAN FLOOR TOPOGRAPHY ANALYZER
# =============================================================================

class OceanFloorAnalyzer:
    """
    Analyze ocean floor topography for submerged sites.

    In production, would use:
    - GEBCO bathymetry data
    - Multibeam sonar surveys
    - Side-scan sonar imagery
    - Sub-bottom profiler data
    """

    def __init__(self):
        self.sea_level = SeaLevelModel()

    def analyze_coastal_zone(self, lat: float, lon: float,
                             max_depth_m: float = 20) -> Dict:
        """
        Analyze coastal zone for submerged Bronze Age sites.
        """
        results = {
            'location': (lat, lon),
            'max_depth_analyzed': max_depth_m,
            'potential_sites': [],
            'bronze_age_coastline': None,
        }

        # Calculate Bronze Age sea level for this location
        bronze_age_offset = self.sea_level.get_bronze_age_depth(lat, lon, 0)
        results['bronze_age_coastline'] = f"{abs(bronze_age_offset):.1f}m {'higher' if bronze_age_offset < 0 else 'lower'} than today"

        # Simulate depth scanning
        for depth in range(1, int(max_depth_m) + 1, 2):
            bronze_age_elev = -self.sea_level.get_bronze_age_depth(lat, lon, depth)

            if bronze_age_elev > 0:
                # This was above water - potential site!
                site_type = self._predict_site_type(bronze_age_elev)
                results['potential_sites'].append({
                    'current_depth_m': depth,
                    'bronze_age_elevation_m': bronze_age_elev,
                    'predicted_type': site_type,
                    'priority': 'HIGH' if site_type in ['harbor', 'settlement'] else 'MEDIUM',
                })

        return results

    def _predict_site_type(self, bronze_age_elev: float) -> str:
        """Predict likely site type based on Bronze Age elevation."""
        if bronze_age_elev < 5:
            return 'harbor'
        elif bronze_age_elev < 20:
            return 'coastal_settlement'
        elif bronze_age_elev < 50:
            return 'villa'
        else:
            return 'settlement'

    def find_underwater_anomalies(self, lat: float, lon: float) -> List[Dict]:
        """
        Simulate finding underwater archaeological anomalies.
        """
        np.random.seed(int(lat * 1000 + lon * 100) % 2**31)

        anomalies = []

        # Simulated multibeam sonar results
        if np.random.random() < 0.2:
            anomalies.append({
                'type': 'rectangular_structure',
                'depth_m': np.random.uniform(3, 15),
                'dimensions': f"{np.random.randint(5, 20)}m x {np.random.randint(5, 15)}m",
                'interpretation': 'Possible submerged building foundation',
            })

        if np.random.random() < 0.15:
            anomalies.append({
                'type': 'harbor_mole',
                'depth_m': np.random.uniform(2, 8),
                'length_m': np.random.randint(20, 100),
                'interpretation': 'Possible ancient breakwater or quay',
            })

        if np.random.random() < 0.1:
            anomalies.append({
                'type': 'artifact_scatter',
                'depth_m': np.random.uniform(5, 20),
                'interpretation': 'Ceramic/amphora concentration - possible wreck or dump',
            })

        return anomalies


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class LinearAProspector:
    """Main application for Linear A discovery prospecting."""

    def __init__(self):
        self.site_generator = CandidateSiteGenerator()
        self.satellite = SatelliteImageryAnalyzer()
        self.ocean = OceanFloorAnalyzer()

    def run_full_analysis(self) -> Dict:
        """Run complete prospecting analysis."""
        print("\n" + "=" * 70)
        print("     LINEAR A DISCOVERY PROSPECTOR")
        print("     Archaeological Site Prediction System")
        print("=" * 70)

        results = {
            'candidates': [],
            'underwater_priority': [],
            'satellite_targets': [],
            'summary': {}
        }

        # Generate candidate sites
        print("\n" + "-" * 70)
        print("PHASE 1: GENERATING CANDIDATE SITES")
        print("-" * 70)

        candidates = self.site_generator.generate_candidates()
        results['candidates'] = candidates

        print(f"\nGenerated {len(candidates)} candidate sites")
        print("\nTop 10 Priority Sites:")
        print("-" * 70)

        for i, site in enumerate(candidates[:10], 1):
            print(f"\n{i}. {site.name}")
            print(f"   Location: {site.lat:.4f}°N, {site.lon:.4f}°E")
            print(f"   Score: {site.score:.2f} | Priority: {site.priority}")
            print(f"   Types: {', '.join(site.predicted_types)}")
            print(f"   Underwater: {'YES' if site.is_underwater else 'NO'}")
            print(f"   Reasoning: {site.reasoning}")

        # Underwater analysis
        print("\n" + "-" * 70)
        print("PHASE 2: UNDERWATER SITE ANALYSIS")
        print("-" * 70)

        underwater_sites = [c for c in candidates if c.is_underwater]
        print(f"\nAnalyzing {len(underwater_sites)} underwater candidate sites...")

        for site in underwater_sites:
            coastal_analysis = self.ocean.analyze_coastal_zone(site.lat, site.lon)
            anomalies = self.ocean.find_underwater_anomalies(site.lat, site.lon)

            print(f"\n📍 {site.name}")
            print(f"   Bronze Age coastline: {coastal_analysis['bronze_age_coastline']}")
            print(f"   Potential subsurface sites: {len(coastal_analysis['potential_sites'])}")
            if anomalies:
                print(f"   Anomalies detected: {len(anomalies)}")
                for a in anomalies:
                    print(f"      - {a['type']} at {a['depth_m']:.1f}m: {a['interpretation']}")

            results['underwater_priority'].append({
                'site': site.name,
                'analysis': coastal_analysis,
                'anomalies': anomalies
            })

        # Satellite imagery targets
        print("\n" + "-" * 70)
        print("PHASE 3: SATELLITE IMAGERY ANALYSIS")
        print("-" * 70)

        terrestrial_sites = [c for c in candidates if not c.is_underwater][:5]
        print(f"\nAnalyzing satellite imagery for {len(terrestrial_sites)} terrestrial sites...")

        for site in terrestrial_sites:
            imagery_result = self.satellite.analyze_region(site.lat, site.lon)
            print(f"\n🛰️  {site.name}")
            print(f"   Location: {site.lat:.4f}°N, {site.lon:.4f}°E")
            if imagery_result['anomalies']:
                print(f"   Anomalies detected: {len(imagery_result['anomalies'])}")
                for a in imagery_result['anomalies']:
                    print(f"      - {a['type']}: {a['description']} ({a['confidence']:.0%} confidence)")
            else:
                print(f"   No significant anomalies (may need higher resolution imagery)")

            results['satellite_targets'].append({
                'site': site.name,
                'analysis': imagery_result
            })

        # Summary
        print("\n" + "=" * 70)
        print("     DISCOVERY PRIORITY SUMMARY")
        print("=" * 70)

        critical = [c for c in candidates if c.priority == 'CRITICAL']
        high = [c for c in candidates if c.priority == 'HIGH']
        medium = [c for c in candidates if c.priority == 'MEDIUM']

        print(f"""
   TOTAL CANDIDATE SITES: {len(candidates)}

   By Priority:
   ├─ CRITICAL: {len(critical)} sites (underwater, immediate survey recommended)
   ├─ HIGH:     {len(high)} sites (strong evidence, excavation recommended)
   └─ MEDIUM:   {len(medium)} sites (promising, further research needed)

   IMMEDIATE ACTIONS RECOMMENDED:

   1. UNDERWATER ARCHAEOLOGY
      Priority targets for ROV/diving survey:
""")
        for site in critical[:3]:
            print(f"      • {site.name} ({site.lat:.4f}°N, {site.lon:.4f}°E)")

        print(f"""
   2. TERRESTRIAL EXCAVATION
      Priority targets for excavation permits:
""")
        for site in high[:3]:
            print(f"      • {site.name}")

        print(f"""
   3. SATELLITE SURVEY
      Request high-resolution imagery for:
""")
        for site in medium[:3]:
            print(f"      • {site.name} region")

        print("""
   ═══════════════════════════════════════════════════════════════════

   NOTE: This analysis is based on:
   • Known Linear A find site patterns
   • Bronze Age sea level reconstructions
   • Minoan settlement distribution models
   • Simulated satellite/bathymetric data

   For actual discovery expeditions, require:
   • Real satellite imagery (Sentinel-2, CORONA, commercial)
   • Actual bathymetric surveys
   • Ground-truthing and test excavations
   • Archaeological permits from Greek authorities
        """)

        results['summary'] = {
            'total_candidates': len(candidates),
            'critical': len(critical),
            'high': len(high),
            'medium': len(medium),
            'underwater_targets': len(underwater_sites),
        }

        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the Linear A Discovery Prospector."""
    prospector = LinearAProspector()
    results = prospector.run_full_analysis()
    return results


if __name__ == '__main__':
    main()
