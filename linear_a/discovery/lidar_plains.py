#!/usr/bin/env python3
"""
LiDAR Archaeological Anomaly Detection for Great Plains

Uses USGS 3DEP LiDAR data to detect potential archaeological features:
- Mound structures (burial mounds, platform mounds)
- Linear earthworks (roads, walls, canals)
- Depressions (house pits, storage pits)
- Geometric patterns (villages, field systems)

Data source: USGS 3DEP via AWS OpenData
"""

import requests
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
import math
from io import BytesIO


# =============================================================================
# KNOWN ARCHAEOLOGICAL SITES FOR REFERENCE
# =============================================================================

# Great Plains archaeological cultures/sites we might detect
KNOWN_SITE_TYPES = {
    'burial_mound': {
        'description': 'Conical or dome-shaped earth mounds, often contain burials',
        'cultures': ['Hopewell', 'Adena', 'Mississippian'],
        'typical_height_m': (0.5, 5.0),
        'typical_diameter_m': (10, 50),
    },
    'platform_mound': {
        'description': 'Flat-topped pyramidal mounds, used for temples/elite residences',
        'cultures': ['Mississippian', 'Caddoan'],
        'typical_height_m': (2.0, 20.0),
        'typical_diameter_m': (30, 200),
    },
    'effigy_mound': {
        'description': 'Mounds shaped like animals or symbols',
        'cultures': ['Effigy Mound culture'],
        'typical_height_m': (0.3, 2.0),
        'typical_diameter_m': (20, 100),
    },
    'house_depression': {
        'description': 'Circular depressions from pit houses or earth lodges',
        'cultures': ['Central Plains Tradition', 'Pawnee', 'Arikara'],
        'typical_depth_m': (0.3, 1.5),
        'typical_diameter_m': (8, 15),
    },
    'earthwork': {
        'description': 'Linear or geometric earth embankments',
        'cultures': ['Hopewell', 'Fort Ancient'],
        'typical_height_m': (0.5, 3.0),
        'typical_length_m': (50, 1000),
    },
}

# Known major sites in Great Plains for validation
KNOWN_SITES = [
    {'name': 'Cahokia Mounds', 'lat': 38.6555, 'lon': -90.0619, 'type': 'platform_mound', 'state': 'IL'},
    {'name': 'Poverty Point', 'lat': 32.6347, 'lon': -91.4072, 'type': 'earthwork', 'state': 'LA'},
    {'name': 'Serpent Mound', 'lat': 39.0253, 'lon': -83.4303, 'type': 'effigy_mound', 'state': 'OH'},
    {'name': 'Pawnee Indian Village', 'lat': 39.9108, 'lon': -98.8828, 'type': 'house_depression', 'state': 'KS'},
    {'name': 'Wichita Indian Village', 'lat': 38.7283, 'lon': -98.6289, 'type': 'house_depression', 'state': 'KS'},
    {'name': 'Council Grove', 'lat': 38.6611, 'lon': -96.4919, 'type': 'earthwork', 'state': 'KS'},
    {'name': 'Knife River Indian Villages', 'lat': 47.3542, 'lon': -101.3856, 'type': 'house_depression', 'state': 'ND'},
    {'name': 'Double Ditch Village', 'lat': 46.9750, 'lon': -100.6917, 'type': 'house_depression', 'state': 'ND'},
]


# =============================================================================
# USGS 3DEP DATA ACCESS
# =============================================================================

class USGS3DEPClient:
    """Access USGS 3DEP LiDAR data via public APIs."""

    # USGS TNM (The National Map) API
    TNM_API = "https://tnmaccess.nationalmap.gov/api/v1/products"

    # OpenTopography API for elevation data
    OPENTOPO_API = "https://portal.opentopography.org/API/globaldem"

    def search_lidar_coverage(self, min_lat: float, min_lon: float,
                              max_lat: float, max_lon: float) -> List[Dict]:
        """Search for available LiDAR datasets in a region."""
        params = {
            'bbox': f'{min_lon},{min_lat},{max_lon},{max_lat}',
            'datasets': 'Lidar Point Cloud (LPC)',
            'prodFormats': 'LAZ',
            'max': 50,
        }

        try:
            response = requests.get(self.TNM_API, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data.get('items', [])
        except Exception as e:
            print(f"    TNM API error: {e}")
        return []

    def get_elevation_data(self, min_lat: float, min_lon: float,
                           max_lat: float, max_lon: float,
                           resolution: str = 'SRTMGL1') -> Optional[np.ndarray]:
        """Get elevation data for analysis."""
        # Use SRTM for quick analysis (actual LiDAR would need LAZ processing)
        params = {
            'demtype': resolution,  # SRTMGL1 = 30m, SRTMGL3 = 90m
            'south': min_lat,
            'north': max_lat,
            'west': min_lon,
            'east': max_lon,
            'outputFormat': 'AAIGrid',
        }

        try:
            response = requests.get(self.OPENTOPO_API, params=params, timeout=60)
            if response.status_code == 200:
                # Parse ASCII Grid format
                return self._parse_ascii_grid(response.text)
        except Exception as e:
            print(f"    OpenTopo API error: {e}")
        return None

    def _parse_ascii_grid(self, text: str) -> Optional[np.ndarray]:
        """Parse ESRI ASCII Grid format."""
        lines = text.strip().split('\n')

        # Parse header
        header = {}
        data_start = 0
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) == 2 and parts[0].lower() in ['ncols', 'nrows', 'xllcorner',
                                                          'yllcorner', 'cellsize', 'nodata_value']:
                header[parts[0].lower()] = float(parts[1])
                data_start = i + 1
            else:
                break

        if 'ncols' not in header or 'nrows' not in header:
            return None

        # Parse data
        ncols = int(header['ncols'])
        nrows = int(header['nrows'])
        nodata = header.get('nodata_value', -9999)

        data = []
        for line in lines[data_start:]:
            values = [float(v) for v in line.split()]
            data.extend(values)

        if len(data) != ncols * nrows:
            return None

        arr = np.array(data).reshape(nrows, ncols)
        arr[arr == nodata] = np.nan
        return arr


# =============================================================================
# ARCHAEOLOGICAL FEATURE DETECTION
# =============================================================================

class ArchaeologicalDetector:
    """Detect potential archaeological features in elevation data."""

    def __init__(self):
        self.min_mound_height = 0.3  # meters
        self.min_mound_diameter = 5  # meters
        self.min_depression_depth = 0.2  # meters

    def analyze_dem(self, dem: np.ndarray, cell_size_m: float = 30) -> Dict:
        """Analyze a DEM for archaeological anomalies."""
        results = {
            'mound_candidates': [],
            'depression_candidates': [],
            'linear_features': [],
            'statistics': {},
        }

        if dem is None or dem.size == 0:
            return results

        # Calculate statistics
        valid_data = dem[~np.isnan(dem)]
        if len(valid_data) == 0:
            return results

        results['statistics'] = {
            'min_elev': float(np.min(valid_data)),
            'max_elev': float(np.max(valid_data)),
            'mean_elev': float(np.mean(valid_data)),
            'std_elev': float(np.std(valid_data)),
            'relief': float(np.max(valid_data) - np.min(valid_data)),
        }

        # Detect mounds (local maxima)
        mounds = self._detect_mounds(dem, cell_size_m)
        results['mound_candidates'] = mounds

        # Detect depressions (local minima)
        depressions = self._detect_depressions(dem, cell_size_m)
        results['depression_candidates'] = depressions

        # Detect linear features
        linears = self._detect_linear_features(dem, cell_size_m)
        results['linear_features'] = linears

        return results

    def _detect_mounds(self, dem: np.ndarray, cell_size_m: float) -> List[Dict]:
        """Detect potential mound features."""
        mounds = []

        # Create local relief model (difference from neighborhood mean)
        from scipy import ndimage

        # Use different kernel sizes for different mound scales
        for kernel_size in [3, 5, 7]:  # 90m, 150m, 210m at 30m resolution
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            local_mean = ndimage.convolve(dem, kernel, mode='reflect')
            relief = dem - local_mean

            # Find local maxima
            max_filter = ndimage.maximum_filter(relief, size=kernel_size)
            is_local_max = (relief == max_filter) & (relief > self.min_mound_height)

            # Get coordinates of maxima
            rows, cols = np.where(is_local_max & ~np.isnan(dem))

            for r, c in zip(rows, cols):
                height = relief[r, c]
                if height > self.min_mound_height:
                    mounds.append({
                        'row': int(r),
                        'col': int(c),
                        'height_m': float(height),
                        'diameter_estimate_m': kernel_size * cell_size_m,
                        'confidence': min(1.0, height / 2.0),  # Higher mounds = more confident
                    })

        # Remove duplicates (keep highest confidence)
        mounds = self._deduplicate_features(mounds, cell_size_m * 3)

        return mounds[:50]  # Top 50 candidates

    def _detect_depressions(self, dem: np.ndarray, cell_size_m: float) -> List[Dict]:
        """Detect potential pit house or storage pit depressions."""
        depressions = []

        from scipy import ndimage

        for kernel_size in [3, 5]:
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            local_mean = ndimage.convolve(dem, kernel, mode='reflect')
            relief = local_mean - dem  # Inverted for depressions

            # Find local minima (maxima in inverted relief)
            max_filter = ndimage.maximum_filter(relief, size=kernel_size)
            is_local_min = (relief == max_filter) & (relief > self.min_depression_depth)

            rows, cols = np.where(is_local_min & ~np.isnan(dem))

            for r, c in zip(rows, cols):
                depth = relief[r, c]
                if depth > self.min_depression_depth:
                    depressions.append({
                        'row': int(r),
                        'col': int(c),
                        'depth_m': float(depth),
                        'diameter_estimate_m': kernel_size * cell_size_m,
                        'confidence': min(1.0, depth / 1.0),
                    })

        depressions = self._deduplicate_features(depressions, cell_size_m * 3)
        return depressions[:50]

    def _detect_linear_features(self, dem: np.ndarray, cell_size_m: float) -> List[Dict]:
        """Detect potential linear earthworks."""
        linears = []

        from scipy import ndimage

        # Edge detection to find linear features
        # Sobel filters for gradient
        sobel_x = ndimage.sobel(dem, axis=1, mode='reflect')
        sobel_y = ndimage.sobel(dem, axis=0, mode='reflect')
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Find high-gradient linear features
        threshold = np.nanpercentile(gradient_magnitude, 95)
        high_gradient = gradient_magnitude > threshold

        # Label connected components
        labeled, num_features = ndimage.label(high_gradient)

        for i in range(1, min(num_features + 1, 20)):  # Top 20 features
            feature_mask = labeled == i
            feature_size = np.sum(feature_mask)

            if feature_size > 10:  # Minimum size
                rows, cols = np.where(feature_mask)
                # Calculate linearity (how linear vs. blob-like)
                if len(rows) > 2:
                    # Fit line and calculate R²
                    try:
                        coeffs = np.polyfit(cols, rows, 1)
                        predicted = np.polyval(coeffs, cols)
                        ss_res = np.sum((rows - predicted) ** 2)
                        ss_tot = np.sum((rows - np.mean(rows)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                        if r_squared > 0.7:  # Reasonably linear
                            length_m = np.sqrt((rows.max() - rows.min())**2 +
                                             (cols.max() - cols.min())**2) * cell_size_m
                            linears.append({
                                'start_row': int(rows.min()),
                                'start_col': int(cols[np.argmin(rows)]),
                                'end_row': int(rows.max()),
                                'end_col': int(cols[np.argmax(rows)]),
                                'length_m': float(length_m),
                                'linearity_r2': float(r_squared),
                                'confidence': float(r_squared * min(1.0, length_m / 100)),
                            })
                    except:
                        pass

        return sorted(linears, key=lambda x: -x['confidence'])[:10]

    def _deduplicate_features(self, features: List[Dict], min_distance: float) -> List[Dict]:
        """Remove duplicate features that are too close together."""
        if not features:
            return []

        # Sort by confidence
        features = sorted(features, key=lambda x: -x.get('confidence', 0))

        kept = []
        for f in features:
            is_duplicate = False
            for k in kept:
                dist = np.sqrt((f['row'] - k['row'])**2 + (f['col'] - k['col'])**2) * 30  # Assume 30m cells
                if dist < min_distance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(f)

        return kept


# =============================================================================
# ANALYSIS REGIONS
# =============================================================================

ANALYSIS_REGIONS = [
    {
        'name': 'Central Kansas (Republican River)',
        'description': 'Known Pawnee and Wichita territory',
        'min_lat': 39.5, 'max_lat': 40.0,
        'min_lon': -99.5, 'max_lon': -99.0,
        'known_cultures': ['Pawnee', 'Wichita', 'Central Plains Tradition'],
    },
    {
        'name': 'Eastern Nebraska (Missouri Valley)',
        'description': 'Confluence region with many known sites',
        'min_lat': 40.5, 'max_lat': 41.0,
        'min_lon': -96.5, 'max_lon': -96.0,
        'known_cultures': ['Nebraska phase', 'Oneota'],
    },
    {
        'name': 'Knife River region (North Dakota)',
        'description': 'Mandan and Hidatsa earthlodge villages',
        'min_lat': 47.2, 'max_lat': 47.5,
        'min_lon': -101.5, 'max_lon': -101.2,
        'known_cultures': ['Mandan', 'Hidatsa', 'Arikara'],
    },
    {
        'name': 'Flint Hills (Kansas)',
        'description': 'Potential unexplored upland sites',
        'min_lat': 38.5, 'max_lat': 39.0,
        'min_lon': -96.8, 'max_lon': -96.3,
        'known_cultures': ['Wichita', 'Kansa'],
    },
    {
        'name': 'Smoky Hill River (Kansas)',
        'description': 'Central Plains Tradition territory',
        'min_lat': 38.8, 'max_lat': 39.2,
        'min_lon': -98.0, 'max_lon': -97.5,
        'known_cultures': ['Smoky Hill phase', 'Central Plains Tradition'],
    },
]


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("   LIDAR ARCHAEOLOGICAL ANOMALY DETECTION - GREAT PLAINS")
    print("=" * 70)
    print("\nData sources:")
    print("  • USGS 3DEP (The National Map)")
    print("  • OpenTopography SRTM (30m resolution)")
    print("\n" + "=" * 70)

    # Check for scipy
    try:
        from scipy import ndimage
    except ImportError:
        print("\nInstalling scipy for image processing...")
        import subprocess
        subprocess.run(['pip', 'install', 'scipy', '-q'])
        from scipy import ndimage

    usgs = USGS3DEPClient()
    detector = ArchaeologicalDetector()

    all_results = []

    for region in ANALYSIS_REGIONS:
        print(f"\n{'='*70}")
        print(f"REGION: {region['name']}")
        print(f"{'='*70}")
        print(f"Description: {region['description']}")
        print(f"Known cultures: {', '.join(region['known_cultures'])}")
        print(f"Bounds: {region['min_lat']:.2f}°N to {region['max_lat']:.2f}°N, "
              f"{region['min_lon']:.2f}°W to {region['max_lon']:.2f}°W")

        # Check LiDAR availability
        print("\n  Checking LiDAR coverage...")
        lidar_products = usgs.search_lidar_coverage(
            region['min_lat'], region['min_lon'],
            region['max_lat'], region['max_lon']
        )
        print(f"  ✓ Found {len(lidar_products)} LiDAR datasets available")

        if lidar_products:
            print(f"  Latest dataset: {lidar_products[0].get('title', 'Unknown')[:60]}...")

        # Get elevation data for analysis
        print("\n  Fetching elevation data (SRTM 30m)...")
        dem = usgs.get_elevation_data(
            region['min_lat'], region['min_lon'],
            region['max_lat'], region['max_lon']
        )

        if dem is not None:
            print(f"  ✓ Retrieved DEM: {dem.shape[0]} x {dem.shape[1]} pixels")

            # Run detection
            print("\n  Running anomaly detection...")
            results = detector.analyze_dem(dem, cell_size_m=30)

            print(f"\n  DETECTION RESULTS:")
            print(f"  ├─ Mound candidates: {len(results['mound_candidates'])}")
            print(f"  ├─ Depression candidates: {len(results['depression_candidates'])}")
            print(f"  └─ Linear features: {len(results['linear_features'])}")

            if results['mound_candidates']:
                print(f"\n  Top mound candidates:")
                for i, m in enumerate(results['mound_candidates'][:3], 1):
                    # Convert pixel coords to lat/lon
                    lat = region['max_lat'] - (m['row'] / dem.shape[0]) * (region['max_lat'] - region['min_lat'])
                    lon = region['min_lon'] + (m['col'] / dem.shape[1]) * (region['max_lon'] - region['min_lon'])
                    print(f"    {i}. Height: {m['height_m']:.1f}m, Diameter: ~{m['diameter_estimate_m']:.0f}m")
                    print(f"       Location: {lat:.4f}°N, {abs(lon):.4f}°W")
                    print(f"       Confidence: {m['confidence']:.0%}")

            if results['depression_candidates']:
                print(f"\n  Top depression candidates:")
                for i, d in enumerate(results['depression_candidates'][:3], 1):
                    lat = region['max_lat'] - (d['row'] / dem.shape[0]) * (region['max_lat'] - region['min_lat'])
                    lon = region['min_lon'] + (d['col'] / dem.shape[1]) * (region['max_lon'] - region['min_lon'])
                    print(f"    {i}. Depth: {d['depth_m']:.1f}m, Diameter: ~{d['diameter_estimate_m']:.0f}m")
                    print(f"       Location: {lat:.4f}°N, {abs(lon):.4f}°W")

            results['region'] = region['name']
            all_results.append(results)

        else:
            print("  ✗ Could not retrieve elevation data")

    # Summary
    print("\n" + "=" * 70)
    print("   ANALYSIS SUMMARY")
    print("=" * 70)

    total_mounds = sum(len(r['mound_candidates']) for r in all_results)
    total_depressions = sum(len(r['depression_candidates']) for r in all_results)
    total_linears = sum(len(r['linear_features']) for r in all_results)

    print(f"""
   TOTAL ANOMALIES DETECTED:
   ├─ Potential mounds: {total_mounds}
   ├─ Potential depressions: {total_depressions}
   └─ Potential linear features: {total_linears}
    """)

    # Caveats
    print("=" * 70)
    print("   IMPORTANT CAVEATS")
    print("=" * 70)
    print("""
   ⚠️  LIMITATIONS OF THIS ANALYSIS:

   1. RESOLUTION: Using 30m SRTM, not full LiDAR resolution (0.5-1m)
      - Cannot detect features smaller than ~30m
      - Many archaeological sites are too small to detect

   2. DATA: Full LiDAR point clouds require:
      - LAZ/LAS file processing (not available via simple API)
      - Significant storage (gigabytes per small area)
      - Professional GIS software (PDAL, LAStools, ArcGIS)

   3. FALSE POSITIVES: Detected "anomalies" are likely:
      - Natural terrain features (hills, sinkholes)
      - Modern disturbances (roads, farms, construction)
      - Geological features (glacial deposits)

   4. KNOWN SITES: Most detectable sites in Great Plains are
      ALREADY KNOWN and documented in state archaeological surveys

   FOR REAL DISCOVERY YOU NEED:
   - Full-resolution LiDAR (1m or better)
   - Ground-truthing field surveys
   - State archaeological permits
   - Consultation with tribal nations (NAGPRA compliance)
    """)

    return all_results


if __name__ == '__main__':
    results = main()
