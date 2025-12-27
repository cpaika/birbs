#!/usr/bin/env python3
"""
Real Discovery Opportunities Using Existing Open Data

These are actual datasets where computational analysis could
lead to genuine new discoveries without fieldwork.
"""

OPPORTUNITIES = {

    # =========================================================================
    # HIGH POTENTIAL - Genuinely underexplored
    # =========================================================================

    "corona_satellite": {
        "name": "CORONA Declassified Spy Satellite Imagery",
        "description": """
            800,000+ high-resolution photos from 1960-1972 showing
            archaeological sites BEFORE modern development destroyed them.
            Many areas never systematically analyzed.
        """,
        "data_source": "https://earthexplorer.usgs.gov/",
        "alternative": "https://corona.cast.uark.edu/",
        "format": "TIFF images, some orthorectified",
        "size": "Terabytes total, individual frames ~100MB",
        "underexplored_regions": [
            "Central Asia (Kazakhstan, Turkmenistan)",
            "Iran (limited modern access)",
            "Iraq (destroyed by conflict since 1990s)",
            "Syria (destroyed by conflict since 2011)",
            "Yemen (ongoing conflict)",
            "Western China (restricted access)",
        ],
        "discovery_potential": "HIGH",
        "skills_needed": ["Image analysis", "GIS", "Archaeology knowledge"],
        "success_examples": [
            "Ur's Jason detected 14,000 sites in Syria from CORONA",
            "Lost irrigation systems in Iraq found",
            "Nomad camp patterns in Iran documented",
        ],
    },

    "cuneiform_tablets": {
        "name": "Unread Cuneiform Tablets",
        "description": """
            ~600,000 tablets excavated, most never read. CDLI has
            350,000+ digitized. Pattern recognition and ML could
            help read, classify, and cross-reference them.
        """,
        "data_source": "https://cdli.ucla.edu/",
        "format": "Images, some with transliterations",
        "size": "350,000+ records",
        "underexplored": [
            "Tablets without transliteration",
            "Fragment matching (joining broken pieces)",
            "Cross-collection prosopography (finding same people)",
            "Undeciphered scripts (Proto-Elamite)",
        ],
        "discovery_potential": "HIGH",
        "skills_needed": ["OCR/ML", "Cuneiform knowledge or willingness to learn"],
        "success_examples": [
            "AI fragment matching at British Museum",
            "Automated sign recognition for damaged tablets",
        ],
    },

    "herculaneum_scrolls": {
        "name": "Vesuvius Challenge - Carbonized Scrolls",
        "description": """
            CT scans of scrolls carbonized by Vesuvius eruption in 79 CE.
            AI successfully read first words in 2023. Hundreds more scrolls
            await analysis. Active competition with cash prizes.
        """,
        "data_source": "https://scrollprize.org/",
        "format": "3D CT scan volumes",
        "size": "~100GB per scroll",
        "status": "ACTIVE COMPETITION - $100k+ prizes available",
        "discovery_potential": "VERY HIGH",
        "skills_needed": ["Deep learning", "3D image processing", "GPU computing"],
        "success_examples": [
            "2023: First words read after 2000 years",
            "2024: Multiple passages deciphered",
        ],
    },

    # =========================================================================
    # MEDIUM POTENTIAL - Some work done but more possible
    # =========================================================================

    "latin_inscriptions": {
        "name": "EDCS Latin Inscription Database",
        "description": """
            ~500,000 Latin inscriptions from Roman world. Digitized
            text available but many analyses not done: name patterns,
            social networks, statistical linguistics.
        """,
        "data_source": "http://www.manfredclauss.de/",
        "format": "Text database, searchable online",
        "size": "~500,000 inscriptions",
        "underexplored": [
            "Prosopography automation (linking people across inscriptions)",
            "Dialectal/regional variation analysis",
            "Dating refinement through language patterns",
            "Formula evolution over time",
        ],
        "discovery_potential": "MEDIUM",
        "skills_needed": ["NLP", "Latin", "Statistics"],
    },

    "hexagon_satellite": {
        "name": "HEXAGON Satellite Imagery (1971-1986)",
        "description": """
            Successor to CORONA, declassified 2011, higher resolution.
            Only recently becoming available for download (2020-2022).
            Much less analyzed than CORONA.
        """,
        "data_source": "https://earthexplorer.usgs.gov/",
        "format": "Film scans",
        "size": "Even larger than CORONA archive",
        "discovery_potential": "HIGH (less explored than CORONA)",
        "skills_needed": ["Same as CORONA"],
    },

    "shipwreck_databases": {
        "name": "Mediterranean Shipwreck Databases",
        "description": """
            Multiple databases of ancient shipwrecks with cargo data.
            Trade network reconstruction, statistical analysis of
            routes, dating refinement possible.
        """,
        "data_sources": [
            "Oxford Roman Economy Project",
            "Strasbourg shipwreck database",
        ],
        "discovery_potential": "MEDIUM",
        "skills_needed": ["Network analysis", "GIS", "Statistics"],
    },

    # =========================================================================
    # SPECIFIC TO LINEAR A
    # =========================================================================

    "linear_a_sigla": {
        "name": "SigLA Project - Linear A Sign Analysis",
        "description": """
            Digital paleography project for Linear A signs.
            Statistical analysis of sign variants could reveal
            scribal hands, regional differences, chronology.
        """,
        "data_source": "https://sigla.phis.me/",
        "format": "Vector sign drawings, measurements",
        "discovery_potential": "MEDIUM-HIGH for Linear A specifically",
        "skills_needed": ["Paleography", "Statistics", "Image analysis"],
        "what_to_look_for": [
            "Scribal hand identification",
            "Regional variants",
            "Chronological development",
            "Connections to other Aegean scripts",
        ],
    },

    "minoan_trade_networks": {
        "name": "Minoan Pottery Distribution Analysis",
        "description": """
            Published data on Minoan pottery finds across Mediterranean.
            Trade network reconstruction could identify missing nodes
            (potential undiscovered sites).
        """,
        "data_sources": [
            "Aegean Dendrochronology Project data",
            "Published excavation reports (JSTOR, etc.)",
        ],
        "discovery_potential": "MEDIUM",
        "method": "Network analysis to find 'gaps' in trade routes",
    },
}


def print_opportunities():
    """Print all opportunities in readable format."""
    print("=" * 70)
    print("   REAL DISCOVERY OPPORTUNITIES WITH EXISTING OPEN DATA")
    print("=" * 70)

    high_potential = {k: v for k, v in OPPORTUNITIES.items()
                      if 'HIGH' in v.get('discovery_potential', '')}
    medium_potential = {k: v for k, v in OPPORTUNITIES.items()
                        if v.get('discovery_potential', '') == 'MEDIUM'}

    print("\n🔴 HIGH POTENTIAL OPPORTUNITIES:\n")
    for key, opp in high_potential.items():
        print(f"  {opp['name']}")
        print(f"  {'─' * 50}")
        desc = opp['description'].strip().replace('\n', ' ')
        while '  ' in desc:
            desc = desc.replace('  ', ' ')
        print(f"  {desc[:100]}...")
        print(f"  📁 Data: {opp.get('data_source', 'See description')}")
        print(f"  🎯 Potential: {opp['discovery_potential']}")
        print()

    print("\n🟡 MEDIUM POTENTIAL OPPORTUNITIES:\n")
    for key, opp in medium_potential.items():
        print(f"  {opp['name']}")
        print(f"  📁 Data: {opp.get('data_source', opp.get('data_sources', ['See description'])[0] if isinstance(opp.get('data_sources'), list) else 'See description')}")
        print()

    print("=" * 70)
    print("   RECOMMENDED STARTING POINT")
    print("=" * 70)
    print("""
   For immediate impact, consider:

   1. VESUVIUS CHALLENGE (scrollprize.org)
      - Active competition with cash prizes
      - Clear success metric
      - Global attention

   2. CORONA IMAGERY of conflict zones
      - Sites in Syria/Iraq visible in 1960s
      - Now destroyed - your analysis preserves them
      - University of Arkansas has orthorectified data

   3. CUNEIFORM FRAGMENT MATCHING
      - British Museum has millions of fragments
      - ML can match pieces across collections
      - Could reconstruct lost texts

   4. For LINEAR A specifically: SigLA project
      - Statistical analysis of sign shapes
      - Could identify scribal hands/workshops
      - Might reveal regional dialects
    """)


if __name__ == '__main__':
    print_opportunities()
