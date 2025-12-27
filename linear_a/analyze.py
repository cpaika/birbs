#!/usr/bin/env python3
"""
Linear A Analysis - Main Entry Point

Run the complete decipherment analysis on the Linear A corpus.
"""

import sys
import os

# Ensure we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.analyzer import LinearAAnalyzer
from corpus.inscriptions import get_all_inscriptions, build_sign_frequency
from signs.inventory import SYLLABIC_SIGNS, get_syllabary_grid


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def main():
    print_header("LINEAR A DECIPHERMENT PROJECT")

    print("""
This project attempts to decipher Linear A, the undeciphered writing
system of Minoan Crete (ca. 1800-1450 BCE).

Our approach:
1. Apply Linear B phonetic values to homomorphic signs
2. Analyze statistical patterns in the corpus
3. Identify recurring formulas and vocabulary
4. Compare with known Aegean languages
5. Propose hypothetical readings and meanings
""")

    # Show syllabary
    print_header("LINEAR A SYLLABARY (Using Linear B Values)")
    print(get_syllabary_grid())

    # Corpus statistics
    print_header("CORPUS STATISTICS")
    inscriptions = get_all_inscriptions()
    print(f"Total inscriptions in corpus: {len(inscriptions)}")

    freq = build_sign_frequency()
    print(f"Unique signs attested: {len(freq)}")
    print(f"Total sign tokens: {sum(freq.values())}")

    # Run analysis
    print_header("RUNNING ANALYSIS")
    analyzer = LinearAAnalyzer()
    print(analyzer.run_full_analysis())

    # Summary of findings
    print_header("DECIPHERMENT HYPOTHESES")
    print("""
Based on our analysis, we propose the following working hypotheses:

1. LANGUAGE TYPE
   - Likely agglutinative with suffixing morphology
   - Not Indo-European (no clear cognates despite centuries of comparison)
   - May be a language isolate or related to pre-Greek Aegean substrates

2. CONFIRMED READINGS
   - pa-i-to = Phaistos (place name, confirmed via Linear B)
   - ku-ro = "total" (administrative term, appears at list ends)

3. PROBABLE DIVINE NAMES
   - i-da-ma-te: Appears in libation contexts; possibly related to
     Mt. Ida and/or early form of Demeter
   - ja-sa-sa-ra: Recurring religious formula; possibly a goddess name
     (cf. Semitic Asherah? Or indigenous Minoan)
   - a-ta-i-*301-wa-ja: Libation formula opening; deity invocation?

4. GRAMMATICAL OBSERVATIONS
   - Words average 3-4 signs (typical for syllabaries)
   - Endings show more variation than beginnings (suffixing language)
   - Signs like -te, -na, -i appear frequently word-finally
     (possible case/number markers)

5. VOCABULARY DOMAINS
   - Religious: Libation formulas, offerings, sanctuary texts
   - Administrative: Personnel lists, commodities, totals
   - Toponyms: Place names (many correspond to Linear B)

CHALLENGES:
- Small corpus (~1,400 inscriptions vs. ~6,000 for Linear B)
- No bilingual text (no "Rosetta Stone")
- Unknown language family makes comparison difficult
- Many signs are hapax legomena (occur only once)

NEXT STEPS:
- Expand corpus with newly published inscriptions
- Apply machine learning for pattern recognition
- Compare with Luwian, Hurrian, Etruscan for possible cognates
- Analyze semantic fields of ideograms
""")


if __name__ == "__main__":
    main()
