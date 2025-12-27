#!/usr/bin/env python3
"""
Sound Law Discovery for Linear A

Instead of matching individual words, look for SYSTEMATIC sound correspondences.
If Linear A is related to a known language, we should see regular patterns like:
- Linear A /k/ always corresponds to Hurrian /g/
- Linear A /a/ always corresponds to Etruscan /e/

This is how real language relationships are proven (Grimm's Law, etc.)
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import sys
import os
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.inscriptions import get_all_inscriptions
from signs.inventory import SYLLABIC_SIGNS


def get_phonetic_reading(word: List[str]) -> Optional[str]:
    """Convert a word to its phonetic reading."""
    phonetics = []
    for sign in word:
        if sign in SYLLABIC_SIGNS:
            p = SYLLABIC_SIGNS[sign].phonetic
            if p and p != "?":
                phonetics.append(p)
            else:
                return None
        else:
            return None
    return '-'.join(phonetics) if phonetics else None


# =============================================================================
# SOUND CORRESPONDENCE EXTRACTOR
# =============================================================================

class SoundCorrespondenceExtractor:
    """Extract sound correspondences between two sets of words."""

    def __init__(self):
        self.vowels = set('aeiou')
        self.consonants = set('bcdfghjklmnpqrstvwxyz')

    def extract_correspondences(self, pairs: List[Tuple[str, str]]) -> Dict[Tuple, int]:
        """
        Extract sound correspondences from word pairs.
        Returns count of each correspondence.
        """
        correspondences = Counter()

        for word1, word2 in pairs:
            # Align the words
            alignment = self._align(word1, word2)
            for s1, s2 in alignment:
                if s1 and s2:  # Both present (not gap)
                    correspondences[(s1, s2)] += 1

        return correspondences

    def _align(self, w1: str, w2: str) -> List[Tuple[str, str]]:
        """Simple alignment based on position."""
        segs1 = self._segment(w1)
        segs2 = self._segment(w2)

        # Pad shorter one
        max_len = max(len(segs1), len(segs2))
        segs1 = segs1 + [''] * (max_len - len(segs1))
        segs2 = segs2 + [''] * (max_len - len(segs2))

        return list(zip(segs1, segs2))

    def _segment(self, word: str) -> List[str]:
        """Segment word into sounds."""
        word = word.lower().replace('-', '')
        segments = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i:i+2] in ['sh', 'ch', 'th', 'ph']:
                segments.append(word[i:i+2])
                i += 2
            else:
                segments.append(word[i])
                i += 1
        return segments

    def find_regular_correspondences(self, correspondences: Dict[Tuple, int],
                                     min_count: int = 3) -> List[Tuple]:
        """
        Find correspondences that occur regularly (not just once).
        Regular correspondences suggest a real relationship.
        """
        regular = []
        for (s1, s2), count in correspondences.items():
            if count >= min_count and s1 != s2:
                regular.append((s1, s2, count))

        regular.sort(key=lambda x: -x[2])
        return regular

    def calculate_regularity_score(self, correspondences: Dict[Tuple, int]) -> float:
        """
        Calculate how regular the correspondences are.
        High regularity = possible real relationship
        """
        if not correspondences:
            return 0.0

        # Count how many correspondences occur multiple times
        multi_occur = sum(1 for c in correspondences.values() if c >= 2)
        total = len(correspondences)

        # Count most common correspondences
        top_counts = sorted(correspondences.values(), reverse=True)[:10]
        regularity = sum(top_counts) / sum(correspondences.values()) if correspondences else 0

        return regularity


# =============================================================================
# CANDIDATE LANGUAGE VOCABULARIES
# =============================================================================

LANGUAGES = {
    'hurrian': {
        'teshub': 'storm god', 'hebat': 'goddess', 'ewri': 'lord',
        'allai': 'lady', 'tahe': 'man', 'asti': 'woman',
        'eni': 'god', 'sena': 'brother', 'ela': 'sister',
        'mena': 'twin', 'pahi': 'head', 'arde': 'city',
        'keli': 'house', 'eshi': 'earth', 'shuki': 'one',
        'shena': 'two', 'kig': 'three', 'tumni': 'four',
        'nariya': 'five', 'eman': 'ten', 'kade': 'barley',
        'ami': 'wine', 'shuri': 'kingship', 'hawirni': 'sheep',
    },
    'etruscan': {
        'tinia': 'sky god', 'uni': 'goddess', 'turan': 'Venus',
        'lucumo': 'king', 'clan': 'son', 'sec': 'daughter',
        'puia': 'wife', 'apa': 'father', 'ati': 'mother',
        'spur': 'city', 'fler': 'offering', 'tur': 'give',
        'thu': 'one', 'zal': 'two', 'ci': 'three',
        'avil': 'year', 'ais': 'god', 'calu': 'death',
        'aita': 'Hades', 'vacl': 'vessel', 'nefts': 'grandson',
    },
    'luwian': {
        'tarhunt': 'storm god', 'tiwat': 'sun god', 'arma': 'moon',
        'wana': 'stele', 'ura': 'great', 'hantili': 'first',
        'parha': 'high', 'ziti': 'man', 'wanati': 'woman',
        'tati': 'father', 'anni': 'mother', 'nimuwiza': 'son',
        'parna': 'house', 'harna': 'sanctuary',
        'awi': 'to come', 'tawa': 'to put',
    },
    'pre_greek': {
        'labyrinthos': 'labyrinth', 'hyakinthos': 'hyacinth',
        'terebinthos': 'tree', 'thalassa': 'sea', 'parnassos': 'mountain',
        'korinthos': 'Corinth', 'zakynthos': 'island',
        'narkissos': 'narcissus', 'kyparissos': 'cypress',
        'labrys': 'double axe', 'tyrannos': 'ruler',
        'basileus': 'king', 'diktynna': 'goddess',
    },
}


# =============================================================================
# MAIN SOUND LAW DISCOVERY
# =============================================================================

class SoundLawDiscoverer:
    """Discover systematic sound correspondences."""

    def __init__(self):
        self.extractor = SoundCorrespondenceExtractor()

    def discover(self) -> Dict:
        """Run sound law discovery on all candidate languages."""
        print("\n" + "=" * 70)
        print(" " * 15 + "SOUND LAW DISCOVERY FOR LINEAR A")
        print("=" * 70)

        # Get Linear A words
        inscriptions = get_all_inscriptions()
        la_words = []
        for insc in inscriptions:
            for word in insc.get_words():
                reading = get_phonetic_reading(word)
                if reading:
                    la_words.append(reading)

        la_words = list(set(la_words))
        print(f"\nAnalyzing {len(la_words)} Linear A words...")

        results = {}

        for lang_name, vocab in LANGUAGES.items():
            print(f"\n{'='*70}")
            print(f"Analyzing: {lang_name.upper()}")
            print(f"{'='*70}")

            lang_words = list(vocab.keys())

            # Create all possible pairs
            pairs = []
            for la in la_words:
                for lw in lang_words:
                    # Only pair words of similar length
                    la_clean = la.replace('-', '')
                    if 0.5 < len(la_clean) / len(lw) < 2.0:
                        pairs.append((la, lw))

            print(f"  Created {len(pairs)} potential pairs...")

            # Extract correspondences
            correspondences = self.extractor.extract_correspondences(pairs)

            # Find regular ones
            regular = self.extractor.find_regular_correspondences(correspondences, min_count=2)

            # Calculate regularity score
            regularity = self.extractor.calculate_regularity_score(correspondences)

            results[lang_name] = {
                'regularity_score': regularity,
                'regular_correspondences': regular[:10],
                'total_correspondences': len(correspondences),
            }

            print(f"\n  Regularity Score: {regularity:.3f}")
            print(f"  Total unique correspondences: {len(correspondences)}")

            if regular:
                print(f"\n  Most frequent correspondences:")
                for s1, s2, count in regular[:5]:
                    print(f"    Linear A /{s1}/ ~ {lang_name} /{s2}/  (x{count})")

            # Look for sound laws
            sound_laws = self._detect_sound_laws(correspondences)
            if sound_laws:
                print(f"\n  Potential Sound Laws:")
                for law in sound_laws[:3]:
                    print(f"    {law}")

        # Final ranking
        print("\n" + "=" * 70)
        print(" " * 20 + "SOUND LAW RANKING")
        print("=" * 70)

        ranked = sorted(results.items(), key=lambda x: -x[1]['regularity_score'])

        print("\n📊 Languages by Sound Correspondence Regularity:")
        for lang, data in ranked:
            bar = '█' * int(data['regularity_score'] * 50)
            print(f"   {lang:15} {bar} ({data['regularity_score']:.3f})")

        # Check for meaningful patterns
        best = ranked[0]
        print(f"\n" + "=" * 70)
        print(" " * 25 + "ANALYSIS")
        print("=" * 70)

        if best[1]['regularity_score'] > 0.3:
            print(f"""
   POSSIBLE RELATIONSHIP: {best[0].upper()}

   Regular sound correspondences detected:""")
            for s1, s2, count in best[1]['regular_correspondences'][:5]:
                print(f"   • Linear A /{s1}/ → {best[0]} /{s2}/  ({count} times)")

            print(f"""
   These patterns COULD indicate a genetic relationship.
   However, without more data, this remains speculative.
            """)
        else:
            print(f"""
   NO REGULAR SOUND CORRESPONDENCES FOUND

   All tested languages show similar (low) regularity scores.
   This suggests the phonetic similarities are random,
   not the result of systematic sound change.

   Linear A appears to be a TRUE LANGUAGE ISOLATE.
            """)

        return results

    def _detect_sound_laws(self, correspondences: Dict[Tuple, int]) -> List[str]:
        """Try to detect sound laws from correspondences."""
        laws = []

        # Group by first element
        by_source = defaultdict(list)
        for (s1, s2), count in correspondences.items():
            if count >= 2:
                by_source[s1].append((s2, count))

        # Look for consistent mappings
        for source, targets in by_source.items():
            if len(targets) == 1:
                # Consistent mapping
                target, count = targets[0]
                if source != target and count >= 3:
                    laws.append(f"/{source}/ → /{target}/ (regular, {count} examples)")
            elif len(targets) >= 2:
                # Conditioned change?
                targets.sort(key=lambda x: -x[1])
                main_target = targets[0]
                if main_target[1] > sum(t[1] for t in targets[1:]):
                    laws.append(f"/{source}/ → /{main_target[0]}/ (primary, {main_target[1]} examples)")

        return laws


# =============================================================================
# DEEP ANALYSIS: SEMANTIC FIELD MATCHING
# =============================================================================

class SemanticFieldAnalyzer:
    """
    Check if semantic fields match between Linear A context and translations.
    If pa-i-to appears with grain ideograms, does its "match" word also mean grain-related?
    """

    def __init__(self):
        self.la_semantic_contexts = {
            # Words appearing with specific ideograms
            'ku-ro': 'total/summation',
            'pa-i-to': 'place_name',
            'ja-sa-sa-ra': 'religious',
            'da-ma-te': 'agricultural',
            'a-ta-na-te': 'religious',
        }

    def check_semantic_match(self, la_word: str, translation: str,
                            translation_meaning: str) -> bool:
        """Check if the semantic fields match."""
        if la_word in self.la_semantic_contexts:
            la_field = self.la_semantic_contexts[la_word]

            # Check translation meaning
            religious_words = ['god', 'goddess', 'sacred', 'deity', 'divine']
            place_words = ['city', 'place', 'location', 'mountain']
            economic_words = ['total', 'count', 'number']
            agri_words = ['grain', 'barley', 'wheat', 'agricultural']

            if la_field == 'religious' and any(w in translation_meaning.lower() for w in religious_words):
                return True
            if la_field == 'place_name' and any(w in translation_meaning.lower() for w in place_words):
                return True
            if la_field == 'total/summation' and any(w in translation_meaning.lower() for w in economic_words):
                return True

        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    discoverer = SoundLawDiscoverer()
    results = discoverer.discover()

    # Additional semantic analysis
    print("\n" + "=" * 70)
    print(" " * 15 + "SEMANTIC FIELD CROSS-CHECK")
    print("=" * 70)

    semantic = SemanticFieldAnalyzer()
    print("""
   Checking if matched words have compatible semantic fields...

   Known Linear A semantic fields:
   • ku-ro: appears at end of lists (= "total")
   • pa-i-to: place name (= Phaistos)
   • ja-sa-sa-ra: religious formula (= goddess epithet)

   Cross-checking proposed matches:""")

    # Check specific matches against semantic expectations
    checks = [
        ('ku-ro', 'eman', 'ten (Hurrian)'),
        ('pa-i-to', 'aita', 'Hades (Etruscan)'),
        ('ja-sa-sa-ra', 'thalassa', 'sea (Pre-Greek)'),
    ]

    for la, match, meaning in checks:
        compatible = semantic.check_semantic_match(la, match, meaning)
        status = "✓ COMPATIBLE" if compatible else "✗ MISMATCH"
        print(f"   {la:15} ~ {match:12} '{meaning}' → {status}")

    print("""
   CONCLUSION:
   Semantic fields often DON'T match between Linear A words and
   their phonetic "matches" in other languages. This suggests
   the similarities are coincidental, not meaningful.
    """)

    return results


if __name__ == '__main__':
    main()
