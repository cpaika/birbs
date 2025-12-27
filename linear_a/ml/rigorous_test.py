#!/usr/bin/env python3
"""
Rigorous Hypothesis Testing with Statistical Controls

The initial test showed high scores for Hurrian and Etruscan.
This module adds:
1. Control languages (Semitic, Sumerian) to detect false positives
2. Bootstrap significance testing
3. Stricter phonetic matching
4. Morphological structure comparison
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.inscriptions import get_all_inscriptions
from signs.inventory import SYLLABIC_SIGNS, IDEOGRAMS


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
        elif sign in IDEOGRAMS:
            return None
        else:
            return None
    return '-'.join(phonetics) if phonetics else None


# =============================================================================
# EXPANDED LANGUAGE DATA (including controls)
# =============================================================================

LANGUAGE_DATA = {
    'hurrian': {
        'type': 'test',  # Language we think might be related
        'family': 'Hurro-Urartian',
        'vocabulary': {
            'teshub': 'storm god', 'hebat': 'goddess', 'ewri': 'lord',
            'allai': 'lady', 'tahe': 'man', 'asti': 'woman', 'eni': 'god',
            'pahi': 'head', 'arde': 'city', 'keli': 'house', 'eshi': 'earth',
            'ar': 'give', 'un': 'come', 'tan': 'make', 'fur': 'see',
            'shuki': 'one', 'shena': 'two', 'kig': 'three', 'tumni': 'four',
            'nariya': 'five', 'eman': 'ten', 'kade': 'barley', 'ami': 'wine',
            'shuri': 'kingship', 'nirhhe': 'libation', 'hawirni': 'sheep',
            'sena': 'brother', 'ela': 'sister', 'mena': 'twin',
            'hiari': 'hand', 'turi': 'foot', 'tive': 'word',
        },
        'suffixes': ['-ne', '-sh', '-wa', '-da', '-ae', '-nna', '-ash', '-i', '-o'],
        'structure': 'agglutinative',
    },

    'etruscan': {
        'type': 'test',
        'family': 'Tyrsenian',
        'vocabulary': {
            'tinia': 'sky god', 'uni': 'queen goddess', 'turan': 'Venus',
            'lucumo': 'king', 'zilath': 'magistrate', 'clan': 'son',
            'sec': 'daughter', 'puia': 'wife', 'apa': 'father', 'ati': 'mother',
            'spur': 'city', 'fler': 'offering', 'tur': 'give', 'mul': 'dedicate',
            'thu': 'one', 'zal': 'two', 'ci': 'three', 'sa': 'four',
            'avil': 'year', 'tiv': 'month', 'ais': 'god', 'suth': 'tomb',
            'calu': 'death', 'aita': 'Hades', 'vacl': 'vessel',
            'lauchum': 'lord', 'lautni': 'freedman', 'nefts': 'grandson',
        },
        'suffixes': ['-s', '-l', '-si', '-ri', '-ale', '-ar', '-ce', '-as'],
        'structure': 'agglutinative',
    },

    'akkadian': {
        'type': 'control',  # Semitic control
        'family': 'Semitic',
        'vocabulary': {
            'ilum': 'god', 'sharrum': 'king', 'beltum': 'lady', 'abum': 'father',
            'ummum': 'mother', 'marum': 'son', 'martum': 'daughter',
            'bitum': 'house', 'alum': 'city', 'eqlum': 'field', 'shamash': 'sun god',
            'ishtar': 'goddess', 'marduk': 'god', 'nabu': 'god',
            'kashpum': 'silver', 'hurashu': 'gold', 'siparru': 'bronze',
            'shattum': 'year', 'arhu': 'month', 'umum': 'day',
            'ishtenu': 'one', 'shina': 'two', 'shalash': 'three',
            'erbe': 'four', 'hamish': 'five', 'esher': 'ten',
            'shipru': 'work', 'nakru': 'enemy', 'kalbu': 'dog',
        },
        'suffixes': ['-um', '-am', '-im', '-at', '-u', '-i', '-a'],
        'structure': 'fusional',
    },

    'sumerian': {
        'type': 'control',  # Isolate control
        'family': 'Sumerian',
        'vocabulary': {
            'dingir': 'god', 'lugal': 'king', 'nin': 'lady', 'ab': 'father',
            'ama': 'mother', 'dumu': 'child', 'e': 'house', 'uru': 'city',
            'a': 'water', 'an': 'heaven', 'ki': 'earth', 'ud': 'sun',
            'iti': 'moon', 'mu': 'year', 'kur': 'mountain', 'id': 'river',
            'gal': 'big', 'tur': 'small', 'sig': 'good', 'hul': 'bad',
            'gu': 'ox', 'udu': 'sheep', 'sila': 'lamb', 'mash': 'goat',
            'she': 'barley', 'zid': 'flour', 'ninda': 'bread', 'kas': 'beer',
            'kug': 'silver', 'za': 'stone', 'urudu': 'copper',
        },
        'suffixes': ['-ak', '-ra', '-ta', '-da', '-e', '-a', '-bi', '-ani'],
        'structure': 'agglutinative',
    },

    'hittite': {
        'type': 'control',  # Indo-European control
        'family': 'Anatolian IE',
        'vocabulary': {
            'shiu': 'god', 'hassu': 'king', 'ishha': 'lord', 'anna': 'mother',
            'atta': 'father', 'parna': 'house', 'watar': 'water', 'pahhur': 'fire',
            'nepis': 'heaven', 'tekan': 'earth', 'hassa': 'hearth',
            'eshar': 'blood', 'kurur': 'enemy', 'ishpant': 'night',
            'kuish': 'who', 'kuwat': 'what', 'kuwapi': 'where',
            'sher': 'one', 'daan': 'two',
            'eku': 'drink', 'ed': 'eat', 'pahs': 'protect', 'kuen': 'kill',
            'uwa': 'come', 'pai': 'go', 'ep': 'take', 'dai': 'put',
        },
        'suffixes': ['-as', '-an', '-i', '-us', '-un', '-anza', '-anti'],
        'structure': 'fusional',
    },

    'proto_kartvelian': {
        'type': 'test',  # Caucasian test
        'family': 'Kartvelian',
        'vocabulary': {
            'ghmerti': 'god', 'mepe': 'king', 'deda': 'mother', 'mama': 'father',
            'shvili': 'child', 'saxli': 'house', 'kalaki': 'city', 'mta': 'mountain',
            'tsqali': 'water', 'cecxli': 'fire', 'mze': 'sun', 'mtvare': 'moon',
            'dge': 'day', 'ghame': 'night', 'tseli': 'year',
            'erti': 'one', 'ori': 'two', 'sami': 'three', 'otxi': 'four',
            'xuti': 'five', 'ati': 'ten',
            'dzaghli': 'dog', 'kata': 'cat', 'cxeni': 'horse', 'dzroxa': 'cow',
            'ghvino': 'wine', 'puri': 'bread',
        },
        'suffixes': ['-i', '-a', '-s', '-ma', '-is', '-it', '-ad'],
        'structure': 'agglutinative',
    },

    'lemnian': {
        'type': 'test',  # Aegean test (crucial!)
        'family': 'Tyrsenian',
        'vocabulary': {
            'holaies': 'name', 'sialchvis': 'title', 'avis': 'year',
            'maras': 'name', 'shivai': 'unknown', 'zeronai': 'unknown',
            'morinail': 'unknown', 'haralio': 'unknown', 'evisth': 'suffix',
        },
        'suffixes': ['-ail', '-ai', '-is', '-th'],
        'structure': 'agglutinative',
    },

    'pre_greek': {
        'type': 'test',  # Substrate test
        'family': 'Pre-Greek',
        'vocabulary': {
            'labyrinthos': 'labyrinth', 'hyakinthos': 'hyacinth', 'terebinthos': 'tree',
            'thalassa': 'sea', 'parnassos': 'mountain', 'narkissos': 'narcissus',
            'oinos': 'wine', 'elaia': 'olive', 'kyparissos': 'cypress',
            'labrys': 'double axe', 'tyrannos': 'ruler', 'basileus': 'king',
            'diktynna': 'goddess', 'britomartis': 'goddess', 'ariadne': 'goddess',
        },
        'suffixes': ['-nthos', '-ssos', '-ssa', '-eus', '-na'],
        'structure': 'unknown',
    },
}


# =============================================================================
# STRICTER PHONETIC MATCHING
# =============================================================================

class StrictPhoneticMatcher:
    """Stricter phonetic matching with penalties."""

    def __init__(self):
        # Exact matches only for consonants, some flexibility for vowels
        self.vowels = set('aeiou')
        self.consonants = set('bcdfghjklmnpqrstvwxyz')

    def score(self, word1: str, word2: str) -> float:
        """
        Score phonetic similarity with strict criteria.
        Returns 0-1 score.
        """
        # Normalize
        w1 = word1.lower().replace('-', '')
        w2 = word2.lower().replace('-', '')

        if not w1 or not w2:
            return 0.0

        # Length penalty - words should be similar length
        len_ratio = min(len(w1), len(w2)) / max(len(w1), len(w2))
        if len_ratio < 0.5:
            return 0.0  # Too different in length

        # Extract consonant skeletons
        cons1 = ''.join(c for c in w1 if c in self.consonants)
        cons2 = ''.join(c for c in w2 if c in self.consonants)

        # Consonant match is more important
        cons_score = self._sequence_similarity(cons1, cons2)

        # Vowel pattern
        vow1 = ''.join(c for c in w1 if c in self.vowels)
        vow2 = ''.join(c for c in w2 if c in self.vowels)
        vow_score = self._sequence_similarity(vow1, vow2)

        # Weighted combination
        score = 0.7 * cons_score + 0.3 * vow_score

        # Apply length penalty
        score *= len_ratio

        return score

    def _sequence_similarity(self, s1: str, s2: str) -> float:
        """LCS-based similarity."""
        if not s1 or not s2:
            return 0.5  # Neutral if one is empty

        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs = dp[m][n]
        return 2 * lcs / (m + n)


# =============================================================================
# BOOTSTRAP SIGNIFICANCE TESTING
# =============================================================================

class SignificanceTester:
    """Test if matches are better than random chance."""

    def __init__(self, n_bootstrap: int = 100):
        self.n_bootstrap = n_bootstrap
        self.matcher = StrictPhoneticMatcher()

    def test_significance(self, linear_a_words: List[str],
                         language_vocab: Dict[str, str]) -> Dict:
        """
        Test if the match between Linear A and language is significant.
        """
        # Get real score
        real_scores = []
        for la_word in linear_a_words:
            best_score = 0
            for lang_word in language_vocab.keys():
                score = self.matcher.score(la_word, lang_word)
                best_score = max(best_score, score)
            real_scores.append(best_score)

        real_mean = np.mean(real_scores)

        # Bootstrap: shuffle language words and recompute
        bootstrap_means = []
        lang_words = list(language_vocab.keys())

        for _ in range(self.n_bootstrap):
            shuffled = lang_words.copy()
            random.shuffle(shuffled)

            boot_scores = []
            for la_word in linear_a_words:
                best_score = 0
                for lang_word in shuffled:
                    score = self.matcher.score(la_word, lang_word)
                    best_score = max(best_score, score)
                boot_scores.append(best_score)

            bootstrap_means.append(np.mean(boot_scores))

        # Calculate p-value
        p_value = np.mean([b >= real_mean for b in bootstrap_means])

        # Effect size
        if np.std(bootstrap_means) > 0:
            effect_size = (real_mean - np.mean(bootstrap_means)) / np.std(bootstrap_means)
        else:
            effect_size = 0

        return {
            'real_score': real_mean,
            'bootstrap_mean': np.mean(bootstrap_means),
            'bootstrap_std': np.std(bootstrap_means),
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05 and effect_size > 0.5
        }


# =============================================================================
# MORPHOLOGICAL STRUCTURE COMPARISON
# =============================================================================

class MorphologyComparator:
    """Compare morphological structures."""

    def analyze_structure(self, words: List[str]) -> Dict:
        """Analyze morphological patterns in word list."""
        # Suffix frequency
        suffix_freq = Counter()
        prefix_freq = Counter()

        for word in words:
            syllables = word.split('-') if '-' in word else [word]
            if len(syllables) >= 2:
                suffix_freq[syllables[-1]] += 1
                prefix_freq[syllables[0]] += 1

        # Word length distribution
        lengths = [len(word.split('-')) if '-' in word else 1 for word in words]

        return {
            'common_suffixes': suffix_freq.most_common(5),
            'common_prefixes': prefix_freq.most_common(5),
            'avg_length': np.mean(lengths),
            'length_std': np.std(lengths),
            'suffix_diversity': len(suffix_freq) / len(words) if words else 0,
        }

    def compare_structures(self, structure1: Dict, structure2: Dict) -> float:
        """Compare two morphological structures."""
        score = 0.0

        # Compare average lengths
        len_diff = abs(structure1['avg_length'] - structure2['avg_length'])
        score += max(0, 1 - len_diff / 3)

        # Compare suffix patterns
        suffs1 = set(s for s, _ in structure1['common_suffixes'])
        suffs2 = set(s for s, _ in structure2['common_suffixes'])
        if suffs1 and suffs2:
            overlap = len(suffs1 & suffs2) / len(suffs1 | suffs2)
            score += overlap

        # Compare diversity
        div_diff = abs(structure1['suffix_diversity'] - structure2['suffix_diversity'])
        score += max(0, 1 - div_diff * 5)

        return score / 3


# =============================================================================
# MAIN RIGOROUS TESTER
# =============================================================================

class RigorousHypothesisTester:
    """Rigorous testing with controls and significance."""

    def __init__(self):
        self.matcher = StrictPhoneticMatcher()
        self.significance = SignificanceTester(n_bootstrap=50)
        self.morphology = MorphologyComparator()

    def run_tests(self) -> Dict:
        """Run rigorous hypothesis tests."""
        print("\n" + "=" * 70)
        print(" " * 10 + "RIGOROUS HYPOTHESIS TESTING WITH CONTROLS")
        print("=" * 70)

        # Get Linear A words
        inscriptions = get_all_inscriptions()
        la_words = set()
        for insc in inscriptions:
            for word in insc.get_words():
                reading = get_phonetic_reading(word)
                if reading:
                    la_words.add(reading)

        la_words = list(la_words)
        print(f"\nAnalyzing {len(la_words)} Linear A words...")

        # Analyze Linear A morphology
        la_morph = self.morphology.analyze_structure(la_words)
        print(f"\nLinear A Morphology:")
        print(f"  • Average word length: {la_morph['avg_length']:.1f} syllables")
        print(f"  • Common suffixes: {[s for s, _ in la_morph['common_suffixes']]}")

        results = {}

        for lang_name, lang_data in LANGUAGE_DATA.items():
            print(f"\n{'='*70}")
            print(f"Testing: {lang_name.upper()} ({lang_data['type'].upper()} - {lang_data['family']})")
            print(f"{'='*70}")

            vocab = lang_data['vocabulary']

            # Basic matching
            matches = []
            for la_word in la_words:
                best_match = None
                best_score = 0
                for lang_word, meaning in vocab.items():
                    score = self.matcher.score(la_word, lang_word)
                    if score > best_score:
                        best_score = score
                        best_match = (lang_word, meaning)
                if best_match and best_score > 0.3:
                    matches.append((la_word, best_match[0], best_match[1], best_score))

            matches.sort(key=lambda x: -x[3])

            # Significance testing
            print("\n  Running bootstrap significance test...")
            sig_result = self.significance.test_significance(la_words, vocab)

            # Morphology comparison
            lang_word_list = list(vocab.keys())
            lang_morph = self.morphology.analyze_structure(lang_word_list)
            morph_score = self.morphology.compare_structures(la_morph, lang_morph)

            # Combined score
            vocab_score = sig_result['real_score']
            significance_bonus = 0.2 if sig_result['significant'] else 0
            combined = (vocab_score * 0.5 + morph_score * 0.3 + significance_bonus)

            results[lang_name] = {
                'type': lang_data['type'],
                'family': lang_data['family'],
                'vocab_score': vocab_score,
                'p_value': sig_result['p_value'],
                'effect_size': sig_result['effect_size'],
                'significant': sig_result['significant'],
                'morph_score': morph_score,
                'combined_score': combined,
                'top_matches': matches[:5],
            }

            # Print results
            print(f"\n  Vocabulary Score: {vocab_score:.3f}")
            print(f"  Bootstrap p-value: {sig_result['p_value']:.3f}")
            print(f"  Effect size: {sig_result['effect_size']:.2f}")
            print(f"  Significant: {'YES' if sig_result['significant'] else 'NO'}")
            print(f"  Morphology Score: {morph_score:.3f}")
            print(f"  COMBINED SCORE: {combined:.3f}")

            if matches:
                print(f"\n  Top Matches:")
                for la, lang, meaning, score in matches[:3]:
                    print(f"    {la:15} ~ {lang:15} '{meaning}' ({score:.0%})")

        # Final ranking
        print("\n" + "=" * 70)
        print(" " * 25 + "FINAL RANKING")
        print("=" * 70)

        # Separate test and control languages
        test_results = {k: v for k, v in results.items() if v['type'] == 'test'}
        control_results = {k: v for k, v in results.items() if v['type'] == 'control'}

        print("\n📊 CONTROL LANGUAGES (should score LOW if method is valid):")
        for lang, data in sorted(control_results.items(), key=lambda x: -x[1]['combined_score']):
            sig = "✓" if data['significant'] else "✗"
            print(f"   {lang:20} {data['combined_score']:.3f}  [p={data['p_value']:.3f}] {sig}")

        print("\n📊 TEST LANGUAGES (looking for HIGH scores):")
        for lang, data in sorted(test_results.items(), key=lambda x: -x[1]['combined_score']):
            sig = "✓ SIGNIFICANT" if data['significant'] else ""
            print(f"   {lang:20} {data['combined_score']:.3f}  [p={data['p_value']:.3f}] {sig}")

        # Conclusion
        best_test = max(test_results.items(), key=lambda x: x[1]['combined_score'])
        best_control = max(control_results.items(), key=lambda x: x[1]['combined_score'])

        print("\n" + "=" * 70)
        print(" " * 25 + "CONCLUSION")
        print("=" * 70)

        if best_test[1]['combined_score'] > best_control[1]['combined_score'] * 1.2:
            if best_test[1]['significant']:
                print(f"""
   ✓ MEANINGFUL RESULT: {best_test[0].upper()} scores significantly higher
     than control languages.

   This suggests a possible relationship between Linear A and {best_test[0]}.

   Key evidence:
   • Vocabulary score: {best_test[1]['vocab_score']:.3f}
   • Statistically significant (p={best_test[1]['p_value']:.3f})
   • Effect size: {best_test[1]['effect_size']:.2f}

   Recommended: Deep investigation of {best_test[0]} connection
                """)
            else:
                print(f"""
   ⚠ SUGGESTIVE but NOT SIGNIFICANT: {best_test[0].upper()} scores higher
     than controls, but did not reach statistical significance.

   This could indicate:
   • A weak relationship
   • Coincidental similarities
   • Need for more data
                """)
        else:
            print(f"""
   ✗ NO CLEAR RELATIONSHIP FOUND

   Test languages do not score meaningfully higher than controls.
   Linear A appears to be a true language isolate with no detectable
   relationship to any tested language family.

   Control scores similar to test scores suggest our matches
   may be coincidental phonetic similarities.
            """)

        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(42)  # Reproducibility
    np.random.seed(42)

    tester = RigorousHypothesisTester()
    results = tester.run_tests()
    return results


if __name__ == '__main__':
    main()
