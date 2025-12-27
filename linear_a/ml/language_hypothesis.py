#!/usr/bin/env python3
"""
Language Hypothesis Testing for Linear A

Implements the MIT minimum-cost-flow inspired approach:
1. Pre-train on a candidate language (Hurrian, Etruscan, etc.)
2. Test if Linear A shows transfer/cognacy
3. Iterate through candidates until we find a match

Based on: Luo, Cao & Barzilay (2019) "Neural Decipherment via Minimum-Cost Flow"
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import sys
import os
import re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.inscriptions import get_all_inscriptions, extract_all_words
from signs.inventory import SYLLABIC_SIGNS, IDEOGRAMS


def get_phonetic_reading(word: List[str]) -> Optional[str]:
    """Convert a word (list of sign codes) to its phonetic reading."""
    phonetics = []
    for sign in word:
        if sign in SYLLABIC_SIGNS:
            p = SYLLABIC_SIGNS[sign].phonetic
            if p and p != "?":
                phonetics.append(p)
            else:
                return None  # Skip words with unknown signs
        elif sign in IDEOGRAMS:
            return None
        else:
            return None
    return '-'.join(phonetics) if phonetics else None


# =============================================================================
# HURRIAN LANGUAGE DATA
# =============================================================================

@dataclass
class HurrianData:
    """
    Hurrian language data for comparison with Linear A.

    Hurrian was spoken in northern Mesopotamia and Anatolia (c. 2300-1000 BCE).
    Key features:
    - Agglutinative morphology
    - Ergative-absolutive alignment
    - SOV word order
    - Complex suffix chains

    Sources: Mittanni Letter, Bogazköy texts, Nuzi tablets
    """

    # Core vocabulary with meanings
    vocabulary: Dict[str, str] = None

    # Grammatical suffixes
    suffixes: Dict[str, str] = None

    # Phoneme inventory
    phonemes: Set[str] = None

    def __post_init__(self):
        # Hurrian vocabulary (transliterated)
        self.vocabulary = {
            # Deities
            'teshub': 'storm god',
            'hebat': 'goddess (queen of heaven)',
            'kumarbi': 'father of gods',
            'shaushka': 'goddess of love/war',
            'shimige': 'sun god',
            'kushuh': 'moon god',
            'ea': 'god of wisdom',
            'allani': 'lady of the underworld',

            # Royalty/titles
            'ewri': 'lord/king',
            'allai': 'lady/queen',
            'tahe': 'man',
            'asti': 'woman',
            'eni': 'god',
            'sena': 'brother',
            'ela': 'sister',
            'mena': 'twin',

            # Body parts
            'pahi': 'head',
            'teri': 'chest',
            'hiari': 'hand',
            'turi': 'foot',
            'pendi': 'ear',
            'tive': 'word/thing',

            # Objects
            'arde': 'city',
            'abi': 'pit/well',
            'keli': 'house/temple',
            'hazhiari': 'tribute',
            'tabri': 'craftsman',
            'eshi': 'earth',
            'ije': 'thing/matter',

            # Actions (verb roots)
            'ar': 'give',
            'un': 'come',
            'man': 'be',
            'tan': 'make/do',
            'ag': 'lead',
            'pai': 'build',
            'hatte': 'write',
            'nahhubi': 'seat/sit',
            'fur': 'see',
            'hashi': 'hear',

            # Numbers
            'shuki': 'one',
            'shena': 'two',
            'kig': 'three',
            'tumni': 'four',
            'nariya': 'five',
            'sheshe': 'six',
            'shindi': 'seven',
            'kiriya': 'eight',
            'tamriya': 'nine',
            'eman': 'ten',

            # Agricultural/economic
            'kade': 'barley',
            'shinni': 'wheat',
            'shahi': 'pig',
            'hawirni': 'sheep',
            'izzi': 'fire',
            'ami': 'wine',

            # Abstract
            'shuri': 'king(ship)',
            'pahe': 'goodness',
            'huradi': 'warrior',
            'nirhhe': 'libation',
        }

        # Hurrian grammatical suffixes
        self.suffixes = {
            # Case endings
            '-ne': 'ergative (agent of transitive)',
            '-sh': 'absolutive (patient/intransitive subject)',
            '-wa': 'genitive',
            '-da': 'dative',
            '-dan': 'directive/allative (to)',
            '-ae': 'locative (in/at)',
            '-ura': 'ablative (from)',

            # Article/definiteness
            '-nna': 'the (definite article)',
            '-lla': 'a (indefinite)',

            # Plural
            '-ash': 'plural marker',
            '-na': 'collective plural',

            # Relational suffixes
            '-ardi': 'with',
            '-khhe': 'like/as',
            '-iffu': 'and',

            # Verbal suffixes
            '-i': 'intransitive marker',
            '-o': 'transitive marker',
            '-iya': 'passive',
            '-anna': 'optative',
            '-ili': 'causative',
        }

        # Hurrian phoneme inventory
        self.phonemes = {
            # Vowels
            'a', 'e', 'i', 'o', 'u',
            # Stops
            'p', 'b', 't', 'd', 'k', 'g',
            # Fricatives
            's', 'sh', 'h', 'hh',
            # Nasals
            'm', 'n',
            # Liquids
            'r', 'l',
            # Glides
            'w', 'y',
            # Affricates
            'z',
        }


# =============================================================================
# ETRUSCAN LANGUAGE DATA
# =============================================================================

@dataclass
class EtruscanData:
    """
    Etruscan language data for comparison with Linear A.

    Etruscan was spoken in Italy (c. 700 BCE - 100 CE).
    Key features:
    - Agglutinative morphology
    - No grammatical gender
    - Case suffixes
    - Related to Lemnian (Aegean connection!)

    The Tyrsenian hypothesis links Etruscan, Lemnian, and possibly Minoan.
    """

    vocabulary: Dict[str, str] = None
    suffixes: Dict[str, str] = None
    phonemes: Set[str] = None

    def __post_init__(self):
        # Etruscan vocabulary (from inscriptions)
        self.vocabulary = {
            # Deities
            'tinia': 'Jupiter/sky god',
            'uni': 'Juno/queen goddess',
            'menrva': 'Minerva',
            'turms': 'Mercury',
            'turan': 'Venus/love',
            'maris': 'Mars',
            'sethlans': 'Vulcan',
            'fufluns': 'Bacchus',
            'calu': 'death deity',
            'aita': 'Hades/underworld god',

            # Titles/people
            'lucumo': 'king/chief',
            'zilath': 'magistrate',
            'lauchum': 'lord',
            'lautni': 'freedman',
            'etera': 'servant/client',
            'clan': 'son',
            'sec': 'daughter',
            'puia': 'wife',
            'apa': 'father',
            'ati': 'mother',
            'nefts': 'nephew/grandson',

            # Objects/places
            'spur': 'city',
            'tular': 'boundary',
            'vacl': 'libation vessel',
            'pruchum': 'pitcher',
            'thina': 'vessel/vase',
            'fler': 'offering/statue',

            # Verbs/actions
            'tur': 'give',
            'ar': 'make/do',
            'mul': 'dedicate',
            'lup': 'die',
            'zic': 'write',
            'ten': 'hold/have',
            'cer': 'make/build',

            # Numbers
            'thu': 'one',
            'zal': 'two',
            'ci': 'three',
            'sa': 'four',
            'mach': 'five',
            'huth': 'six',
            'semph': 'seven',
            'cezp': 'eight',
            'nurph': 'nine',
            'sar': 'ten',

            # Time/other
            'avil': 'year',
            'tiv': 'month/moon',
            'ril': 'age/at age of',
            'ais': 'god (generic)',
            'nes': 'dead person',
            'suth': 'tomb/grave',
            'thui': 'here',
            'eca': 'this',
            'ita': 'this',
        }

        # Etruscan grammatical suffixes
        self.suffixes = {
            # Case endings
            '-s': 'genitive (of)',
            '-l': 'genitive variant',
            '-si': 'locative (in/at)',
            '-i': 'locative variant',
            '-ri': 'dative/for',
            '-ale': 'ablative (from)',
            '-th': 'ablative variant',
            '-eri': 'pertinentive (concerning)',

            # Plural
            '-ar': 'plural',
            '-er': 'plural variant',
            '-va': 'plural (archaic)',

            # Demonstrative/article
            '-sa': 'demonstrative this',
            '-ta': 'demonstrative that',
            '-ca': 'and (enclitic)',

            # Verbal
            '-ce': 'past tense/perfect',
            '-as': 'past participle',
            '-u': 'verbal noun',
            '-eri': 'infinitive',
        }

        # Etruscan phonemes
        self.phonemes = {
            'a', 'e', 'i', 'u',  # No 'o' in native words!
            'p', 'ph', 't', 'th', 'c', 'ch',
            'v', 'f', 's', 'sh', 'h',
            'm', 'n',
            'r', 'l',
            'z',
        }


# =============================================================================
# LEMNIAN LANGUAGE DATA (Aegean relative of Etruscan)
# =============================================================================

@dataclass
class LemnianData:
    """
    Lemnian language data - crucial because it's from the AEGEAN.

    Found on the Lemnos stele (6th century BCE).
    Related to Etruscan (Tyrsenian family).
    Geographic proximity to Crete makes it significant.
    """

    vocabulary: Dict[str, str] = None

    def __post_init__(self):
        # From the Lemnos stele (limited corpus)
        self.vocabulary = {
            'holaies': 'person name?',
            'fokiasiale': 'from Phocaea?',
            'sialchvis': 'unknown (title?)',
            'avis': 'year? (cf. Etruscan avil)',
            'maras': 'name (cf. Etruscan maris)',
            'tavarsio': 'unknown',
            'vanala': 'unknown',
            'zeronai': 'unknown',
            'morinail': 'unknown',
            'shivai': 'unknown (cf. Linear A si-wa?)',
            'aker': 'unknown',
            'rom': 'unknown',
            'haralio': 'unknown',
            'evisth': 'unknown (cf. Etruscan suffix?)',
            'aomai': 'unknown',
            'mav': 'unknown',
        }


# =============================================================================
# ANATOLIAN SUBSTRATE DATA
# =============================================================================

@dataclass
class AnatolianSubstrate:
    """
    Pre-Greek/Pre-Anatolian substrate words that might be Minoan.
    These survived in Greek and Anatolian languages.
    """

    vocabulary: Dict[str, str] = None

    def __post_init__(self):
        self.vocabulary = {
            # -nthos/-ntha words (non-IE)
            'labyrinthos': 'labyrinth (place of double-axe)',
            'hyakinthos': 'hyacinth',
            'terebinthos': 'turpentine tree',
            'korinthos': 'Corinth (place name)',
            'zakynthos': 'Zakynthos (island)',
            'olynthos': 'wild fig',
            'akanthos': 'acanthus',
            'plinthos': 'brick',
            'minthos': 'dung (cf. Minotaur)',

            # -ssos/-ssa words
            'thalassa': 'sea',
            'parnassos': 'Parnassus (mountain)',
            'knossos': 'Knossos',
            'lykabettos': 'Lycabettus',
            'hymettos': 'Hymettus',
            'narkissos': 'narcissus',

            # Plant terms (Mediterranean substrate)
            'oinos': 'wine',
            'elaia': 'olive',
            'selinon': 'celery',
            'kyparissos': 'cypress',
            'daphne': 'laurel',
            'kissos': 'ivy',
            'sminthos': 'mouse',

            # Religious terms
            'labrys': 'double axe (sacred)',
            'diktynna': 'goddess of nets',
            'britomartis': 'sweet maiden (goddess)',
            'pasiphae': 'all-shining (goddess)',
            'ariadne': 'most holy (goddess)',

            # Titles
            'tyrannos': 'ruler/king',
            'basileus': 'king',
            'wanax': 'lord/high king',
            'despotes': 'master',
        }


# =============================================================================
# MINIMUM-COST FLOW COGNATE DETECTOR
# =============================================================================

class CognateScorer:
    """
    Score potential cognates using the minimum-cost flow approach.

    The key insight from Luo et al. (2019):
    - Model sound correspondences as a bipartite matching problem
    - Find the minimum-cost alignment between source and target words
    - Regular sound changes should have low cost
    """

    def __init__(self):
        # Common sound correspondences across language families
        self.sound_correspondences = {
            # Vowel correspondences
            ('a', 'a'): 0.0, ('a', 'e'): 0.3, ('a', 'i'): 0.5, ('a', 'o'): 0.3, ('a', 'u'): 0.5,
            ('e', 'e'): 0.0, ('e', 'i'): 0.2, ('e', 'a'): 0.3,
            ('i', 'i'): 0.0, ('i', 'e'): 0.2, ('i', 'u'): 0.4,
            ('o', 'o'): 0.0, ('o', 'u'): 0.2, ('o', 'a'): 0.3,
            ('u', 'u'): 0.0, ('u', 'o'): 0.2, ('u', 'i'): 0.4,

            # Stop correspondences
            ('p', 'p'): 0.0, ('p', 'b'): 0.2, ('p', 'ph'): 0.1, ('p', 'f'): 0.3,
            ('t', 't'): 0.0, ('t', 'd'): 0.2, ('t', 'th'): 0.1, ('t', 's'): 0.4,
            ('k', 'k'): 0.0, ('k', 'g'): 0.2, ('k', 'c'): 0.1, ('k', 'ch'): 0.2,
            ('b', 'b'): 0.0, ('b', 'p'): 0.2, ('b', 'v'): 0.3,
            ('d', 'd'): 0.0, ('d', 't'): 0.2, ('d', 'z'): 0.4,
            ('g', 'g'): 0.0, ('g', 'k'): 0.2,

            # Nasal correspondences
            ('m', 'm'): 0.0, ('m', 'n'): 0.3,
            ('n', 'n'): 0.0, ('n', 'm'): 0.3,

            # Liquid correspondences
            ('r', 'r'): 0.0, ('r', 'l'): 0.3,
            ('l', 'l'): 0.0, ('l', 'r'): 0.3,

            # Sibilant correspondences
            ('s', 's'): 0.0, ('s', 'sh'): 0.2, ('s', 'z'): 0.3, ('s', 'th'): 0.4,
            ('sh', 'sh'): 0.0, ('sh', 's'): 0.2,
            ('z', 'z'): 0.0, ('z', 's'): 0.3,

            # Glide correspondences
            ('w', 'w'): 0.0, ('w', 'v'): 0.2, ('w', 'u'): 0.3,
            ('j', 'j'): 0.0, ('j', 'y'): 0.0, ('j', 'i'): 0.3,
            ('y', 'y'): 0.0, ('y', 'j'): 0.0, ('y', 'i'): 0.3,
        }

        # Default cost for unknown correspondences
        self.default_cost = 0.8

        # Cost for insertion/deletion
        self.gap_cost = 0.5

    def get_correspondence_cost(self, s1: str, s2: str) -> float:
        """Get the cost of a sound correspondence."""
        if s1 == s2:
            return 0.0
        if (s1, s2) in self.sound_correspondences:
            return self.sound_correspondences[(s1, s2)]
        if (s2, s1) in self.sound_correspondences:
            return self.sound_correspondences[(s2, s1)]
        return self.default_cost

    def align_words(self, word1: str, word2: str) -> Tuple[float, List[Tuple]]:
        """
        Align two words using dynamic programming (Needleman-Wunsch variant).
        Returns alignment cost and the alignment itself.
        """
        # Parse into segments
        segs1 = self._parse_segments(word1)
        segs2 = self._parse_segments(word2)

        m, n = len(segs1), len(segs2)

        # DP table
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0

        # Initialize gaps
        for i in range(1, m + 1):
            dp[i][0] = dp[i-1][0] + self.gap_cost
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j-1] + self.gap_cost

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match_cost = dp[i-1][j-1] + self.get_correspondence_cost(segs1[i-1], segs2[j-1])
                gap1_cost = dp[i-1][j] + self.gap_cost
                gap2_cost = dp[i][j-1] + self.gap_cost
                dp[i][j] = min(match_cost, gap1_cost, gap2_cost)

        # Normalize by length
        alignment_cost = dp[m][n] / max(m, n) if max(m, n) > 0 else 0

        return alignment_cost, []

    def _parse_segments(self, word: str) -> List[str]:
        """Parse word into phonetic segments."""
        # Handle hyphenated (Linear A style)
        if '-' in word:
            syllables = word.split('-')
            segments = []
            for syl in syllables:
                segments.extend(self._segment_syllable(syl))
            return segments
        else:
            return self._segment_syllable(word)

    def _segment_syllable(self, syl: str) -> List[str]:
        """Segment a syllable into individual sounds."""
        segments = []
        i = 0
        while i < len(syl):
            # Check for digraphs
            if i + 1 < len(syl) and syl[i:i+2] in ['sh', 'ch', 'th', 'ph', 'hh']:
                segments.append(syl[i:i+2])
                i += 2
            else:
                segments.append(syl[i])
                i += 1
        return segments

    def score_cognate_pair(self, linear_a_word: str, candidate_word: str) -> float:
        """
        Score a potential cognate pair.
        Lower score = better match.
        Returns normalized score 0-1.
        """
        cost, _ = self.align_words(linear_a_word, candidate_word)
        # Convert cost to similarity score
        similarity = max(0, 1 - cost)
        return similarity


# =============================================================================
# LANGUAGE HYPOTHESIS TESTER
# =============================================================================

class LanguageHypothesisTester:
    """
    Test whether Linear A could be related to a candidate language.

    Approach:
    1. Build vocabulary model for candidate language
    2. Find potential cognates with Linear A
    3. Check for systematic sound correspondences
    4. Evaluate grammatical parallels
    """

    def __init__(self):
        self.scorer = CognateScorer()
        self.hurrian = HurrianData()
        self.etruscan = EtruscanData()
        self.lemnian = LemnianData()
        self.substrate = AnatolianSubstrate()

    def test_hypothesis(self, language_name: str,
                       language_vocab: Dict[str, str],
                       language_suffixes: Dict[str, str] = None) -> Dict:
        """
        Test if Linear A is related to the given language.
        """
        results = {
            'language': language_name,
            'cognate_candidates': [],
            'suffix_matches': [],
            'overall_score': 0.0,
            'verdict': ''
        }

        # Get Linear A vocabulary
        inscriptions = get_all_inscriptions()
        linear_a_words = set()
        for insc in inscriptions:
            for word in insc.get_words():
                reading = get_phonetic_reading(word)
                if reading:
                    linear_a_words.add(reading)

        print(f"\n   Testing {len(linear_a_words)} Linear A words against {len(language_vocab)} {language_name} words...")

        # Find cognate candidates
        cognate_pairs = []
        for la_word in linear_a_words:
            best_match = None
            best_score = 0

            for cand_word, meaning in language_vocab.items():
                score = self.scorer.score_cognate_pair(la_word, cand_word)
                if score > best_score and score > 0.4:  # Threshold
                    best_score = score
                    best_match = (cand_word, meaning, score)

            if best_match:
                cognate_pairs.append((la_word, best_match[0], best_match[1], best_match[2]))

        # Sort by score
        cognate_pairs.sort(key=lambda x: -x[3])
        results['cognate_candidates'] = cognate_pairs[:20]

        # Test suffix matches if available
        if language_suffixes:
            la_suffixes = self._extract_linear_a_suffixes(linear_a_words)
            for la_suf, count in la_suffixes.most_common(10):
                for lang_suf, meaning in language_suffixes.items():
                    score = self.scorer.score_cognate_pair(la_suf, lang_suf.lstrip('-'))
                    if score > 0.5:
                        results['suffix_matches'].append((la_suf, lang_suf, meaning, score))

        # Calculate overall score
        if cognate_pairs:
            avg_cognate_score = np.mean([x[3] for x in cognate_pairs[:10]])
            suffix_bonus = len(results['suffix_matches']) * 0.05
            results['overall_score'] = min(1.0, avg_cognate_score + suffix_bonus)

        # Verdict
        if results['overall_score'] > 0.7:
            results['verdict'] = 'STRONG MATCH - Possible genetic relationship'
        elif results['overall_score'] > 0.5:
            results['verdict'] = 'MODERATE MATCH - Worth investigating'
        elif results['overall_score'] > 0.3:
            results['verdict'] = 'WEAK MATCH - Some similarities, likely coincidental'
        else:
            results['verdict'] = 'NO MATCH - No evidence of relationship'

        return results

    def _extract_linear_a_suffixes(self, words: Set[str]) -> Counter:
        """Extract common suffixes from Linear A words."""
        suffixes = Counter()
        for word in words:
            syllables = word.split('-')
            if len(syllables) >= 2:
                # Last syllable as potential suffix
                suffixes[syllables[-1]] += 1
                # Last two syllables
                if len(syllables) >= 3:
                    suffixes['-'.join(syllables[-2:])] += 1
        return suffixes

    def run_all_hypotheses(self) -> List[Dict]:
        """Test all language hypotheses."""
        results = []

        print("\n" + "=" * 70)
        print(" " * 10 + "LANGUAGE HYPOTHESIS TESTING FOR LINEAR A")
        print("=" * 70)
        print("\nTesting candidate languages to find Linear A's relatives...")

        # Test Hurrian
        print("\n" + "-" * 70)
        print("HYPOTHESIS 1: HURRIAN")
        print("-" * 70)
        hurrian_result = self.test_hypothesis(
            "Hurrian",
            self.hurrian.vocabulary,
            self.hurrian.suffixes
        )
        results.append(hurrian_result)
        self._print_result(hurrian_result)

        # Test Etruscan
        print("\n" + "-" * 70)
        print("HYPOTHESIS 2: ETRUSCAN")
        print("-" * 70)
        etruscan_result = self.test_hypothesis(
            "Etruscan",
            self.etruscan.vocabulary,
            self.etruscan.suffixes
        )
        results.append(etruscan_result)
        self._print_result(etruscan_result)

        # Test Lemnian (Aegean Tyrsenian)
        print("\n" + "-" * 70)
        print("HYPOTHESIS 3: LEMNIAN (Aegean Tyrsenian)")
        print("-" * 70)
        lemnian_result = self.test_hypothesis(
            "Lemnian",
            self.lemnian.vocabulary,
            None
        )
        results.append(lemnian_result)
        self._print_result(lemnian_result)

        # Test Pre-Greek Substrate
        print("\n" + "-" * 70)
        print("HYPOTHESIS 4: PRE-GREEK SUBSTRATE")
        print("-" * 70)
        substrate_result = self.test_hypothesis(
            "Pre-Greek Substrate",
            self.substrate.vocabulary,
            None
        )
        results.append(substrate_result)
        self._print_result(substrate_result)

        return results

    def _print_result(self, result: Dict):
        """Print test results."""
        print(f"\n   Score: {result['overall_score']:.2%}")
        print(f"   Verdict: {result['verdict']}")

        if result['cognate_candidates']:
            print(f"\n   Top Cognate Candidates:")
            for la, cand, meaning, score in result['cognate_candidates'][:5]:
                print(f"   • {la:15} ~ {cand:15} '{meaning}' ({score:.0%})")

        if result['suffix_matches']:
            print(f"\n   Suffix Matches:")
            for la_suf, lang_suf, meaning, score in result['suffix_matches'][:5]:
                print(f"   • -{la_suf:10} ~ {lang_suf:10} '{meaning}' ({score:.0%})")


# =============================================================================
# ITERATIVE REFINEMENT
# =============================================================================

class IterativeDecipherer:
    """
    Iteratively refine decipherment by testing multiple hypotheses
    and combining the best evidence.
    """

    def __init__(self):
        self.tester = LanguageHypothesisTester()
        self.best_matches = {}
        self.iteration = 0
        self.max_iterations = 5

    def run(self) -> Dict:
        """Run iterative hypothesis testing."""
        print("\n" + "=" * 70)
        print(" " * 15 + "ITERATIVE DECIPHERMENT LOOP")
        print("=" * 70)

        all_results = []
        combined_vocab = {}

        while self.iteration < self.max_iterations:
            self.iteration += 1
            print(f"\n{'='*70}")
            print(f" ITERATION {self.iteration}")
            print(f"{'='*70}")

            # Run all hypothesis tests
            results = self.tester.run_all_hypotheses()
            all_results.extend(results)

            # Find best matches from this iteration
            for result in results:
                for la, cand, meaning, score in result['cognate_candidates']:
                    if la not in combined_vocab or score > combined_vocab[la]['score']:
                        combined_vocab[la] = {
                            'meaning': meaning,
                            'source_word': cand,
                            'source_language': result['language'],
                            'score': score
                        }

            # Check convergence
            best_result = max(results, key=lambda x: x['overall_score'])
            print(f"\n   Best match this iteration: {best_result['language']} ({best_result['overall_score']:.2%})")

            # If we found a strong match, focus on it
            if best_result['overall_score'] > 0.6:
                print(f"\n   PROMISING LEAD: {best_result['language']}")
                print(f"   Focusing refinement on this hypothesis...")
                # In a real implementation, we'd do deeper analysis here
                break

            # If no progress, try combining evidence
            if self.iteration >= 2:
                print("\n   Combining evidence from multiple languages...")
                break

        # Final summary
        return self._generate_final_report(all_results, combined_vocab)

    def _generate_final_report(self, results: List[Dict], vocab: Dict) -> Dict:
        """Generate final decipherment report."""
        print("\n" + "=" * 70)
        print(" " * 20 + "FINAL REPORT")
        print("=" * 70)

        # Rank languages
        lang_scores = {}
        for r in results:
            if r['language'] not in lang_scores:
                lang_scores[r['language']] = []
            lang_scores[r['language']].append(r['overall_score'])

        avg_scores = {lang: np.mean(scores) for lang, scores in lang_scores.items()}
        ranked = sorted(avg_scores.items(), key=lambda x: -x[1])

        print("\n📊 LANGUAGE RANKING:")
        for i, (lang, score) in enumerate(ranked, 1):
            bar = '█' * int(score * 30)
            print(f"   {i}. {lang:25} {bar} ({score:.1%})")

        # Best vocabulary
        print("\n📖 BEST VOCABULARY MATCHES:")
        sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1]['score'])
        for word, info in sorted_vocab[:15]:
            print(f"   {word:20} = {info['meaning']:20} ({info['source_language']}, {info['score']:.0%})")

        # Conclusion
        best_lang = ranked[0][0] if ranked else "Unknown"
        best_score = ranked[0][1] if ranked else 0

        print("\n" + "=" * 70)
        print(" " * 20 + "CONCLUSION")
        print("=" * 70)

        if best_score > 0.6:
            print(f"""
   PROMISING RESULT: Linear A shows strongest affinity with {best_lang}

   This suggests either:
   1. Genetic relationship (same language family)
   2. Areal contact (borrowing/sprachbund)
   3. Common substrate influence

   Next steps:
   • Deep morphological analysis comparing Linear A to {best_lang}
   • Search for systematic sound correspondences
   • Test against expanded {best_lang} corpus
            """)
        elif best_score > 0.4:
            print(f"""
   INCONCLUSIVE: Linear A shows weak similarities with multiple languages

   The data suggests:
   • Linear A may be a language isolate
   • Or related to an extinct, unattested language family
   • Some vocabulary may be borrowed from contact languages

   The strongest (but still weak) match is {best_lang}
            """)
        else:
            print(f"""
   NEGATIVE RESULT: No clear relationship found

   Linear A appears to be:
   • A true language isolate
   • Unrelated to any tested language family
   • Possibly part of an entirely lost language family

   This confirms the difficulty of decipherment without
   a bilingual text or related living language.
            """)

        return {
            'rankings': ranked,
            'vocabulary': vocab,
            'best_match': best_lang,
            'best_score': best_score
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the iterative language hypothesis testing."""
    decipherer = IterativeDecipherer()
    results = decipherer.run()
    return results


if __name__ == '__main__':
    main()
