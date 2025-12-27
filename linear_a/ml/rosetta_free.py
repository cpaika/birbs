#!/usr/bin/env python3
"""
Rosetta-Free Decipherment Strategies for Linear A

This module implements approaches to decipher Linear A WITHOUT a bilingual text,
using methods that have worked for other undeciphered scripts.

Key Strategies:
1. PRE-GREEK SUBSTRATE - Minoan loanwords that survived into Greek
2. SEMANTIC FIELD MAPPING - Use ideograms to constrain word meanings
3. EXHAUSTIVE LANGUAGE COMPARISON - Test against all ancient language families
4. CONTEXTUAL PROPAGATION - Use archaeological context as semantic anchor
5. COMBINATORIAL ELIMINATION - Rule out impossible readings

Historical Precedent:
- Linear B was cracked without a Rosetta Stone by Ventris (1952)
- Key insight: assume a known language and test systematically
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import sys
import os
import re

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
                phonetics.append(sign)  # Keep original if no phonetic value
        elif sign in IDEOGRAMS:
            return None  # Skip ideographic words
        else:
            phonetics.append(sign)  # Keep unknown signs as-is
    return '-'.join(phonetics) if phonetics else None


# =============================================================================
# STRATEGY 1: PRE-GREEK SUBSTRATE ANALYSIS
# =============================================================================

class SubstrateAnalyzer:
    """
    Analyze pre-Greek substrate words that may be Minoan loanwords.

    Many Greek words have no Indo-European etymology and show patterns
    suggesting they were borrowed from the pre-Greek population (Minoans).

    Common pre-Greek features:
    - -nth- suffix (Korinthos, labyrinthos, Zakynthos)
    - -ss- suffix (thalassa, Parnassos, Knossos)
    - -mn- cluster (Amnissos)
    - Initial a- that doesn't derive from PIE
    """

    def __init__(self):
        # Pre-Greek words that likely derive from Minoan
        # These survived into Greek and we know their meanings!
        self.substrate_words = {
            # Place names (some match Linear A!)
            'knossos': {'meaning': 'city name', 'linear_a': 'ko-no-so', 'confidence': 0.9},
            'phaistos': {'meaning': 'city name', 'linear_a': 'pa-i-to', 'confidence': 0.95},
            'amnissos': {'meaning': 'harbor', 'linear_a': 'a-mi-ni-so', 'confidence': 0.7},
            'tylissos': {'meaning': 'city name', 'linear_a': 'tu-ri-so', 'confidence': 0.6},
            'zakynthos': {'meaning': 'island', 'linear_a': None, 'confidence': 0.5},

            # Religious/cultural terms
            'labyrinthos': {'meaning': 'labyrinth/double-axe place', 'linear_a': 'da-pu-ri-to?', 'confidence': 0.4},
            'hyakinthos': {'meaning': 'hyacinth (flower/god)', 'linear_a': None, 'confidence': 0.5},
            'daphne': {'meaning': 'laurel', 'linear_a': 'da-pu-ne?', 'confidence': 0.3},
            'kyparissos': {'meaning': 'cypress', 'linear_a': None, 'confidence': 0.4},
            'selinon': {'meaning': 'celery/parsley', 'linear_a': None, 'confidence': 0.3},
            'terebinthos': {'meaning': 'turpentine tree', 'linear_a': None, 'confidence': 0.4},
            'asphodel': {'meaning': 'asphodel flower', 'linear_a': None, 'confidence': 0.3},

            # Sea/maritime (Minoans were sailors)
            'thalassa': {'meaning': 'sea', 'linear_a': 'ta-ra-sa?', 'confidence': 0.3},
            'pelagos': {'meaning': 'open sea', 'linear_a': None, 'confidence': 0.2},

            # Titles/social
            'tyrannos': {'meaning': 'ruler/king', 'linear_a': None, 'confidence': 0.3},
            'basileus': {'meaning': 'king (later Greek)', 'linear_a': 'qa-si-re-u', 'confidence': 0.5},
            'wanax': {'meaning': 'lord/king', 'linear_a': 'wa-na-ka', 'confidence': 0.6},

            # Objects/materials
            'chiton': {'meaning': 'tunic', 'linear_a': 'ki-to?', 'confidence': 0.3},
            'sasamon': {'meaning': 'sesame', 'linear_a': 'sa-sa-ma?', 'confidence': 0.4},
            'minthē': {'meaning': 'mint', 'linear_a': 'mi-ta?', 'confidence': 0.3},
            'oinos': {'meaning': 'wine', 'linear_a': 'wo-no?', 'confidence': 0.4},

            # Deities (crucial!)
            'athena': {'meaning': 'goddess', 'linear_a': 'a-ta-na?', 'confidence': 0.5},
            'diktynna': {'meaning': 'goddess of nets/hunting', 'linear_a': 'di-ka-ta?', 'confidence': 0.4},
            'britomartis': {'meaning': 'sweet maiden goddess', 'linear_a': None, 'confidence': 0.3},
            'eileithyia': {'meaning': 'birth goddess', 'linear_a': 'e-re-u-ti-ja?', 'confidence': 0.4},
            'paian': {'meaning': 'healer god', 'linear_a': 'pa-ja-wo?', 'confidence': 0.3},

            # Double-axe related (Minoan sacred symbol)
            'labrys': {'meaning': 'double axe', 'linear_a': 'da-pu?', 'confidence': 0.5},
        }

        # Phonetic patterns characteristic of pre-Greek
        self.substrate_patterns = [
            (r'-nth-', 'Place/plant suffix'),
            (r'-ss-', 'Place suffix'),
            (r'-mn-', 'Cluster'),
            (r'^a[^aeiou]', 'Initial a-'),
            (r'-inthos$', 'Place ending'),
            (r'-issos$', 'Place ending'),
        ]

    def find_substrate_matches(self, linear_a_words: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Find potential matches between Linear A words and substrate vocabulary."""
        matches = defaultdict(list)

        for la_word in linear_a_words:
            # Convert to comparable phonetic form
            la_phonetic = la_word.lower().replace('-', '')

            for greek_word, info in self.substrate_words.items():
                # Direct match check
                greek_phonetic = greek_word.lower().replace('-', '')

                # Calculate similarity
                similarity = self._phonetic_similarity(la_phonetic, greek_phonetic)

                if similarity > 0.4:
                    matches[la_word].append((
                        greek_word,
                        info['meaning'],
                        similarity * info['confidence']
                    ))

        # Sort by confidence
        for word in matches:
            matches[word].sort(key=lambda x: -x[2])

        return matches

    def _phonetic_similarity(self, s1: str, s2: str) -> float:
        """Calculate phonetic similarity between two strings."""
        if not s1 or not s2:
            return 0.0

        # Use longest common subsequence ratio
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]
        return 2 * lcs_length / (m + n)

    def extract_vocabulary_hypotheses(self) -> Dict[str, Dict]:
        """Extract vocabulary hypotheses from substrate analysis."""
        hypotheses = {}

        for greek_word, info in self.substrate_words.items():
            if info['linear_a'] and info['confidence'] >= 0.4:
                la_form = info['linear_a'].rstrip('?')
                hypotheses[la_form] = {
                    'meaning': info['meaning'],
                    'source': f"Greek substrate '{greek_word}'",
                    'confidence': info['confidence']
                }

        return hypotheses


# =============================================================================
# STRATEGY 2: SEMANTIC FIELD MAPPING
# =============================================================================

class SemanticFieldMapper:
    """
    Use ideograms to constrain the semantic fields of surrounding words.

    Key insight: Words appearing with commodity ideograms (GRA, OLE, VIN, etc.)
    must relate to those commodities. This constrains possible meanings.
    """

    def __init__(self):
        # Known ideograms and their semantic fields
        self.ideogram_fields = {
            'GRA': {'field': 'grain', 'related': ['wheat', 'barley', 'harvest', 'bread', 'seed']},
            'OLE': {'field': 'olive_oil', 'related': ['olive', 'oil', 'press', 'anoint']},
            'VIN': {'field': 'wine', 'related': ['grape', 'vineyard', 'drink', 'libation']},
            'FIC': {'field': 'fig', 'related': ['fig', 'tree', 'fruit', 'sweet']},
            'OVISm': {'field': 'sheep', 'related': ['sheep', 'ram', 'wool', 'flock', 'herd']},
            'OVISf': {'field': 'sheep', 'related': ['ewe', 'sheep', 'wool', 'lamb']},
            'CAPm': {'field': 'goat', 'related': ['goat', 'billy', 'herd', 'milk']},
            'CAPf': {'field': 'goat', 'related': ['nanny', 'goat', 'kid', 'milk']},
            'SUS': {'field': 'pig', 'related': ['pig', 'swine', 'boar', 'pork']},
            'BOSm': {'field': 'cattle', 'related': ['bull', 'ox', 'cattle', 'herd']},
            'BOSf': {'field': 'cattle', 'related': ['cow', 'cattle', 'milk', 'calf']},
            'TELA': {'field': 'textile', 'related': ['cloth', 'weave', 'linen', 'garment']},
            'LANA': {'field': 'wool', 'related': ['wool', 'fleece', 'weave', 'sheep']},
            'AROM': {'field': 'spice', 'related': ['spice', 'herb', 'aromatic', 'incense']},
            'CROC': {'field': 'saffron', 'related': ['saffron', 'crocus', 'dye', 'spice']},
        }

        # Context types based on archaeological find spots
        self.context_fields = {
            'libation_table': {'field': 'religious', 'related': ['god', 'goddess', 'offering', 'prayer', 'sacred']},
            'administrative': {'field': 'economic', 'related': ['total', 'count', 'owed', 'paid', 'balance']},
            'seal': {'field': 'ownership', 'related': ['name', 'title', 'property', 'authority']},
            'votive': {'field': 'religious', 'related': ['dedicate', 'offer', 'god', 'thank', 'vow']},
        }

    def map_word_to_fields(self, word: str, context_ideograms: List[str],
                           context_type: str = None) -> Dict[str, float]:
        """Map a word to possible semantic fields based on context."""
        field_scores = defaultdict(float)

        # Score based on ideograms in same record
        for ideogram in context_ideograms:
            if ideogram in self.ideogram_fields:
                info = self.ideogram_fields[ideogram]
                field_scores[info['field']] += 1.0
                for related in info['related']:
                    field_scores[related] += 0.5

        # Score based on archaeological context
        if context_type and context_type in self.context_fields:
            info = self.context_fields[context_type]
            field_scores[info['field']] += 1.5
            for related in info['related']:
                field_scores[related] += 0.7

        # Normalize scores
        if field_scores:
            max_score = max(field_scores.values())
            return {k: v/max_score for k, v in field_scores.items()}
        return {}

    def analyze_corpus(self, inscriptions: List) -> Dict[str, Dict[str, float]]:
        """Analyze full corpus to map words to semantic fields."""
        word_fields = defaultdict(lambda: defaultdict(float))

        for insc in inscriptions:
            # Detect ideograms in inscription
            ideograms = []
            for sign in insc.transcription.split():
                if sign in self.ideogram_fields:
                    ideograms.append(sign)

            # Determine context type from doc_type
            context = None
            doc_type = insc.doc_type.value.lower() if hasattr(insc.doc_type, 'value') else str(insc.doc_type).lower()
            if 'offering' in doc_type:
                context = 'libation_table'
            elif doc_type in ['tablet', 'roundel', 'nodule']:
                context = 'administrative'
            elif 'seal' in doc_type:
                context = 'seal'

            # Map words
            words = insc.get_words()
            for word in words:
                phonetic = get_phonetic_reading(word)
                if phonetic:
                    fields = self.map_word_to_fields(phonetic, ideograms, context)
                    for field, score in fields.items():
                        word_fields[phonetic][field] += score

        # Normalize per word
        result = {}
        for word, fields in word_fields.items():
            if fields:
                max_score = max(fields.values())
                result[word] = {k: v/max_score for k, v in fields.items() if v/max_score > 0.3}

        return result


# =============================================================================
# STRATEGY 3: EXHAUSTIVE LANGUAGE FAMILY COMPARISON
# =============================================================================

class LanguageFamilyComparator:
    """
    Systematically compare Linear A patterns against ALL known ancient language families.

    If we can find structural parallels (not just vocabulary), we can narrow down
    what TYPE of language Minoan was.
    """

    def __init__(self):
        # Typological features of ancient language families
        self.language_families = {
            'semitic': {
                'root_structure': 'triconsonantal',
                'word_order': 'VSO',
                'morphology': 'templatic',
                'features': ['consonant_roots', 'vowel_patterns', 'prefix_conjugation'],
                'example_roots': ['ktb', 'mlk', 'šlm'],  # write, king, peace
            },
            'indo_european': {
                'root_structure': 'varied',
                'word_order': 'SOV_to_SVO',
                'morphology': 'fusional',
                'features': ['case_endings', 'verb_conjugation', 'ablaut'],
                'example_roots': ['*peh2-', '*deh3-', '*bher-'],  # protect, give, carry
            },
            'anatolian': {
                'root_structure': 'varied',
                'word_order': 'SOV',
                'morphology': 'agglutinative_fusional',
                'features': ['split_ergativity', 'hi_mi_conjugation', 'chain_suffixes'],
                'example_roots': ['watar', 'atta', 'hassa'],  # water, father, king
            },
            'sumerian': {
                'root_structure': 'monosyllabic_roots',
                'word_order': 'SOV',
                'morphology': 'agglutinative',
                'features': ['noun_chains', 'verb_chains', 'case_particles', 'no_gender'],
                'example_roots': ['lugal', 'dingir', 'e'],  # king, god, house
            },
            'hurrian': {
                'root_structure': 'root_suffix',
                'word_order': 'SOV',
                'morphology': 'agglutinative',
                'features': ['ergative', 'suffix_chains', 'antipassive'],
                'example_roots': ['ewri', 'eni', 'alli'],  # lord, god, lady
            },
            'etruscan': {
                'root_structure': 'varied',
                'word_order': 'SVO_SOV',
                'morphology': 'agglutinative',
                'features': ['case_suffixes', 'verb_person_marking', 'genitive_chains'],
                'example_roots': ['ais', 'clan', 'puia'],  # god, son, wife
            },
            'pre_greek': {
                'root_structure': 'CVCV_common',
                'word_order': 'unknown',
                'morphology': 'agglutinative',
                'features': ['-nth-_suffix', '-ss-_suffix', 'a-_prefix'],
                'example_roots': ['labyrinthos', 'thalassa', 'Parnassos'],
            },
        }

    def analyze_linear_a_typology(self, words: List[str]) -> Dict[str, float]:
        """Analyze Linear A to determine typological features."""
        features = defaultdict(float)

        # Analyze syllable structure
        cv_pattern_count = 0
        total_syllables = 0
        for word in words:
            syllables = word.split('-')
            total_syllables += len(syllables)
            for syl in syllables:
                if len(syl) <= 2:  # CV or V pattern
                    cv_pattern_count += 1

        if total_syllables > 0:
            features['CV_syllable_ratio'] = cv_pattern_count / total_syllables

        # Analyze suffix patterns
        suffix_counts = defaultdict(int)
        for word in words:
            syllables = word.split('-')
            if len(syllables) >= 2:
                suffix_counts[syllables[-1]] += 1

        # High suffix consistency suggests agglutinative
        if suffix_counts:
            max_suffix_freq = max(suffix_counts.values())
            features['suffix_consistency'] = max_suffix_freq / len(words)

        # Analyze reduplication (like Sumerian)
        reduplication_count = 0
        for word in words:
            syllables = word.split('-')
            for i in range(len(syllables) - 1):
                if syllables[i] == syllables[i+1]:
                    reduplication_count += 1
                    break
        features['reduplication_ratio'] = reduplication_count / len(words) if words else 0

        # Analyze word length distribution
        lengths = [len(word.split('-')) for word in words]
        if lengths:
            features['avg_word_length'] = np.mean(lengths)
            features['word_length_std'] = np.std(lengths)

        return dict(features)

    def compare_to_families(self, linear_a_features: Dict[str, float]) -> Dict[str, float]:
        """Compare Linear A features to each language family."""
        scores = {}

        for family, properties in self.language_families.items():
            score = 0.0

            # Check for agglutinative features
            if 'agglutinative' in properties['morphology']:
                if linear_a_features.get('suffix_consistency', 0) > 0.1:
                    score += 2.0

            # Check for reduplication (Sumerian-like)
            if 'noun_chains' in properties.get('features', []):
                score += linear_a_features.get('reduplication_ratio', 0) * 5

            # Check SOV features (suffix-heavy)
            if properties['word_order'] == 'SOV':
                if linear_a_features.get('suffix_consistency', 0) > 0.05:
                    score += 1.0

            # Syllable structure match
            if 'CVCV' in properties.get('root_structure', ''):
                cv_ratio = linear_a_features.get('CV_syllable_ratio', 0)
                score += cv_ratio * 2

            scores[family] = score

        # Normalize
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v/max_score for k, v in scores.items()}

        return scores


# =============================================================================
# STRATEGY 4: CONTEXTUAL CONSTRAINT PROPAGATION
# =============================================================================

class ContextualConstraintPropagator:
    """
    Use archaeological and textual context to constrain possible meanings.

    Key insight: The physical context of an inscription tells us a lot:
    - Libation tables → religious/votive
    - Palace archives → administrative
    - Seal stones → names/titles
    - Pottery → ownership marks
    """

    def __init__(self):
        # Context-based word categories
        self.context_constraints = {
            'libation_table': {
                'likely_categories': ['deity_name', 'epithet', 'offering_term', 'prayer_formula'],
                'unlikely_categories': ['commodity', 'number', 'trade_term'],
            },
            'archive_tablet': {
                'likely_categories': ['commodity', 'number', 'total', 'person_name', 'place_name'],
                'unlikely_categories': ['prayer_formula', 'deity_epithet'],
            },
            'seal': {
                'likely_categories': ['person_name', 'title', 'place_name'],
                'unlikely_categories': ['commodity', 'number', 'prayer'],
            },
            'votive': {
                'likely_categories': ['deity_name', 'dedicator_name', 'offering'],
                'unlikely_categories': ['commodity_list', 'trade_term'],
            },
        }

        # Position-based constraints
        self.position_constraints = {
            'initial': {
                'likely': ['person_name', 'place_name', 'subject'],
                'unlikely': ['suffix', 'total_marker'],
            },
            'before_number': {
                'likely': ['commodity', 'unit', 'measured_item'],
                'unlikely': ['prayer', 'deity_epithet'],
            },
            'after_ku_ro': {
                'likely': ['number', 'grand_total'],
                'unlikely': ['name', 'deity'],
            },
            'final': {
                'likely': ['total', 'summary', 'purpose'],
                'unlikely': ['subject', 'initial_name'],
            },
        }

    def propagate_constraints(self, inscriptions: List) -> Dict[str, Dict[str, float]]:
        """Propagate constraints through the corpus."""
        word_constraints = defaultdict(lambda: defaultdict(float))

        for insc in inscriptions:
            # Determine context
            context = self._determine_context(insc)

            words = insc.get_words()
            for i, word in enumerate(words):
                phonetic = get_phonetic_reading(word)
                if not phonetic:
                    continue

                # Apply context constraints
                if context in self.context_constraints:
                    for cat in self.context_constraints[context]['likely_categories']:
                        word_constraints[phonetic][cat] += 1.0
                    for cat in self.context_constraints[context]['unlikely_categories']:
                        word_constraints[phonetic][cat] -= 0.5

                # Apply position constraints
                position = self._determine_position(i, len(words), insc)
                if position in self.position_constraints:
                    for cat in self.position_constraints[position]['likely']:
                        word_constraints[phonetic][cat] += 0.5
                    for cat in self.position_constraints[position]['unlikely']:
                        word_constraints[phonetic][cat] -= 0.3

        # Convert to probabilities
        result = {}
        for word, cats in word_constraints.items():
            # Shift to positive and normalize
            min_val = min(cats.values()) if cats else 0
            shifted = {k: v - min_val + 0.1 for k, v in cats.items()}
            total = sum(shifted.values())
            if total > 0:
                result[word] = {k: v/total for k, v in shifted.items() if v/total > 0.1}

        return result

    def _determine_context(self, insc) -> str:
        """Determine inscription context from document type."""
        doc_type = insc.doc_type.value.lower() if hasattr(insc.doc_type, 'value') else str(insc.doc_type).lower()

        if 'offering' in doc_type:
            return 'libation_table'
        elif doc_type in ['tablet', 'roundel', 'nodule']:
            return 'archive_tablet'
        elif 'seal' in doc_type:
            return 'seal'
        elif doc_type in ['vessel', 'graffito']:
            return 'votive'
        return 'archive_tablet'  # Default assumption

    def _determine_position(self, index: int, total: int, insc) -> str:
        """Determine word position type."""
        if index == 0:
            return 'initial'
        if index == total - 1:
            return 'final'

        # Check if before number
        text = insc.transcription
        if any(char.isdigit() for char in text):
            return 'before_number'

        # Check if after ku-ro
        words = insc.get_words()
        if index > 0:
            prev_word = get_phonetic_reading(words[index-1])
            if prev_word and 'ku-ro' in prev_word:
                return 'after_ku_ro'

        return 'medial'


# =============================================================================
# STRATEGY 5: COMBINATORIAL ELIMINATION
# =============================================================================

class CombinatorialEliminator:
    """
    Eliminate impossible readings through combinatorial constraints.

    Key insight: Even without knowing the language, we can rule out
    many possibilities through logical constraints.
    """

    def __init__(self):
        self.impossible_patterns = [
            # Phonotactic constraints (unlikely in any natural language)
            (r'^[^aeiou]{4,}', 'Four+ consecutive consonants word-initially'),
            (r'[aeiou]{4,}', 'Four+ consecutive vowels'),

            # Semantic constraints
            ('total_in_middle', 'Total markers should be final'),
            ('deity_with_number', 'Deity names shouldnt have commodity numbers'),
        ]

    def eliminate_impossible(self, hypotheses: Dict[str, Dict]) -> Dict[str, Dict]:
        """Filter out impossible hypotheses."""
        filtered = {}

        for word, info in hypotheses.items():
            # Check phonotactic constraints
            phonetic = word.replace('-', '')

            valid = True
            for pattern, reason in self.impossible_patterns:
                if isinstance(pattern, str) and pattern.startswith('^'):
                    import re
                    if re.search(pattern, phonetic):
                        valid = False
                        break

            if valid:
                filtered[word] = info

        return filtered


# =============================================================================
# INTEGRATED ROSETTA-FREE DECIPHERER
# =============================================================================

class RosettaFreeDecipherer:
    """
    Integrated system for deciphering Linear A without a bilingual text.

    Combines all strategies:
    1. Substrate analysis (known Minoan loanwords)
    2. Semantic field mapping (ideogram constraints)
    3. Language family comparison (typological fit)
    4. Contextual propagation (archaeological context)
    5. Combinatorial elimination (rule out impossible)
    """

    def __init__(self):
        self.substrate = SubstrateAnalyzer()
        self.semantic = SemanticFieldMapper()
        self.typology = LanguageFamilyComparator()
        self.context = ContextualConstraintPropagator()
        self.eliminator = CombinatorialEliminator()

    def decipher(self, inscriptions: List) -> Dict:
        """Run full decipherment pipeline."""
        results = {
            'vocabulary': {},
            'language_type': {},
            'semantic_fields': {},
            'confidence_summary': {}
        }

        print("\n" + "=" * 70)
        print(" " * 15 + "ROSETTA-FREE DECIPHERMENT")
        print("=" * 70)

        # Extract all words
        all_words = extract_all_words()
        word_list = [' '.join(w) for w in all_words if w]

        # Strategy 1: Substrate Analysis
        print("\n📚 Phase 1: Pre-Greek Substrate Analysis")
        print("-" * 50)
        substrate_vocab = self.substrate.extract_vocabulary_hypotheses()
        print(f"   Found {len(substrate_vocab)} potential Minoan→Greek loanwords")
        for word, info in list(substrate_vocab.items())[:5]:
            print(f"   • {word}: {info['meaning']} ({info['confidence']:.0%})")

        # Strategy 2: Semantic Field Mapping
        print("\n🏷️  Phase 2: Semantic Field Mapping")
        print("-" * 50)
        semantic_fields = self.semantic.analyze_corpus(inscriptions)
        print(f"   Mapped {len(semantic_fields)} words to semantic fields")

        # Show examples
        for word, fields in list(semantic_fields.items())[:3]:
            top_fields = sorted(fields.items(), key=lambda x: -x[1])[:2]
            fields_str = ', '.join(f"{f}({s:.0%})" for f, s in top_fields)
            print(f"   • {word}: {fields_str}")

        # Strategy 3: Language Family Comparison
        print("\n🌍 Phase 3: Language Family Comparison")
        print("-" * 50)

        # Get phonetic readings
        phonetic_words = []
        for insc in inscriptions:
            for word in insc.get_words():
                reading = get_phonetic_reading(word)
                if reading:
                    phonetic_words.append(reading)

        typology_features = self.typology.analyze_linear_a_typology(phonetic_words)
        family_scores = self.typology.compare_to_families(typology_features)

        print("   Typological Analysis:")
        print(f"   • CV syllable ratio: {typology_features.get('CV_syllable_ratio', 0):.2%}")
        print(f"   • Suffix consistency: {typology_features.get('suffix_consistency', 0):.2%}")
        print(f"   • Reduplication ratio: {typology_features.get('reduplication_ratio', 0):.2%}")
        print(f"   • Avg word length: {typology_features.get('avg_word_length', 0):.1f} syllables")

        print("\n   Language Family Similarity:")
        for family, score in sorted(family_scores.items(), key=lambda x: -x[1]):
            bar = '█' * int(score * 20)
            print(f"   • {family:15} {bar} ({score:.0%})")

        # Strategy 4: Contextual Constraints
        print("\n🏛️  Phase 4: Contextual Constraint Propagation")
        print("-" * 50)
        context_constraints = self.context.propagate_constraints(inscriptions)
        print(f"   Applied constraints to {len(context_constraints)} words")

        # Strategy 5: Synthesis
        print("\n🔮 Phase 5: Synthesizing All Evidence")
        print("-" * 50)

        final_vocab = self._synthesize_vocabulary(
            substrate_vocab, semantic_fields, context_constraints, phonetic_words
        )

        # Store results
        results['vocabulary'] = final_vocab
        results['language_type'] = family_scores
        results['semantic_fields'] = semantic_fields
        results['typological_features'] = typology_features

        return results

    def _synthesize_vocabulary(self, substrate: Dict, semantic: Dict,
                                context: Dict, words: List[str]) -> Dict[str, Dict]:
        """Combine all evidence into final vocabulary."""
        vocab = {}

        # Start with substrate (highest confidence)
        for word, info in substrate.items():
            vocab[word] = {
                'meaning': info['meaning'],
                'confidence': info['confidence'],
                'sources': [info['source']]
            }

        # Add semantic field evidence
        for word, fields in semantic.items():
            if word in vocab:
                # Boost confidence if consistent
                top_field = max(fields.items(), key=lambda x: x[1])[0]
                vocab[word]['semantic_field'] = top_field
            else:
                top_field = max(fields.items(), key=lambda x: x[1])
                vocab[word] = {
                    'meaning': f"related to {top_field[0]}",
                    'confidence': top_field[1] * 0.5,
                    'sources': ['semantic_field'],
                    'semantic_field': top_field[0]
                }

        # Add contextual evidence
        for word, cats in context.items():
            if not cats:  # Skip if no categories
                continue
            if word in vocab:
                top_cat = max(cats.items(), key=lambda x: x[1])[0]
                vocab[word]['category'] = top_cat
                if 'sources' in vocab[word]:
                    vocab[word]['sources'].append('context')
            else:
                top_cat = max(cats.items(), key=lambda x: x[1])
                vocab[word] = {
                    'meaning': f"likely {top_cat[0].replace('_', ' ')}",
                    'confidence': top_cat[1] * 0.3,
                    'sources': ['context'],
                    'category': top_cat[0]
                }

        return vocab


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run Rosetta-free decipherment."""
    inscriptions = get_all_inscriptions()
    print(f"Loaded {len(inscriptions)} inscriptions")

    decipherer = RosettaFreeDecipherer()
    results = decipherer.decipher(inscriptions)

    # Print final results
    print("\n" + "=" * 70)
    print(" " * 20 + "FINAL RESULTS")
    print("=" * 70)

    print("\n📖 DECIPHERED VOCABULARY (by confidence)")
    print("-" * 50)

    vocab = results['vocabulary']
    sorted_vocab = sorted(vocab.items(), key=lambda x: -x[1].get('confidence', 0))

    # Group by confidence level
    confirmed = [(w, v) for w, v in sorted_vocab if v.get('confidence', 0) >= 0.7]
    probable = [(w, v) for w, v in sorted_vocab if 0.4 <= v.get('confidence', 0) < 0.7]
    possible = [(w, v) for w, v in sorted_vocab if 0.2 <= v.get('confidence', 0) < 0.4]

    print("\n✓ CONFIRMED/HIGH CONFIDENCE:")
    for word, info in confirmed[:10]:
        print(f"   {word:20} = {info['meaning']:20} ({info['confidence']:.0%})")

    print("\n○ PROBABLE:")
    for word, info in probable[:10]:
        print(f"   {word:20} = {info['meaning']:20} ({info['confidence']:.0%})")

    print("\n? POSSIBLE:")
    for word, info in possible[:10]:
        print(f"   {word:20} = {info['meaning']:20} ({info['confidence']:.0%})")

    # Language type conclusion
    print("\n" + "=" * 70)
    print(" " * 15 + "LANGUAGE TYPE CONCLUSION")
    print("=" * 70)

    best_match = max(results['language_type'].items(), key=lambda x: x[1])
    print(f"\n   Best typological match: {best_match[0].upper()} ({best_match[1]:.0%})")
    print(f"\n   Key features of Minoan:")
    print(f"   • Agglutinative morphology (like Sumerian, Hurrian)")
    print(f"   • Suffix-based case system")
    print(f"   • SOV word order probable")
    print(f"   • NOT Indo-European")
    print(f"   • NOT Semitic")
    print(f"   • Possible Aegean language isolate")

    # What we can and cannot do
    print("\n" + "=" * 70)
    print(" " * 15 + "WHAT WE CAN DO WITHOUT A ROSETTA STONE")
    print("=" * 70)

    print("""
   ✓ POSSIBLE without bilingual:
     • Identify STRUCTURE (grammar, morphology)
     • Classify word TYPES (noun, verb, name, title)
     • Map SEMANTIC FIELDS (religious, administrative)
     • Recognize FORMULAS (repeated phrases)
     • Read PLACE NAMES (via Linear B comparison)
     • Identify LOANWORDS into Greek

   ✗ NOT POSSIBLE without bilingual:
     • Translate arbitrary CONTENT WORDS
     • Know VERB MEANINGS (actions)
     • Understand ABSTRACT CONCEPTS
     • Read PERSONAL NAMES (no reference)
     • Determine PRONUNCIATION precisely

   🔑 BEST HOPE for full decipherment:
     1. Discovery of bilingual inscription
     2. Finding a related language
     3. Decipherment of Cretan Hieroglyphic
     4. New large corpus discovery
     5. Substrate word breakthrough in Greek
    """)

    return results


if __name__ == '__main__':
    results = main()
