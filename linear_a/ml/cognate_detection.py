"""
Cognate Detection for Linear A

Compare Linear A vocabulary with ancient Mediterranean languages
to find potential cognates or loanwords.

Candidate languages:
1. Greek (Mycenaean and Classical)
2. Semitic (Akkadian, Ugaritic, Hebrew)
3. Anatolian (Luwian, Hittite)
4. Etruscan (also undeciphered, but some vocabulary known)
5. Egyptian
6. Pre-Greek substrate words in Greek

Methods:
1. Phonetic distance metrics
2. Sequence alignment (like DNA alignment)
3. Sound correspondence patterns
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.inscriptions import get_all_inscriptions, extract_all_words
from signs.inventory import SYLLABIC_SIGNS


# ============================================================================
# COMPARATIVE VOCABULARY DATABASES
# ============================================================================

# Words from ancient languages that might have Minoan cognates
# Format: (word, language, meaning, phonetic_form)

GREEK_VOCABULARY = [
    # Pre-Greek substrate words (possibly Minoan loans)
    ("ἀσάμινθος", "Greek", "bathtub", "a-sa-min-thos"),
    ("λαβύρινθος", "Greek", "labyrinth", "la-bu-rin-thos"),
    ("θάλασσα", "Greek", "sea", "tha-las-sa"),
    ("κυπάρισσος", "Greek", "cypress", "ku-pa-ris-sos"),
    ("νάρκισσος", "Greek", "narcissus", "nar-kis-sos"),
    ("ὑάκινθος", "Greek", "hyacinth", "hu-a-kin-thos"),
    ("τέρμινθος", "Greek", "terebinth", "ter-min-thos"),
    ("πλίνθος", "Greek", "brick", "plin-thos"),
    ("σέλινον", "Greek", "celery", "se-li-non"),
    ("μίνθη", "Greek", "mint", "min-the"),
    ("κισσός", "Greek", "ivy", "kis-sos"),
    ("ἐλαία", "Greek", "olive", "e-lai-a"),
    ("δάφνη", "Greek", "laurel", "daph-ne"),
    ("κρόκος", "Greek", "saffron", "kro-kos"),

    # Mycenaean Linear B words
    ("pa-i-to", "Linear B", "Phaistos", "pa-i-to"),
    ("ko-no-so", "Linear B", "Knossos", "ko-no-so"),
    ("a-mi-ni-so", "Linear B", "Amnisos", "a-mi-ni-so"),
    ("da-pu-ri-to", "Linear B", "labyrinth", "da-pu-ri-to"),
    ("po-ti-ni-ja", "Linear B", "lady/mistress", "po-ti-ni-ja"),
    ("wa-na-ka", "Linear B", "king", "wa-na-ka"),
    ("da-mo", "Linear B", "people/district", "da-mo"),
    ("te-o", "Linear B", "god", "te-o"),
    ("i-je-ro", "Linear B", "sacred", "i-je-ro"),

    # Divine names
    ("Δημήτηρ", "Greek", "Demeter", "de-me-ter"),
    ("Ἀθήνη", "Greek", "Athena", "a-the-ne"),
    ("Ἥρα", "Greek", "Hera", "he-ra"),
    ("Δίκτυννα", "Greek", "Diktynna", "dik-tun-na"),
    ("Βριτόμαρτις", "Greek", "Britomartis", "bri-to-mar-tis"),
]

SEMITIC_VOCABULARY = [
    # Potential Semitic cognates (controversial)
    ("אשרה", "Hebrew", "Asherah (goddess)", "a-she-ra"),
    ("כתרת", "Ugaritic", "Kothar (craftsman god)", "ko-thar"),
    ("מלך", "Hebrew", "king", "me-lek"),
    ("שמן", "Hebrew", "oil", "she-men"),
    ("יין", "Hebrew", "wine", "ya-yin"),
    ("זית", "Hebrew", "olive", "za-yit"),
    ("חטה", "Hebrew", "wheat", "hit-ta"),
]

ANATOLIAN_VOCABULARY = [
    # Luwian words (Anatolian, related to Hittite)
    ("tarḫunt", "Luwian", "storm god", "tar-hunt"),
    ("wanax", "Luwian", "king", "wa-na"),
    ("atti", "Hittite", "father", "at-ti"),
    ("anna", "Hittite", "mother", "an-na"),
    ("ḫaššu", "Hittite", "king", "has-su"),
]

ETRUSCAN_VOCABULARY = [
    # Etruscan words (language isolate, some known vocabulary)
    ("ati", "Etruscan", "mother", "a-ti"),
    ("clan", "Etruscan", "son", "clan"),
    ("sec", "Etruscan", "daughter", "sec"),
    ("tur", "Etruscan", "give", "tur"),
    ("mul", "Etruscan", "dedicate", "mul"),
    ("θu", "Etruscan", "one", "thu"),
    ("zal", "Etruscan", "two", "zal"),
]


class PhoneticDistance:
    """Compute phonetic distance between syllable sequences."""

    # Phonetic feature vectors for sounds
    # Format: (manner, place, voice, vowel_height, vowel_front)
    FEATURES = {
        # Vowels
        'a': (0, 2, 1, 0, 2),  # open, central
        'e': (0, 2, 1, 1, 3),  # mid, front
        'i': (0, 2, 1, 2, 3),  # close, front
        'o': (0, 2, 1, 1, 1),  # mid, back
        'u': (0, 2, 1, 2, 1),  # close, back

        # Stops
        'p': (1, 0, 0, 0, 0),  # bilabial, voiceless
        'b': (1, 0, 1, 0, 0),  # bilabial, voiced
        't': (1, 1, 0, 0, 0),  # alveolar, voiceless
        'd': (1, 1, 1, 0, 0),  # alveolar, voiced
        'k': (1, 2, 0, 0, 0),  # velar, voiceless
        'g': (1, 2, 1, 0, 0),  # velar, voiced
        'q': (1, 3, 0, 0, 0),  # labiovelar

        # Nasals
        'm': (2, 0, 1, 0, 0),
        'n': (2, 1, 1, 0, 0),

        # Fricatives
        's': (3, 1, 0, 0, 0),
        'z': (3, 1, 1, 0, 0),
        'h': (3, 4, 0, 0, 0),
        'th': (3, 1, 0, 0, 0),

        # Liquids/Glides
        'r': (4, 1, 1, 0, 0),
        'l': (4, 1, 1, 0, 0),
        'w': (5, 0, 1, 0, 0),
        'j': (5, 3, 1, 0, 0),
    }

    @classmethod
    def sound_distance(cls, s1: str, s2: str) -> float:
        """Compute distance between two sounds."""
        f1 = cls.FEATURES.get(s1.lower(), (0, 0, 0, 0, 0))
        f2 = cls.FEATURES.get(s2.lower(), (0, 0, 0, 0, 0))

        # Euclidean distance in feature space
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(f1, f2)))

    @classmethod
    def syllable_to_sounds(cls, syllable: str) -> List[str]:
        """Convert syllable to list of sounds."""
        syllable = syllable.lower()
        sounds = []

        i = 0
        while i < len(syllable):
            # Check for digraphs
            if i + 1 < len(syllable) and syllable[i:i + 2] in ['th', 'ph', 'kh']:
                sounds.append(syllable[i:i + 2])
                i += 2
            else:
                sounds.append(syllable[i])
                i += 1

        return sounds


class SequenceAligner:
    """
    Align two phonetic sequences using dynamic programming.
    Similar to DNA sequence alignment (Needleman-Wunsch).
    """

    def __init__(self, match_score: float = 2.0,
                 mismatch_penalty: float = -1.0,
                 gap_penalty: float = -0.5):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    def align(self, seq1: List[str], seq2: List[str]) -> Tuple[float, List, List]:
        """
        Align two sequences and return score and alignments.
        """
        m, n = len(seq1), len(seq2)

        # Initialize score matrix
        score = np.zeros((m + 1, n + 1))
        for i in range(m + 1):
            score[i, 0] = i * self.gap_penalty
        for j in range(n + 1):
            score[0, j] = j * self.gap_penalty

        # Fill matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Match/mismatch
                s1, s2 = seq1[i - 1], seq2[j - 1]
                if s1.lower() == s2.lower():
                    match = score[i - 1, j - 1] + self.match_score
                else:
                    # Use phonetic distance
                    dist = PhoneticDistance.sound_distance(s1, s2)
                    match = score[i - 1, j - 1] + self.match_score - dist

                gap1 = score[i - 1, j] + self.gap_penalty
                gap2 = score[i, j - 1] + self.gap_penalty

                score[i, j] = max(match, gap1, gap2)

        # Traceback
        align1, align2 = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                s1, s2 = seq1[i - 1], seq2[j - 1]
                if s1.lower() == s2.lower():
                    expected = score[i - 1, j - 1] + self.match_score
                else:
                    dist = PhoneticDistance.sound_distance(s1, s2)
                    expected = score[i - 1, j - 1] + self.match_score - dist

                if abs(score[i, j] - expected) < 0.01:
                    align1.append(seq1[i - 1])
                    align2.append(seq2[j - 1])
                    i -= 1
                    j -= 1
                    continue

            if i > 0 and abs(score[i, j] - (score[i - 1, j] + self.gap_penalty)) < 0.01:
                align1.append(seq1[i - 1])
                align2.append("-")
                i -= 1
            else:
                align1.append("-")
                align2.append(seq2[j - 1])
                j -= 1

        return score[m, n], list(reversed(align1)), list(reversed(align2))


class CognateDetector:
    """
    Detect potential cognates between Linear A and other languages.
    """

    def __init__(self):
        self.aligner = SequenceAligner()
        self.linear_a_words = self._build_linear_a_vocabulary()

    def _build_linear_a_vocabulary(self) -> List[Dict]:
        """Build vocabulary from Linear A corpus with phonetic readings."""
        words = extract_all_words()
        vocab = []

        for word in words:
            if len(word) < 2:
                continue

            # Build phonetic reading
            phonetics = []
            valid = True
            for sign in word:
                if sign in SYLLABIC_SIGNS:
                    p = SYLLABIC_SIGNS[sign].phonetic
                    if p and p != "?":
                        phonetics.append(p)
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break

            if valid and phonetics:
                vocab.append({
                    "signs": word,
                    "phonetic": "-".join(phonetics),
                    "sounds": self._phonetic_to_sounds("-".join(phonetics))
                })

        return vocab

    def _phonetic_to_sounds(self, phonetic: str) -> List[str]:
        """Convert phonetic string to list of sounds."""
        sounds = []
        for syllable in phonetic.split("-"):
            sounds.extend(PhoneticDistance.syllable_to_sounds(syllable))
        return sounds

    def find_cognates(self, comparison_vocab: List[Tuple],
                      threshold: float = 0.5) -> List[Dict]:
        """
        Find potential cognates between Linear A and another language.

        Returns matches above threshold, sorted by similarity.
        """
        results = []

        for linear_a_word in self.linear_a_words:
            la_sounds = linear_a_word["sounds"]

            for word, language, meaning, phonetic in comparison_vocab:
                other_sounds = self._phonetic_to_sounds(phonetic)

                # Align sequences
                score, align1, align2 = self.aligner.align(la_sounds, other_sounds)

                # Normalize score by length
                max_len = max(len(la_sounds), len(other_sounds))
                if max_len > 0:
                    normalized_score = score / max_len
                else:
                    normalized_score = 0

                if normalized_score >= threshold:
                    results.append({
                        "linear_a": linear_a_word["phonetic"],
                        "linear_a_signs": "-".join(linear_a_word["signs"]),
                        "other_word": word,
                        "other_phonetic": phonetic,
                        "language": language,
                        "meaning": meaning,
                        "score": normalized_score,
                        "alignment": (align1, align2)
                    })

        # Sort by score
        results.sort(key=lambda x: -x["score"])
        return results

    def analyze_sound_correspondences(self, cognates: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze systematic sound correspondences from cognate pairs.

        If there's a genetic relationship, we expect regular sound changes.
        """
        correspondences = defaultdict(Counter)

        for cognate in cognates:
            align1, align2 = cognate["alignment"]
            for s1, s2 in zip(align1, align2):
                if s1 != "-" and s2 != "-":
                    correspondences[s1][s2] += 1

        # Find regular correspondences
        regular = {}
        for sound, counter in correspondences.items():
            if sum(counter.values()) >= 2:  # At least 2 examples
                most_common = counter.most_common(1)[0]
                if most_common[1] >= 2:  # At least 2 of same correspondence
                    regular[sound] = {
                        "corresponds_to": most_common[0],
                        "count": most_common[1],
                        "total": sum(counter.values()),
                        "regularity": most_common[1] / sum(counter.values())
                    }

        return regular


def run_cognate_analysis():
    """Run comprehensive cognate detection analysis."""
    print("\n" + "=" * 70)
    print("COGNATE DETECTION ANALYSIS")
    print("=" * 70)

    detector = CognateDetector()
    print(f"\nLinear A vocabulary: {len(detector.linear_a_words)} words")

    # Compare with each language
    languages = [
        ("Greek (including substrate)", GREEK_VOCABULARY),
        ("Semitic", SEMITIC_VOCABULARY),
        ("Anatolian", ANATOLIAN_VOCABULARY),
        ("Etruscan", ETRUSCAN_VOCABULARY),
    ]

    all_cognates = []

    for lang_name, vocab in languages:
        print(f"\n--- Comparing with {lang_name} ---")
        cognates = detector.find_cognates(vocab, threshold=0.3)

        if cognates:
            print(f"\nTop potential cognates (score >= 0.3):")
            for cog in cognates[:8]:
                print(f"  Linear A: {cog['linear_a']}")
                print(f"    ~ {cog['other_phonetic']} ({cog['language']})")
                print(f"    Meaning: {cog['meaning']}")
                print(f"    Score: {cog['score']:.3f}")
                print()
            all_cognates.extend(cognates)
        else:
            print("  No significant matches found.")

    # Sound correspondence analysis
    if all_cognates:
        print("\n--- Sound Correspondence Analysis ---")
        correspondences = detector.analyze_sound_correspondences(all_cognates)

        if correspondences:
            print("\nPotential regular sound correspondences:")
            for sound, data in sorted(correspondences.items(),
                                       key=lambda x: -x[1]["regularity"]):
                if data["regularity"] >= 0.5:
                    print(f"  Linear A '{sound}' ~ '{data['corresponds_to']}': "
                          f"{data['count']}/{data['total']} ({data['regularity']:.0%})")

    # Key findings
    print("\n--- Key Findings ---")
    print("""
1. PLACE NAMES (Most Reliable)
   - Linear A pa-i-to = Greek Phaistos (confirmed)
   - Suggests phonetic values are largely correct

2. RELIGIOUS VOCABULARY
   - Linear A ja-sa-sa-ra ~ Semitic Asherah?
   - Linear A i-da-ma-te ~ Greek Ida + mother?
   - Caution: May be coincidence or later borrowing

3. PRE-GREEK SUBSTRATE
   - Many Greek words with non-IE etymology
   - -nth-, -ss- suffixes may be Minoan
   - Examples: labyrinth, thalassa, hyacinth

4. METHODOLOGICAL CAUTION
   - Small corpus limits statistical power
   - Surface similarities may be coincidental
   - Need systematic correspondence to prove relationship
   - One cognate pair proves nothing; patterns matter
""")


if __name__ == "__main__":
    run_cognate_analysis()
