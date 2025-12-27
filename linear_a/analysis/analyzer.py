"""
Linear A Analysis Engine

Comprehensive statistical and linguistic analysis tools for
attempting to decipher Linear A inscriptions.

Analysis methods:
1. Frequency analysis - Sign distribution patterns
2. N-gram analysis - Common sign sequences
3. Positional analysis - Signs at word/text boundaries
4. Pattern matching - Recurring formulas and phrases
5. Comparative analysis - Comparison with Linear B readings
"""

from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.inscriptions import (
    get_all_inscriptions, extract_all_words, build_sign_frequency,
    Inscription
)
from signs.inventory import (
    SYLLABIC_SIGNS, get_sign_by_phonetic, LinearASign
)


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    name: str
    data: dict
    interpretation: str


class LinearAAnalyzer:
    """Main analysis class for Linear A decipherment."""

    def __init__(self):
        self.inscriptions = get_all_inscriptions()
        self.words = extract_all_words()
        self.sign_freq = build_sign_frequency()
        self._precompute()

    def _precompute(self):
        """Precompute commonly needed data structures."""
        # All signs in corpus
        self.all_signs = []
        for inscription in self.inscriptions:
            self.all_signs.extend(inscription.get_signs())

        # Build bigrams and trigrams
        self.bigrams = Counter()
        self.trigrams = Counter()
        for word in self.words:
            for i in range(len(word) - 1):
                self.bigrams[(word[i], word[i + 1])] += 1
            for i in range(len(word) - 2):
                self.trigrams[(word[i], word[i + 1], word[i + 2])] += 1

        # Word-initial and word-final signs
        self.initial_signs = Counter()
        self.final_signs = Counter()
        for word in self.words:
            if word:
                self.initial_signs[word[0]] += 1
                self.final_signs[word[-1]] += 1

    def frequency_analysis(self) -> AnalysisResult:
        """Analyze sign frequencies and compare to expected distributions."""
        total = len(self.all_signs)

        # Calculate percentages
        freq_pct = {sign: (count / total * 100)
                    for sign, count in self.sign_freq.items()}

        # Zipf's law analysis - natural languages follow power law distribution
        rank_freq = list(enumerate(self.sign_freq.items(), 1))

        # Identify potential vowels (often high frequency)
        high_freq_signs = [sign for sign, pct in freq_pct.items() if pct > 3]

        # Identify rare signs (might be ideograms or special markers)
        low_freq_signs = [sign for sign, count in self.sign_freq.items()
                         if count <= 2]

        interpretation = f"""
Frequency Analysis Results:
- Total signs in corpus: {total}
- Unique signs: {len(self.sign_freq)}

Top 10 Most Frequent Signs:
{self._format_top_signs(10)}

Observations:
- High frequency signs ({len(high_freq_signs)}): Likely common syllables or vowels
- Very rare signs ({len(low_freq_signs)}): May be ideograms, variants, or errors

If Minoan follows typical syllabic language patterns:
- Vowels should be very common (CV syllabaries favor V signs)
- The sign AB08 (hypothetically 'a') is frequent, supporting this
"""
        return AnalysisResult(
            name="Frequency Analysis",
            data={
                "frequencies": self.sign_freq,
                "percentages": freq_pct,
                "high_freq": high_freq_signs,
                "low_freq": low_freq_signs
            },
            interpretation=interpretation
        )

    def _format_top_signs(self, n: int) -> str:
        """Format top n signs with phonetic values."""
        lines = []
        for sign, count in list(self.sign_freq.items())[:n]:
            phonetic = ""
            if sign in SYLLABIC_SIGNS:
                p = SYLLABIC_SIGNS[sign].phonetic
                phonetic = f" [{p}]" if p else ""
            lines.append(f"  {sign}{phonetic}: {count}")
        return "\n".join(lines)

    def bigram_analysis(self) -> AnalysisResult:
        """Analyze two-sign sequences (bigrams)."""
        common_bigrams = self.bigrams.most_common(20)

        # Find sign pairs that never occur
        all_sign_set = set(self.sign_freq.keys())
        possible_pairs = len(all_sign_set) ** 2
        actual_pairs = len(self.bigrams)

        # Look for patterns - which signs tend to follow which
        followers = defaultdict(Counter)
        for (s1, s2), count in self.bigrams.items():
            followers[s1][s2] += count

        interpretation = f"""
Bigram Analysis Results:
- Unique bigrams found: {actual_pairs}
- Possible combinations: {possible_pairs}
- Coverage: {actual_pairs / possible_pairs * 100:.1f}%

Most Common Bigrams:
"""
        for (s1, s2), count in common_bigrams[:10]:
            p1 = SYLLABIC_SIGNS.get(s1, LinearASign(s1, None, None, "?", None)).phonetic or "?"
            p2 = SYLLABIC_SIGNS.get(s2, LinearASign(s2, None, None, "?", None)).phonetic or "?"
            interpretation += f"  {s1}-{s2} [{p1}-{p2}]: {count}\n"

        interpretation += """
Linguistic Implications:
- Common bigrams may represent:
  * Frequent morphemes (prefixes, suffixes, roots)
  * Common word patterns
  * Grammatical markers
"""

        return AnalysisResult(
            name="Bigram Analysis",
            data={
                "bigrams": dict(self.bigrams),
                "followers": dict(followers)
            },
            interpretation=interpretation
        )

    def positional_analysis(self) -> AnalysisResult:
        """Analyze which signs appear at word boundaries."""
        # Compare initial vs final vs medial frequencies
        total_initial = sum(self.initial_signs.values())
        total_final = sum(self.final_signs.values())

        # Signs that strongly prefer initial position
        initial_preference = {}
        for sign in self.sign_freq:
            init_count = self.initial_signs.get(sign, 0)
            final_count = self.final_signs.get(sign, 0)
            total = init_count + final_count
            if total >= 3:  # Minimum threshold
                initial_preference[sign] = init_count / total if total else 0

        strongly_initial = {s: p for s, p in initial_preference.items() if p > 0.7}
        strongly_final = {s: p for s, p in initial_preference.items() if p < 0.3}

        interpretation = f"""
Positional Analysis Results:
- Total word-initial positions: {total_initial}
- Total word-final positions: {total_final}

Signs Preferring Initial Position (>70%):
"""
        for sign, pref in sorted(strongly_initial.items(), key=lambda x: -x[1])[:10]:
            phonetic = SYLLABIC_SIGNS.get(sign, LinearASign(sign, None, None, "?", None)).phonetic or "?"
            interpretation += f"  {sign} [{phonetic}]: {pref*100:.0f}% initial\n"

        interpretation += """
Signs Preferring Final Position (>70%):
"""
        for sign, pref in sorted(strongly_final.items(), key=lambda x: x[1])[:10]:
            phonetic = SYLLABIC_SIGNS.get(sign, LinearASign(sign, None, None, "?", None)).phonetic or "?"
            interpretation += f"  {sign} [{phonetic}]: {(1-pref)*100:.0f}% final\n"

        interpretation += """
Linguistic Implications:
- Initial-preferring signs may represent:
  * Definite articles or determiners
  * Common prefixes
  * First syllables of frequent words
- Final-preferring signs may represent:
  * Case endings
  * Verb suffixes
  * Plural markers
"""

        return AnalysisResult(
            name="Positional Analysis",
            data={
                "initial_signs": dict(self.initial_signs),
                "final_signs": dict(self.final_signs),
                "initial_preference": initial_preference
            },
            interpretation=interpretation
        )

    def recurring_formulas(self) -> AnalysisResult:
        """Identify recurring phrases and formulas."""
        # Build word-level sequences
        word_strings = []
        for inscription in self.inscriptions:
            words = inscription.get_words()
            word_strings.append(["-".join(w) for w in words])

        # Count word occurrences
        word_freq = Counter()
        for ws in word_strings:
            for w in ws:
                word_freq[w] += 1

        # Find repeated multi-word sequences
        phrase_freq = Counter()
        for ws in word_strings:
            for i in range(len(ws) - 1):
                phrase = " | ".join(ws[i:i + 2])
                phrase_freq[phrase] += 1
            for i in range(len(ws) - 2):
                phrase = " | ".join(ws[i:i + 3])
                phrase_freq[phrase] += 1

        common_words = word_freq.most_common(20)
        common_phrases = [(p, c) for p, c in phrase_freq.most_common(20) if c > 1]

        interpretation = """
Recurring Formulas Analysis:

Most Common Words/Terms:
"""
        for word, count in common_words[:15]:
            # Try to provide phonetic reading
            signs = word.split("-")
            phonetics = []
            for s in signs:
                p = SYLLABIC_SIGNS.get(s, LinearASign(s, None, None, "?", None)).phonetic
                phonetics.append(p if p else "?")
            interpretation += f"  {word} [{'-'.join(phonetics)}]: {count}\n"

        interpretation += """
Recurring Phrases:
"""
        for phrase, count in common_phrases[:10]:
            interpretation += f"  {phrase}: {count}\n"

        interpretation += """
Key Recurring Elements:

1. 'ku-ro' (AB77-AB26) - Appears frequently at text ends
   Likely meaning: "total" or "sum" (cf. Linear B)

2. 'ja-sa-sa-ra' (AB57-AB31-AB31-AB60) - Religious formula
   Possibly a divine name or epithet

3. 'i-da-ma-te' (AB28-AB01-AB80-AB04) - Appears in libation texts
   Possibly a deity name (cf. Greek "Ida", "Demeter"?)

4. 'pa-i-to' (AB03-AB28-AB05) - Toponym
   Same as Linear B pa-i-to = Phaistos
"""

        return AnalysisResult(
            name="Recurring Formulas",
            data={
                "word_freq": dict(word_freq),
                "phrase_freq": dict(phrase_freq)
            },
            interpretation=interpretation
        )

    def word_length_analysis(self) -> AnalysisResult:
        """Analyze word length distribution."""
        lengths = [len(w) for w in self.words]
        length_freq = Counter(lengths)

        avg_length = sum(lengths) / len(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0

        interpretation = f"""
Word Length Analysis:
- Average word length: {avg_length:.2f} signs
- Maximum word length: {max_length} signs
- Most common length: {length_freq.most_common(1)[0][0]} signs

Length Distribution:
"""
        for length in range(1, min(max_length + 1, 10)):
            count = length_freq.get(length, 0)
            bar = "█" * (count // 2)
            interpretation += f"  {length} signs: {count:3d} {bar}\n"

        interpretation += f"""
Comparison with Other Syllabaries:
- Linear B average: ~3-4 signs/word
- Japanese (hiragana): ~4-5 signs/word
- Linear A: {avg_length:.1f} signs/word

This suggests Minoan words are {'similar to' if 3 <= avg_length <= 5 else 'different from'} typical syllabic language patterns.
"""

        return AnalysisResult(
            name="Word Length Analysis",
            data={
                "lengths": lengths,
                "distribution": dict(length_freq),
                "average": avg_length
            },
            interpretation=interpretation
        )

    def comparative_analysis(self) -> AnalysisResult:
        """Compare Linear A readings with Linear B and other languages."""
        # Known Linear B words that appear in Linear A
        known_correspondences = {
            "AB03-AB28-AB05": {
                "linear_a": "pa-i-to",
                "linear_b": "pa-i-to",
                "meaning": "Phaistos (city name)",
                "confidence": "high"
            },
            "AB77-AB26": {
                "linear_a": "ku-ro",
                "linear_b": "to-so (different word but same function)",
                "meaning": "total/sum",
                "confidence": "medium"
            },
            "AB01-AB06-AB52": {
                "linear_a": "da-na-no",
                "linear_b": "-",
                "meaning": "possibly ethnic/demonym",
                "confidence": "low"
            },
        }

        # Possible loanwords or cognates
        potential_cognates = [
            ("AB28-AB01-AB80-AB04", "i-da-ma-te", "cf. Greek Ida + Demeter?", "speculative"),
            ("AB57-AB31-AB31-AB60", "ja-sa-sa-ra", "cf. Semitic 'ashera'?", "speculative"),
            ("AB08-AB59-AB08-AB06-AB08", "a-ta-a-na-a", "cf. Athena/Athana?", "speculative"),
        ]

        interpretation = """
Comparative Analysis:

Confirmed Correspondences with Linear B:
"""
        for signs, data in known_correspondences.items():
            interpretation += f"""
  {signs}
    Linear A: {data['linear_a']}
    Linear B: {data['linear_b']}
    Meaning: {data['meaning']}
    Confidence: {data['confidence']}
"""

        interpretation += """
Potential Cognates (Speculative):
"""
        for signs, reading, comparison, confidence in potential_cognates:
            interpretation += f"  {reading}: {comparison} ({confidence})\n"

        interpretation += """
Methodological Notes:
- Place names are our best anchor points (same places, same names)
- Divine names may have been borrowed or influenced later Greek
- Agricultural/trade terms might show regional Aegean vocabulary
- Caution: Similar sounds don't prove related meaning
"""

        return AnalysisResult(
            name="Comparative Analysis",
            data={
                "correspondences": known_correspondences,
                "cognates": potential_cognates
            },
            interpretation=interpretation
        )

    def grammatical_patterns(self) -> AnalysisResult:
        """Attempt to identify grammatical patterns."""
        # Look for recurring endings
        endings = Counter()
        for word in self.words:
            if len(word) >= 2:
                endings[word[-1]] += 1
                endings[tuple(word[-2:])] += 1

        # Look for recurring beginnings (possible prefixes/articles)
        beginnings = Counter()
        for word in self.words:
            if len(word) >= 2:
                beginnings[word[0]] += 1
                beginnings[tuple(word[:2])] += 1

        # Look for alternations that might indicate inflection
        # Words that share roots but differ in endings
        word_families = defaultdict(list)
        for word in self.words:
            if len(word) >= 3:
                root = tuple(word[:2])  # First two signs as "root"
                word_families[root].append(word)

        # Filter to families with variations
        inflected = {root: variants for root, variants in word_families.items()
                    if len(set(tuple(v) for v in variants)) > 1}

        interpretation = """
Grammatical Pattern Analysis:

Potential Suffixes (Common Word Endings):
"""
        # Single-sign endings
        for sign, count in endings.most_common(10):
            if isinstance(sign, str):
                phonetic = SYLLABIC_SIGNS.get(sign, LinearASign(sign, None, None, "?", None)).phonetic or "?"
                interpretation += f"  -{sign} [-{phonetic}]: {count} occurrences\n"

        interpretation += """
Potential Prefixes (Common Word Beginnings):
"""
        for sign, count in beginnings.most_common(10):
            if isinstance(sign, str):
                phonetic = SYLLABIC_SIGNS.get(sign, LinearASign(sign, None, None, "?", None)).phonetic or "?"
                interpretation += f"  {sign}- [{phonetic}-]: {count} occurrences\n"

        interpretation += f"""
Possible Word Families (Shared Roots):
Found {len(inflected)} potential root groups with variants.

"""
        for root, variants in list(inflected.items())[:5]:
            root_reading = "-".join(
                SYLLABIC_SIGNS.get(s, LinearASign(s, None, None, "?", None)).phonetic or "?"
                for s in root
            )
            interpretation += f"  Root '{root_reading}':\n"
            for v in variants[:3]:
                v_reading = "-".join(
                    SYLLABIC_SIGNS.get(s, LinearASign(s, None, None, "?", None)).phonetic or "?"
                    for s in v
                )
                interpretation += f"    {v_reading}\n"

        interpretation += """
Hypothetical Grammar Features:
Based on patterns, Minoan may have:
- Suffixing morphology (endings vary more than beginnings)
- Possible case system (different endings for same roots)
- Possible number marking (singular/plural)
"""

        return AnalysisResult(
            name="Grammatical Patterns",
            data={
                "endings": dict(endings),
                "beginnings": dict(beginnings),
                "word_families": {str(k): v for k, v in inflected.items()}
            },
            interpretation=interpretation
        )

    def run_full_analysis(self) -> str:
        """Run all analyses and compile results."""
        output = []
        output.append("=" * 70)
        output.append("LINEAR A DECIPHERMENT ANALYSIS")
        output.append("=" * 70)
        output.append("")

        analyses = [
            self.frequency_analysis(),
            self.bigram_analysis(),
            self.positional_analysis(),
            self.word_length_analysis(),
            self.recurring_formulas(),
            self.comparative_analysis(),
            self.grammatical_patterns(),
        ]

        for analysis in analyses:
            output.append("-" * 70)
            output.append(f"[{analysis.name}]")
            output.append("-" * 70)
            output.append(analysis.interpretation)
            output.append("")

        return "\n".join(output)


def main():
    """Run the analysis."""
    print("Initializing Linear A Analyzer...")
    analyzer = LinearAAnalyzer()
    print("Running full analysis...\n")
    print(analyzer.run_full_analysis())


if __name__ == "__main__":
    main()
