"""
Linear A Sign Inventory

This module contains the known Linear A signs with their:
- Standard sign numbers (AB/LAxx notation)
- Unicode code points (U+10600-U+1077F)
- Hypothetical phonetic values (based on Linear B cognates)
- Sign categories (syllabic, ideographic, numeric)

The phonetic values are HYPOTHETICAL - derived from Linear B cognates
where the signs appear similar. The actual pronunciation in Minoan
may have differed.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum


class SignCategory(Enum):
    SYLLABIC = "syllabic"      # Represents a syllable (CV or V)
    IDEOGRAM = "ideogram"      # Represents a word/concept
    NUMERIC = "numeric"        # Numeric values
    FRACTION = "fraction"      # Fractional values
    LIGATURE = "ligature"      # Combined signs
    UNKNOWN = "unknown"        # Unclassified


@dataclass
class LinearASign:
    """Represents a single Linear A sign."""
    sign_number: str           # Standard notation (e.g., "AB01", "LA301")
    unicode: Optional[str]     # Unicode character if assigned
    unicode_hex: Optional[str] # Unicode code point (e.g., "10600")
    phonetic: Optional[str]    # Hypothetical phonetic value from Linear B
    category: SignCategory
    description: Optional[str] = None
    frequency: int = 0         # Corpus frequency (to be computed)
    linear_b_cognate: Optional[str] = None  # Corresponding Linear B sign

    def __str__(self):
        if self.phonetic:
            return f"{self.sign_number} ({self.phonetic})"
        return self.sign_number


# Core syllabic signs with Linear B-derived phonetic values
# These are the most common and best-understood signs
SYLLABIC_SIGNS: Dict[str, LinearASign] = {
    # Vowels
    "AB08": LinearASign("AB08", "𐘀", "10600", "a", SignCategory.SYLLABIC,
                        linear_b_cognate="*08"),
    "AB28": LinearASign("AB28", "𐘆", "10606", "i", SignCategory.SYLLABIC,
                        linear_b_cognate="*28"),
    "AB61": LinearASign("AB61", "𐘉", "10609", "o", SignCategory.SYLLABIC,
                        linear_b_cognate="*61"),
    "AB10": LinearASign("AB10", "𐘂", "10602", "u", SignCategory.SYLLABIC,
                        linear_b_cognate="*10"),
    "AB38": LinearASign("AB38", "𐘈", "10608", "e", SignCategory.SYLLABIC,
                        linear_b_cognate="*38"),

    # D-series
    "AB01": LinearASign("AB01", "𐘁", "10601", "da", SignCategory.SYLLABIC,
                        linear_b_cognate="*01"),
    "AB07": LinearASign("AB07", "𐘃", "10603", "di", SignCategory.SYLLABIC,
                        linear_b_cognate="*07"),
    "AB14": LinearASign("AB14", "𐘄", "10604", "do", SignCategory.SYLLABIC,
                        linear_b_cognate="*14"),
    "AB51": LinearASign("AB51", "𐘅", "10605", "du", SignCategory.SYLLABIC,
                        linear_b_cognate="*51"),
    "AB45": LinearASign("AB45", "𐘇", "10607", "de", SignCategory.SYLLABIC,
                        linear_b_cognate="*45"),

    # J-series
    "AB57": LinearASign("AB57", "𐘊", "1060A", "ja", SignCategory.SYLLABIC,
                        linear_b_cognate="*57"),
    "AB65": LinearASign("AB65", "𐘋", "1060B", "ju", SignCategory.SYLLABIC,
                        linear_b_cognate="*65"),
    "AB46": LinearASign("AB46", "𐘌", "1060C", "je", SignCategory.SYLLABIC,
                        linear_b_cognate="*46"),

    # K-series
    "AB77": LinearASign("AB77", "𐘍", "1060D", "ka", SignCategory.SYLLABIC,
                        linear_b_cognate="*77"),
    "AB67": LinearASign("AB67", "𐘎", "1060E", "ki", SignCategory.SYLLABIC,
                        linear_b_cognate="*67"),
    "AB70": LinearASign("AB70", "𐘏", "1060F", "ko", SignCategory.SYLLABIC,
                        linear_b_cognate="*70"),
    "AB81": LinearASign("AB81", "𐘐", "10610", "ku", SignCategory.SYLLABIC,
                        linear_b_cognate="*81"),

    # M-series
    "AB80": LinearASign("AB80", "𐘤", "10624", "ma", SignCategory.SYLLABIC,
                        linear_b_cognate="*80"),
    "AB73": LinearASign("AB73", "𐘥", "10625", "mi", SignCategory.SYLLABIC,
                        linear_b_cognate="*73"),
    "AB15": LinearASign("AB15", "𐘦", "10626", "mo", SignCategory.SYLLABIC,
                        linear_b_cognate="*15"),
    "AB23": LinearASign("AB23", "𐘧", "10627", "mu", SignCategory.SYLLABIC,
                        linear_b_cognate="*23"),
    "AB13": LinearASign("AB13", "𐘨", "10628", "me", SignCategory.SYLLABIC,
                        linear_b_cognate="*13"),

    # N-series
    "AB06": LinearASign("AB06", "𐘝", "1061D", "na", SignCategory.SYLLABIC,
                        linear_b_cognate="*06"),
    "AB30": LinearASign("AB30", "𐘞", "1061E", "ni", SignCategory.SYLLABIC,
                        linear_b_cognate="*30"),
    "AB52": LinearASign("AB52", "𐘟", "1061F", "no", SignCategory.SYLLABIC,
                        linear_b_cognate="*52"),
    "AB55": LinearASign("AB55", "𐘠", "10620", "nu", SignCategory.SYLLABIC,
                        linear_b_cognate="*55"),
    "AB24": LinearASign("AB24", "𐘡", "10621", "ne", SignCategory.SYLLABIC,
                        linear_b_cognate="*24"),

    # P-series
    "AB03": LinearASign("AB03", "𐘒", "10612", "pa", SignCategory.SYLLABIC,
                        linear_b_cognate="*03"),
    "AB39": LinearASign("AB39", "𐘓", "10613", "pi", SignCategory.SYLLABIC,
                        linear_b_cognate="*39"),
    "AB11": LinearASign("AB11", "𐘔", "10614", "po", SignCategory.SYLLABIC,
                        linear_b_cognate="*11"),
    "AB50": LinearASign("AB50", "𐘕", "10615", "pu", SignCategory.SYLLABIC,
                        linear_b_cognate="*50"),
    "AB72": LinearASign("AB72", "𐘖", "10616", "pe", SignCategory.SYLLABIC,
                        linear_b_cognate="*72"),

    # Q-series
    "AB16": LinearASign("AB16", "𐘗", "10617", "qa", SignCategory.SYLLABIC,
                        linear_b_cognate="*16"),
    "AB21": LinearASign("AB21", "𐘘", "10618", "qi", SignCategory.SYLLABIC,
                        linear_b_cognate="*21"),
    "AB32": LinearASign("AB32", "𐘙", "10619", "qo", SignCategory.SYLLABIC,
                        linear_b_cognate="*32"),

    # R-series
    "AB60": LinearASign("AB60", "𐘚", "1061A", "ra", SignCategory.SYLLABIC,
                        linear_b_cognate="*60"),
    "AB53": LinearASign("AB53", "𐘛", "1061B", "ri", SignCategory.SYLLABIC,
                        linear_b_cognate="*53"),
    "AB02": LinearASign("AB02", "𐘜", "1061C", "ro", SignCategory.SYLLABIC,
                        linear_b_cognate="*02"),
    "AB26": LinearASign("AB26", "𐘢", "10622", "ru", SignCategory.SYLLABIC,
                        linear_b_cognate="*26"),
    "AB27": LinearASign("AB27", "𐘣", "10623", "re", SignCategory.SYLLABIC,
                        linear_b_cognate="*27"),

    # S-series
    "AB31": LinearASign("AB31", "𐘩", "10629", "sa", SignCategory.SYLLABIC,
                        linear_b_cognate="*31"),
    "AB41": LinearASign("AB41", "𐘪", "1062A", "si", SignCategory.SYLLABIC,
                        linear_b_cognate="*41"),
    "AB12": LinearASign("AB12", "𐘫", "1062B", "so", SignCategory.SYLLABIC,
                        linear_b_cognate="*12"),
    "AB58": LinearASign("AB58", "𐘬", "1062C", "su", SignCategory.SYLLABIC,
                        linear_b_cognate="*58"),
    "AB09": LinearASign("AB09", "𐘭", "1062D", "se", SignCategory.SYLLABIC,
                        linear_b_cognate="*09"),

    # T-series
    "AB59": LinearASign("AB59", "𐘮", "1062E", "ta", SignCategory.SYLLABIC,
                        linear_b_cognate="*59"),
    "AB37": LinearASign("AB37", "𐘯", "1062F", "ti", SignCategory.SYLLABIC,
                        linear_b_cognate="*37"),
    "AB05": LinearASign("AB05", "𐘰", "10630", "to", SignCategory.SYLLABIC,
                        linear_b_cognate="*05"),
    "AB69": LinearASign("AB69", "𐘱", "10631", "tu", SignCategory.SYLLABIC,
                        linear_b_cognate="*69"),
    "AB04": LinearASign("AB04", "𐘲", "10632", "te", SignCategory.SYLLABIC,
                        linear_b_cognate="*04"),

    # W-series
    "AB54": LinearASign("AB54", "𐘳", "10633", "wa", SignCategory.SYLLABIC,
                        linear_b_cognate="*54"),
    "AB40": LinearASign("AB40", "𐘴", "10634", "wi", SignCategory.SYLLABIC,
                        linear_b_cognate="*40"),
    "AB75": LinearASign("AB75", "𐘵", "10635", "wo", SignCategory.SYLLABIC,
                        linear_b_cognate="*75"),
    "AB91": LinearASign("AB91", "𐘶", "10636", "we", SignCategory.SYLLABIC,
                        linear_b_cognate="*91"),

    # Z-series
    "AB17": LinearASign("AB17", "𐘷", "10637", "za", SignCategory.SYLLABIC,
                        linear_b_cognate="*17"),
    "AB74": LinearASign("AB74", "𐘸", "10638", "ze", SignCategory.SYLLABIC,
                        linear_b_cognate="*74"),
    "AB20": LinearASign("AB20", "𐘹", "10639", "zo", SignCategory.SYLLABIC,
                        linear_b_cognate="*20"),

    # Additional syllabic signs with less certain values
    "AB22": LinearASign("AB22", None, None, "?", SignCategory.SYLLABIC,
                        description="Uncertain value"),
    "AB25": LinearASign("AB25", None, None, "a2", SignCategory.SYLLABIC,
                        linear_b_cognate="*25", description="Variant of 'a'"),
    "AB29": LinearASign("AB29", None, None, "pu2", SignCategory.SYLLABIC,
                        linear_b_cognate="*29"),
    "AB34": LinearASign("AB34", None, None, "?", SignCategory.SYLLABIC),
    "AB44": LinearASign("AB44", None, None, "ke", SignCategory.SYLLABIC,
                        linear_b_cognate="*44"),
    "AB47": LinearASign("AB47", None, None, "?", SignCategory.SYLLABIC),
    "AB49": LinearASign("AB49", None, None, "?", SignCategory.SYLLABIC),
    "AB56": LinearASign("AB56", None, None, "pa3", SignCategory.SYLLABIC,
                        linear_b_cognate="*56"),
    "AB63": LinearASign("AB63", None, None, "?", SignCategory.SYLLABIC),
    "AB66": LinearASign("AB66", None, None, "ta2", SignCategory.SYLLABIC,
                        linear_b_cognate="*66"),
    "AB68": LinearASign("AB68", None, None, "ro2", SignCategory.SYLLABIC,
                        linear_b_cognate="*68"),
    "AB76": LinearASign("AB76", None, None, "ra2", SignCategory.SYLLABIC,
                        linear_b_cognate="*76"),
    "AB78": LinearASign("AB78", None, None, "qe", SignCategory.SYLLABIC,
                        linear_b_cognate="*78"),
    "AB79": LinearASign("AB79", None, None, "?", SignCategory.SYLLABIC),
    "AB82": LinearASign("AB82", None, None, "?", SignCategory.SYLLABIC),
    "AB83": LinearASign("AB83", None, None, "?", SignCategory.SYLLABIC),
    "AB85": LinearASign("AB85", None, None, "au", SignCategory.SYLLABIC,
                        linear_b_cognate="*85"),
    "AB86": LinearASign("AB86", None, None, "?", SignCategory.SYLLABIC),
    "AB87": LinearASign("AB87", None, None, "twe", SignCategory.SYLLABIC,
                        linear_b_cognate="*87"),
}


# Common ideograms (logograms) - represent whole words
IDEOGRAMS: Dict[str, LinearASign] = {
    # Agricultural products
    "AB100": LinearASign("AB100", None, None, None, SignCategory.IDEOGRAM,
                         description="VIR (man)"),
    "AB102": LinearASign("AB102", None, None, None, SignCategory.IDEOGRAM,
                         description="MUL (woman)"),
    "AB120": LinearASign("AB120", None, None, None, SignCategory.IDEOGRAM,
                         description="GRA (grain/wheat)"),
    "AB121": LinearASign("AB121", None, None, None, SignCategory.IDEOGRAM,
                         description="HORD (barley)"),
    "AB122": LinearASign("AB122", None, None, None, SignCategory.IDEOGRAM,
                         description="OLIV (olives)"),
    "AB123": LinearASign("AB123", None, None, None, SignCategory.IDEOGRAM,
                         description="AROM (aromatics/spices)"),
    "AB130": LinearASign("AB130", None, None, None, SignCategory.IDEOGRAM,
                         description="OLE (oil)"),
    "AB131": LinearASign("AB131", None, None, None, SignCategory.IDEOGRAM,
                         description="VIN (wine)"),

    # Animals
    "AB104": LinearASign("AB104", None, None, None, SignCategory.IDEOGRAM,
                         description="SUS (pig)"),
    "AB105": LinearASign("AB105", None, None, None, SignCategory.IDEOGRAM,
                         description="OVIS (sheep)"),
    "AB106": LinearASign("AB106", None, None, None, SignCategory.IDEOGRAM,
                         description="CAP (goat)"),
    "AB107": LinearASign("AB107", None, None, None, SignCategory.IDEOGRAM,
                         description="BOS (cattle)"),

    # Vessels and containers
    "AB200": LinearASign("AB200", None, None, None, SignCategory.IDEOGRAM,
                         description="VAS (vessel)"),
    "AB201": LinearASign("AB201", None, None, None, SignCategory.IDEOGRAM,
                         description="Vessel type"),

    # Textiles
    "AB145": LinearASign("AB145", None, None, None, SignCategory.IDEOGRAM,
                         description="TELA (cloth)"),
    "AB146": LinearASign("AB146", None, None, None, SignCategory.IDEOGRAM,
                         description="LANA (wool)"),
}


# Numeric signs
NUMERICS: Dict[str, LinearASign] = {
    "NUM1": LinearASign("NUM1", "|", None, "1", SignCategory.NUMERIC,
                        description="Unit stroke"),
    "NUM10": LinearASign("NUM10", "●", None, "10", SignCategory.NUMERIC,
                         description="Dot for 10"),
    "NUM100": LinearASign("NUM100", "◯", None, "100", SignCategory.NUMERIC,
                          description="Circle for 100"),
    "NUM1000": LinearASign("NUM1000", "◎", None, "1000", SignCategory.NUMERIC,
                           description="Circle with center dot for 1000"),
}


# Fractions
FRACTIONS: Dict[str, LinearASign] = {
    "J": LinearASign("J", None, None, "1/2", SignCategory.FRACTION),
    "E": LinearASign("E", None, None, "1/4", SignCategory.FRACTION),
    "F": LinearASign("F", None, None, "1/8", SignCategory.FRACTION),
    "K": LinearASign("K", None, None, "1/16", SignCategory.FRACTION),
}


def get_all_signs() -> Dict[str, LinearASign]:
    """Return all known Linear A signs."""
    all_signs = {}
    all_signs.update(SYLLABIC_SIGNS)
    all_signs.update(IDEOGRAMS)
    all_signs.update(NUMERICS)
    all_signs.update(FRACTIONS)
    return all_signs


def get_sign_by_phonetic(phonetic: str) -> Optional[LinearASign]:
    """Look up a sign by its phonetic value."""
    for sign in SYLLABIC_SIGNS.values():
        if sign.phonetic == phonetic:
            return sign
    return None


def get_syllabary_grid() -> str:
    """Return a formatted syllabary grid similar to Linear B."""
    consonants = ['', 'd', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'w', 'z']
    vowels = ['a', 'e', 'i', 'o', 'u']

    # Build lookup
    grid_signs = {}
    for sign in SYLLABIC_SIGNS.values():
        if sign.phonetic and sign.phonetic != "?":
            grid_signs[sign.phonetic] = sign

    # Format grid
    lines = []
    header = "     " + "   ".join(f"{v:>3}" for v in vowels)
    lines.append(header)
    lines.append("-" * len(header))

    for c in consonants:
        row = f"{c:>3}  " if c else "     "
        cells = []
        for v in vowels:
            syllable = c + v if c else v
            if syllable in grid_signs:
                sign = grid_signs[syllable]
                cells.append(f"{sign.sign_number:>5}")
            else:
                cells.append("    -")
        lines.append(row + " ".join(cells))

    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Linear A Sign Inventory ===\n")
    print(f"Total syllabic signs: {len(SYLLABIC_SIGNS)}")
    print(f"Total ideograms: {len(IDEOGRAMS)}")
    print(f"Total signs: {len(get_all_signs())}\n")

    print("=== Syllabary Grid (Sign Numbers) ===\n")
    print(get_syllabary_grid())

    print("\n\n=== Signs with Known Phonetic Values ===\n")
    for sign_id, sign in sorted(SYLLABIC_SIGNS.items()):
        if sign.phonetic and sign.phonetic != "?":
            print(f"  {sign_id}: {sign.phonetic}")
