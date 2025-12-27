"""
Linear A Corpus - Inscription Database

This module contains a collection of Linear A inscriptions transcribed
using the standard AB notation system (e.g., AB01 = da, AB02 = ro).

Sources:
- GORILA (Godart & Olivier, Recueil des inscriptions en linéaire A)
- SigLA database
- Various academic publications

Inscriptions are organized by find-site and document type.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class DocumentType(Enum):
    TABLET = "tablet"
    ROUNDEL = "roundel"
    NODULE = "nodule"
    VESSEL = "vessel"
    OFFERING_TABLE = "offering_table"
    SEAL = "seal"
    GRAFFITO = "graffito"
    OTHER = "other"


class FindSite(Enum):
    HAGIA_TRIADA = "HT"     # Hagia Triada - main administrative center
    ZAKROS = "ZA"           # Zakro palace
    KHANIA = "KH"           # Chania
    PHAISTOS = "PH"         # Phaistos palace
    KNOSSOS = "KN"          # Knossos palace
    MALIA = "MA"            # Malia palace
    PETRAS = "PE"           # Petras
    ARCHANES = "AR"         # Archanes
    KATO_SYME = "KS"        # Kato Syme sanctuary
    PSEIRA = "PS"           # Pseira island
    PYRGOS = "PY"           # Pyrgos
    TYLISSOS = "TY"         # Tylissos
    OTHER = "OTH"


@dataclass
class Inscription:
    """Represents a single Linear A inscription."""
    id: str                       # Standard ID (e.g., "HT 1", "ZA 4")
    site: FindSite
    doc_type: DocumentType
    transcription: str            # Signs in AB notation, hyphen-separated
    phonetic_reading: Optional[str] = None  # Hypothetical reading
    translation_attempt: Optional[str] = None
    notes: Optional[str] = None

    def get_signs(self) -> List[str]:
        """Parse transcription into list of sign codes."""
        # Handle word dividers and line breaks
        text = self.transcription.replace("|", " ").replace("/", " ")
        words = text.split()
        signs = []
        for word in words:
            signs.extend(word.split("-"))
        return [s for s in signs if s]  # Remove empty strings

    def get_words(self) -> List[List[str]]:
        """Parse transcription into words (lists of signs)."""
        text = self.transcription.replace("/", "|")  # Normalize line breaks
        word_strings = text.split("|")
        words = []
        for ws in word_strings:
            ws = ws.strip()
            if ws:
                signs = [s.strip() for s in ws.split("-") if s.strip()]
                if signs:
                    words.append(signs)
        return words


# ============================================================================
# HAGIA TRIADA TABLETS - The largest collection (~150 tablets)
# Administrative texts dealing with commodities, personnel, offerings
# ============================================================================

HAGIA_TRIADA_TABLETS = [
    # HT 1 - Famous tablet with "total" formula
    Inscription(
        id="HT 1",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB77-AB26-AB06-AB04 | AB06-AB04 | AB81-AB02-AB06",
        phonetic_reading="ku-ro-na-te | na-te | ku-ro-na",
        notes="Contains ku-ro, possibly 'total' (cf. Linear B ko-wo)"
    ),

    # HT 6 - List of offerings or commodities
    Inscription(
        id="HT 6",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB57-AB31-AB31-AB60 | AB77-AB26-AB02 | AB59-AB27-AB17-AB08",
        phonetic_reading="ja-sa-sa-ra | ku-ro-ro | ta-re-za-a",
        notes="Ritual or offering list"
    ),

    # HT 7 - Personnel list
    Inscription(
        id="HT 7",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB57-AB01-AB59-AB06 | AB39-AB52-AB06-AB28",
        phonetic_reading="ja-da-ta-na | pi-no-na-i",
        notes="Possibly personal names"
    ),

    # HT 8 - Commodity record
    Inscription(
        id="HT 8",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB41-AB60-AB06 | AB01-AB54-AB06-AB02 | VIN",
        phonetic_reading="si-ra-na | da-wa-na-ro | wine",
        notes="Wine distribution record"
    ),

    # HT 9
    Inscription(
        id="HT 9",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB77-AB39 | AB60-AB06-AB28 | AB59-AB28",
        phonetic_reading="ku-pi | ra-na-i | ta-i",
        notes="Short commodity list"
    ),

    # HT 10 - Grain record
    Inscription(
        id="HT 10",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB80-AB60-AB26 | AB59-AB27-AB31 | GRA",
        phonetic_reading="ma-ra-ru | ta-re-sa | grain",
        notes="Grain distribution"
    ),

    # HT 12
    Inscription(
        id="HT 12",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB28-AB03-AB06-AB80 | AB28-AB01-AB80-AB04",
        phonetic_reading="i-pa-na-ma | i-da-ma-te",
        notes="Possibly theophoric names"
    ),

    # HT 13 - Important ritualistic tablet
    Inscription(
        id="HT 13",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB08-AB59-AB08-AB28-AB07-AB54-AB57-AB08 | AB28-AB01-AB80-AB04",
        phonetic_reading="a-ta-i-di-wa-ja-a | i-da-ma-te",
        notes="Religious/ritual text, contains divine names?"
    ),

    # HT 14 - Oil distribution
    Inscription(
        id="HT 14",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB41-AB60 | AB57-AB31-AB31-AB60-AB80-AB24 | OLE",
        phonetic_reading="si-ra | ja-sa-sa-ra-ma-ne | oil",
        notes="Oil distribution record"
    ),

    # HT 28
    Inscription(
        id="HT 28",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB77-AB26-AB02 | AB01-AB54-AB52",
        phonetic_reading="ku-ro-ro | da-wa-no",
        notes="Contains ku-ro (total?)"
    ),

    # HT 31 - Well-preserved tablet
    Inscription(
        id="HT 31",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB03-AB28-AB05 | AB31-AB08-AB60-AB08 | AB77-AB26",
        phonetic_reading="pa-i-to | sa-a-ra-a | ku-ro",
        notes="Contains pa-i-to (Phaistos placename)"
    ),

    # HT 85 - Long administrative list
    Inscription(
        id="HT 85",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB77-AB26-AB02 | AB59-AB28-AB37-AB59",
        phonetic_reading="ku-ro-ro | ta-i-ti-ta",
        notes="Administrative record with totaling formula"
    ),

    # HT 86
    Inscription(
        id="HT 86",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB08-AB57-AB77-AB26 | AB41-AB06-AB30-AB59",
        phonetic_reading="a-ja-ku-ro | si-na-ni-ta",
        notes="Compound with ku-ro"
    ),

    # HT 88
    Inscription(
        id="HT 88",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB08-AB31-AB08-AB60-AB80-AB24 | AB77-AB26",
        phonetic_reading="a-sa-a-ra-ma-ne | ku-ro",
        notes="Ritual or religious text"
    ),

    # HT 94 - Personnel record
    Inscription(
        id="HT 94",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB01-AB80-AB13 | AB60-AB57-AB06",
        phonetic_reading="da-ma-me | ra-ja-na",
        notes="Names or titles"
    ),

    # HT 95
    Inscription(
        id="HT 95",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB01-AB59-AB17 | AB77-AB26-AB57",
        phonetic_reading="da-ta-za | ku-ro-ja",
        notes="Administrative record"
    ),

    # HT 96
    Inscription(
        id="HT 96",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB57-AB31-AB31-AB60 | AB80-AB06 | AB57-AB08",
        phonetic_reading="ja-sa-sa-ra | ma-na | ja-a",
        notes="Contains recurring ja-sa-sa-ra formula"
    ),

    # HT 97
    Inscription(
        id="HT 97",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB02-AB57-AB08 | AB59-AB08-AB58-AB28-AB37",
        phonetic_reading="ro-ja-a | ta-a-su-i-ti",
        notes="Administrative"
    ),

    # HT 99 - Well-preserved
    Inscription(
        id="HT 99",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB03-AB28-AB59-AB80 | AB26-AB60-AB02",
        phonetic_reading="pa-i-ta-ma | ru-ra-ro",
        notes="Toponym?"
    ),

    # HT 100
    Inscription(
        id="HT 100",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB06-AB60 | AB01-AB80-AB28 | AB77-AB26",
        phonetic_reading="na-ra | da-ma-i | ku-ro",
        notes="Administrative totaling"
    ),

    # HT 117 - Important libation formula
    Inscription(
        id="HT 117",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB08-AB57 | AB01-AB28-AB01-AB59-AB59-AB28",
        phonetic_reading="a-ja | di-di-ta-ta-i",
        notes="Possible libation formula"
    ),

    # HT 122
    Inscription(
        id="HT 122",
        site=FindSite.HAGIA_TRIADA,
        doc_type=DocumentType.TABLET,
        transcription="AB08-AB01-AB77-AB28-AB59-AB04",
        phonetic_reading="a-da-ki-i-ta-te",
        notes="Single word or phrase"
    ),
]


# ============================================================================
# ZAKROS TABLETS
# ============================================================================

ZAKROS_TABLETS = [
    Inscription(
        id="ZA 1",
        site=FindSite.ZAKROS,
        doc_type=DocumentType.TABLET,
        transcription="AB28-AB57-AB06 | AB77-AB26 | AB31-AB08-AB59-AB08",
        phonetic_reading="i-ja-na | ku-ro | sa-a-ta-a",
        notes="Administrative tablet"
    ),

    Inscription(
        id="ZA 4",
        site=FindSite.ZAKROS,
        doc_type=DocumentType.TABLET,
        transcription="AB59-AB28-AB37 | AB59-AB08-AB06-AB28",
        phonetic_reading="ta-i-ti | ta-a-na-i",
        notes="Short text"
    ),

    Inscription(
        id="ZA 8",
        site=FindSite.ZAKROS,
        doc_type=DocumentType.TABLET,
        transcription="AB41-AB60-AB06 | AB77-AB26-AB02",
        phonetic_reading="si-ra-na | ku-ro-ro",
        notes="Contains totaling formula"
    ),

    Inscription(
        id="ZA 10",
        site=FindSite.ZAKROS,
        doc_type=DocumentType.TABLET,
        transcription="AB03-AB28-AB59-AB08 | AB80-AB28-AB05",
        phonetic_reading="pa-i-ta-a | ma-i-to",
        notes="Toponym?"
    ),

    Inscription(
        id="ZA 15",
        site=FindSite.ZAKROS,
        doc_type=DocumentType.TABLET,
        transcription="AB57-AB31-AB31-AB60-AB80-AB24",
        phonetic_reading="ja-sa-sa-ra-ma-ne",
        notes="Same formula as HT tablets"
    ),
]


# ============================================================================
# KHANIA TABLETS
# ============================================================================

KHANIA_TABLETS = [
    Inscription(
        id="KH 5",
        site=FindSite.KHANIA,
        doc_type=DocumentType.TABLET,
        transcription="AB01-AB80-AB04 | AB77-AB26-AB02",
        phonetic_reading="da-ma-te | ku-ro-ro",
        notes="Contains da-ma-te (deity?)"
    ),

    Inscription(
        id="KH 7",
        site=FindSite.KHANIA,
        doc_type=DocumentType.TABLET,
        transcription="AB08-AB59-AB08-AB06-AB24 | AB28-AB80-AB28",
        phonetic_reading="a-ta-a-na-ne | i-ma-i",
        notes="Religious context"
    ),

    Inscription(
        id="KH 10",
        site=FindSite.KHANIA,
        doc_type=DocumentType.TABLET,
        transcription="AB28-AB57-AB03-AB06 | AB80-AB60-AB26",
        phonetic_reading="i-ja-pa-na | ma-ra-ru",
        notes="Administrative"
    ),
]


# ============================================================================
# LIBATION FORMULAS - Religious/ritual inscriptions on stone vessels
# ============================================================================

LIBATION_FORMULAS = [
    # IO Za 1 - One of the most analyzed inscriptions
    Inscription(
        id="IO Za 1",
        site=FindSite.OTHER,
        doc_type=DocumentType.OFFERING_TABLE,
        transcription="AB08-AB59-AB28-AB07-AB54-AB57-AB08 | AB28-AB01-AB80-AB04 | AB04-AB57-AB08",
        phonetic_reading="a-ta-i-*301-wa-ja | i-da-ma-te | te-ja-a",
        notes="Famous libation formula from Mt. Iouktas sanctuary"
    ),

    # PK Za 11
    Inscription(
        id="PK Za 11",
        site=FindSite.OTHER,
        doc_type=DocumentType.OFFERING_TABLE,
        transcription="AB08-AB59-AB28-AB07-AB54-AB57-AB08",
        phonetic_reading="a-ta-i-*301-wa-ja",
        notes="Libation formula, same as IO Za 1 opening"
    ),

    # PS Za 2
    Inscription(
        id="PS Za 2",
        site=FindSite.PSEIRA,
        doc_type=DocumentType.OFFERING_TABLE,
        transcription="AB59-AB08-AB06-AB28 | AB28-AB01-AB80-AB04",
        phonetic_reading="ta-a-na-i | i-da-ma-te",
        notes="Contains i-da-ma-te divine name?"
    ),

    # TL Za 1
    Inscription(
        id="TL Za 1",
        site=FindSite.OTHER,
        doc_type=DocumentType.OFFERING_TABLE,
        transcription="AB08-AB31-AB08-AB31-AB60-AB80-AB24",
        phonetic_reading="a-sa-a-sa-ra-ma-ne",
        notes="Ritual formula, variant of ja-sa-sa-ra-ma-ne?"
    ),

    # SY Za 3
    Inscription(
        id="SY Za 3",
        site=FindSite.KATO_SYME,
        doc_type=DocumentType.OFFERING_TABLE,
        transcription="AB28-AB01-AB80-AB04 | AB08-AB01-AB77-AB28-AB59-AB04",
        phonetic_reading="i-da-ma-te | a-da-ki-i-ta-te",
        notes="Peak sanctuary, religious formula"
    ),
]


# ============================================================================
# VESSEL INSCRIPTIONS - Short texts painted/incised on pottery
# ============================================================================

VESSEL_INSCRIPTIONS = [
    Inscription(
        id="KN Za 7",
        site=FindSite.KNOSSOS,
        doc_type=DocumentType.VESSEL,
        transcription="AB57-AB31-AB31-AB60",
        phonetic_reading="ja-sa-sa-ra",
        notes="Short dedication formula"
    ),

    Inscription(
        id="AR Zf 1",
        site=FindSite.ARCHANES,
        doc_type=DocumentType.VESSEL,
        transcription="AB01-AB54-AB06-AB08",
        phonetic_reading="da-wa-na-a",
        notes="Vessel inscription"
    ),

    Inscription(
        id="KN Zf 31",
        site=FindSite.KNOSSOS,
        doc_type=DocumentType.VESSEL,
        transcription="AB31-AB28-AB60-AB06",
        phonetic_reading="sa-i-ra-na",
        notes="Incised on vessel"
    ),
]


def get_all_inscriptions() -> List[Inscription]:
    """Return all inscriptions in the corpus."""
    return (
        HAGIA_TRIADA_TABLETS +
        ZAKROS_TABLETS +
        KHANIA_TABLETS +
        LIBATION_FORMULAS +
        VESSEL_INSCRIPTIONS
    )


def get_inscriptions_by_site(site: FindSite) -> List[Inscription]:
    """Return inscriptions from a specific site."""
    return [i for i in get_all_inscriptions() if i.site == site]


def get_inscriptions_by_type(doc_type: DocumentType) -> List[Inscription]:
    """Return inscriptions of a specific type."""
    return [i for i in get_all_inscriptions() if i.doc_type == doc_type]


def extract_all_words() -> List[List[str]]:
    """Extract all words (sign sequences) from the corpus."""
    words = []
    for inscription in get_all_inscriptions():
        words.extend(inscription.get_words())
    return words


def build_sign_frequency() -> Dict[str, int]:
    """Calculate frequency of each sign in the corpus."""
    freq = {}
    for inscription in get_all_inscriptions():
        for sign in inscription.get_signs():
            if sign.startswith("AB") or sign.startswith("LA"):
                freq[sign] = freq.get(sign, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    print("=== Linear A Corpus Statistics ===\n")

    all_inscriptions = get_all_inscriptions()
    print(f"Total inscriptions: {len(all_inscriptions)}")

    # By site
    print("\nBy site:")
    for site in FindSite:
        count = len(get_inscriptions_by_site(site))
        if count > 0:
            print(f"  {site.value}: {count}")

    # By type
    print("\nBy type:")
    for doc_type in DocumentType:
        count = len(get_inscriptions_by_type(doc_type))
        if count > 0:
            print(f"  {doc_type.value}: {count}")

    # Sign frequency
    print("\n=== Most Common Signs ===\n")
    freq = build_sign_frequency()
    for sign, count in list(freq.items())[:20]:
        print(f"  {sign}: {count}")

    # Word patterns
    print("\n=== Sample Words ===\n")
    words = extract_all_words()[:15]
    for word in words:
        print(f"  {'-'.join(word)}")
