#!/usr/bin/env python3
"""
Linear A to English Translation System

This module attempts translation using multiple strategies:

1. SEMANTIC ANCHORING
   - Use ideograms (GRA, VIN, OLE) as meaning anchors
   - Use numbers to infer quantity expressions
   - Use confirmed words (pa-i-to = Phaistos)

2. TEMPLATE MATCHING
   - Administrative texts follow predictable patterns
   - "Word + Ideogram + Number" = "X units of commodity"
   - "ku-ro + Number" = "Total: X"

3. TRANSFER FROM LINEAR B
   - Linear B is deciphered Mycenaean Greek
   - Similar administrative vocabulary
   - Map Linear A patterns to Linear B meanings

4. SEMANTIC ROLE PREDICTION
   - Use position to predict grammatical role
   - Initial position = subject/agent
   - Pre-numeric position = commodity
   - Final position = grammatical marker

5. NEURAL CONSTRAINED GENERATION
   - Generate English that satisfies semantic constraints
   - Use learned patterns to fill slots

This is EXPERIMENTAL - translations are hypothetical!
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.inscriptions import (
    get_all_inscriptions, extract_all_words, Inscription,
    DocumentType, LIBATION_FORMULAS
)
from signs.inventory import SYLLABIC_SIGNS


# ============================================================================
# KNOWN VOCABULARY (Confirmed or High-Confidence)
# ============================================================================

class Confidence(Enum):
    CONFIRMED = "confirmed"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SPECULATIVE = "speculative"


@dataclass
class LexiconEntry:
    """Entry in our Linear A lexicon."""
    linear_a: str           # Sign sequence (hyphenated)
    english: str            # English translation
    part_of_speech: str     # noun, verb, adj, name, etc.
    confidence: Confidence
    notes: str = ""


# Our working lexicon based on analysis
LEXICON: Dict[str, LexiconEntry] = {
    # Confirmed
    "AB03-AB28-AB05": LexiconEntry(
        "pa-i-to", "Phaistos", "proper_noun", Confidence.CONFIRMED,
        "Place name, confirmed via Linear B"
    ),

    # High confidence (administrative terms)
    "AB77-AB26": LexiconEntry(
        "ku-ro", "total", "noun", Confidence.HIGH,
        "Appears at end of lists before numbers"
    ),
    "AB77-AB26-AB02": LexiconEntry(
        "ku-ro-ro", "totals/grand total", "noun", Confidence.MEDIUM,
        "Extended form of ku-ro"
    ),

    # Medium confidence (religious vocabulary)
    "AB57-AB31-AB31-AB60": LexiconEntry(
        "ja-sa-sa-ra", "Lady/Goddess", "noun", Confidence.MEDIUM,
        "Divine name/epithet in religious texts"
    ),
    "AB57-AB31-AB31-AB60-AB80-AB24": LexiconEntry(
        "ja-sa-sa-ra-ma-ne", "to the Lady", "phrase", Confidence.MEDIUM,
        "Dative/dedicatory form"
    ),
    "AB28-AB01-AB80-AB04": LexiconEntry(
        "i-da-ma-te", "Mother of Ida", "proper_noun", Confidence.MEDIUM,
        "Deity name, appears in libation formulas"
    ),
    "AB08-AB59-AB28-AB07-AB54-AB57-AB08": LexiconEntry(
        "a-ta-i-*301-wa-ja", "to the...", "phrase", Confidence.LOW,
        "Libation formula opening"
    ),

    # Low confidence (inferred from patterns)
    "AB01-AB80-AB04": LexiconEntry(
        "da-ma-te", "lady/mother", "noun", Confidence.LOW,
        "Possibly related to i-da-ma-te"
    ),
    "AB01-AB80": LexiconEntry(
        "da-ma", "house?/give?", "noun", Confidence.SPECULATIVE,
        "Common root, meaning unclear"
    ),
    "AB06-AB04": LexiconEntry(
        "na-te", "for/to", "particle", Confidence.SPECULATIVE,
        "Appears to mark recipients"
    ),
}


# Ideogram meanings (these are more certain)
IDEOGRAM_MEANINGS = {
    "VIN": ("wine", Confidence.HIGH),
    "OLE": ("oil/olive oil", Confidence.HIGH),
    "GRA": ("grain/wheat", Confidence.HIGH),
    "HORD": ("barley", Confidence.HIGH),
    "OLIV": ("olives", Confidence.HIGH),
    "AROM": ("aromatics/spices", Confidence.MEDIUM),
    "AB120": ("grain", Confidence.HIGH),
    "AB130": ("oil", Confidence.HIGH),
    "AB131": ("wine", Confidence.HIGH),
    "AB100": ("man/person", Confidence.MEDIUM),
    "AB102": ("woman", Confidence.MEDIUM),
    "AB104": ("pig", Confidence.MEDIUM),
    "AB105": ("sheep", Confidence.MEDIUM),
    "AB106": ("goat", Confidence.MEDIUM),
    "AB107": ("cattle/ox", Confidence.MEDIUM),
}


# Grammatical endings (hypothetical)
ENDINGS = {
    "AB08": ("-a", "nominative?"),
    "AB04": ("-te", "dative/recipient?"),
    "AB06": ("-na", "locative?"),
    "AB28": ("-i", "genitive?"),
    "AB02": ("-ro", "ablative/from?"),
    "AB24": ("-ne", "benefactive/for?"),
}


# ============================================================================
# TRANSLATION TEMPLATES
# ============================================================================

class TranslationTemplate:
    """Template for pattern-based translation."""

    def __init__(self, pattern: str, template: str, confidence: Confidence):
        self.pattern = pattern  # Regex-like pattern
        self.template = template  # English template with slots
        self.confidence = confidence


TEMPLATES = [
    # Administrative templates
    TranslationTemplate(
        pattern="WORD+ IDEOGRAM NUMBER",
        template="{number} units of {commodity} (for {word})",
        confidence=Confidence.MEDIUM
    ),
    TranslationTemplate(
        pattern="ku-ro NUMBER",
        template="Total: {number}",
        confidence=Confidence.HIGH
    ),
    TranslationTemplate(
        pattern="WORD+ ku-ro NUMBER",
        template="{word}: total {number}",
        confidence=Confidence.MEDIUM
    ),

    # Religious templates
    TranslationTemplate(
        pattern="a-ta-i-* i-da-ma-te",
        template="To the Mother of Ida (libation)",
        confidence=Confidence.MEDIUM
    ),
    TranslationTemplate(
        pattern="ja-sa-sa-ra-ma-ne",
        template="To the Lady/Goddess",
        confidence=Confidence.MEDIUM
    ),
]


# ============================================================================
# LINEAR B TRANSFER KNOWLEDGE
# ============================================================================

# Linear B patterns we can transfer
LINEAR_B_PATTERNS = {
    # Administrative patterns
    "commodity_list": {
        "pattern": "NAME COMMODITY QUANTITY",
        "meaning": "Person X receives/provides Y amount of commodity",
    },
    "total_formula": {
        "pattern": "to-so QUANTITY",  # Linear B for "total"
        "linear_a_equiv": "ku-ro",
        "meaning": "Total: X",
    },
    "offering_formula": {
        "pattern": "DEITY OFFERING",
        "meaning": "Offering to deity",
    },
}


# ============================================================================
# TRANSLATOR CLASS
# ============================================================================

class LinearATranslator:
    """
    Multi-strategy translator for Linear A → English.
    """

    def __init__(self):
        self.lexicon = LEXICON
        self.ideograms = IDEOGRAM_MEANINGS
        self.templates = TEMPLATES

    def translate_inscription(self, inscription: Inscription) -> Dict:
        """
        Translate a complete inscription using multiple strategies.
        """
        result = {
            "original": inscription.transcription,
            "id": inscription.id,
            "type": inscription.doc_type.value,
            "translations": [],
            "confidence": Confidence.SPECULATIVE,
            "analysis": {}
        }

        # Get words and signs
        words = inscription.get_words()
        signs = inscription.get_signs()

        # Strategy 1: Direct lexicon lookup
        lexicon_translation = self._translate_by_lexicon(words)
        if lexicon_translation:
            result["translations"].append({
                "method": "lexicon",
                "text": lexicon_translation["text"],
                "confidence": lexicon_translation["confidence"]
            })

        # Strategy 2: Template matching
        template_translation = self._translate_by_template(inscription)
        if template_translation:
            result["translations"].append({
                "method": "template",
                "text": template_translation["text"],
                "confidence": template_translation["confidence"]
            })

        # Strategy 3: Semantic role analysis
        role_translation = self._translate_by_roles(words, signs)
        if role_translation:
            result["translations"].append({
                "method": "semantic_roles",
                "text": role_translation["text"],
                "confidence": role_translation["confidence"]
            })

        # Strategy 4: Document type inference
        type_translation = self._translate_by_type(inscription)
        if type_translation:
            result["translations"].append({
                "method": "document_type",
                "text": type_translation["text"],
                "confidence": type_translation["confidence"]
            })

        # Combine translations into best guess
        result["best_translation"] = self._combine_translations(result["translations"])
        result["confidence"] = self._overall_confidence(result["translations"])

        return result

    def _translate_by_lexicon(self, words: List[List[str]]) -> Optional[Dict]:
        """Translate using direct lexicon lookup."""
        translations = []
        overall_confidence = Confidence.SPECULATIVE

        for word in words:
            word_key = "-".join(word)

            if word_key in self.lexicon:
                entry = self.lexicon[word_key]
                translations.append(entry.english)
                if entry.confidence.value < overall_confidence.value:
                    overall_confidence = entry.confidence
            else:
                # Try to translate individual signs
                sign_translations = []
                for sign in word:
                    if sign in self.ideograms:
                        meaning, conf = self.ideograms[sign]
                        sign_translations.append(meaning)
                    elif sign in SYLLABIC_SIGNS:
                        p = SYLLABIC_SIGNS[sign].phonetic
                        sign_translations.append(f"[{p}]" if p else f"[{sign}]")
                    else:
                        sign_translations.append(f"[{sign}]")
                translations.append("-".join(sign_translations))

        if translations:
            return {
                "text": " | ".join(translations),
                "confidence": overall_confidence
            }
        return None

    def _translate_by_template(self, inscription: Inscription) -> Optional[Dict]:
        """Match inscription against known templates."""
        text = inscription.transcription

        # Check for ku-ro (total) pattern
        if "AB77-AB26" in text or "ku-ro" in text.lower():
            # Extract any numbers
            return {
                "text": "Total: [number] (administrative record)",
                "confidence": Confidence.HIGH
            }

        # Check for religious formulas
        if "AB57-AB31-AB31-AB60" in text:  # ja-sa-sa-ra
            if "AB80-AB24" in text:  # -ma-ne
                return {
                    "text": "Dedication to the Lady/Goddess",
                    "confidence": Confidence.MEDIUM
                }
            return {
                "text": "Invocation of the Lady/Goddess",
                "confidence": Confidence.MEDIUM
            }

        if "AB28-AB01-AB80-AB04" in text:  # i-da-ma-te
            return {
                "text": "Offering/prayer to the Mother of Ida",
                "confidence": Confidence.MEDIUM
            }

        # Check for commodity ideograms
        for ideogram, (meaning, conf) in self.ideograms.items():
            if ideogram in text:
                return {
                    "text": f"Record concerning {meaning}",
                    "confidence": Confidence.MEDIUM
                }

        return None

    def _translate_by_roles(self, words: List[List[str]],
                           signs: List[str]) -> Optional[Dict]:
        """Translate by inferring semantic roles from position."""
        if not words:
            return None

        roles = []

        for i, word in enumerate(words):
            word_key = "-".join(word)
            phonetic = "-".join(
                SYLLABIC_SIGNS.get(s, type('', (), {'phonetic': '?'})()).phonetic or '?'
                for s in word
            )

            # Infer role based on position
            if i == 0:
                role = "SUBJECT/AGENT"
            elif i == len(words) - 1:
                # Check if it's ku-ro
                if word_key == "AB77-AB26":
                    role = "TOTAL"
                else:
                    role = "PREDICATE/END"
            else:
                # Check for ideograms
                if any(s in self.ideograms for s in word):
                    role = "COMMODITY"
                else:
                    role = "MODIFIER/OBJECT"

            roles.append(f"{role}:{phonetic}")

        return {
            "text": " ".join(roles),
            "confidence": Confidence.LOW
        }

    def _translate_by_type(self, inscription: Inscription) -> Optional[Dict]:
        """Translate based on document type."""
        doc_type = inscription.doc_type

        if doc_type == DocumentType.TABLET:
            return {
                "text": "Administrative record (commodities, personnel, or transactions)",
                "confidence": Confidence.MEDIUM
            }
        elif doc_type == DocumentType.OFFERING_TABLE:
            return {
                "text": "Religious dedication/libation to deity",
                "confidence": Confidence.MEDIUM
            }
        elif doc_type == DocumentType.VESSEL:
            return {
                "text": "Ownership mark or contents label",
                "confidence": Confidence.LOW
            }

        return None

    def _combine_translations(self, translations: List[Dict]) -> str:
        """Combine multiple translation strategies into best guess."""
        if not translations:
            return "[Unable to translate]"

        # Prioritize by confidence
        priority = {
            Confidence.CONFIRMED: 5,
            Confidence.HIGH: 4,
            Confidence.MEDIUM: 3,
            Confidence.LOW: 2,
            Confidence.SPECULATIVE: 1
        }

        sorted_trans = sorted(
            translations,
            key=lambda t: priority.get(t["confidence"], 0),
            reverse=True
        )

        # Return highest confidence translation
        return sorted_trans[0]["text"]

    def _overall_confidence(self, translations: List[Dict]) -> Confidence:
        """Determine overall confidence level."""
        if not translations:
            return Confidence.SPECULATIVE

        confidences = [t["confidence"] for t in translations]

        # If any confirmed, overall is high
        if Confidence.CONFIRMED in confidences:
            return Confidence.HIGH

        # Otherwise, return the highest confidence
        priority = {
            Confidence.HIGH: 4,
            Confidence.MEDIUM: 3,
            Confidence.LOW: 2,
            Confidence.SPECULATIVE: 1
        }

        best = max(confidences, key=lambda c: priority.get(c, 0))
        return best

    def translate_word(self, word: List[str]) -> Dict:
        """Translate a single word."""
        word_key = "-".join(word)
        phonetic = "-".join(
            SYLLABIC_SIGNS.get(s, type('', (), {'phonetic': '?'})()).phonetic or '?'
            for s in word
        )

        result = {
            "signs": word_key,
            "phonetic": phonetic,
            "translations": []
        }

        # Direct lookup
        if word_key in self.lexicon:
            entry = self.lexicon[word_key]
            result["translations"].append({
                "english": entry.english,
                "confidence": entry.confidence,
                "source": "lexicon"
            })

        # Check for ideograms
        for sign in word:
            if sign in self.ideograms:
                meaning, conf = self.ideograms[sign]
                result["translations"].append({
                    "english": meaning,
                    "confidence": conf,
                    "source": "ideogram"
                })

        # Check for known endings
        if word and word[-1] in ENDINGS:
            ending, meaning = ENDINGS[word[-1]]
            result["grammatical_note"] = f"Ends in {ending} ({meaning})"

        return result


# ============================================================================
# NEURAL TRANSLATION MODEL
# ============================================================================

class NeuralTranslator:
    """
    Neural network-based translator using learned representations.

    This model learns to:
    1. Encode Linear A sequences
    2. Predict semantic categories
    3. Generate constrained English output
    """

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Build vocabulary
        self.sign_to_idx = {}
        self.idx_to_sign = {}

        # Semantic categories
        self.categories = [
            "PERSON", "PLACE", "DEITY", "COMMODITY", "QUANTITY",
            "ACTION", "TOTAL", "OFFERING", "UNKNOWN"
        ]
        self.cat_to_idx = {c: i for i, c in enumerate(self.categories)}

        # Network weights (to be trained)
        self.embeddings = None
        self.W_hidden = None
        self.W_output = None

    def _build_training_data(self, inscriptions: List[Inscription]) -> List[Dict]:
        """Build training examples from corpus with inferred labels."""
        examples = []

        for insc in inscriptions:
            words = insc.get_words()

            for i, word in enumerate(words):
                word_key = "-".join(word)

                # Infer category based on context and position
                if word_key in LEXICON:
                    entry = LEXICON[word_key]
                    if entry.part_of_speech == "proper_noun":
                        category = "PLACE" if "Phaistos" in entry.english else "DEITY"
                    elif "total" in entry.english.lower():
                        category = "TOTAL"
                    else:
                        category = "UNKNOWN"
                elif any(s in IDEOGRAM_MEANINGS for s in word):
                    category = "COMMODITY"
                elif i == 0:
                    category = "PERSON"  # Names often come first
                else:
                    category = "UNKNOWN"

                examples.append({
                    "word": word,
                    "category": category,
                    "context": words,
                    "position": i
                })

        return examples

    def train(self, inscriptions: List[Inscription], epochs: int = 50):
        """Train the neural translator."""
        # Build vocabulary
        all_signs = set()
        for insc in inscriptions:
            for sign in insc.get_signs():
                all_signs.add(sign)

        for idx, sign in enumerate(sorted(all_signs)):
            self.sign_to_idx[sign] = idx
            self.idx_to_sign[idx] = sign

        vocab_size = len(self.sign_to_idx)
        n_categories = len(self.categories)

        # Initialize weights
        np.random.seed(42)
        self.embeddings = np.random.randn(vocab_size, self.embedding_dim) * 0.1
        self.W_hidden = np.random.randn(self.embedding_dim, self.hidden_dim) * 0.1
        self.W_output = np.random.randn(self.hidden_dim, n_categories) * 0.1

        # Build training data
        examples = self._build_training_data(inscriptions)

        print(f"Training neural translator on {len(examples)} examples...")

        # Training loop
        learning_rate = 0.01
        for epoch in range(epochs):
            total_loss = 0
            correct = 0

            for ex in examples:
                word = ex["word"]
                target_cat = self.cat_to_idx[ex["category"]]

                # Forward pass
                # Average word embedding
                word_emb = np.zeros(self.embedding_dim)
                for sign in word:
                    if sign in self.sign_to_idx:
                        word_emb += self.embeddings[self.sign_to_idx[sign]]
                word_emb /= max(len(word), 1)

                # Hidden layer
                hidden = np.tanh(word_emb @ self.W_hidden)

                # Output (softmax)
                logits = hidden @ self.W_output
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / exp_logits.sum()

                # Loss
                loss = -np.log(probs[target_cat] + 1e-10)
                total_loss += loss

                # Accuracy
                if np.argmax(probs) == target_cat:
                    correct += 1

                # Backward pass
                grad_output = probs.copy()
                grad_output[target_cat] -= 1

                grad_W_output = np.outer(hidden, grad_output)
                grad_hidden = grad_output @ self.W_output.T * (1 - hidden ** 2)
                grad_W_hidden = np.outer(word_emb, grad_hidden)

                # Update weights
                self.W_output -= learning_rate * grad_W_output
                self.W_hidden -= learning_rate * grad_W_hidden

            if (epoch + 1) % 10 == 0:
                acc = correct / len(examples)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(examples):.4f}, Acc: {acc:.2%}")

    def predict_category(self, word: List[str]) -> Tuple[str, float]:
        """Predict semantic category for a word."""
        if self.embeddings is None:
            return "UNKNOWN", 0.0

        # Get word embedding
        word_emb = np.zeros(self.embedding_dim)
        count = 0
        for sign in word:
            if sign in self.sign_to_idx:
                word_emb += self.embeddings[self.sign_to_idx[sign]]
                count += 1
        if count > 0:
            word_emb /= count

        # Forward pass
        hidden = np.tanh(word_emb @ self.W_hidden)
        logits = hidden @ self.W_output
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Get best category
        best_idx = np.argmax(probs)
        return self.categories[best_idx], probs[best_idx]

    def translate_word(self, word: List[str]) -> Dict:
        """Generate translation for a word using neural model."""
        category, confidence = self.predict_category(word)

        phonetic = "-".join(
            SYLLABIC_SIGNS.get(s, type('', (), {'phonetic': '?'})()).phonetic or '?'
            for s in word
        )

        # Generate English based on category
        category_templates = {
            "PERSON": f"[Person: {phonetic}]",
            "PLACE": f"[Place: {phonetic}]",
            "DEITY": f"[Deity: {phonetic}]",
            "COMMODITY": f"[commodity]",
            "QUANTITY": f"[amount]",
            "ACTION": f"[action: {phonetic}]",
            "TOTAL": "total",
            "OFFERING": "offering",
            "UNKNOWN": f"[{phonetic}]"
        }

        return {
            "phonetic": phonetic,
            "category": category,
            "confidence": confidence,
            "english": category_templates.get(category, f"[{phonetic}]")
        }


# ============================================================================
# MAIN TRANSLATION PIPELINE
# ============================================================================

def run_translation_experiments():
    """Run translation experiments on the corpus."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "LINEAR A → ENGLISH TRANSLATION EXPERIMENTS".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    inscriptions = get_all_inscriptions()
    print(f"\nCorpus: {len(inscriptions)} inscriptions")

    # Initialize translators
    rule_translator = LinearATranslator()
    neural_translator = NeuralTranslator()

    # Train neural model
    print("\n" + "=" * 70)
    print(" TRAINING NEURAL TRANSLATOR")
    print("=" * 70)
    neural_translator.train(inscriptions, epochs=30)

    # Translate sample inscriptions
    print("\n" + "=" * 70)
    print(" TRANSLATION RESULTS")
    print("=" * 70)

    # Select diverse samples
    samples = [
        inscriptions[0],  # Administrative
        inscriptions[5],  # Another administrative
    ]

    # Add libation formulas if present
    for insc in inscriptions:
        if insc.doc_type == DocumentType.OFFERING_TABLE:
            samples.append(insc)
            break

    for insc in samples[:8]:
        print(f"\n{'─' * 60}")
        print(f"ID: {insc.id} ({insc.doc_type.value})")
        print(f"Original: {insc.transcription}")
        if insc.phonetic_reading:
            print(f"Phonetic: {insc.phonetic_reading}")

        # Rule-based translation
        result = rule_translator.translate_inscription(insc)
        print(f"\n📜 Rule-based translation:")
        print(f"   {result['best_translation']}")
        print(f"   Confidence: {result['confidence'].value}")

        # Neural translation
        words = insc.get_words()
        neural_parts = []
        for word in words:
            trans = neural_translator.translate_word(word)
            neural_parts.append(trans["english"])
        print(f"\n🧠 Neural translation:")
        print(f"   {' '.join(neural_parts)}")

    # Translate key formulas
    print("\n" + "=" * 70)
    print(" KEY VOCABULARY TRANSLATIONS")
    print("=" * 70)

    key_terms = [
        ["AB77", "AB26"],  # ku-ro
        ["AB03", "AB28", "AB05"],  # pa-i-to
        ["AB57", "AB31", "AB31", "AB60"],  # ja-sa-sa-ra
        ["AB28", "AB01", "AB80", "AB04"],  # i-da-ma-te
        ["AB57", "AB31", "AB31", "AB60", "AB80", "AB24"],  # ja-sa-sa-ra-ma-ne
    ]

    for term in key_terms:
        rule_result = rule_translator.translate_word(term)
        neural_result = neural_translator.translate_word(term)

        print(f"\n{rule_result['phonetic']}:")
        if rule_result["translations"]:
            for t in rule_result["translations"]:
                print(f"  📜 {t['english']} ({t['confidence'].value})")
        print(f"  🧠 Category: {neural_result['category']} ({neural_result['confidence']:.1%})")

    # Summary
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "TRANSLATION SUMMARY".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    print("""
WHAT WE CAN TRANSLATE WITH CONFIDENCE:

✓ CONFIRMED:
  • pa-i-to = "Phaistos" (place name)
  • ku-ro = "total" (administrative term)
  • Commodity ideograms (wine, oil, grain, etc.)

○ PROBABLE:
  • ja-sa-sa-ra = divine name/epithet ("Lady"?)
  • i-da-ma-te = "Mother of Ida" (deity name?)
  • ku-ro-ro = "totals" or "grand total"
  • -ma-ne suffix = dative/dedicatory ("to/for")

? SPECULATIVE:
  • da-ma-te = "lady" or "mother"
  • -te ending = dative case
  • -na ending = locative case
  • Personal names in initial position

DOCUMENT TYPE TRANSLATIONS:

Administrative tablets:
  "[Name] [commodity] [amount]"
  "Total: [sum]"

Libation formulas:
  "To the Mother of Ida..."
  "O Lady [divine name]..."

Vessel inscriptions:
  "[Owner's name]" or "[Contents]"

LIMITATIONS:
  • No bilingual text to verify translations
  • Unknown language family limits inference
  • Many words remain untranslatable
  • Grammatical structure still unclear
""")


if __name__ == "__main__":
    run_translation_experiments()
