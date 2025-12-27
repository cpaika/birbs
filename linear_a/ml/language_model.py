"""
Language Model for Linear A

Learn the sequential structure of Linear A using:
1. N-gram models (traditional)
2. Simple neural language model (RNN-like)
3. Attention-based analysis

Goals:
- Understand which sign sequences are "grammatical"
- Identify predictable vs. unpredictable positions
- Learn implicit grammatical rules
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.inscriptions import get_all_inscriptions, extract_all_words
from signs.inventory import SYLLABIC_SIGNS


class NGramModel:
    """Traditional n-gram language model for Linear A."""

    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_totals = Counter()
        self.unigram_counts = Counter()
        self.vocab = set()

        # Special tokens
        self.START = "<S>"
        self.END = "</S>"

    def train(self, words: List[List[str]]):
        """Train the n-gram model on word sequences."""
        for word in words:
            # Add start/end tokens
            padded = [self.START] * (self.n - 1) + word + [self.END]

            # Count unigrams
            for sign in word:
                self.unigram_counts[sign] += 1
                self.vocab.add(sign)

            # Count n-grams
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                target = padded[i + self.n - 1]
                self.ngram_counts[context][target] += 1
                self.context_totals[context] += 1

    def probability(self, context: Tuple[str, ...], target: str,
                   smoothing: float = 0.1) -> float:
        """
        Compute P(target | context) with add-k smoothing.
        """
        context = tuple(context[-(self.n - 1):])  # Take last n-1 tokens

        count = self.ngram_counts[context][target]
        total = self.context_totals[context]
        vocab_size = len(self.vocab) + 2  # +2 for START/END

        # Add-k smoothing
        prob = (count + smoothing) / (total + smoothing * vocab_size)
        return prob

    def perplexity(self, word: List[str]) -> float:
        """
        Compute perplexity of a word.
        Lower = more predictable, higher = more surprising.
        """
        padded = [self.START] * (self.n - 1) + word + [self.END]
        log_prob_sum = 0
        count = 0

        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            target = padded[i]
            prob = self.probability(context, target)
            log_prob_sum += np.log2(prob + 1e-10)
            count += 1

        avg_log_prob = log_prob_sum / count if count > 0 else 0
        perplexity = 2 ** (-avg_log_prob)
        return perplexity

    def predict_next(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict most likely next signs given context.
        """
        context = tuple(context[-(self.n - 1):])

        predictions = []
        for sign in self.vocab:
            prob = self.probability(context, sign)
            predictions.append((sign, prob))

        # Add END probability
        predictions.append((self.END, self.probability(context, self.END)))

        predictions.sort(key=lambda x: -x[1])
        return predictions[:top_k]

    def analyze_predictability(self) -> Dict[str, float]:
        """
        Analyze how predictable each sign is in context.
        High predictability = grammatical function (articles, endings)
        Low predictability = content words
        """
        sign_predictability = {}

        for sign in self.vocab:
            # Find all contexts where this sign appears
            total_prob = 0
            count = 0

            for context, targets in self.ngram_counts.items():
                if sign in targets:
                    prob = self.probability(context, sign)
                    total_prob += prob
                    count += 1

            if count > 0:
                sign_predictability[sign] = total_prob / count

        return sign_predictability


class NeuralLM:
    """
    Simple neural language model for Linear A.

    Uses a basic feedforward network with embeddings.
    For a tiny corpus, this is more appropriate than RNN/Transformer.
    """

    def __init__(self, embedding_dim: int = 16, hidden_dim: int = 32,
                 context_size: int = 2):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size

        self.vocab = set()
        self.sign_to_idx = {}
        self.idx_to_sign = {}

        # Weights (to be initialized after seeing data)
        self.embeddings = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _build_vocab(self, words: List[List[str]]):
        """Build vocabulary from training data."""
        for word in words:
            for sign in word:
                self.vocab.add(sign)

        self.vocab.add("<S>")
        self.vocab.add("</S>")
        self.vocab.add("<UNK>")

        for idx, sign in enumerate(sorted(self.vocab)):
            self.sign_to_idx[sign] = idx
            self.idx_to_sign[idx] = sign

        self.vocab_size = len(self.vocab)

    def _initialize_weights(self):
        """Initialize neural network weights."""
        np.random.seed(42)

        # Embeddings
        self.embeddings = np.random.randn(
            self.vocab_size, self.embedding_dim
        ) * 0.1

        # Hidden layer
        input_dim = self.embedding_dim * self.context_size
        self.W1 = np.random.randn(input_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)

        # Output layer
        self.W2 = np.random.randn(self.hidden_dim, self.vocab_size) * 0.1
        self.b2 = np.zeros(self.vocab_size)

    def _get_context_vector(self, context: List[str]) -> np.ndarray:
        """Convert context signs to concatenated embedding vector."""
        # Pad context if needed
        padded = ["<S>"] * self.context_size
        padded = (padded + context)[-self.context_size:]

        indices = [self.sign_to_idx.get(s, self.sign_to_idx["<UNK>"])
                   for s in padded]
        vectors = [self.embeddings[idx] for idx in indices]
        return np.concatenate(vectors)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (exp_x.sum() + 1e-10)

    def forward(self, context: List[str]) -> np.ndarray:
        """
        Forward pass: context -> probability distribution over next signs.
        """
        # Context embedding
        x = self._get_context_vector(context)

        # Hidden layer (ReLU)
        h = np.maximum(0, x @ self.W1 + self.b1)

        # Output layer (softmax)
        logits = h @ self.W2 + self.b2
        probs = self._softmax(logits)

        return probs

    def train(self, words: List[List[str]], epochs: int = 50,
              learning_rate: float = 0.01):
        """Train the neural language model."""
        self._build_vocab(words)
        self._initialize_weights()

        # Build training examples: (context, target)
        examples = []
        for word in words:
            padded = ["<S>"] * self.context_size + word + ["</S>"]
            for i in range(self.context_size, len(padded)):
                context = padded[i - self.context_size:i]
                target = padded[i]
                target_idx = self.sign_to_idx.get(target, self.sign_to_idx["<UNK>"])
                examples.append((context, target_idx))

        print(f"Training on {len(examples)} examples...")

        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(examples)

            for context, target_idx in examples:
                # Forward pass
                x = self._get_context_vector(context)
                h = np.maximum(0, x @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = self._softmax(logits)

                # Cross-entropy loss
                loss = -np.log(probs[target_idx] + 1e-10)
                total_loss += loss

                # Backward pass
                # Gradient of cross-entropy + softmax
                grad_logits = probs.copy()
                grad_logits[target_idx] -= 1

                # Output layer gradients
                grad_W2 = np.outer(h, grad_logits)
                grad_b2 = grad_logits

                # Hidden layer gradients
                grad_h = grad_logits @ self.W2.T
                grad_h = grad_h * (h > 0)  # ReLU derivative

                # Input layer gradients
                grad_W1 = np.outer(x, grad_h)
                grad_b1 = grad_h

                # Embedding gradients
                grad_x = grad_h @ self.W1.T

                # Update weights
                self.W2 -= learning_rate * grad_W2
                self.b2 -= learning_rate * grad_b2
                self.W1 -= learning_rate * grad_W1
                self.b1 -= learning_rate * grad_b1

                # Update embeddings
                grad_emb = grad_x.reshape(self.context_size, self.embedding_dim)
                padded = ["<S>"] * self.context_size
                padded = (padded + context)[-self.context_size:]
                for i, sign in enumerate(padded):
                    idx = self.sign_to_idx.get(sign, self.sign_to_idx["<UNK>"])
                    self.embeddings[idx] -= learning_rate * grad_emb[i]

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(examples)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict_next(self, context: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict next sign given context."""
        probs = self.forward(context)
        predictions = [(self.idx_to_sign[i], p) for i, p in enumerate(probs)]
        predictions.sort(key=lambda x: -x[1])
        return predictions[:top_k]

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """Return learned sign embeddings."""
        return {sign: self.embeddings[idx]
                for sign, idx in self.sign_to_idx.items()
                if sign not in ["<S>", "</S>", "<UNK>"]}


class PositionalEntropyAnalyzer:
    """
    Analyze entropy at each position in words.

    High entropy = many possible signs (content position)
    Low entropy = few possible signs (grammatical position)
    """

    def __init__(self):
        self.position_distributions = defaultdict(Counter)
        self.max_length = 0

    def analyze(self, words: List[List[str]]) -> Dict[int, float]:
        """
        Compute entropy at each position in words.
        """
        for word in words:
            self.max_length = max(self.max_length, len(word))
            for pos, sign in enumerate(word):
                self.position_distributions[pos][sign] += 1

            # Also track from end
            for pos, sign in enumerate(reversed(word)):
                self.position_distributions[f"end-{pos}"][sign] += 1

        entropies = {}
        for pos, counter in self.position_distributions.items():
            total = sum(counter.values())
            probs = [c / total for c in counter.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            entropies[pos] = entropy

        return entropies

    def get_most_predictable_positions(self, entropies: Dict) -> List[Tuple]:
        """Find positions with lowest entropy (most predictable)."""
        # Separate numeric and end-relative positions
        numeric = [(k, v) for k, v in entropies.items() if isinstance(k, int)]
        end_rel = [(k, v) for k, v in entropies.items() if isinstance(k, str)]

        numeric.sort(key=lambda x: x[1])
        end_rel.sort(key=lambda x: x[1])

        return numeric[:3], end_rel[:3]


def run_language_model_analysis():
    """Run comprehensive language model analysis."""
    print("\n" + "=" * 70)
    print("LANGUAGE MODEL ANALYSIS")
    print("=" * 70)

    words = extract_all_words()
    print(f"\nCorpus: {len(words)} words")

    # N-gram model
    print("\n--- N-gram Model (trigram) ---")
    ngram = NGramModel(n=3)
    ngram.train(words)

    # Sample perplexities
    print("\nWord perplexities:")
    sample_words = words[:10]
    for word in sample_words:
        ppl = ngram.perplexity(word)
        reading = "-".join(SYLLABIC_SIGNS.get(s, type('', (), {'phonetic': '?'})()).phonetic or '?'
                          for s in word)
        print(f"  {reading}: {ppl:.2f}")

    # Predictions
    print("\nNext-sign predictions:")
    contexts = [["AB77"], ["AB57", "AB31"], ["AB28", "AB01"]]
    for ctx in contexts:
        preds = ngram.predict_next(ctx, top_k=3)
        ctx_reading = "-".join(SYLLABIC_SIGNS.get(s, type('', (), {'phonetic': '?'})()).phonetic or '?'
                               for s in ctx)
        print(f"\n  After '{ctx_reading}':")
        for sign, prob in preds:
            if sign not in ["<S>", "</S>"]:
                p = SYLLABIC_SIGNS.get(sign, type('', (), {'phonetic': '?'})()).phonetic or '?'
                print(f"    {sign} [{p}]: {prob:.3f}")

    # Predictability analysis
    print("\n--- Sign Predictability ---")
    predictability = ngram.analyze_predictability()
    sorted_pred = sorted(predictability.items(), key=lambda x: -x[1])

    print("\nMost predictable signs (likely grammatical):")
    for sign, pred in sorted_pred[:8]:
        p = SYLLABIC_SIGNS.get(sign, type('', (), {'phonetic': '?'})()).phonetic or '?'
        print(f"  {sign} [{p}]: {pred:.4f}")

    print("\nLeast predictable signs (likely content):")
    for sign, pred in sorted_pred[-8:]:
        p = SYLLABIC_SIGNS.get(sign, type('', (), {'phonetic': '?'})()).phonetic or '?'
        print(f"  {sign} [{p}]: {pred:.4f}")

    # Neural LM
    print("\n--- Neural Language Model ---")
    nlm = NeuralLM(embedding_dim=12, hidden_dim=24, context_size=2)
    nlm.train(words, epochs=30, learning_rate=0.02)

    print("\nNeural predictions:")
    for ctx in contexts:
        preds = nlm.predict_next(ctx, top_k=3)
        ctx_reading = "-".join(SYLLABIC_SIGNS.get(s, type('', (), {'phonetic': '?'})()).phonetic or '?'
                               for s in ctx)
        print(f"\n  After '{ctx_reading}':")
        for sign, prob in preds:
            if sign not in ["<S>", "</S>", "<UNK>"]:
                p = SYLLABIC_SIGNS.get(sign, type('', (), {'phonetic': '?'})()).phonetic or '?'
                print(f"    {sign} [{p}]: {prob:.3f}")

    # Positional entropy
    print("\n--- Positional Entropy Analysis ---")
    entropy_analyzer = PositionalEntropyAnalyzer()
    entropies = entropy_analyzer.analyze(words)

    print("\nEntropy by position (from start):")
    for pos in range(min(6, entropy_analyzer.max_length)):
        if pos in entropies:
            print(f"  Position {pos}: {entropies[pos]:.3f} bits")

    print("\nEntropy by position (from end):")
    for pos in range(3):
        key = f"end-{pos}"
        if key in entropies:
            label = "final" if pos == 0 else f"{pos} from end"
            print(f"  {label}: {entropies[key]:.3f} bits")

    print("\nInterpretation:")
    print("  - Lower entropy = more constrained/predictable")
    print("  - Word-final position often has low entropy")
    print("    (suggests grammatical endings like case markers)")
    print("  - Initial positions may have moderate entropy")
    print("    (mix of roots and prefixes)")


if __name__ == "__main__":
    run_language_model_analysis()
