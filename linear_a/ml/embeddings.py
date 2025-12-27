"""
Sign Embeddings for Linear A

Learn distributed representations of Linear A signs using:
1. Co-occurrence matrices with SVD (like LSA)
2. Skip-gram style learning (simplified word2vec)
3. Contextual analysis

The hypothesis: Signs that appear in similar contexts have similar
functions (phonetic, grammatical, or semantic).
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus.inscriptions import get_all_inscriptions, extract_all_words
from signs.inventory import SYLLABIC_SIGNS


class SignEmbeddings:
    """Learn and analyze sign embeddings for Linear A."""

    def __init__(self, embedding_dim: int = 32, window_size: int = 2):
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        # Load corpus
        self.inscriptions = get_all_inscriptions()
        self.words = extract_all_words()

        # Build vocabulary
        self._build_vocabulary()

        # Embeddings (to be learned)
        self.embeddings: Optional[np.ndarray] = None
        self.cooccurrence_matrix: Optional[np.ndarray] = None

    def _build_vocabulary(self):
        """Build sign vocabulary from corpus."""
        # Count all signs
        sign_counts = Counter()
        for inscription in self.inscriptions:
            for sign in inscription.get_signs():
                if sign.startswith("AB") or sign.startswith("LA"):
                    sign_counts[sign] += 1

        # Create mappings
        self.sign_to_idx = {}
        self.idx_to_sign = {}
        for idx, (sign, count) in enumerate(sign_counts.most_common()):
            self.sign_to_idx[sign] = idx
            self.idx_to_sign[idx] = sign

        self.vocab_size = len(self.sign_to_idx)
        self.sign_counts = sign_counts

    def build_cooccurrence_matrix(self) -> np.ndarray:
        """
        Build co-occurrence matrix from corpus.

        For each pair of signs that appear within window_size of each other,
        increment their co-occurrence count. Weight by inverse distance.
        """
        cooc = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)

        # Process each word
        for word in self.words:
            indices = [self.sign_to_idx.get(s) for s in word]
            indices = [i for i in indices if i is not None]

            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    if i != j:
                        distance = abs(i - j)
                        if distance <= self.window_size:
                            # Weight by inverse distance
                            weight = 1.0 / distance
                            cooc[idx_i, idx_j] += weight

        # Apply log transformation (like GloVe)
        cooc = np.log1p(cooc)

        self.cooccurrence_matrix = cooc
        return cooc

    def learn_embeddings_svd(self) -> np.ndarray:
        """
        Learn embeddings via SVD on co-occurrence matrix.

        This is similar to LSA/LSI - decompose the co-occurrence matrix
        and use the left singular vectors as embeddings.
        """
        if self.cooccurrence_matrix is None:
            self.build_cooccurrence_matrix()

        # Apply PPMI (Positive Pointwise Mutual Information)
        ppmi = self._compute_ppmi(self.cooccurrence_matrix)

        # SVD decomposition
        U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)

        # Take top dimensions
        dim = min(self.embedding_dim, U.shape[1])
        self.embeddings = U[:, :dim] * np.sqrt(S[:dim])

        return self.embeddings

    def _compute_ppmi(self, cooc: np.ndarray) -> np.ndarray:
        """Compute Positive Pointwise Mutual Information matrix."""
        # Total sum
        total = cooc.sum() + 1e-10

        # Row and column sums (marginals)
        row_sum = cooc.sum(axis=1, keepdims=True) + 1e-10
        col_sum = cooc.sum(axis=0, keepdims=True) + 1e-10

        # PMI = log(P(x,y) / (P(x) * P(y)))
        # P(x,y) = cooc / total
        # P(x) = row_sum / total
        # P(y) = col_sum / total
        expected = (row_sum * col_sum) / total
        pmi = np.log2((cooc * total) / expected + 1e-10)

        # PPMI - zero out negative values
        ppmi = np.maximum(pmi, 0)

        return ppmi

    def learn_embeddings_skipgram(self, epochs: int = 100,
                                   learning_rate: float = 0.01) -> np.ndarray:
        """
        Learn embeddings using simplified skip-gram with negative sampling.

        For each (target, context) pair, learn to predict context from target.
        """
        # Initialize embeddings randomly
        np.random.seed(42)
        target_emb = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1
        context_emb = np.random.randn(self.vocab_size, self.embedding_dim) * 0.1

        # Build training pairs
        pairs = []
        for word in self.words:
            indices = [self.sign_to_idx.get(s) for s in word]
            indices = [i for i in indices if i is not None]

            for i, target_idx in enumerate(indices):
                for j in range(max(0, i - self.window_size),
                              min(len(indices), i + self.window_size + 1)):
                    if i != j:
                        context_idx = indices[j]
                        pairs.append((target_idx, context_idx))

        if not pairs:
            self.embeddings = target_emb
            return target_emb

        # Training with negative sampling
        num_negative = 5
        neg_probs = np.array([self.sign_counts.get(self.idx_to_sign[i], 1)
                              for i in range(self.vocab_size)])
        neg_probs = np.power(neg_probs, 0.75)
        neg_probs = neg_probs / neg_probs.sum()

        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(pairs)

            for target_idx, context_idx in pairs:
                # Positive sample
                target_vec = target_emb[target_idx]
                context_vec = context_emb[context_idx]

                # Sigmoid of dot product
                score = 1.0 / (1.0 + np.exp(-np.dot(target_vec, context_vec)))
                loss = -np.log(score + 1e-10)
                total_loss += loss

                # Gradient for positive sample
                grad = (score - 1) * learning_rate
                target_emb[target_idx] -= grad * context_vec
                context_emb[context_idx] -= grad * target_vec

                # Negative samples
                neg_indices = np.random.choice(self.vocab_size, num_negative,
                                               p=neg_probs, replace=False)
                for neg_idx in neg_indices:
                    if neg_idx == context_idx:
                        continue
                    neg_vec = context_emb[neg_idx]
                    score = 1.0 / (1.0 + np.exp(-np.dot(target_vec, neg_vec)))
                    loss += -np.log(1 - score + 1e-10)

                    grad = score * learning_rate
                    target_emb[target_idx] -= grad * neg_vec
                    context_emb[neg_idx] -= grad * target_vec

            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(pairs)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        # Final embeddings are average of target and context
        self.embeddings = (target_emb + context_emb) / 2
        return self.embeddings

    def get_similar_signs(self, sign: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find signs most similar to the given sign."""
        if self.embeddings is None:
            self.learn_embeddings_svd()

        if sign not in self.sign_to_idx:
            return []

        idx = self.sign_to_idx[sign]
        query_vec = self.embeddings[idx]

        # Compute cosine similarities
        similarities = []
        for other_idx in range(self.vocab_size):
            if other_idx == idx:
                continue
            other_vec = self.embeddings[other_idx]
            cos_sim = np.dot(query_vec, other_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(other_vec) + 1e-10
            )
            similarities.append((self.idx_to_sign[other_idx], cos_sim))

        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

    def cluster_signs(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Cluster signs based on their embeddings using k-means.
        """
        if self.embeddings is None:
            self.learn_embeddings_svd()

        # Simple k-means implementation
        np.random.seed(42)
        centroids = self.embeddings[
            np.random.choice(self.vocab_size, n_clusters, replace=False)
        ]

        for iteration in range(50):
            # Assign to nearest centroid
            distances = np.zeros((self.vocab_size, n_clusters))
            for k in range(n_clusters):
                diff = self.embeddings - centroids[k]
                distances[:, k] = np.linalg.norm(diff, axis=1)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = self.embeddings[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        # Build cluster dictionary
        clusters = defaultdict(list)
        for idx, cluster_id in enumerate(assignments):
            sign = self.idx_to_sign[idx]
            clusters[cluster_id].append(sign)

        return dict(clusters)

    def analyze_sign_roles(self) -> Dict[str, Dict]:
        """
        Analyze likely roles of signs based on embedding patterns.

        Hypothesis: Vowels cluster together, grammatical markers cluster,
        consonants with similar articulation cluster, etc.
        """
        if self.embeddings is None:
            self.learn_embeddings_svd()

        results = {}

        # Get clusters
        clusters = self.cluster_signs(n_clusters=6)

        # Analyze each cluster
        for cluster_id, signs in clusters.items():
            # Get phonetic values
            phonetics = []
            for sign in signs:
                if sign in SYLLABIC_SIGNS:
                    p = SYLLABIC_SIGNS[sign].phonetic
                    if p and p != "?":
                        phonetics.append(p)

            # Analyze patterns
            vowel_count = sum(1 for p in phonetics if len(p) == 1)
            cv_count = sum(1 for p in phonetics if len(p) == 2)

            # Initial letters (consonants)
            initials = [p[0] for p in phonetics if len(p) >= 2]

            results[f"Cluster {cluster_id}"] = {
                "signs": signs,
                "phonetics": phonetics,
                "vowel_ratio": vowel_count / len(phonetics) if phonetics else 0,
                "common_initials": Counter(initials).most_common(3),
                "size": len(signs)
            }

        return results

    def print_analysis(self):
        """Print comprehensive embedding analysis."""
        print("\n" + "=" * 70)
        print("SIGN EMBEDDING ANALYSIS")
        print("=" * 70)

        print(f"\nVocabulary size: {self.vocab_size} signs")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Context window: {self.window_size}")

        # Learn embeddings
        print("\n--- Learning embeddings via SVD ---")
        self.learn_embeddings_svd()

        # Show similar signs for key signs
        print("\n--- Sign Similarities ---")
        key_signs = ["AB08", "AB28", "AB77", "AB59", "AB06", "AB31"]
        for sign in key_signs:
            if sign in self.sign_to_idx:
                phonetic = SYLLABIC_SIGNS.get(sign)
                p_str = f" [{phonetic.phonetic}]" if phonetic else ""
                print(f"\n{sign}{p_str} most similar to:")
                similar = self.get_similar_signs(sign, top_k=5)
                for other_sign, sim in similar:
                    other_p = SYLLABIC_SIGNS.get(other_sign)
                    op_str = f" [{other_p.phonetic}]" if other_p else ""
                    print(f"  {other_sign}{op_str}: {sim:.3f}")

        # Cluster analysis
        print("\n--- Sign Clusters ---")
        clusters = self.cluster_signs(n_clusters=5)
        for cluster_id, signs in sorted(clusters.items()):
            phonetics = []
            for sign in signs[:10]:
                if sign in SYLLABIC_SIGNS:
                    p = SYLLABIC_SIGNS[sign].phonetic
                    phonetics.append(p if p else "?")
            print(f"\nCluster {cluster_id} ({len(signs)} signs):")
            print(f"  Signs: {', '.join(signs[:10])}")
            print(f"  Phonetics: {', '.join(phonetics)}")

        # Role analysis
        print("\n--- Sign Role Analysis ---")
        roles = self.analyze_sign_roles()
        for role_name, data in roles.items():
            if data['phonetics']:
                print(f"\n{role_name}:")
                print(f"  Phonetics: {', '.join(data['phonetics'][:8])}")
                print(f"  Vowel ratio: {data['vowel_ratio']:.1%}")
                if data['common_initials']:
                    print(f"  Common consonants: {data['common_initials']}")


def main():
    """Run embedding analysis."""
    embedder = SignEmbeddings(embedding_dim=16, window_size=2)
    embedder.print_analysis()


if __name__ == "__main__":
    main()
