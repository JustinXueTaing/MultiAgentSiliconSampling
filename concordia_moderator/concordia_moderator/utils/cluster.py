from __future__ import annotations
from typing import List
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from ..embeddings.base import EmbeddingProvider


class AnswerClusterer:
    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder

    def dominant_share(self, answers: List[str]) -> float:
        if len(answers) <= 1:
            return 1.0 if answers else 0.0

        n = len(answers)

        # Special-case for exactly two answers (silhouette_score can’t handle n=2)
        if n == 2:
            return 1.0 if answers[0] == answers[1] else 0.5

        X = self.embedder.encode(answers)
        # choose k in [2..min(6, n)] by silhouette
        best_k, best_score = 2, -1.0
        for k in range(2, min(6, n) + 1):
            labels = AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(X)

            n_labels = len(set(labels))
            # Guard against degenerate clustering
            if n_labels <= 1 or n_labels >= n:
                continue

            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score

        # Fallback: if no valid k was found
        if best_score < 0:
            return 1.0  # treat as unanimous

        labels = AgglomerativeClustering(n_clusters=best_k, linkage="average").fit_predict(X)
        counts = np.bincount(labels)
        return counts.max() / n







