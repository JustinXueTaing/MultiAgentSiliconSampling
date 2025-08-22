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
        X = self.embedder.encode(answers)
            # choose k in [2..min(6, n)] by silhouette
        n = len(answers)
        best_k, best_score = 2, -1.0
        for k in range(2, min(6, n) + 1):
            labels = AgglomerativeClustering(n_clusters=k, linkage="average").fit_predict(X)
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        labels = AgglomerativeClustering(n_clusters=best_k, linkage="average").fit_predict(X)
            # compute majority cluster proportion
        counts = np.bincount(labels)
        return counts.max() / n