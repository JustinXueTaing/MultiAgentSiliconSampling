from __future__ import annotations
import numpy as np
try:
    import faiss # type: ignore
except Exception as e: # pragma: no cover
    raise ImportError("faiss not installed. Install with `pip install .[faiss]`.")


from .base import VectorIndex


class FaissIPIndex(VectorIndex):
    def __init__(self, dim: int, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize
        self.index = faiss.IndexFlatIP(dim)


    def _normalize(self, v: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return v
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms


    def add(self, vectors: np.ndarray):
        v = self._normalize(vectors.astype(np.float32))
        self.index.add(v)


    def search(self, queries: np.ndarray, k: int = 5):
        q = self._normalize(queries.astype(np.float32))
        scores, ids = self.index.search(q, k)
        return scores, ids