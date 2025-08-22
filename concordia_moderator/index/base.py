from __future__ import annotations
from typing import Protocol
import numpy as np


class VectorIndex(Protocol):
    def add(self, vectors: np.ndarray): ...
    def search(self, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]: ... # (scores, ids)