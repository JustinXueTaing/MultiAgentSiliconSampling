from __future__ import annotations
import os
import numpy as np
from typing import List
from openai import OpenAI
from .base import EmbeddingProvider


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-large"):
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY not set")
        self.client = OpenAI()
        self.model = model
        # Known dims: 3072 for text-embedding-3-large (subject to provider docs)
        self._dim = 3072


    @property
    def dim(self) -> int:
            return self._dim


    def encode(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return arr