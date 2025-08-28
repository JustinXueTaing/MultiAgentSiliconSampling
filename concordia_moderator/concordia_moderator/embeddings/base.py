
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingProvider(ABC):
	"""Minimal interface for embedding providers used by the project.

	Subclasses should implement the `encode` method returning a numpy array
	of shape (len(texts), dim) and the `dim` property.
	"""

	@property
	@abstractmethod
	def dim(self) -> int:
		"""Return the dimensionality of embeddings produced by this provider."""
		raise NotImplementedError()

	@abstractmethod
	def encode(self, texts: List[str]) -> np.ndarray:
		"""Encode a list of texts into a numpy array of embeddings.

		Args:
			texts: list of input strings.

		Returns:
			numpy.ndarray with shape (len(texts), self.dim)
		"""
		raise NotImplementedError()


__all__ = ["EmbeddingProvider"]

