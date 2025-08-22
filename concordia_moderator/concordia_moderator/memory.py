from __future__ import annotations
from typing import List, Tuple


class AssociativeMemory:
    def __init__(self):
        self._store: List[Tuple[str, str]] = []


    def add(self, key: str, value: str):
        self._store.append((key, value))


    def search(self, key: str, k: int = 3) -> List[Tuple[str, str]]:
    # returns last-k entries with this key
        return [kv for kv in self._store if kv[0] == key][-k:]