from __future__ import annotations
from pathlib import Path
from typing import Any
import json, sys


class JsonlLogger:
    def __init__(self, path: str | None):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8") if path else None


    def write(self, obj: Any):
        line = json.dumps(obj, ensure_ascii=False)
        # If the file handle exists but was closed elsewhere, reopen it.
        if self._fh is not None:
            try:
                closed = getattr(self._fh, "closed", False)
            except Exception:
                closed = True
            if closed:
                # Try to reopen the path if available; otherwise fallback to stdout
                if self.path:
                    self._fh = open(self.path, "a", encoding="utf-8")
                else:
                    self._fh = None

        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        else:
            sys.stdout.write(line + "\n")


    def close(self):
        if self._fh:
            try:
                self._fh.close()
            finally:
                self._fh = None