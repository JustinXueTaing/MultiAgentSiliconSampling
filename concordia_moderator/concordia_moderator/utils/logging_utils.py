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
        if self._fh:
            self._fh.write(line + ""); self._fh.flush()
        else:
            sys.stdout.write(line + "")


    def close(self):
        if self._fh: self._fh.close()