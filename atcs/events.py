"""Hodisalar jurnali (JSON Lines): har bir buyum, xato va UV-C sikli yoziladi."""
from __future__ import annotations

import json
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, t: float, kind: str, **data) -> None:
        rec = {"t": round(t, 3), "wall": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **data}
        self.records.append(rec)
        if len(self.records) > 1000:
            del self.records[:500]
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def of(self, kind: str) -> list[dict]:
        return [r for r in self.records if r["kind"] == kind]
