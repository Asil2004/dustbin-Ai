"""Sozlamalarni TOML fayldan o'qish. Natija - nuqta orqali murojaat qilinadigan obyekt (cfg.sorting.entry_x)."""
from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "atcs.toml"


def _ns(value):
    if isinstance(value, dict):
        # raqamli kalitlar (us_echo = { 2 = 18 }) lug'at bo'lib qoladi
        if all(str(k).isdigit() for k in value):
            return {int(k): v for k, v in value.items()}
        return SimpleNamespace(**{k: _ns(v) for k, v in value.items()})
    return value


def load(path: str | Path | None = None) -> SimpleNamespace:
    with open(path or DEFAULT_PATH, "rb") as f:
        cfg = _ns(tomllib.load(f))
    _validate(cfg)
    return cfg


def _validate(cfg) -> None:
    s = cfg.sorting
    if len(s.deflector_x) != 4 or sorted(s.deflector_x) != list(s.deflector_x):
        raise ValueError("sorting.deflector_x: 4 ta o'sib boruvchi qiymat bo'lishi kerak")
    if not s.entry_x < s.deflector_x[0] < s.belt_end_x:
        raise ValueError("sorting: entry_x < deflector_x[0] < belt_end_x bo'lishi kerak")
    if s.min_spacing_mm <= s.lead_mm + s.pass_mm:
        raise ValueError("sorting.min_spacing_mm > lead_mm + pass_mm bo'lishi kerak, aks holda "
                         "deflektor qo'shni buyumni ham surib yuboradi")
    if s.unknown_action not in ("operator", "bin"):
        raise ValueError('sorting.unknown_action: "operator" yoki "bin" bo\'lishi kerak')
    if not (0 < cfg.vision.presence_min_fraction < 1 and 0 < cfg.vision.presence_pixel_diff < 255):
        raise ValueError("vision.presence_min_fraction (0..1) yoki presence_pixel_diff (0..255) noto'g'ri")
    if cfg.vision.camera not in ("usb", "csi"):
        raise ValueError("vision.camera 'usb' yoki 'csi' bo'lishi kerak")
    if len(cfg.vision.labels) != 5 or len(cfg.bins.names) != 5:
        raise ValueError("vision.labels va bins.names 5 tadan bo'lishi kerak")
