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
    if not 1 <= cfg.timing.presence_confirm <= 10:
        raise ValueError("timing.presence_confirm 1..10 bo'lishi kerak")
    if s.unknown_action not in ("operator", "bin"):
        raise ValueError('sorting.unknown_action: "operator" yoki "bin" bo\'lishi kerak')
    if not (0 < cfg.vision.presence_min_fraction < 1 and 0 < cfg.vision.presence_pixel_diff < 255):
        raise ValueError("vision.presence_min_fraction (0..1) yoki presence_pixel_diff (0..255) noto'g'ri")
    for key in ("usb_focus", "usb_exposure"):
        val = getattr(cfg.vision, key)
        if (isinstance(val, str) and val != "auto") or isinstance(val, bool):
            raise ValueError(f"vision.{key} \"auto\" yoki son bo'lishi kerak")
    if cfg.vision.camera not in ("usb", "csi"):
        raise ValueError("vision.camera 'usb' yoki 'csi' bo'lishi kerak")
    labels = list(cfg.vision.labels)
    item_labels = [lb for lb in labels if lb != "bosh"]
    if len(item_labels) != 5 or len(cfg.bins.names) != 5 or len(labels) - len(item_labels) > 1:
        raise ValueError("vision.labels: 5 ta bo'lim sinfi (+ ixtiyoriy \"bosh\"), bins.names: 5 ta")
    if "igna" not in item_labels:
        raise ValueError("vision.labels da \"igna\" bo'lishi kerak (igna ustuvorligi va xavfsizlik hisoboti)")
    if not 0 < cfg.sorting.needle_priority <= 1:
        raise ValueError("sorting.needle_priority 0..1 oralig'ida bo'lishi kerak (1 = o'chirilgan)")
