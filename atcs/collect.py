"""O'qitish uchun rasm yig'ish.

Kamera config dagi vision.camera bo'yicha ochiladi, o'lcham, fokus va ekspozitsiya ish rejimidagi bilan bir xil.
Noutbukda ham, Pi'da ham bir xil buyruq (stoldagi USB webkamerani noutbukka ulab yig'sa bo'ladi):
  python -m atcs.collect --label shisha
Noutbukning o'z kamerasi emas, tashqi USB kamera kerak bo'lsa: --camera-index 1
Bo'sh lenta ("bosh" sinfi) uchun avtomatik: python -m atcs.collect --label bosh --auto 300 --interval 2
  (yorug'likni, soyalarni, lenta holatini o'zgartirib turing: model shularni "bo'sh" deb o'rganadi)

Buyumni shlyuz oldiga qo'ying -> Enter = suratga olish, buyumni buring/almashtiring -> yana Enter.
  r + Enter  - oxirgi rasmni o'chirish
  q + Enter  - chiqish
Har kadrdan vision.roi zonasi kesib saqlanadi (ish rejimidagi kabi): dataset/<label>/<vaqt>.jpg
dataset/roi_tekshiruv.jpg - to'liq kadr va ROI ramkasi: zona to'g'ri joydami, tekshiring.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from . import config as config_mod
from .vision import crop_roi


def save_roi_check(frame: np.ndarray, roi, path: Path) -> None:
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    ImageDraw.Draw(img).rectangle((x * w, y * h, (x + rw) * w, (y + rh) * h), outline=(255, 40, 40), width=6)
    img.save(path, quality=90)


def counts(out: Path, labels) -> str:
    return "  ".join(f"{lb}:{len(list((out / lb).glob('*.jpg')))}" for lb in labels)


def main() -> int:
    ap = argparse.ArgumentParser(prog="atcs.collect", description="O'qitish uchun rasm yig'ish")
    ap.add_argument("--label", required=True, help="sinf nomi (config: vision.labels)")
    ap.add_argument("--source", choices=["usb", "csi"], help="standart: config vision.camera")
    ap.add_argument("--camera-index", type=int, help="standart: config vision.usb_index")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--note", default="", help="izoh, masalan: 'yangi partiya', 'qonli' (meta.csv ga yoziladi)")
    ap.add_argument("--auto", type=int, default=0, help="Enter kutmasdan shuncha rasm olish (bo'sh lenta uchun)")
    ap.add_argument("--interval", type=float, default=2.0, help="--auto da rasmlar orasidagi vaqt, s")
    ap.add_argument("--config")
    a = ap.parse_args()

    cfg = config_mod.load(a.config)
    labels = list(cfg.vision.labels) + ([] if "bosh" in cfg.vision.labels else ["bosh"])
    if a.label not in labels:
        raise SystemExit(f"Noma'lum sinf '{a.label}'. Mumkin: {', '.join(labels)}")
    roi = cfg.vision.roi
    out = Path(a.out)
    (out / a.label).mkdir(parents=True, exist_ok=True)

    from .cameras import open_camera
    source = a.source or cfg.vision.camera
    cam = open_camera(cfg.vision, source, a.camera_index)

    from PIL import Image
    meta_path = out / "meta.csv"
    new_meta = not meta_path.exists()
    last: Path | None = None
    first = cam.capture()
    save_roi_check(first, roi, out / "roi_tekshiruv.jpg")
    print(f"Kadr {first.shape[1]}x{first.shape[0]}, ROI tekshiruvi: {out / 'roi_tekshiruv.jpg'}")
    print(f"Sinf: {a.label}.  Enter = suratga olish, r = oxirgisini o'chirish, q = chiqish")
    rows: list[list] = []
    auto_left = a.auto
    try:
        while True:
            if a.auto:
                if auto_left <= 0:
                    break
                auto_left -= 1
                time.sleep(a.interval)
                cmd = ""
            else:
                cmd = input(f"[{counts(out, labels)}] > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "r":
                if last and last.exists():
                    last.unlink()
                    rows.pop()
                    print(f"o'chirildi: {last.name}")
                    last = None
                continue
            crop = crop_roi(cam.capture(), roi)
            name = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}.jpg"
            last = out / a.label / name
            Image.fromarray(np.ascontiguousarray(crop)).save(last, quality=95)
            rows.append([f"{a.label}/{name}", a.label, time.strftime("%Y-%m-%d %H:%M:%S"),
                         source, crop.shape[1], crop.shape[0], a.note])
            print(f"saqlandi: {last}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        cam.close()
        with open(meta_path, "a", newline="", encoding="utf-8") as mf:
            w = csv.writer(mf)
            if new_meta:
                w.writerow(["file", "label", "time", "source", "width", "height", "note"])
            w.writerows(rows)
    print(f"Jami: {counts(out, labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
