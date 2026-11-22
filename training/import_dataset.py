"""Tashqi datasetni (Roboflow, Kaggle va h.k.) ATCS sinflariga o'tkazish.

  python training/import_dataset.py --src ~/Downloads/medical-waste --name rf_medwaste \\
      --map "Gloves=qolqop_shlang,Syringes=igna,Masks=bint_paxta"

Natija: dataset_ext/<sinf>/<name>_*.jpg, dataset_ext/<name>_manifest.csv va dataset_ext/<name>_ATTRIBUTION.txt.
Keyin o'qitishda: python train.py --data dataset --extra dataset_ext --test test

Qo'llab-quvvatlanadigan formatlar (avtomatik aniqlanadi):
  - papka: <har qanday>/<sinf>/*.jpg (Roboflow "Folder Structure", Kaggle sinf papkalari);
  - YOLO: images/*.jpg + labels/*.txt va data.yaml (names). Har bir ramka padding bilan kesib olinadi.

Qoidalar:
  - --map da yo'q sinflar o'tkazib yuboriladi va hisobotda ko'rsatiladi. Shubhali sinfni (masalan "Medicines")
    qo'shmang: noto'g'ri yorliq modelni buzadi.
  - Roboflow bitta rasmdan bir nechta augmentatsiyalangan nusxa yasaydi ("x_jpg.rf.<hash>.jpg"). Ular bitta
    guruh hisoblanadi va standart bo'yicha guruhdan bittasi olinadi (train.py augmentatsiyani o'zi qiladi).
  - Bir xil rasmlar (8x8 o'rtacha xesh bo'yicha) bir marta olinadi.
  - Tashqi rasmlar boshqa kamerada olingan: train.py ularni faqat o'qitishga qo'shadi, tekshirish va test
    to'plamlariga qo'shmaydi. Shunda natija stol kamerasidagi haqiqiy aniqlikni ko'rsatadi.
  - Litsenziya: dataset fayllaridagi README/LICENSE ATTRIBUTION.txt ga ko'chiriladi. CC BY talab qiladi:
    manbani ko'rsating.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ATCS_LABELS = ["igna", "qolqop_shlang", "shisha", "plastik", "bint_paxta", "bosh"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = {"train", "valid", "val", "validation", "test", "images", "labels"}
LICENSE_HINTS = ("readme", "license", "licence", "attribution")


def parse_map(text: str) -> dict[str, str]:
    """"Gloves=qolqop_shlang, Syringes=igna" -> {"gloves": "qolqop_shlang", ...} (kichik harflarda)."""
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        if "=" not in part:
            raise SystemExit(f"--map elementi noto'g'ri: '{part}' (kerak: Asl=atcs_sinf)")
        src, dst = (x.strip() for x in part.split("=", 1))
        if dst not in ATCS_LABELS:
            raise SystemExit(f"'{dst}' ATCS sinfi emas. Mumkin: {', '.join(ATCS_LABELS)}")
        out[src.lower()] = dst
    return out


def group_id(path: Path) -> str:
    """Bir manba rasmning nusxalari bitta guruh: Roboflow "<asl>_jpg.rf.<hash>.jpg" -> "<asl>"."""
    stem = path.name
    m = re.match(r"(.+?)[._-](jpe?g|png|bmp|webp)\.rf\.[0-9a-f]+", stem, re.I)
    return (m.group(1) if m else path.stem).lower()


def ahash(img) -> int:
    """8x8 o'rtacha xesh: bir xil (qayta saqlangan, o'lchami o'zgargan) rasmlarni topish uchun."""
    from PIL import Image
    a = np.asarray(img.convert("L").resize((8, 8), Image.BILINEAR), dtype=np.float32)
    bits = (a > a.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def yolo_names(root: Path) -> list[str]:
    """data.yaml dagi names: ['a', 'b'] yoki ro'yxat ko'rinishi (yaml kutubxonasiz)."""
    for y in list(root.rglob("data.yaml")) + list(root.rglob("*.yaml")):
        text = y.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"^names:\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
        m = re.search(r"^names:\s*\n((?:\s+(?:-|\d+:).*\n?)+)", text, re.M)
        if m:
            return [re.sub(r"^\s*(?:-|\d+:)\s*", "", ln).strip().strip("'\"")
                    for ln in m.group(1).splitlines() if ln.strip()]
    raise SystemExit(f"{root} da YOLO sinf nomlari (data.yaml: names) topilmadi")


def is_yolo(root: Path) -> bool:
    return any(p.parent.name == "labels" for p in root.rglob("*.txt"))


def class_of_folder_image(path: Path, root: Path) -> str | None:
    """Rasmning sinf papkasi: train/valid/test kabi bo'linish papkalari tashlab o'tiladi."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        if part.lower() not in SPLIT_DIRS:
            return part
    return None


def iter_folder(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in EXTS:
            cls = class_of_folder_image(p, root)
            if cls:
                yield p, cls, None


def iter_yolo(root: Path, names: list[str], pad: float):
    from PIL import Image
    for lab in sorted(root.rglob("*.txt")):
        if lab.parent.name != "labels":
            continue
        img_dir = lab.parent.parent / "images"
        img = next((img_dir / (lab.stem + e) for e in EXTS if (img_dir / (lab.stem + e)).exists()), None)
        if img is None:
            continue
        with Image.open(img) as im:
            w, h = im.size
        for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            k = int(float(parts[0]))
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
            px, py = (x1 - x0) * pad, (y1 - y0) * pad
            box = (max(0, int(x0 - px)), max(0, int(y0 - py)), min(w, int(x1 + px)), min(h, int(y1 + py)))
            yield img, names[k] if k < len(names) else str(k), box


def copy_attribution(root: Path, dst: Path, name: str, src_note: str) -> None:
    texts = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".txt", ".md", "") and p.stem.lower().startswith(LICENSE_HINTS):
            texts.append(f"===== {p.relative_to(root)}\n{p.read_text(encoding='utf-8', errors='ignore')[:20000]}")
    head = (f"Dataset: {name}\nManba: {src_note or root}\n"
            "Bu rasmlar tashqi datasetdan olingan. Litsenziya shartlariga rioya qiling (CC BY: manbani ko'rsating).\n\n")
    dst.write_text(head + ("\n\n".join(texts) if texts else "LITSENZIYA FAYLI TOPILMADI: manbadan tekshiring.\n"),
                   encoding="utf-8")


def run(src: Path, out: Path, name: str, mapping: dict[str, str], fmt: str = "auto", pad: float = 0.10,
        min_size: int = 32, one_per_group: bool = True, source_note: str = "", log=print) -> dict:
    from PIL import Image
    if fmt == "auto":
        fmt = "yolo" if is_yolo(src) else "folder"
    items = iter_yolo(src, yolo_names(src), pad) if fmt == "yolo" else iter_folder(src)
    out.mkdir(parents=True, exist_ok=True)
    seen_groups: set[tuple[str, str]] = set()
    seen_hash: dict[str, set[int]] = {}
    kept, skipped_cls, rows = Counter(), Counter(), []
    dup = small = bad = 0
    for path, cls, box in items:
        dst_label = mapping.get(cls.lower())
        if dst_label is None:
            skipped_cls[cls] += 1
            continue
        g = group_id(path)
        if one_per_group and box is None and (dst_label, g) in seen_groups:
            dup += 1
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if box is not None:
                    im = im.crop(box)
                if min(im.size) < min_size:
                    small += 1
                    continue
                hsh = ahash(im)
                if hsh in seen_hash.setdefault(dst_label, set()):
                    dup += 1
                    continue
                seen_hash[dst_label].add(hsh)
                seen_groups.add((dst_label, g))
                (out / dst_label).mkdir(exist_ok=True)
                tag = hashlib.sha1(f"{path}|{box}".encode()).hexdigest()[:10]
                dst = out / dst_label / f"{name}_{tag}.jpg"
                im.save(dst, quality=92)
        except (OSError, ValueError):
            bad += 1
            continue
        kept[dst_label] += 1
        rows.append([str(dst.relative_to(out)), dst_label, cls, str(path.relative_to(src)), g,
                     "" if box is None else ",".join(map(str, box))])
    with open(out / f"{name}_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "atcs_label", "source_class", "source_file", "group", "box"])
        w.writerows(rows)
    copy_attribution(src, out / f"{name}_ATTRIBUTION.txt", name, source_note)
    report = {"format": fmt, "kept": dict(kept), "skipped_classes": dict(skipped_cls),
              "duplicates": dup, "too_small": small, "unreadable": bad}
    log(f"Format: {fmt}. Olindi: {dict(kept)}")
    if skipped_cls:
        log(f"O'tkazib yuborilgan sinflar (--map da yo'q): {dict(skipped_cls)}")
    log(f"Takror: {dup}, juda kichik: {small}, o'qilmadi: {bad}. Manifest: {out / (name + '_manifest.csv')}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Tashqi datasetni ATCS sinflariga o'tkazish")
    ap.add_argument("--src", required=True, help="ochilgan (unzip) dataset papkasi")
    ap.add_argument("--name", required=True, help="qisqa nom (fayl nomi boshi), masalan rf_medwaste")
    ap.add_argument("--map", required=True, help='"Asl=atcs_sinf,..." masalan "Gloves=qolqop_shlang,Syringes=igna"')
    ap.add_argument("--out", default="dataset_ext")
    ap.add_argument("--format", choices=["auto", "folder", "yolo"], default="auto")
    ap.add_argument("--pad", type=float, default=0.10, help="YOLO ramkasini kengaytirish ulushi")
    ap.add_argument("--min-size", type=int, default=32)
    ap.add_argument("--all-copies", action="store_true", help="Roboflow nusxalarining hammasini olish")
    ap.add_argument("--source", default="", help="manba URL (ATTRIBUTION.txt ga yoziladi)")
    a = ap.parse_args()
    src = Path(a.src).expanduser()
    if not src.is_dir():
        raise SystemExit(f"{src} papkasi yo'q")
    rep = run(src, Path(a.out), a.name, parse_map(a.map), a.format, a.pad, a.min_size,
              not a.all_copies, a.source)
    return 0 if sum(rep["kept"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
