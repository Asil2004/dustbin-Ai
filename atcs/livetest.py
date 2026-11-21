"""Modelni kamerada jonli sinash (noutbuk yoki stol). Stol, lenta va motorlar kerak emas.

  python -m atcs.livetest                      # config dagi kamera (vision.camera), models/atcs.tflite
  python -m atcs.livetest --model natija/atcs.tflite --camera-index 1
Kamera ish rejimidagi sozlamalar bilan ochiladi (o'lcham, qotirilgan fokus va ekspozitsiya).

Oynada: ROI ramkasi, model qarori (bo'lim yoki OPERATOR) va 5 sinf ehtimoli.
  q / Esc - chiqish,  f - ROI va to'liq kadr orasida almashtirish,  s - kadrni livetest/ ga saqlash
Kompyuterda kerak: pip install ai-edge-litert opencv-python
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import config as config_mod

FULL_ROI = (0.0, 0.0, 1.0, 1.0)


def decide(probs: np.ndarray, labels, threshold: float) -> tuple[str, int | None, float]:
    """Stol nima qiladi: (matn, bo'lim yoki None = operator, ishonch). Controller bilan bir xil qoida."""
    k = int(np.argmax(probs))
    conf = float(probs[k])
    if conf < threshold:
        return f"OPERATOR ({labels[k]}? {conf:.0%})", None, conf
    return f"{labels[k]} -> {k + 1}-bo'lim ({conf:.0%})", k + 1, conf


def annotate(bgr: np.ndarray, probs: np.ndarray, labels, threshold: float, roi, fps: float = 0.0) -> np.ndarray:
    """Kadr ustiga ROI, qaror va ehtimollar ustunlarini chizish (BGR, joyida)."""
    import cv2
    h, w = bgr.shape[:2]
    x, y, rw, rh = roi
    text, bin_no, _ = decide(probs, labels, threshold)
    color = (0, 200, 255) if bin_no is None else (60, 220, 60)            # sariq = operator, yashil = bo'lim
    cv2.rectangle(bgr, (int(x * w), int(y * h)), (int((x + rw) * w) - 1, int((y + rh) * h) - 1), color, 3)
    cv2.rectangle(bgr, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(bgr, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    if fps:
        cv2.putText(bgr, f"{fps:.0f} fps", (w - 110, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    top = int(np.argmax(probs))
    for i, (lb, p) in enumerate(zip(labels, probs)):
        y0 = h - 22 * (len(labels) - i) - 6
        cv2.rectangle(bgr, (8, y0 - 14), (8 + 300, y0 + 4), (40, 40, 40), -1)
        cv2.rectangle(bgr, (8, y0 - 14), (8 + int(300 * float(p)), y0 + 4),
                      color if i == top else (120, 120, 120), -1)
        cv2.putText(bgr, f"{i + 1} {lb} {float(p):.2f}", (14, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def main() -> int:
    ap = argparse.ArgumentParser(prog="atcs.livetest", description="Modelni kamerada jonli sinash")
    ap.add_argument("--model", help="TFLite model (standart: config vision.model)")
    ap.add_argument("--source", choices=["usb", "csi"], help="standart: config vision.camera")
    ap.add_argument("--camera-index", type=int, help="standart: config vision.usb_index")
    ap.add_argument("--full", action="store_true", help="ROI o'rniga butun kadrni tahlil qilish")
    ap.add_argument("--threshold", type=float, help="standart: sorting.min_confidence")
    ap.add_argument("--config")
    a = ap.parse_args()

    import cv2
    from .vision import TfliteClassifier

    cfg = config_mod.load(a.config)
    labels = list(cfg.vision.labels)
    threshold = a.threshold if a.threshold is not None else cfg.sorting.min_confidence
    model = Path(a.model or cfg.vision.model)
    if not model.exists():
        raise SystemExit(f"Model topilmadi: {model}. Avval training/train.py bilan o'qiting.")
    clf = TfliteClassifier(model, cfg.vision.input_size, roi=None)
    roi = FULL_ROI if a.full else tuple(cfg.vision.roi)

    from .cameras import open_camera
    cam = open_camera(cfg.vision, a.source, a.camera_index)

    out = Path("livetest")
    win = "ATCS livetest"
    print(f"Model: {model}, chegara {threshold:.2f}.  q = chiqish, f = ROI/to'liq kadr, s = saqlash")
    fps, t_prev = 0.0, time.perf_counter()
    try:
        while True:
            rgb = cam.capture()
            probs = clf.probs(rgb if roi == FULL_ROI else _crop(rgb, roi))
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev) if fps else 1 / max(1e-3, now - t_prev)
            t_prev = now
            bgr = annotate(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), probs, labels, threshold, roi, fps)
            if bgr.shape[1] > 1280:                                       # Pi kamerasining katta kadri
                bgr = cv2.resize(bgr, (1280, int(bgr.shape[0] * 1280 / bgr.shape[1])))
            cv2.imshow(win, bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("f"):
                roi = tuple(cfg.vision.roi) if roi == FULL_ROI else FULL_ROI
            if key == ord("s"):
                out.mkdir(exist_ok=True)
                p = out / (time.strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(str(p), bgr)
                print(f"saqlandi: {p}  [{decide(probs, labels, threshold)[0]}]")
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        cv2.destroyAllWindows()
    return 0


def _crop(frame, roi):
    from .vision import crop_roi
    return crop_roi(frame, roi)


if __name__ == "__main__":
    raise SystemExit(main())
