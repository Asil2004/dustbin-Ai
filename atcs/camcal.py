"""USB kamerani kalibrlash: ekspozitsiya va fokusni avtomatik topish.

  python3 -m atcs.camcal            # o'lchaydi va natijani ko'rsatadi
  python3 -m atcs.camcal --save     # natijani config/atcs.toml ga yozadi (usb_exposure, usb_focus)

Tayyorlash: kamera ish joyida (lentadan 115 mm balandda) tursin, yorug'lik ish rejimidagidek bo'lsin.
ROI ramkasi ostiga MATN BOSILGAN OQ QOG'OZ qo'ying: oq qog'oz ekspozitsiya uchun, harflar fokus uchun.

Ko'p webkameralar avto-fokus va avto-ekspozitsiya topgan qiymatni dasturga aytmaydi, shuning uchun ularni
bir marta shu yerda topib, sozlamaga yozib qo'yamiz.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from . import config as config_mod

WIN = sys.platform == "win32"


def sharpness(gray: np.ndarray) -> float:
    """Tiniqlik: Laplas operatori dispersiyasi. Tiniq qirralar ko'p bo'lsa, katta."""
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> tuple[float, float]:
    """(o'rtacha yorqinlik - mediana, oq bo'lib "kuyib" ketgan piksellar ulushi)."""
    return float(np.median(gray)), float((gray >= 250).mean())


def exposure_candidates(fps: float, mains_hz: int) -> list[float]:
    """Sinab ko'riladigan ekspozitsiyalar, qisqasidan uzuniga.
    Linux (V4L2): birlik 100 mks. 50 Hz tarmoqda lampalar 100 Hz da miltillaydi: ekspozitsiya 10 ms ga
    (100 birlik) karrali bo'lsa, kadrlar orasida yorqinlik sakramaydi.
    Windows (DSHOW): log2 soniya (-11 = 0,5 ms ... -3 = 125 ms)."""
    if WIN:
        return [float(e) for e in range(-11, -2)]
    frame_units = int(1e4 / max(1.0, fps))                     # bir kadr vaqti, 100 mks birlikda
    step = int(1e4 / (2 * mains_hz)) if mains_hz else 50        # 50 Hz -> 100, 60 Hz -> 83
    short = [5.0, 10.0, 20.0, 30.0, 50.0, 70.0]
    flicker_free = [float(k * step) for k in range(1, 40) if k * step <= frame_units]
    return sorted(set(short + flicker_free))


def pick_exposure(measured: list[tuple[float, float, float]], target: float) -> tuple[float, str]:
    """measured: [(ekspozitsiya, mediana, kuygan ulush)], qisqasidan uzuniga.
    Kuymagan (kuygan < 1 %) va mediana target+15 dan oshmagan eng uzun ekspozitsiya: shovqin eng kam."""
    ok = [m for m in measured if m[2] < 0.01 and m[1] <= target + 15]
    if not ok:
        return measured[0][0], "hatto eng qisqa ekspozitsiyada ham qog'oz kuyib ketyapti: yorug'likni kamaytiring"
    best = ok[-1]
    note = ""
    if best[1] < target - 60:
        note = "yorug'lik kam: tasvir qorong'i, LED yorug'likni kuchaytiring"
    return best[0], note


def focus_sweep(cam, values, settle_s: float, log) -> list[tuple[float, float]]:
    out = []
    for v in values:
        cam.set_focus(v)
        s = sharpness(cam.roi_gray(settle_s))
        out.append((float(v), s))
        log(f"  fokus {v:>6g}: tiniqlik {s:10.1f}")
    return out


def find_focus(cam, focus_max: int, settle_s: float, log=print) -> tuple[float, list[tuple[float, float]], str]:
    coarse_step = max(1, focus_max // 32)
    coarse = focus_sweep(cam, range(0, focus_max + 1, coarse_step), settle_s, log)
    best = max(coarse, key=lambda r: r[1])[0]
    fine_step = max(1, coarse_step // 4)
    lo, hi = max(0, int(best) - coarse_step), min(focus_max, int(best) + coarse_step)
    log("Aniqlashtirish:")
    fine = focus_sweep(cam, range(lo, hi + 1, fine_step), settle_s, log)
    allr = coarse + fine
    best_s = max(r[1] for r in allr)
    # tiniqlik "platosi" (chuqurlik bo'yicha tiniq zona) o'rtasi: buyum balandligi o'zgarsa ham tiniq qoladi
    near = sorted(v for v, s in fine if s >= 0.97 * best_s) or [max(allr, key=lambda r: r[1])[0]]
    best_v = near[len(near) // 2]
    worst = min(r[1] for r in allr)
    note = ""
    if best_s < 1.5 * max(worst, 1e-6):
        note = ("fokus o'zgarganda tiniqlik deyarli o'zgarmadi: kamera fokusni boshqarmaydi "
                "yoki ramka ostida harfli qog'oz yo'q")
    return best_v, allr, note


def save_to_config(path: Path, focus: float | None, exposure: float | None) -> None:
    text = path.read_text(encoding="utf-8")
    for key, val in (("usb_focus", focus), ("usb_exposure", exposure)):
        if val is None:
            continue
        num = f"{int(val)}" if float(val).is_integer() else f"{val:g}"
        text, n = re.subn(rf'(?m)^({key}\s*=\s*)("auto"|-?[\d.]+)', rf"\g<1>{num}", text)
        if n != 1:
            raise SystemExit(f"{path} da '{key} = ...' qatori topilmadi")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(prog="atcs.camcal", description="USB kamera ekspozitsiyasi va fokusini topish")
    ap.add_argument("--save", action="store_true", help="natijani config ga yozish")
    ap.add_argument("--camera-index", type=int)
    ap.add_argument("--target", type=float, default=190, help="oq qog'ozning maqsad yorqinligi (0-255)")
    ap.add_argument("--mains-hz", type=int, default=50, help="elektr tarmog'i chastotasi (O'zbekiston: 50)")
    ap.add_argument("--focus-max", type=int, default=255 if WIN else 1023,
                    help="kameraning eng katta fokus qiymati (v4l2-ctl --list-ctrls: focus_absolute max)")
    ap.add_argument("--settle", type=float, default=0.6, help="har sozlamadan keyin kutish, s")
    ap.add_argument("--skip-exposure", action="store_true")
    ap.add_argument("--skip-focus", action="store_true")
    ap.add_argument("--yes", action="store_true", help="Enter kutmasdan boshlash")
    ap.add_argument("--config")
    a = ap.parse_args()

    import cv2
    from .cameras import UsbCamera

    cfg_path = Path(a.config) if a.config else config_mod.DEFAULT_PATH
    cfg = config_mod.load(cfg_path)
    if cfg.vision.camera != "usb":
        raise SystemExit("Bu vosita faqat USB kamera uchun (vision.camera = \"usb\")")
    print(__doc__.split("\n\n")[2])
    if not a.yes:
        input("Tayyor bo'lsa Enter bosing...")
    cam = UsbCamera(cfg.vision, a.camera_index)
    exposure = focus = None
    notes = []
    try:
        if not a.skip_exposure:
            print("\n1/2 Ekspozitsiya (oq qog'oz bo'yicha):")
            measured = []
            for e in exposure_candidates(cfg.vision.usb_fps, a.mains_hz):
                cam.set_exposure(e)
                med, burnt = brightness(cam.roi_gray(a.settle))
                measured.append((e, med, burnt))
                print(f"  ekspozitsiya {e:>6g}: yorqinlik {med:5.0f}, kuygan {burnt:6.2%}")
                if burnt > 0.2:
                    break                                     # bundan uzuni faqat yomonroq
            exposure, note = pick_exposure(measured, a.target)
            cam.set_exposure(exposure)
            if note:
                notes.append(note)
            print(f"  -> ekspozitsiya = {exposure:g}")
        if not a.skip_focus:
            print("\n2/2 Fokus (harflar bo'yicha), taxminan 30 s:")
            focus, _, note = find_focus(cam, a.focus_max, a.settle)
            cam.set_focus(focus)
            if note:
                notes.append(note)
            print(f"  -> fokus = {focus:g}")
        out = Path("livetest")
        out.mkdir(exist_ok=True)
        sample = out / "camcal_natija.jpg"
        cv2.imwrite(str(sample), cv2.cvtColor(cam.capture(), cv2.COLOR_RGB2BGR))
    finally:
        cam.close()

    print("\nNatija:")
    if exposure is not None:
        print(f"  usb_exposure = {exposure:g}")
    if focus is not None:
        print(f"  usb_focus = {focus:g}")
    print(f"  namuna kadr: {sample}  (kompyuterda kattalashtirib, harflar tiniqligini tekshiring)")
    for n in notes:
        print(f"  DIQQAT: {n}")
    if a.save:
        save_to_config(cfg_path, focus, exposure)
        print(f"  {cfg_path} ga yozildi. Endi: python3 -m atcs.livetest --no-model")
    else:
        print("  Config ga yozish uchun: python3 -m atcs.camcal --save")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
