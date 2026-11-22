"""Kamerada jonli ko'rish va modelni sinash (noutbukda yoki Pi'da). Lenta va motorlar kerak emas.

  python -m atcs.livetest                      # config dagi kamera; model bo'lsa aniqlash ham
  python -m atcs.livetest --no-model           # faqat kamera: tasvir, ROI, buyumni sezish
  python -m atcs.livetest --web                # oyna o'rniga brauzerda: http://<pi-ip>:8000
  python -m atcs.livetest --model natija/atcs.tflite --camera-index 1

Model fayli bo'lmasa, avtomatik ravishda faqat kamera rejimi ishlaydi.
Monitorsiz Pi'da (grafik muhit yo'q) avtomatik ravishda brauzer rejimi yoqiladi.

Ekranda: ROI ramkasi, buyumni sezish (fon bilan farq), fps, aniqlash vaqti, model qarori va 5 sinf ehtimoli.
Oyna tugmalari:  q / Esc - chiqish,  f - ROI / to'liq kadr,  b - fonni yangilash,  s - kadrni livetest/ ga saqlash
Brauzerda xuddi shu amallar sahifadagi tugmalar bilan.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import config as config_mod

FULL_ROI = (0.0, 0.0, 1.0, 1.0)
VIEW_WIDTH = 960          # ekranga chiqariladigan kadr eni (Pi 3 uchun yengil)


def decide(probs: np.ndarray, labels, threshold: float, needle_priority: float = 1.0) -> tuple[str, int | None, float]:
    """Stol nima qiladi: (matn, bo'lim yoki None, ishonch). Controller bilan bir xil qoida (vision.decide)."""
    from .vision import decide as table_decide
    d = table_decide(probs, list(labels), threshold, needle_priority)
    if d.kind == "empty":
        return f"bo'sh lenta ({d.confidence:.0%})", None, d.confidence
    if d.kind == "unknown":
        why = " igna?" if d.reason else ""
        return f"OPERATOR ({d.label}?{why} {d.confidence:.0%})", None, d.confidence
    if d.reason == "igna_ustuvor":
        return f"igna ehtimoli {d.confidence:.0%} -> 1-bo'lim (ustuvor)", d.bin, d.confidence
    return f"{d.label} -> {d.bin}-bo'lim ({d.confidence:.0%})", d.bin, d.confidence


def annotate(bgr: np.ndarray, probs, labels, threshold: float, roi, fps: float = 0.0,
             presence: tuple[float, float] | None = None, infer_ms: float | None = None,
             info: str = "", needle_priority: float = 1.0) -> np.ndarray:
    """Kadr ustiga ROI, holat va ehtimollarni chizish (BGR, joyida).
    probs = None bo'lsa (modelsiz rejim), tepada buyumni sezish holati chiqadi.
    presence = (o'zgargan piksellar ulushi, chegara), 0..1."""
    import cv2
    h, w = bgr.shape[:2]
    x, y, rw, rh = roi
    font = cv2.FONT_HERSHEY_SIMPLEX
    if probs is not None:
        text, bin_no, _ = decide(probs, labels, threshold, needle_priority)
        color = (0, 200, 255) if bin_no is None else (60, 220, 60)          # sariq = operator, yashil = bo'lim
    elif presence is not None:
        score, thr = presence
        seen = score >= thr
        text = (f"BUYUM BOR  (o'zgargan {score:.2%} >= {thr:.2%})" if seen
                else f"bo'sh  (o'zgargan {score:.2%} / {thr:.2%})")
        color = (60, 220, 60) if seen else (200, 200, 200)
    else:
        text, color = "kamera", (200, 200, 200)
    cv2.rectangle(bgr, (int(x * w), int(y * h)), (int((x + rw) * w) - 1, int((y + rh) * h) - 1), color, 3)
    cv2.rectangle(bgr, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.putText(bgr, text, (10, 31), font, 0.9, color, 2, cv2.LINE_AA)
    right = [f"{fps:.0f} fps"] if fps else []
    if infer_ms is not None:
        right.append(f"model {infer_ms:.0f} ms")
    if probs is not None and presence is not None:
        right.append(f"o'zgargan {presence[0]:.2%}")
    if right:
        s = "  ".join(right)
        (tw, _), _ = cv2.getTextSize(s, font, 0.6, 1)
        cv2.putText(bgr, s, (w - tw - 10, 30), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    if info:
        cv2.rectangle(bgr, (0, h - 24), (w, h), (0, 0, 0), -1)
        cv2.putText(bgr, info, (8, h - 7), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    if probs is not None:
        top = int(np.argmax(probs))
        base = h - (30 if info else 6)
        for i, (lb, p) in enumerate(zip(labels, probs)):
            y0 = base - 22 * (len(labels) - 1 - i)
            cv2.rectangle(bgr, (8, y0 - 14), (8 + 300, y0 + 4), (40, 40, 40), -1)
            cv2.rectangle(bgr, (8, y0 - 14), (8 + int(300 * float(p)), y0 + 4),
                          color if i == top else (120, 120, 120), -1)
            cv2.putText(bgr, f"{i + 1} {lb} {float(p):.2f}", (14, y0), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def has_display() -> bool:
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class WebView:
    """Kadrlarni brauzerga MJPEG oqimi qilib beradi. Sahifadagi tugmalar buyruq navbatiga tushadi."""

    PAGE = """<!doctype html><html lang="uz"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ATCS kamera</title>
<style>body{margin:0;background:#111;color:#ddd;font:15px system-ui,sans-serif}
img{display:block;width:100%;max-width:1280px;margin:0 auto}
.bar{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;padding:10px}
button{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:10px 16px;font-size:15px}
button:active{background:#555}</style></head><body>
<img src="/stream" alt="kamera">
<div class="bar"><button onclick="cmd('b')">Fonni yangilash</button>
<button onclick="cmd('f')">ROI / to'liq kadr</button><button onclick="cmd('s')">Kadrni saqlash</button></div>
<script>function cmd(k){fetch('/cmd?k='+k,{method:'POST'})}</script></body></html>"""

    def __init__(self, port: int):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self._jpeg = b""
        self._cond = threading.Condition()
        self.commands: list[str] = []
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/":
                    body = view.PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            with view._cond:
                                view._cond.wait_for(lambda: view._jpeg is not last, timeout=2)
                                jpg = last = view._jpeg
                            if not jpg:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path.startswith("/cmd?k="):
                    k = self.path[len("/cmd?k="):][:1]
                    if k in ("b", "f", "s"):
                        with view._cond:
                            view.commands.append(k)
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_error(404)

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="livetest-web", daemon=True).start()

    def show(self, bgr: np.ndarray) -> str | None:
        import cv2
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with self._cond:
            if ok:
                self._jpeg = buf.tobytes()
                self._cond.notify_all()
            return self.commands.pop(0) if self.commands else None

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def local_ips() -> list[str]:
    import socket
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))          # paket yuborilmaydi, faqat chiquvchi interfeys aniqlanadi
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips) or ["<pi-ip>"]


def main() -> int:
    ap = argparse.ArgumentParser(prog="atcs.livetest", description="Kamerada jonli ko'rish va modelni sinash")
    ap.add_argument("--model", help="TFLite model (standart: config vision.model)")
    ap.add_argument("--no-model", action="store_true", help="faqat kamera, aniqlashsiz")
    ap.add_argument("--source", choices=["usb", "csi"], help="standart: config vision.camera")
    ap.add_argument("--camera-index", type=int, help="standart: config vision.usb_index")
    ap.add_argument("--full", action="store_true", help="ROI o'rniga butun kadrni tahlil qilish")
    ap.add_argument("--threshold", type=float, help="standart: sorting.min_confidence")
    ap.add_argument("--web", action="store_true", help="oyna o'rniga brauzerda ko'rsatish")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config")
    a = ap.parse_args()

    import cv2
    from .cameras import open_camera
    from .vision import crop_roi

    cfg = config_mod.load(a.config)
    labels = list(cfg.vision.labels)
    threshold = a.threshold if a.threshold is not None else cfg.sorting.min_confidence
    clf = None
    if not a.no_model:
        model = Path(a.model or cfg.vision.model)
        if model.exists():
            from .vision import TfliteClassifier
            clf = TfliteClassifier(model, cfg.vision.input_size, roi=None)
            print(f"Model: {model}, chegara {threshold:.2f}")
        elif a.model:
            raise SystemExit(f"Model topilmadi: {model}")
        else:
            print(f"Model yo'q ({model}): faqat kamera rejimi")
    roi = FULL_ROI if a.full else tuple(cfg.vision.roi)

    cam = open_camera(cfg.vision, a.source, a.camera_index)
    settings = getattr(cam, "settings", "")
    web = a.web or not has_display()
    view = None
    if web:
        view = WebView(a.port)
        for ip in local_ips():
            print(f"Brauzerda oching: http://{ip}:{a.port}   (to'xtatish: Ctrl+C)")
    else:
        print("q = chiqish, f = ROI/to'liq kadr, b = fonni yangilash, s = saqlash")
    win = "ATCS livetest"
    out = Path("livetest")
    fps, t_prev = 0.0, time.perf_counter()
    try:
        while True:
            rgb = cam.capture()
            score = cam.presence_score() if hasattr(cam, "presence_score") else None
            probs, infer_ms = None, None
            if clf is not None:
                t0 = time.perf_counter()
                probs = clf.probs(rgb if roi == FULL_ROI else crop_roi(rgb, roi))
                infer_ms = (time.perf_counter() - t0) * 1000
            now = time.perf_counter()
            dt = max(1e-3, now - t_prev)
            fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
            t_prev = now
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if bgr.shape[1] > VIEW_WIDTH:
                bgr = cv2.resize(bgr, (VIEW_WIDTH, int(bgr.shape[0] * VIEW_WIDTH / bgr.shape[1])),
                                 interpolation=cv2.INTER_AREA)
            info = f"{rgb.shape[1]}x{rgb.shape[0]}  {settings}".strip()
            annotate(bgr, probs, labels, threshold, roi, fps,
                     presence=None if score is None else (score, cfg.vision.presence_min_fraction),
                     infer_ms=infer_ms, info=info, needle_priority=cfg.sorting.needle_priority)
            if view is not None:
                key = view.show(bgr)
            else:
                cv2.imshow(win, bgr)
                k = cv2.waitKey(1) & 0xFF
                key = None if k == 255 else chr(k)
                if key in ("q", "\x1b") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if key == "f":
                roi = tuple(cfg.vision.roi) if roi == FULL_ROI else FULL_ROI
            elif key == "b":
                cam.set_background()
                print("fon yangilandi")
            elif key == "s":
                out.mkdir(exist_ok=True)
                p = out / (time.strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                print(f"saqlandi: {p}" + (f"  [{decide(probs, labels, threshold, cfg.sorting.needle_priority)[0]}]" if probs is not None else ""))
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if view is not None:
            view.close()
        else:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
