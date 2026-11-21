"""Kameralar: USB webkamera (OpenCV) va Raspberry Pi kamerasi (CSI, picamera2).

Rasm yig'ish (atcs.collect), jonli sinash (atcs.livetest) va stol ishi bitta open_camera() orqali
kamerani ochadi. Shunda o'lcham, fokus va ekspozitsiya hamma joyda bir xil bo'ladi.
Ikkala kamera ham bir xil interfeysga ega: presence(), set_background(), capture() -> RGB, close().
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np

PRESENCE_SIZE = (160, 120)


class UsbCamera:
    """USB webkamera. Alohida oqim kadrlarni to'xtovsiz o'qiydi, shuning uchun presence() tez ishlaydi
    va capture() har doim chaqiruvdan keyin olingan yangi kadrni beradi (bufferdagi eskisini emas)."""

    def __init__(self, vision_cfg, index: int | None = None, verbose: bool = True):
        import cv2
        self._cv2 = cv2
        self._log = print if verbose else (lambda *a, **k: None)
        idx = vision_cfg.usb_index if index is None else index
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        self._cap = cv2.VideoCapture(idx, backend)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(idx)
        if not self._cap.isOpened():
            raise RuntimeError(f"USB kamera {idx} ochilmadi")
        w, h = (int(v) for v in vision_cfg.capture_size)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # USB 2.0 da 1080p uchun kerak
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS, float(vision_cfg.usb_fps))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._cap.release()
            raise RuntimeError(f"USB kamera {idx} kadr bermadi")
        if (frame.shape[1], frame.shape[0]) != (w, h):
            self._cap.release()
            raise RuntimeError(f"Kamera {frame.shape[1]}x{frame.shape[0]} berdi, sozlamada {w}x{h} "
                               f"(vision.capture_size). Kamera qo'llaydigan o'lchamni yozing; ROI ga ta'sir qiladi.")

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = frame
        self._stamp = time.monotonic()
        self._error: str | None = None
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="usb-camera", daemon=True)
        self._thread.start()

        self._lock_settings(vision_cfg)
        self._threshold = vision_cfg.presence_threshold
        self._bg = None

    # -- kadr o'qish oqimi --
    def _reader(self):
        fails = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fails += 1
                if fails >= 50:
                    with self._cond:
                        self._error = "USB kamera kadr bermay qo'ydi (kabel yoki quvvat?)"
                        self._cond.notify_all()
                    return
                time.sleep(0.02)
                continue
            fails = 0
            with self._cond:
                self._frame, self._stamp = frame, time.monotonic()
                self._cond.notify_all()

    def _latest(self, newer_than: float | None = None, timeout: float = 1.0) -> np.ndarray:
        with self._cond:
            if newer_than is not None:
                end = time.monotonic() + timeout
                while self._stamp <= newer_than and self._error is None:
                    left = end - time.monotonic()
                    if left <= 0:
                        raise RuntimeError("USB kamera yangi kadr bermadi")
                    self._cond.wait(left)
            if self._error:
                raise RuntimeError(self._error)
            return self._frame

    # -- fokus, ekspozitsiya va oq balansni qotirish (drayverga bog'liq, imkon qadar) --
    def _lock_settings(self, v):
        cv2, cap = self._cv2, self._cap
        win = sys.platform == "win32"
        time.sleep(float(v.usb_warmup_s))                 # avto-fokus va avto-ekspozitsiya joyiga tushsin
        report = []
        if v.usb_focus >= 0:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            cap.set(cv2.CAP_PROP_FOCUS, float(v.usb_focus))
            report.append(f"fokus={v.usb_focus:g} (qo'lda)")
        else:
            focus = cap.get(cv2.CAP_PROP_FOCUS)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            if focus >= 0:
                cap.set(cv2.CAP_PROP_FOCUS, focus)
            report.append(f"fokus={focus:g} (avtodan qotirildi)")
        if v.lock_exposure:
            exp = cap.get(cv2.CAP_PROP_EXPOSURE) if v.usb_exposure < 0 else float(v.usb_exposure)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if win else 1)       # qo'lda rejim (DSHOW / V4L2)
            cap.set(cv2.CAP_PROP_EXPOSURE, exp)
            wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            if wb > 0:
                cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
            report.append(f"ekspozitsiya={exp:g}, oq balans={wb:g} (qotirildi)")
        time.sleep(0.3)
        self.settings = ", ".join(report)
        self._log(f"USB kamera {self._frame.shape[1]}x{self._frame.shape[0]}: {self.settings}")

    # -- Camera interfeysi --
    def _gray(self) -> np.ndarray:
        cv2 = self._cv2
        small = cv2.resize(self._latest(), PRESENCE_SIZE, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)

    def set_background(self):
        self._bg = self._gray()

    def presence(self) -> bool:
        if self._bg is None:
            self.set_background()
            return False
        return float(np.abs(self._gray() - self._bg).mean()) > self._threshold

    def capture(self) -> np.ndarray:
        """Chaqiruvdan keyin boshlangan kadr, RGB."""
        bgr = self._latest(newer_than=time.monotonic())
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


def open_camera(vision_cfg, source: str | None = None, index: int | None = None, verbose: bool = True):
    """source: "usb" | "csi" | None (= vision.camera)."""
    kind = source or vision_cfg.camera
    if kind == "usb":
        return UsbCamera(vision_cfg, index, verbose)
    if kind == "csi":
        from .hal.rpi import PiCamera
        return PiCamera(vision_cfg)
    raise ValueError(f"Noma'lum kamera turi: {kind} (usb yoki csi)")
