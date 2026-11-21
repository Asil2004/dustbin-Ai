"""USB kamera: soxta OpenCV kamerasi bilan (haqiqiy kamera yoqilmaydi)."""
import copy
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from atcs import config
from atcs.cameras import PresenceDetector, UsbCamera, open_camera


class FakeCapture:
    """cv2.VideoCapture o'rnida: berilgan o'lchamda kadr qaytaradi, kadrlarni raqamlaydi."""
    instances = []

    def __init__(self, *args, native=(1920, 1080)):
        self.props = {}
        self.native = native
        self.scene = np.full((native[1], native[0], 3), 60, np.uint8)      # BGR
        self.n = 0
        self.lock = threading.Lock()
        self.released = False
        FakeCapture.instances.append(self)

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FOCUS: 37.0, cv2.CAP_PROP_EXPOSURE: -6.0,
                cv2.CAP_PROP_WB_TEMPERATURE: 4600.0}.get(prop, -1.0)

    def read(self):
        time.sleep(1 / 60)
        with self.lock:
            self.n += 1
            f = self.scene.copy()
        f[0, 0, 0] = self.n % 256          # kadr raqami
        return True, f

    def release(self):
        self.released = True


def vision(**kw):
    v = copy.copy(config.load().vision)
    v.usb_warmup_s = 0.0
    for k, val in kw.items():
        setattr(v, k, val)
    return v


class UsbCameraTest(unittest.TestCase):
    def setUp(self):
        FakeCapture.instances.clear()

    def open(self, native=(1920, 1080), **kw):
        with mock.patch("cv2.VideoCapture", lambda *a: FakeCapture(*a, native=native)):
            return UsbCamera(vision(**kw), verbose=False)

    def test_capture_returns_fresh_rgb_frame(self):
        cam = self.open()
        try:
            fake = FakeCapture.instances[-1]
            with fake.lock:
                fake.scene[:] = (255, 0, 0)          # BGR ko'k
                n_before = fake.n
            img = cam.capture()
            self.assertEqual(img.shape, (1080, 1920, 3))
            self.assertGreater(fake.n, n_before, "chaqiruvdan keyin o'qilgan kadr kerak")
            self.assertEqual(tuple(img[10, 10]), (0, 0, 255), "RGB tartibida qaytishi kerak")
        finally:
            cam.close()
        self.assertTrue(fake.released)

    def test_presence_detects_item(self):
        cam = self.open()
        try:
            fake = FakeCapture.instances[-1]
            self.assertFalse(cam.presence())          # birinchi chaqiruv fonni oladi
            cam.capture()
            self.assertFalse(cam.presence())
            with fake.lock:
                fake.scene[300:800, 700:1300] = (230, 230, 230)
            cam.capture()                              # yangi sahna oqimga yetib kelsin
            self.assertTrue(cam.presence())
            cam.set_background()
            self.assertFalse(cam.presence())
        finally:
            cam.close()

    def test_presence_ignores_change_outside_roi(self):
        cam = self.open()
        try:
            fake = FakeCapture.instances[-1]
            cam.set_background()
            with fake.lock:
                fake.scene[:, :200] = (250, 250, 250)        # ROI dan tashqarida (roi x = 0.30 dan boshlanadi)
            cam.capture()
            self.assertFalse(cam.presence())
        finally:
            cam.close()

    def test_wrong_resolution_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "capture_size"):
            self.open(native=(1280, 720))
        self.assertTrue(FakeCapture.instances[-1].released)

    def test_focus_and_exposure_locked(self):
        cam = self.open()
        cam.close()
        p = FakeCapture.instances[-1].props
        self.assertEqual(p[cv2.CAP_PROP_AUTOFOCUS], 0)
        self.assertEqual(p[cv2.CAP_PROP_FOCUS], 37.0)      # avto-fokus topgani qotirildi
        self.assertEqual(p[cv2.CAP_PROP_EXPOSURE], -6.0)
        self.assertEqual(p[cv2.CAP_PROP_AUTO_WB], 0)
        self.assertEqual(p[cv2.CAP_PROP_FRAME_WIDTH], 1920)

    def test_manual_focus(self):
        cam = self.open(usb_focus=120)
        cam.close()
        self.assertEqual(FakeCapture.instances[-1].props[cv2.CAP_PROP_FOCUS], 120.0)

    def test_open_camera_rejects_unknown(self):
        with self.assertRaises(ValueError):
            open_camera(vision(), "ip")


class PresenceDetectorTest(unittest.TestCase):
    """1080p kadr, ROI ichida ingichka igna: o'rtacha farq sezmaydi, piksel ulushi sezadi."""

    def setUp(self):
        self.v = vision()
        self.det = PresenceDetector(self.v)
        from atcs.vision import crop_roi
        self.crop = lambda g: self.det.shrink(crop_roi(g, self.det.roi))
        self.rng = np.random.default_rng(0)
        self.bg = np.full((1080, 1920), 40, np.uint8)

    def noisy(self, img):
        return np.clip(img.astype(np.int16) + self.rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)

    def test_thin_needle_detected(self):
        self.det.set_background(self.crop(self.noisy(self.bg)))
        g = self.bg.copy()
        g[500:504, 900:1060] = 200                      # ~0,4 mm x 16 mm igna (0,1 mm/piksel)
        score = self.det.score(self.crop(self.noisy(g)))
        self.assertTrue(self.det.present(score), f"igna sezilmadi: {score:.4%}")
        mean_diff = float(np.abs(self.crop(g) - self.crop(self.bg)).mean())
        self.assertLess(mean_diff, 1.0, "eski usul (o'rtacha farq) buni sezmas edi")

    def test_sensor_noise_not_detected(self):
        self.det.set_background(self.crop(self.noisy(self.bg)))
        for _ in range(5):
            self.assertFalse(self.det.present(self.det.score(self.crop(self.noisy(self.bg)))))


if __name__ == "__main__":
    unittest.main()
