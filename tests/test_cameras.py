"""USB kamera: soxta OpenCV kamerasi bilan (haqiqiy kamera yoqilmaydi)."""
import copy
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from atcs import config
from atcs.cameras import UsbCamera, open_camera


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


if __name__ == "__main__":
    unittest.main()
