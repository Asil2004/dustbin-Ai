"""Kamera kalibrlash: ekspozitsiya tanlash, fokus qidirish, config ga yozish (soxta optika bilan)."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from atcs import camcal, config


class FakeOptics:
    """Fokusi `best` da tiniq, undan uzoqlashgan sari xiralashadigan harfli qog'oz."""

    def __init__(self, best=412):
        self.best = best
        self.focus = 0
        page = np.full((300, 400), 200, np.uint8)
        for y in range(20, 300, 24):
            for x in range(10, 390, 14):
                cv2.putText(page, "a", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 20, 1)
        self.page = page

    def set_focus(self, v):
        self.focus = v

    def roi_gray(self, settle_s=0.0):
        sigma = abs(self.focus - self.best) / 40 + 0.01
        return cv2.GaussianBlur(self.page, (0, 0), sigma) if sigma > 0.3 else self.page.copy()


class CamCalTest(unittest.TestCase):
    def test_focus_found_near_sharpest(self):
        best, results, note = camcal.find_focus(FakeOptics(412), 1023, 0, log=lambda *a: None)
        self.assertLessEqual(abs(best - 412), 8, f"topilgan fokus {best}")
        self.assertEqual(note, "")

    def test_focus_warns_when_nothing_changes(self):
        cam = FakeOptics()
        cam.roi_gray = lambda settle_s=0.0: np.full((300, 400), 128, np.uint8)
        _, _, note = camcal.find_focus(cam, 1023, 0, log=lambda *a: None)
        self.assertIn("o'zgarmadi", note)

    def test_exposure_candidates_flicker_free(self):
        with mock.patch.object(camcal, "WIN", False):
            c = camcal.exposure_candidates(15, 50)
        self.assertIn(100.0, c)
        self.assertIn(600.0, c)
        self.assertNotIn(700.0, c, "15 fps da kadr 66,6 ms: undan uzun ekspozitsiya bo'lmaydi")
        self.assertEqual(c, sorted(c))

    def test_pick_exposure(self):
        m = [(100.0, 90, 0.0), (200.0, 160, 0.0), (300.0, 200, 0.002), (400.0, 240, 0.08)]
        self.assertEqual(camcal.pick_exposure(m, 190)[0], 300.0)        # 400 kuygan
        e, note = camcal.pick_exposure([(100.0, 100, 0.0), (200.0, 120, 0.0)], 190)
        self.assertEqual(e, 200.0)
        self.assertIn("yorug'lik kam", note)
        e, note = camcal.pick_exposure([(5.0, 255, 0.9)], 190)
        self.assertIn("kuyib", note)

    def test_save_to_config_keeps_comments(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "atcs.toml"
            shutil.copy(config.DEFAULT_PATH, p)
            camcal.save_to_config(p, 408, -6)               # Windows ekspozitsiyasi manfiy
            cfg = config.load(p)
            self.assertEqual(cfg.vision.usb_focus, 408)
            self.assertEqual(cfg.vision.usb_exposure, -6)
            camcal.save_to_config(p, 410, None)              # qayta yozish ham ishlaydi
            cfg = config.load(p)
            self.assertEqual((cfg.vision.usb_focus, cfg.vision.usb_exposure), (410, -6))
            self.assertIn("camcal --save topib yozadi", p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
