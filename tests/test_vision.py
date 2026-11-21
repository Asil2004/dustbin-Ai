"""Kamera kesimi va model kirishini tayyorlash: yig'ish, o'qitish va ish rejimi bir xil bo'lishi kerak."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from atcs import config
from atcs.vision import crop_roi, prepare


class VisionPrepTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((1296, 2304, 3), np.uint8)
        self.frame[400:600, 1000:1200] = (200, 30, 30)          # qizil buyum

    def test_crop_roi_shape_and_content(self):
        roi = config.load().vision.roi
        c = crop_roi(self.frame, roi)
        self.assertEqual(c.shape[:2], (round(roi[3] * 1296), round(roi[2] * 2304)))
        self.assertTrue((c[..., 0] == 200).any(), "buyum ROI ichida bo'lishi kerak")

    def test_crop_roi_rejects_outside(self):
        with self.assertRaises(ValueError):
            crop_roi(self.frame, (1.2, 0, 0.1, 0.1))

    def test_prepare_float_and_uint8_keep_0_255(self):
        f = prepare(self.frame, (0.4, 0.2, 0.2, 0.4), 224, np.float32)
        u = prepare(self.frame, (0.4, 0.2, 0.2, 0.4), 224, np.uint8)
        self.assertEqual(f.shape, (1, 224, 224, 3))
        self.assertEqual(u.dtype, np.uint8)
        self.assertGreater(f.max(), 150, "normalizatsiya modelda: kirish 0..255 bo'lishi kerak")
        np.testing.assert_array_equal(f.astype(np.uint8), u)

    def test_prepare_matches_training_loader(self):
        """train.py rasmni fayldan o'qib xuddi shunday kichraytiradi."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
        from train import load_dir
        from PIL import Image
        crop = crop_roi(self.frame, (0.4, 0.2, 0.2, 0.4))
        with tempfile.TemporaryDirectory() as d:
            labels = ["a", "b"]
            for lb in labels:
                Path(d, lb).mkdir()
                Image.fromarray(np.ascontiguousarray(crop)).save(Path(d, lb, "x.png"))
            x, y, _ = load_dir(Path(d), labels, 224)
        np.testing.assert_array_equal(x[0], prepare(crop, None, 224, np.uint8)[0])
        self.assertEqual(list(y), [0, 1])


class LiveTestTest(unittest.TestCase):
    labels = ["igna", "qolqop_shlang", "shisha", "plastik", "bint_paxta"]

    def test_decide_matches_controller_rule(self):
        from atcs.livetest import decide
        text, b, c = decide(np.array([0.05, 0.05, 0.8, 0.05, 0.05]), self.labels, 0.70)
        self.assertEqual(b, 3); self.assertIn("shisha", text)
        text, b, _ = decide(np.array([0.6, 0.1, 0.1, 0.1, 0.1]), self.labels, 0.70)
        self.assertIsNone(b); self.assertIn("OPERATOR", text)

    def test_annotate_draws_on_frame(self):
        from atcs.livetest import annotate
        frame = np.zeros((480, 640, 3), np.uint8)
        out = annotate(frame.copy(), np.array([0.9, 0.025, 0.025, 0.025, 0.025]), self.labels, 0.7,
                       (0.3, 0.15, 0.4, 0.7), fps=25)
        self.assertEqual(out.shape, frame.shape)
        self.assertGreater(int(out.sum()), 0)


if __name__ == "__main__":
    unittest.main()
