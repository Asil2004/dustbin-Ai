"""Tashqi datasetni import qilish va o'qitish hisobotidagi igna qoidasi."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import import_dataset as imp  # noqa: E402

from atcs.vision import decide  # noqa: E402


def img(path: Path, seed: int, size=(120, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


class ImportFolderTest(unittest.TestCase):
    def test_roboflow_folder_export(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d, "src"), Path(d, "out")
            # Roboflow: bitta manba rasmdan 3 ta augmentatsiyalangan nusxa
            for i, h in enumerate(("aa11", "bb22", "cc33")):
                img(src / "train" / "Gloves" / f"glove1_jpg.rf.{h}.jpg", seed=10 + i)
            img(src / "valid" / "Gloves" / "glove2_jpg.rf.dd44.jpg", seed=20)
            img(src / "train" / "Syringes" / "syr1_jpg.rf.ee55.jpg", seed=30)
            img(src / "test" / "Medicines" / "med1.jpg", seed=40)                    # mapda yo'q
            img(src / "train" / "Gloves" / "tiny_jpg.rf.ff66.jpg", seed=50, size=(20, 20))
            (src / "README.dataset.txt").write_text("License: CC BY 4.0", encoding="utf-8")

            rep = imp.run(src, out, "rf", imp.parse_map("Gloves=qolqop_shlang, Syringes=igna"),
                          log=lambda *a: None)
            self.assertEqual(rep["kept"], {"qolqop_shlang": 2, "igna": 1})
            self.assertEqual(rep["skipped_classes"], {"Medicines": 1})
            self.assertEqual(rep["duplicates"], 2, "bitta manbaning nusxalari bir marta olinadi")
            self.assertEqual(rep["too_small"], 1)
            self.assertEqual(len(list((out / "qolqop_shlang").glob("rf_*.jpg"))), 2)
            self.assertIn("CC BY 4.0", (out / "rf_ATTRIBUTION.txt").read_text(encoding="utf-8"))
            rows = list(csv.DictReader(open(out / "rf_manifest.csv", encoding="utf-8")))
            self.assertEqual({r["group"] for r in rows if r["atcs_label"] == "qolqop_shlang"}, {"glove1", "glove2"})

    def test_exact_duplicates_removed(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d, "src"), Path(d, "out")
            img(src / "glove" / "a.jpg", seed=1)
            img(src / "glove" / "b.jpg", seed=1)                     # xuddi shu rasm boshqa nom bilan
            rep = imp.run(src, out, "k", {"glove": "qolqop_shlang"}, log=lambda *a: None)
            self.assertEqual(rep["kept"], {"qolqop_shlang": 1})

    def test_bad_map_rejected(self):
        with self.assertRaises(SystemExit):
            imp.parse_map("Gloves=gloves")


class ImportYoloTest(unittest.TestCase):
    def test_yolo_boxes_cropped_with_padding(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d, "src"), Path(d, "out")
            img(src / "train" / "images" / "p1.jpg", seed=3, size=(400, 300))
            lab = src / "train" / "labels"
            lab.mkdir(parents=True)
            # 0 = syringe (markazda 100x60), 1 = mask (mapda yo'q)
            (lab / "p1.txt").write_text("0 0.5 0.5 0.25 0.2\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")
            (src / "data.yaml").write_text("train: x\nnc: 2\nnames: ['syringe', 'mask']\n", encoding="utf-8")
            rep = imp.run(src, out, "y", {"syringe": "igna"}, pad=0.10, log=lambda *a: None)
            self.assertEqual(rep["format"], "yolo")
            self.assertEqual(rep["kept"], {"igna": 1})
            with Image.open(next((out / "igna").glob("*.jpg"))) as crop:
                size = crop.size
            self.assertEqual(size, (120, 72), "100x60 ramka + 10 % padding har tomondan")

    def test_yolo_names_list_style(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "data.yaml").write_text("names:\n  0: gloves\n  1: needle\n", encoding="utf-8")
            self.assertEqual(imp.yolo_names(Path(d)), ["gloves", "needle"])


class TrainingRuleTest(unittest.TestCase):
    """train.py hisobotidagi qaror stol qaroriga (vision.decide) mos bo'lishi kerak."""

    def test_table_pred_matches_decide(self):
        from train import table_pred
        rng = np.random.default_rng(0)
        for labels in (["igna", "qolqop_shlang", "shisha", "plastik", "bint_paxta"],
                       ["igna", "qolqop_shlang", "shisha", "plastik", "bint_paxta", "bosh"]):
            probs = rng.dirichlet(np.full(len(labels), 0.4), size=2000)
            pred, sure, _ = table_pred(probs, labels, 0.70, 0.15)
            for p, k, s in zip(probs, pred, sure):
                d = decide(p, labels, 0.70, 0.15)
                if d.kind == "bin":
                    self.assertTrue(s)
                    self.assertEqual(labels[k], [lb for lb in labels if lb != "bosh"][d.bin - 1])
                elif d.kind == "empty":
                    self.assertEqual(labels[k], "bosh")
                else:
                    self.assertFalse(s and labels[k] != "bosh")

    def test_report_counts_needle_rule(self):
        from train import report
        labels = ["igna", "qolqop_shlang", "shisha", "plastik", "bint_paxta"]
        probs = np.array([[0.20, 0.75, 0.02, 0.02, 0.01],      # igna, model qo'lqop dedi -> qoida qutqaradi
                          [0.18, 0.02, 0.76, 0.02, 0.02],      # shisha -> qoida sharps ga yuboradi (narx)
                          [0.05, 0.90, 0.02, 0.02, 0.01]])     # igna, qoida ishlamaydi -> xavfli xato
        y = np.array([0, 2, 0])
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            r = report(probs, y, labels, 0.70, 0.15)
            off = report(probs, y, labels, 0.70, 1.0)
        self.assertEqual(r["needle_sent_elsewhere"], 1)
        self.assertEqual(r["rule_extra_to_sharps"], 1)
        self.assertEqual(off["needle_sent_elsewhere"], 2)


if __name__ == "__main__":
    unittest.main()
