"""Boshqaruv mantig'i sinovlari simulyatsiyada (apparat kerak emas).
Ishga tushirish: py -m unittest discover -s tests -v"""
import tempfile
import unittest
from pathlib import Path

from atcs import config
from atcs.controller import KUTISH, OPERATOR, PAUZA, QABUL, XATO, Controller
from atcs.hal import sim

DT = 0.01


class Rig:
    def __init__(self, **overrides):
        self.cfg = config.load()
        for dotted, value in overrides.items():
            sec, key = dotted.split("__")
            setattr(getattr(self.cfg, sec), key, value)
        self.hw, self.w = sim.build(self.cfg)
        self.ctl = Controller(self.cfg, self.hw, sim.SimClassifier())
        self.ctl.start(0.0)

    def run(self, seconds, until=None):
        end = self.w.t + seconds
        while self.w.t < end:
            self.ctl.step(self.w.t)
            self.w.advance(DT)
            if until and until():
                return True
        return False

    def wake(self):
        self.w.hand_near = True
        self.run(0.5)
        self.w.hand_near = False


class SortingTest(unittest.TestCase):
    def test_each_category_lands_in_its_bin(self):
        r = Rig()
        for b in (1, 2, 3, 4, 5, 4, 3, 2, 1):
            r.w.add_item(b, 0.95)
        r.wake()
        self.assertTrue(r.run(60, until=lambda: len(r.w.landed()) == 9 and not r.ctl.tracked))
        for name, want, got in r.w.landed():
            self.assertEqual(want, got, name)
        self.assertEqual(r.ctl.counts, [2, 2, 2, 2, 1])
        self.assertFalse(r.ctl.log.of("unexpected_item"))

    def test_bin_mode_low_confidence_goes_to_unknown_bin_after_retries(self):
        r = Rig(sorting__unknown_action="bin")
        r.w.add_item(3, 0.40)
        r.wake()
        self.assertTrue(r.run(20, until=lambda: r.w.landed()))
        self.assertEqual(r.w.landed()[0][2], r.cfg.sorting.unknown_bin)
        rec = r.ctl.log.of("classified")[0]
        self.assertEqual(rec["retries"], r.cfg.sorting.max_retries)
        self.assertEqual(len(r.ctl.log.of("unknown")), 1)

    def test_deflectors_rest_when_belt_empty(self):
        r = Rig()
        r.w.add_item(2)
        r.wake()
        r.run(20, until=lambda: r.w.landed() and not r.ctl.tracked)
        r.run(0.1)
        for ch in r.cfg.pca9685.deflector_channels:
            self.assertEqual(r.w.angles[ch], r.cfg.servo.deflector_rest)

    def test_goes_idle_and_stops_belts(self):
        r = Rig(timing__idle_after_s=2)
        r.wake()
        r.run(4)
        self.assertEqual(r.ctl.state, KUTISH)
        self.assertEqual((r.w.dozator_speed, r.w.main_speed), (0, 0))


class OperatorTest(unittest.TestCase):
    def _door_cycle(self, r):
        r.w.door_is_closed = False
        r.run(1)
        r.w.door_is_closed = True
        r.run(DT * 2)

    def test_low_confidence_stops_and_waits_for_operator(self):
        r = Rig()
        needle = r.w.add_item(1, 0.40)
        r.w.add_item(2, 0.95)
        r.wake()
        self.assertTrue(r.run(15, until=lambda: r.ctl.state == OPERATOR))
        self.assertEqual(r.ctl.log.of("classified"), [], "aniqlanmagan buyum hech qayerga yuborilmasligi kerak")
        r.run(5)                                      # operator kelguncha hech narsa qimirlamaydi
        self.assertEqual(r.ctl.state, OPERATOR)
        self.assertEqual((r.w.dozator_speed, r.w.main_speed), (0, 0))
        self.assertEqual(r.w.angles[r.cfg.pca9685.gate_channel], r.cfg.servo.gate_closed)
        self.assertEqual(needle.stage, "dozator")
        self.assertIn("aniqlanmadi", r.ctl.message)

        self.assertIs(r.w.operator_remove(), needle)
        self._door_cycle(r)
        self.assertNotEqual(r.ctl.state, OPERATOR)
        self.assertTrue(r.run(20, until=lambda: r.w.landed()))
        self.assertEqual(r.w.landed(), [("buyum-2", 2, 2)])
        self.assertIsNone(needle.landed_bin)

    def test_item_left_in_place_is_reclassified(self):
        r = Rig()
        it = r.w.add_item(4, 0.40)
        r.wake()
        self.assertTrue(r.run(15, until=lambda: r.ctl.state == OPERATOR))
        it.confidence = 0.95                          # operator buyumni to'g'rilab qo'ydi
        self._door_cycle(r)
        self.assertTrue(r.run(15, until=lambda: r.w.landed()))
        self.assertEqual(r.w.landed()[0][2], 4)

    def test_unexpected_item_on_main_belt_calls_operator(self):
        r = Rig()
        r.wake()
        stray = r.w.add_item(1)
        stray.stage, stray.pos = "main", r.cfg.sorting.entry_x - 10   # shlyuzni chetlab tushgan buyum
        self.assertTrue(r.run(2, until=lambda: r.ctl.state == OPERATOR))
        self.assertEqual(r.w.main_speed, 0)
        self.assertIs(r.w.operator_remove(), stray)
        self._door_cycle(r)
        r.run(5)
        self.assertEqual(r.ctl.tracked, [])
        self.assertEqual(r.ctl.counts, [0] * 5)

    def test_operator_door_wait_does_not_resume_by_itself(self):
        r = Rig()
        r.w.add_item(3, 0.30)
        r.wake()
        r.run(15, until=lambda: r.ctl.state == OPERATOR)
        r.run(30)
        self.assertEqual(r.ctl.state, OPERATOR)
        self.assertFalse(r.w.uvc_on)


class SafetyTest(unittest.TestCase):
    def test_door_open_pauses_and_resumes(self):
        r = Rig()
        for b in (4, 4):
            r.w.add_item(b)
        r.wake()
        r.run(5)
        r.w.door_is_closed = False
        r.run(0.5)
        self.assertEqual(r.ctl.state, PAUZA)
        self.assertEqual((r.w.dozator_speed, r.w.main_speed), (0, 0))
        r.w.door_is_closed = True
        self.assertTrue(r.run(30, until=lambda: len(r.w.landed()) == 2))
        self.assertEqual([g for _, _, g in r.w.landed()], [4, 4])

    def test_full_bin_blocks_only_its_items(self):
        r = Rig()
        r.w.fill_mm[3] = 50          # 3-bo'lim to'la
        r.w.add_item(3)
        r.wake()
        r.run(10)
        self.assertEqual(r.ctl.state, PAUZA)
        self.assertIn("3-bo'lim", r.ctl.message)
        self.assertFalse(r.w.landed())
        r.w.fill_mm[3] = 470         # bo'shatildi
        self.assertTrue(r.run(15, until=lambda: r.w.landed()))
        self.assertEqual(r.w.landed()[0][2], 3)

    def test_jam_faults_and_door_cycle_recovers(self):
        r = Rig()
        r.w.jam = True
        r.w.add_item(2)
        r.wake()
        self.assertTrue(r.run(15, until=lambda: r.ctl.state == XATO))
        self.assertEqual((r.w.dozator_speed, r.w.main_speed), (0, 0))
        self.assertEqual(r.w.angles[r.cfg.pca9685.gate_channel], r.cfg.servo.gate_closed)
        r.w.jam = False
        r.w.door_is_closed = False
        r.run(1)
        r.w.door_is_closed = True
        r.run(DT * 2)
        self.assertNotEqual(r.ctl.state, XATO)
        self.assertIn({"state": QABUL, "message": "xato tiklandi"},
                      [{"state": e["state"], "message": e["message"]} for e in r.ctl.log.of("state")])
        # tiqilgan buyum qayta aniqlanib, o'z bo'limiga tushadi
        self.assertTrue(r.run(15, until=lambda: r.w.landed()))
        self.assertEqual(r.w.landed()[0][2], 2)

    def test_uvc_only_when_idle_and_closed(self):
        r = Rig(uvc__start_after_idle_s=5, uvc__duration_s=10)
        r.run(4)
        self.assertFalse(r.w.uvc_on)
        r.run(2)
        self.assertTrue(r.w.uvc_on)
        r.w.door_is_closed = False
        r.run(DT * 2)
        self.assertFalse(r.w.uvc_on, "eshik ochilganda UV-C darhol o'chishi kerak")
        r.w.door_is_closed = True
        r.w.hand_near = True
        r.run(0.05)
        self.assertFalse(r.w.uvc_on)
        self.assertTrue(r.w.angles[r.cfg.pca9685.flap_channel] > 0, "qabul eshikchasi ochilishi kerak")

    def test_uvc_cycle_completes(self):
        r = Rig(uvc__start_after_idle_s=1, uvc__duration_s=3)
        r.run(2)
        self.assertTrue(r.w.uvc_on)
        r.run(3)
        self.assertFalse(r.w.uvc_on)
        self.assertEqual([e["reason"] for e in r.ctl.log.of("uvc")], ["dezinfeksiya", "sikl tugadi"])


class FillAndConfigTest(unittest.TestCase):
    def test_sharps_counter(self):
        r = Rig(bins__sharps_capacity=2)
        for _ in range(2):
            r.w.add_item(1)
        r.wake()
        r.run(20, until=lambda: len(r.w.landed()) == 2 and not r.ctl.tracked)
        self.assertTrue(r.ctl.full[0])
        r.ctl.reset_sharps()
        self.assertFalse(r.ctl.full[0])

    def test_bad_spacing_rejected(self):
        src = Path(config.DEFAULT_PATH).read_text(encoding="utf-8")
        bad = src.replace("min_spacing_mm = 150", "min_spacing_mm = 50")
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as f:
            f.write(bad)
        with self.assertRaises(ValueError):
            config.load(f.name)

    def test_display_render(self):
        from atcs.display import render
        r = Rig()
        img = render(r.ctl.status())
        self.assertEqual(img.size, (480, 320))


if __name__ == "__main__":
    unittest.main()
