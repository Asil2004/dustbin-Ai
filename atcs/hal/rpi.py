"""Raspberry Pi 4B haqiqiy apparat drayverlari (ATCS.01.00.000 E1).
Kutubxonalar faqat shu faylda import qilinadi - Windows/simulyatsiyada kerak emas.
O'rnatish: pip install -r requirements-pi.txt"""
from __future__ import annotations

import threading
import time

import numpy as np

from .base import Hardware, Status


class Pca9685Bank:
    def __init__(self, address: int, freq_hz: int):
        import board
        from adafruit_pca9685 import PCA9685
        self._pca = PCA9685(board.I2C(), address=address)
        self._pca.frequency = freq_hz
        self._period_us = 1_000_000 / freq_hz

    def set_angle(self, channel, degrees):
        # MG996R / SG90: 500..2500 us = 0..180 grad
        pulse = 500 + 2000 * max(0.0, min(180.0, degrees)) / 180
        self._pca.channels[channel].duty_cycle = int(pulse / self._period_us * 0xFFFF)

    def set_duty(self, channel, duty):
        self._pca.channels[channel].duty_cycle = int(max(0.0, min(1.0, duty)) * 0xFFFF)

    def close(self):
        for ch in range(16):
            self._pca.channels[ch].duty_cycle = 0
        self._pca.deinit()


class Tb6612Channel:
    def __init__(self, pwm_pin, in1, in2, stby):
        from gpiozero import DigitalOutputDevice, PWMOutputDevice
        self._pwm = PWMOutputDevice(pwm_pin, frequency=1000)
        self._in1 = DigitalOutputDevice(in1)
        self._in2 = DigitalOutputDevice(in2)
        self._stby = stby

    def set_speed(self, fraction):
        f = max(-1.0, min(1.0, fraction))
        self._in1.value = f > 0
        self._in2.value = f < 0
        self._pwm.value = abs(f)
        if f:
            self._stby.on()

    def close(self):
        self._pwm.value = 0
        for d in (self._pwm, self._in1, self._in2):
            d.close()


class QuadEncoder:
    def __init__(self, a, b):
        from gpiozero import RotaryEncoder
        self._enc = RotaryEncoder(a, b, max_steps=0)

    def count(self):
        return abs(self._enc.steps)

    def close(self):
        self._enc.close()


class PulseEncoder:
    """Faqat A kanal ulangan enkoder (dozator, GPIO22)."""
    def __init__(self, a):
        from gpiozero import DigitalInputDevice
        self._n = 0
        self._dev = DigitalInputDevice(a, pull_up=True)
        self._dev.when_activated = self._tick

    def _tick(self):
        self._n += 1

    def count(self):
        return self._n

    def close(self):
        self._dev.close()


class Input:
    """pull_up=True: kontakt GND ga tutashsa (gerkon yopiq / IR buyumni ko'rdi) value() = True."""
    def __init__(self, pin):
        from gpiozero import DigitalInputDevice
        self._dev = DigitalInputDevice(pin, pull_up=True)

    def value(self):
        return bool(self._dev.value)

    def close(self):
        self._dev.close()


class RelayOut:
    def __init__(self, pin, active_high=True):
        from gpiozero import DigitalOutputDevice
        self._dev = DigitalOutputDevice(pin, active_high=active_high, initial_value=False)

    def set(self, on):
        self._dev.value = bool(on)

    def close(self):
        self._dev.off()
        self._dev.close()


class Vl53l0x:
    def __init__(self, address):
        import board
        import adafruit_vl53l0x
        self._s = adafruit_vl53l0x.VL53L0X(board.I2C(), address=address)

    def read_mm(self):
        try:
            mm = self._s.range
        except OSError:
            return None
        return float(mm) if 0 < mm < 2000 else None


class UltrasonicGroup:
    """JSN-SR04T x4, umumiy TRIG. Fon oqimida navbat bilan o'lchaydi, oxirgi qiymatni saqlaydi."""
    def __init__(self, trig, echo_by_bin: dict[int, int], period_s: float):
        from gpiozero import DigitalInputDevice, DigitalOutputDevice
        self._trig = DigitalOutputDevice(trig)
        self._echo = {b: DigitalInputDevice(p) for b, p in echo_by_bin.items()}
        self._last: dict[int, float | None] = {b: None for b in echo_by_bin}
        self._period = period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _measure(self, echo) -> float | None:
        self._trig.on(); time.sleep(20e-6); self._trig.off()
        t0 = time.perf_counter()
        while not echo.value:
            if time.perf_counter() - t0 > 0.03:
                return None
        start = time.perf_counter()
        while echo.value:
            if time.perf_counter() - start > 0.03:
                return None
        return (time.perf_counter() - start) * 343_000 / 2   # mm

    def _run(self):
        while not self._stop.is_set():
            for b, echo in self._echo.items():
                self._last[b] = self._measure(echo)
                time.sleep(0.07)   # aks-sado so'nishi uchun
            self._stop.wait(self._period)

    def read_mm(self, bin_no):
        return self._last.get(bin_no)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1)


class PiCamera:
    """Camera Module 3 Wide. Ish rejimida ham, rasm yig'ishda ham (atcs.collect) aynan shu sozlama ishlatiladi.
    presence() - lores kulrang kadrning ROI qismida fon bilan farq (cameras.PresenceDetector); capture() - to'liq RGB kadr (ROI klassifikatorda kesiladi)."""
    def __init__(self, vision_cfg):
        from picamera2 import Picamera2
        from libcamera import controls
        self._cam = Picamera2()
        # RGB888 picamera2 da xotirada B,G,R tartibida -> "BGR888" so'rab, numpy da RGB olamiz
        cfg = self._cam.create_video_configuration(
            main={"size": tuple(vision_cfg.capture_size), "format": "BGR888"},
            lores={"size": (640, 360), "format": "YUV420"})
        self._cam.configure(cfg)
        self._cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                "LensPosition": float(vision_cfg.lens_position)})
        self._cam.start()
        if vision_cfg.lock_exposure:
            time.sleep(1.5)                       # AE/AWB barqarorlashsin
            md = self._cam.capture_metadata()
            self._cam.set_controls({"AeEnable": False, "AwbEnable": False,
                                    "ExposureTime": md["ExposureTime"], "AnalogueGain": md["AnalogueGain"],
                                    "ColourGains": md["ColourGains"]})
            time.sleep(0.3)
        from ..cameras import PresenceDetector
        self._det = PresenceDetector(vision_cfg)

    def _gray(self):
        from ..vision import crop_roi
        y = self._cam.capture_array("lores")[:360, :640]          # YUV420 ning Y (yorqinlik) qismi
        return self._det.shrink(crop_roi(y, self._det.roi))

    def set_background(self):
        self._det.set_background(self._gray())

    def presence_score(self) -> float:
        """ROI da fon bilan farq qilgan piksellar ulushi (0..1)."""
        return self._det.score(self._gray())

    def presence(self):
        return self._det.present(self.presence_score())

    @property
    def presence_min_fraction(self) -> float:
        return self._det.min_fraction

    def capture(self):
        return self._cam.capture_array("main")

    def close(self):
        self._cam.stop()


class LumaDisplay:
    def __init__(self, cfg):
        from luma.core.interface.serial import spi
        from luma.lcd import device
        from ..display import render
        self._render = render
        drv = getattr(device, cfg.display.driver)
        self._dev = drv(spi(port=0, device=0, gpio_DC=cfg.pins.tft_dc, gpio_RST=cfg.pins.tft_rst),
                        width=cfg.display.width, height=cfg.display.height, rotate=0)
        self._size = (cfg.display.width, cfg.display.height)
        self._last = None

    def show(self, status: Status):
        key = repr(status)
        if key != self._last:
            self._last = key
            self._dev.display(self._render(status, self._size))

    def close(self):
        self._dev.cleanup()


def build(cfg) -> Hardware:
    from ..cameras import open_camera
    from gpiozero import DigitalOutputDevice
    p = cfg.pins
    stby = DigitalOutputDevice(p.tb_stby)
    return Hardware(
        pwm=Pca9685Bank(cfg.pca9685.address, cfg.pca9685.freq_hz),
        dozator=Tb6612Channel(p.tb_pwma, p.tb_ain1, p.tb_ain2, stby),
        main=Tb6612Channel(p.tb_pwmb, p.tb_bin1, p.tb_bin2, stby),
        enc_main=QuadEncoder(p.enc_main_a, p.enc_main_b),
        enc_dozator=PulseEncoder(p.enc_dozator_a),
        ir_channel=Input(p.ir_channel),
        door_closed=Input(p.door_reed),
        uvc=RelayOut(p.uvc_relay),
        tof=Vl53l0x(cfg.tof.address),
        fill=UltrasonicGroup(p.us_trig, p.us_echo, cfg.timing.fill_period_s),
        camera=open_camera(cfg.vision),
        display=LumaDisplay(cfg),
    )
