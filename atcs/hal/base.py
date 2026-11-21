"""Apparat qatlami interfeyslari. Boshqaruv mantig'i faqat shularga tayanadi;
haqiqiy (Raspberry Pi) va simulyatsiya backendlari ularni amalga oshiradi."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class PwmBank(Protocol):
    """PCA9685: servolar (gradus) va LEDlar (0..1 yorqinlik)."""
    def set_angle(self, channel: int, degrees: float) -> None: ...
    def set_duty(self, channel: int, duty: float) -> None: ...


class Motor(Protocol):
    def set_speed(self, fraction: float) -> None:
        """-1..1; 0 = to'xtash."""


class Encoder(Protocol):
    def count(self) -> int: ...


class DigitalIn(Protocol):
    def value(self) -> bool: ...


class Relay(Protocol):
    def set(self, on: bool) -> None: ...


class RangeSensor(Protocol):
    def read_mm(self) -> float | None:
        """None - o'lchov yo'q / xato."""


class FillSensors(Protocol):
    def read_mm(self, bin_no: int) -> float | None:
        """Datchikdan chiqindi sirtigacha masofa (oxirgi o'lchov)."""


class Camera(Protocol):
    def presence(self) -> bool:
        """Shlyuz oldidagi zonada buyum bormi (fon bilan solishtirish)."""
    def capture(self) -> Any:
        """Klassifikator uchun kadr."""
    def set_background(self) -> None:
        """Bo'sh lenta kadrini fon sifatida saqlash."""


@dataclass
class Status:
    state: str = "BOSHLANMOQDA"
    message: str = ""
    fill: list[float | None] = field(default_factory=lambda: [None] * 5)   # 0..1, bo'lim 1..5
    counts: list[int] = field(default_factory=lambda: [0] * 5)
    uvc_on: bool = False
    last_item: str = ""


class Display(Protocol):
    def show(self, status: Status) -> None: ...


@dataclass
class Hardware:
    pwm: PwmBank
    dozator: Motor
    main: Motor
    enc_main: Encoder
    enc_dozator: Encoder
    ir_channel: DigitalIn
    door_closed: DigitalIn
    uvc: Relay
    tof: RangeSensor
    fill: FillSensors
    camera: Camera
    display: Display

    def close(self) -> None:
        for part in vars(self).values():
            closer = getattr(part, "close", None)
            if callable(closer):
                closer()
