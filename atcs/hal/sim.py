"""Stolning soddalashtirilgan fizik simulyatsiyasi: Windows/PC da apparatsiz sinash uchun.
Buyum dozator lentasida kamera ostiga keladi, shlyuz ochilsa asosiy lentaga tushadi,
deflektor burilgan bo'lsa o'sha bo'limga, aks holda lenta oxiridagi 5-bo'limga tushadi."""
from __future__ import annotations

from dataclasses import dataclass, field

from .base import Hardware, Status

CAMERA_POS = 250      # dozator lentasida kamera markazi, mm
GATE_POS = 290        # shlyuz joyi
CAMERA_WINDOW = 40    # kamera ko'radigan zona (shlyuzgacha)
ITEM_GAP = 80         # dozatorda buyumlar orasidagi minimal masofa (tushish bunkeri ajratadi)
DROP_S = 0.35         # shlyuzdan asosiy lentagacha tushish vaqti


@dataclass
class SimItem:
    true_bin: int
    confidence: float = 0.95
    name: str = ""
    stage: str = "dozator"     # dozator | drop | main | done
    pos: float = 0.0           # dozatorda - lenta bo'yicha mm, asosiy lentada - X
    timer: float = 0.0
    landed_bin: int | None = None


@dataclass
class SimWorld:
    cfg: object
    t: float = 0.0
    angles: dict = field(default_factory=dict)
    duties: dict = field(default_factory=dict)
    dozator_speed: float = 0.0
    main_speed: float = 0.0
    main_mm: float = 0.0
    dozator_mm: float = 0.0
    items: list = field(default_factory=list)
    hand_near: bool = False
    door_is_closed: bool = True
    uvc_on: bool = False
    fill_mm: dict = field(default_factory=dict)
    statuses: list = field(default_factory=list)
    jam: bool = False              # True bo'lsa buyum shlyuzdan o'tib tushmaydi (tiqilish sinovi)

    def __post_init__(self):
        for b in (2, 3, 4, 5):
            self.fill_mm.setdefault(b, float(self.cfg.bins.sensor_to_bottom_mm))

    # ---- sinov ssenariylari uchun
    def add_item(self, true_bin: int, confidence: float = 0.95, name: str = "") -> SimItem:
        tail = min((i.pos for i in self.items if i.stage == "dozator"), default=None)
        pos = 0.0 if tail is None else min(0.0, tail - ITEM_GAP)
        item = SimItem(true_bin, confidence, name or f"buyum-{len(self.items) + 1}", pos=pos)
        self.items.append(item)
        return item

    def operator_remove(self) -> SimItem | None:
        """Operator shlyuz oldidagi (bo'lmasa asosiy lentaga oxirgi tushgan) buyumni olib tashlaydi."""
        at_gate = [i for i in self.items if i.stage == "dozator" and i.pos >= GATE_POS - CAMERA_WINDOW]
        on_main = [i for i in self.items if i.stage == "main"]
        pick = max(at_gate, key=lambda i: i.pos) if at_gate else (min(on_main, key=lambda i: i.pos) if on_main else None)
        if pick:
            pick.stage = "removed"
        return pick

    def landed(self) -> list[tuple[str, int, int | None]]:
        return [(i.name, i.true_bin, i.landed_bin) for i in self.items if i.stage == "done"]

    # ---- fizika
    def _deflector_active(self, k: int) -> bool:
        ch = self.cfg.pca9685.deflector_channels[k - 1]
        return self.angles.get(ch, 0) >= self.cfg.servo.deflector_active * 0.8

    def _gate_open(self) -> bool:
        return self.angles.get(self.cfg.pca9685.gate_channel, 0) >= self.cfg.servo.gate_open * 0.8

    def advance(self, dt: float) -> None:
        b, s = self.cfg.belts, self.cfg.sorting
        self.t += dt
        dmm = self.dozator_speed * b.dozator_max_mm_s * dt
        mmm = self.main_speed * b.main_max_mm_s * dt
        self.dozator_mm += abs(dmm)
        self.main_mm += abs(mmm)

        ahead = None
        for it in sorted((i for i in self.items if i.stage == "dozator"), key=lambda i: -i.pos):
            new = it.pos + dmm
            if ahead is not None:
                new = min(new, ahead - ITEM_GAP)
            if new >= GATE_POS and (not self._gate_open() or self.jam):
                new = min(new, GATE_POS - 1)
            it.pos = max(it.pos, new)
            if it.pos >= GATE_POS:
                it.stage, it.timer = "drop", DROP_S
            ahead = it.pos

        for it in self.items:
            if it.stage == "drop":
                it.timer -= dt
                if it.timer <= 0:
                    it.stage, it.pos = "main", s.entry_x - 10
            elif it.stage == "main":
                old, it.pos = it.pos, it.pos + mmm
                for k, x in enumerate(s.deflector_x, start=1):
                    if old < x <= it.pos and self._deflector_active(k):
                        self._land(it, k)
                        break
                else:
                    if it.pos >= s.belt_end_x:
                        self._land(it, 5)

    def _land(self, it: SimItem, bin_no: int) -> None:
        it.stage, it.landed_bin = "done", bin_no
        if bin_no in self.fill_mm:
            self.fill_mm[bin_no] -= 4.0


# ---- apparat obyektlari
class _Pwm:
    def __init__(self, w): self.w = w
    def set_angle(self, ch, deg): self.w.angles[ch] = deg
    def set_duty(self, ch, duty): self.w.duties[ch] = duty


class _Motor:
    def __init__(self, w, attr): self.w, self.attr = w, attr
    def set_speed(self, f): setattr(self.w, self.attr, max(-1.0, min(1.0, f)))


class _Encoder:
    def __init__(self, w, attr, mm_per_count): self.w, self.attr, self.k = w, attr, mm_per_count
    def count(self): return int(getattr(self.w, self.attr) / self.k)


class _Ir:
    def __init__(self, w): self.w = w
    def value(self):
        x0 = self.w.cfg.sorting.entry_x
        return any(i.stage == "main" and x0 - 5 <= i.pos <= x0 + 15 for i in self.w.items)


class _Door:
    def __init__(self, w): self.w = w
    def value(self): return self.w.door_is_closed


class _Relay:
    def __init__(self, w): self.w = w
    def set(self, on): self.w.uvc_on = bool(on)


class _Tof:
    def __init__(self, w): self.w = w
    def read_mm(self): return 80.0 if self.w.hand_near else 900.0


class _Fill:
    def __init__(self, w): self.w = w
    def read_mm(self, b): return self.w.fill_mm.get(b)


class _Camera:
    def __init__(self, w): self.w = w
    def _in_view(self):
        return [i for i in self.w.items
                if i.stage == "dozator" and GATE_POS - CAMERA_WINDOW <= i.pos <= GATE_POS]
    def presence(self): return bool(self._in_view())
    def capture(self):
        v = self._in_view()
        return max(v, key=lambda i: i.pos) if v else None
    def set_background(self): pass


class _Display:
    def __init__(self, w, echo): self.w, self.echo, self.last = w, echo, None
    def show(self, st: Status):
        line = f"[{self.w.t:7.2f}s] {st.state:<14} {st.message}"
        if line.split("] ", 1)[1] != (self.last or "").split("] ", 1)[-1]:
            self.last = line
            self.w.statuses.append((self.w.t, st.state, st.message))
            if self.echo:
                print(line)


class SimClassifier:
    """Kadr - SimItem; haqiqiy bo'limni berilgan ishonch bilan qaytaradi."""
    def classify(self, frame):
        if frame is None:
            return 5, 0.0
        return frame.true_bin, frame.confidence


def build(cfg, echo: bool = False) -> tuple[Hardware, SimWorld]:
    w = SimWorld(cfg)
    hw = Hardware(
        pwm=_Pwm(w),
        dozator=_Motor(w, "dozator_speed"),
        main=_Motor(w, "main_speed"),
        enc_main=_Encoder(w, "main_mm", cfg.belts.main_mm_per_count),
        enc_dozator=_Encoder(w, "dozator_mm", cfg.belts.dozator_mm_per_count),
        ir_channel=_Ir(w),
        door_closed=_Door(w),
        uvc=_Relay(w),
        tof=_Tof(w),
        fill=_Fill(w),
        camera=_Camera(w),
        display=_Display(w, echo),
    )
    return hw, w
