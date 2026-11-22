"""ATCS boshqaruv mantig'i. Har sikl (cfg.loop.hz) step(now) chaqiriladi; bloklovchi kutish yo'q.

Ish tartibi:
  KUTISH   - lentalar to'xtagan; ToF qo'lni sezsa qabul eshikchasi ochiladi va QABUL boshlanadi
  QABUL    - dozator va asosiy lenta yuradi; kamera shlyuz oldida buyumni sezsa dozator to'xtaydi
  ANIQLASH - to'xta-va-suratga ol: kadr klassifikatorga beriladi, past ishonchda qayta suratga olinadi
  NAVBAT   - oldingi buyum yetarlicha uzoqlashishini kutish (deflektorlar to'qnashmasligi uchun)
  UZATISH  - shlyuz ochiladi, IR datchik buyum asosiy lentaga tushganini tasdiqlaydi
  PAUZA    - eshik ochiq yoki kerakli bo'lim to'lgan
  XATO     - tiqilish; operator eshikni ochib-yopsa tiklanadi
  OPERATOR - buyum ishonchli aniqlanmadi (sorting.unknown_action = "operator"): stol to'xtaydi,
             operator buyumni olib tashlaydi yoki to'g'rilaydi va eshikni ochib-yopadi;
             buyum joyida qolgan bo'lsa qayta aniqlanadi
Asosiy lentadagi har buyum enkoder bo'yicha kuzatiladi va o'z deflektori oldida o'sha deflektor buriladi.
UV-C faqat eshik va qabul eshikchasi yopiq, stol bo'sh turganda yonadi (dasturiy blokirovka).
"""
from __future__ import annotations

from dataclasses import dataclass

from .events import EventLog
from .hal.base import Hardware, Status
from .vision import decide

KUTISH, QABUL, ANIQLASH, NAVBAT, UZATISH, PAUZA, XATO, OPERATOR = (
    "KUTISH", "QABUL", "ANIQLASH", "NAVBAT", "UZATISH", "PAUZA", "XATO", "OPERATOR")


@dataclass
class Tracked:
    bin: int
    start_mm: float
    label: str


class Controller:
    def __init__(self, cfg, hw: Hardware, classifier, log: EventLog | None = None):
        self.cfg, self.hw, self.clf = cfg, hw, classifier
        self.log = log or EventLog()
        self.state = KUTISH
        self.message = ""
        self.tracked: list[Tracked] = []
        self.counts = [0] * 5
        self.fill: list[float | None] = [None] * 5
        self.full = [False] * 5
        self.warn = [False] * 5

        self._now = 0.0
        self._last_activity = 0.0
        self._idle_since = 0.0
        self._next_presence = 0.0
        self._next_fill = 0.0
        self._capture_at = 0.0
        self._retries = 0
        self._presence_hits = 0
        self._pending: tuple[int, float] | None = None   # (bo'lim, ishonch) uzatilishi kutilayotgan buyum
        self._gate_open_at = 0.0
        self._gate_open_mm = 0.0
        self._gate_close_at = 0.0
        self._gate_open = False
        self._ir_deadline = 0.0
        self._ir_prev = False
        self._flap_until = 0.0
        self._flap_open = False
        self._resume_to = QABUL
        self._fault_door_opened = False
        self._operator_reason = ""
        self._defl = [False] * 4
        self._uv_on = False
        self._uv_until = 0.0
        self._uv_last_done: float | None = None
        self._last_label = ""

    # ------------------------------------------------------------------ yordamchilar
    def _main_mm(self) -> float:
        return self.hw.enc_main.count() * self.cfg.belts.main_mm_per_count

    def _dozator_mm(self) -> float:
        return self.hw.enc_dozator.count() * self.cfg.belts.dozator_mm_per_count

    def _set_state(self, state: str, message: str = "") -> None:
        if state != self.state or message != self.message:
            self.log.write(self._now, "state", state=state, message=message)
        self.state, self.message = state, message
        if state == KUTISH:
            self._idle_since = self._now

    def _belts(self, dozator: bool, main: bool) -> None:
        b = self.cfg.belts
        self.hw.dozator.set_speed(b.dozator_mm_s / b.dozator_max_mm_s if dozator else 0.0)
        self.hw.main.set_speed(b.main_mm_s / b.main_max_mm_s if main else 0.0)

    def _gate(self, open_: bool) -> None:
        sv = self.cfg.servo
        self._gate_open = open_
        self.hw.pwm.set_angle(self.cfg.pca9685.gate_channel, sv.gate_open if open_ else sv.gate_closed)

    def _flap(self, open_: bool) -> None:
        sv = self.cfg.servo
        if open_ != self._flap_open:
            self._flap_open = open_
            self.hw.pwm.set_angle(self.cfg.pca9685.flap_channel, sv.flap_open if open_ else sv.flap_closed)

    def _uvc(self, on: bool, reason: str = "") -> None:
        if on != self._uv_on:
            self._uv_on = on
            self.hw.uvc.set(on)
            self.log.write(self._now, "uvc", on=on, reason=reason)

    # ------------------------------------------------------------------ hayot sikli
    def start(self, now: float) -> None:
        self._now = now
        self.hw.uvc.set(False)
        self._belts(False, False)
        for ch in self.cfg.pca9685.deflector_channels:
            self.hw.pwm.set_angle(ch, self.cfg.servo.deflector_rest)
        self._gate(False)
        self._flap_open = True
        self._flap(False)
        self.hw.camera.set_background()
        self._read_fill()
        self._set_state(KUTISH)
        self.log.write(now, "start")

    def shutdown(self) -> None:
        self._uvc(False, "to'xtatildi")
        self._belts(False, False)
        self._gate(False)
        self._flap(False)

    def step(self, now: float) -> None:
        self._now = now
        door_closed = self.hw.door_closed.value()
        hand = self._hand()

        self._intake(hand)
        self._uvc_control(door_closed, hand)
        if now >= self._next_fill:
            self._next_fill = now + self.cfg.timing.fill_period_s
            self._read_fill()

        ir = self.hw.ir_channel.value()
        ir_edge = ir and not self._ir_prev
        self._ir_prev = ir

        self._sequence(door_closed, hand, ir_edge)
        self._track()
        self._leds()
        self.hw.display.show(self.status())

    def status(self) -> Status:
        return Status(state=self.state, message=self.message, fill=list(self.fill),
                      counts=list(self.counts), uvc_on=self._uv_on, last_item=self._last_label)

    # ------------------------------------------------------------------ qabul eshikchasi
    def _hand(self) -> bool:
        mm = self.hw.tof.read_mm()
        return mm is not None and mm < self.cfg.tof.hand_mm

    def _intake(self, hand: bool) -> None:
        if hand:
            self._uvc(False, "qabul eshigida qo'l")
            self._flap_until = self._now + self.cfg.servo.flap_hold_s
            self._last_activity = self._now
        self._flap(self._now < self._flap_until)

    # ------------------------------------------------------------------ ketma-ketlik
    def _sequence(self, door_closed: bool, hand: bool, ir_edge: bool) -> None:
        t, tm, st = self._now, self.cfg.timing, self.state

        if st in (XATO, OPERATOR):
            # Operator eshikni ochib-yopguncha hammasi to'xtaydi. Fon qayta olinmaydi: buyum hali
            # lentada qolgan bo'lsa, kamera uni yana ko'rib, qayta aniqlashi kerak.
            self._belts(False, False)
            if not door_closed:
                self._fault_door_opened = True
            elif self._fault_door_opened:
                self._fault_door_opened = False
                self._gate(False)
                self._pending = None
                if self._operator_reason == "unexpected" and self.tracked:
                    self.tracked.pop()        # operator lentadan olib tashladi
                self._operator_reason = ""
                self._last_activity = t
                self._next_presence = t + tm.settle_s
                self._set_state(QABUL, "xato tiklandi" if st == XATO else "operator tasdiqladi")
            return

        if not door_closed:
            if st != PAUZA:
                self._resume_to = st
            self._belts(False, False)
            self._set_state(PAUZA, "Eshik ochiq")
            return

        if ir_edge:
            self._on_ir()
            if self.state == OPERATOR:
                return

        if st == PAUZA:
            if self._pending and self.full[self._pending[0] - 1]:
                self._belts(False, bool(self.tracked))
                self._set_state(PAUZA, f"{self._pending[0]}-bo'lim to'ldi")
                return
            if self._resume_to == UZATISH:      # pauza vaqti IR muddatiga kirmasin
                self._ir_deadline = max(self._ir_deadline, t + tm.ir_timeout_s)
                self._gate_close_at = max(self._gate_close_at, t + 0.2)
            self._belts(self._resume_to in (QABUL, UZATISH), self._resume_to != KUTISH or bool(self.tracked))
            self._set_state(self._resume_to)
            return

        if st == KUTISH:
            self._belts(False, False)
            if hand:
                self._set_state(QABUL)
            return

        if st == QABUL:
            self._belts(True, True)
            if t >= self._next_presence:
                self._next_presence = t + tm.presence_period_s
                # bitta kadrdagi shovqin (soya, buzilgan kadr) stolni to'xtatmasin: bir necha marta ketma-ket
                self._presence_hits = self._presence_hits + 1 if self.hw.camera.presence() else 0
                if self._presence_hits >= tm.presence_confirm:
                    self._presence_hits = 0
                    self._belts(False, True)
                    self._capture_at = t + tm.settle_s
                    self._retries = 0
                    self._set_state(ANIQLASH)
                    return
            if not self.tracked and t - self._last_activity > tm.idle_after_s:
                self._belts(False, False)
                self.hw.camera.set_background()
                self._set_state(KUTISH)
            return

        if st == ANIQLASH:
            self._belts(False, True)
            if t >= self._capture_at:
                self._classify()
            return

        if st == NAVBAT:
            self._belts(False, True)
            b = self._pending[0]
            if self.full[b - 1]:
                self._resume_to = NAVBAT
                self._set_state(PAUZA, f"{b}-bo'lim to'ldi")
                return
            if self._spacing_ok():
                self._gate(True)
                self._gate_open_at = t
                self._gate_open_mm = self._dozator_mm()
                self._gate_close_at = t + self.cfg.servo.gate_open_s
                self._ir_deadline = t + tm.ir_timeout_s
                self._belts(True, True)
                self._set_state(UZATISH)
            return

        if st == UZATISH:
            self._belts(self._gate_open, True)
            if self._gate_open:
                # Orqadagi buyum aniqlanmasdan o'tib ketmasligi uchun shlyuz yopiladi, agar:
                # buyum kamera zonasidan chiqsa, dozator release_mm yursa yoki gate_open_s tugasa
                moved = self._dozator_mm() - self._gate_open_mm
                if (t >= self._gate_close_at or moved >= self.cfg.sorting.release_mm
                        or (t - self._gate_open_at >= 0.1 and not self.hw.camera.presence())):
                    self._gate(False)
                    self._belts(False, True)
            if self._pending is None:          # IR tasdiqladi
                if not self._gate_open:
                    self._next_presence = t + 0.3   # tushgan buyum kameradan chiqib ketishi uchun
                    self._set_state(QABUL)
            elif t >= self._ir_deadline:
                self._gate(False)
                self._belts(False, False)
                self.log.write(t, "fault", reason="jam", bin=self._pending[0])
                self._set_state(XATO, "Tiqilib qoldi: shlyuzni tekshiring")
            return

    def _classify(self) -> None:
        s = self.cfg.sorting
        frame = self.hw.camera.capture()
        d = decide(self.clf.probs(frame), list(self.cfg.vision.labels), s.min_confidence, s.needle_priority)
        if d.kind != "bin" and self._retries < s.max_retries:
            self._retries += 1
            self._capture_at = self._now + self.cfg.timing.settle_s
            return
        if d.kind == "empty":
            # buyumni sezish adashgan (soya, yorug'lik): model bir necha marta "bo'sh" dedi
            self.log.write(self._now, "false_presence", confidence=round(d.confidence, 3))
            self.hw.camera.set_background()
            self._next_presence = self._now + self.cfg.timing.settle_s
            self._set_state(QABUL, "lenta bo'sh")
            return
        if d.kind == "unknown":
            self.log.write(self._now, "unknown", guess=d.label, confidence=round(d.confidence, 3),
                           reason=d.reason, action=s.unknown_action)
            if s.unknown_action == "operator":
                self._call_operator("low_confidence",
                                    f"Buyum aniqlanmadi ({d.label}? {d.confidence:.0%}). Olib tashlang yoki "
                                    "to'g'rilang, keyin eshikni ochib-yoping")
                return
            if d.reason == "igna_ehtimoli":           # igna bo'lishi mumkin: bint bo'limiga emas, sharps ga
                bin_no, label = 1, "aniqlanmadi (igna?)"
            else:
                bin_no, label = s.unknown_bin, "aniqlanmadi"
        else:
            bin_no, label = d.bin, self.cfg.bins.names[d.bin - 1]
        self._pending = (bin_no, d.confidence)
        self._last_label = f"{label} -> {bin_no}"
        self.log.write(self._now, "classified", bin=bin_no, confidence=round(d.confidence, 3),
                       retries=self._retries, **({"reason": d.reason} if d.reason else {}))
        self._set_state(NAVBAT)

    def _call_operator(self, reason: str, message: str) -> None:
        self._belts(False, False)
        self._gate(False)
        self._pending = None
        self._fault_door_opened = False
        self._operator_reason = reason
        self.log.write(self._now, "operator_call", reason=reason)
        self._set_state(OPERATOR, message)

    def _spacing_ok(self) -> bool:
        if not self.tracked:
            return True
        last = self.tracked[-1]
        return self._main_mm() - last.start_mm >= self.cfg.sorting.min_spacing_mm

    def _on_ir(self) -> None:
        self._last_activity = self._now
        if self._pending is None:
            b, label = self.cfg.sorting.unknown_bin, "kutilmagan"
            self.log.write(self._now, "unexpected_item")
            if self.cfg.sorting.unknown_action == "operator":
                self.tracked.append(Tracked(b, self._main_mm(), label))
                self._call_operator("unexpected", "Aniqlanmagan buyum asosiy lentaga tushdi. "
                                                  "Olib tashlang, keyin eshikni ochib-yoping")
                return
        else:
            b, label = self._pending[0], self._last_label
            self._pending = None
        self.tracked.append(Tracked(b, self._main_mm(), label))

    # ------------------------------------------------------------------ deflektorlar
    def _track(self) -> None:
        s, sv, pca = self.cfg.sorting, self.cfg.servo, self.cfg.pca9685
        mm = self._main_mm()
        want = [False] * 4
        keep = []
        for it in self.tracked:
            x = s.entry_x + (mm - it.start_mm)
            if it.bin <= 4:
                dx = s.deflector_x[it.bin - 1]
                if x > dx + s.pass_mm:
                    self._done(it)
                    continue
                if x >= dx - s.lead_mm:
                    want[it.bin - 1] = True
            elif x > s.belt_end_x + 20:
                self._done(it)
                continue
            keep.append(it)
        self.tracked = keep
        for k in range(4):
            if want[k] != self._defl[k]:
                self._defl[k] = want[k]
                self.hw.pwm.set_angle(pca.deflector_channels[k], sv.deflector_active if want[k] else sv.deflector_rest)

    def _done(self, it: Tracked) -> None:
        self.counts[it.bin - 1] += 1
        self._last_activity = self._now
        self.log.write(self._now, "sorted", bin=it.bin, label=it.label)
        if it.bin == 1:
            self._update_sharps()

    # ------------------------------------------------------------------ to'lish darajasi
    def _update_sharps(self) -> None:
        cap = self.cfg.bins.sharps_capacity
        self.fill[0] = min(1.0, self.counts[0] / cap)
        self.full[0] = self.counts[0] >= cap
        self.warn[0] = self.counts[0] >= 0.8 * cap

    def reset_sharps(self) -> None:
        """1-bo'lim (sharps konteyner) almashtirilgach chaqiriladi."""
        self.counts[0] = 0
        self._update_sharps()
        self.log.write(self._now, "sharps_reset")

    def _read_fill(self) -> None:
        bn = self.cfg.bins
        self._update_sharps()
        for b in (2, 3, 4, 5):
            mm = self.hw.fill.read_mm(b)
            if mm is None:
                self.fill[b - 1] = None
                continue
            span = bn.sensor_to_bottom_mm - bn.full_mm
            self.fill[b - 1] = max(0.0, min(1.0, 1 - (mm - bn.full_mm) / span))
            was = self.full[b - 1]
            self.full[b - 1] = mm <= bn.full_mm
            self.warn[b - 1] = mm <= bn.warn_mm
            if self.full[b - 1] and not was:
                self.log.write(self._now, "bin_full", bin=b)

    def _leds(self) -> None:
        pairs = self.cfg.pca9685.led_channels
        blink = int(self._now * 2) % 2 == 0
        for i, (red, green) in enumerate(pairs):
            if self.state == XATO:
                r, g = (1.0 if blink else 0.0), 0.0
            elif self.state == OPERATOR:            # sariq miltillaydi
                r, g = (1.0, 0.6) if blink else (0.0, 0.0)
            elif self.full[i]:
                r, g = 1.0, 0.0
            elif self.warn[i]:
                r, g = 1.0, 0.6
            else:
                r, g = 0.0, 1.0
            self.hw.pwm.set_duty(red, r)
            self.hw.pwm.set_duty(green, g)

    # ------------------------------------------------------------------ UV-C
    def _uvc_control(self, door_closed: bool, hand: bool) -> None:
        u, t = self.cfg.uvc, self._now
        safe = door_closed and not self._flap_open and not hand and self.state == KUTISH
        if self._uv_on:
            if not safe:
                self._uvc(False, "blokirovka")
            elif t >= self._uv_until:
                self._uvc(False, "sikl tugadi")
                self._uv_last_done = t
            return
        due = self._uv_last_done is None or t - self._uv_last_done >= u.repeat_every_s
        if safe and due and t - self._idle_since >= u.start_after_idle_s:
            self._uv_until = t + u.duration_s
            self._uvc(True, "dezinfeksiya")
