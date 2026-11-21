"""Ishga tushirish:
  python -m atcs --sim                 # apparatsiz simulyatsiya (Windows/PC)
  python -m atcs --sim --items 20 --screen ekran.png
  python -m atcs                       # Raspberry Pi da haqiqiy apparat bilan
"""
from __future__ import annotations

import argparse
import random
import time

from . import config as config_mod
from .controller import OPERATOR, Controller
from .events import EventLog


def run_sim(cfg, n_items: int, seed: int, screen: str | None, quiet: bool) -> int:
    from .hal import sim
    hw, world = sim.build(cfg, echo=not quiet)
    log = EventLog()
    ctl = Controller(cfg, hw, sim.SimClassifier(), log)
    rng = random.Random(seed)
    names = {1: "ochiq igna", 2: "qo'lqop", 3: "shisha ampula", 4: "shprits korpusi", 5: "bint"}
    for i in range(n_items):
        b = rng.randint(1, 5)
        conf = rng.choice([0.97, 0.94, 0.92, 0.88, 0.81, 0.55])
        world.add_item(b, conf, f"{i + 1:02d} {names[b]}")

    dt = 1 / cfg.loop.hz
    ctl.start(0.0)
    world.hand_near = True
    limit = 20 + n_items * 6
    operator_at = None
    while world.t < limit:
        if world.t > 1.0:
            world.hand_near = False
        # operator chaqirilsa: 3 s da yetib keladi, buyumni olib tashlaydi, eshikni ochib-yopadi
        if ctl.state == OPERATOR and operator_at is None:
            operator_at = world.t + 3.0
        if operator_at is not None and world.t >= operator_at:
            if world.door_is_closed:
                world.operator_remove()
                world.door_is_closed = False
                operator_at = world.t + 1.0
            else:
                world.door_is_closed = True
                operator_at = None
        ctl.step(world.t)
        world.advance(dt)
        finished = sum(i.stage in ("done", "removed") for i in world.items)
        if finished == n_items and not ctl.tracked:
            break

    ok = low = removed = 0
    print("\nBuyum                 kerak  tushdi")
    for it in (i for i in world.items if i.stage in ("done", "removed")):
        if it.stage == "removed":
            mark = f"ishonch past ({it.confidence:.2f}) -> operator olib tashladi"; removed += 1
        elif it.landed_bin == it.true_bin:
            mark = "OK"; ok += 1
        elif it.confidence < cfg.sorting.min_confidence and it.landed_bin == cfg.sorting.unknown_bin:
            mark = f"ishonch past ({it.confidence:.2f}) -> {cfg.sorting.unknown_bin}-bo'lim"; low += 1
        else:
            mark = "XATO"
        print(f"  {it.name:<20} {it.true_bin:>5}  {str(it.landed_bin or '-'):>6}  {mark}")
    bad = len(world.landed()) - ok - low
    lost = n_items - len(world.landed()) - removed
    print(f"\nTo'g'ri: {ok}, operatorga: {removed}, 5-bo'limga (ishonch past): {low}, xato: {bad}, "
          f"yo'qolgan: {lost}; vaqt {world.t:.1f} s, holat: {ctl.state}")
    if screen:
        from .display import render
        render(ctl.status()).save(screen)
        print(f"Ekran rasmi: {screen}")
    return 0 if bad == 0 and lost == 0 else 1


def run_pi(cfg) -> int:
    from .hal import rpi
    from .vision import TfliteClassifier
    hw = rpi.build(cfg)
    ctl = Controller(cfg, hw, TfliteClassifier(cfg.vision.model, cfg.vision.input_size, cfg.vision.roi), EventLog(cfg.log.events))
    period = 1 / cfg.loop.hz
    ctl.start(time.monotonic())
    try:
        nxt = time.monotonic()
        while True:
            ctl.step(time.monotonic())
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        ctl.shutdown()
        hw.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="atcs", description="Aqlli tibbiy chiqindi saralash stoli")
    ap.add_argument("--config", help="sozlamalar fayli (standart: config/atcs.toml)")
    ap.add_argument("--sim", action="store_true", help="apparatsiz simulyatsiya")
    ap.add_argument("--items", type=int, default=12, help="simulyatsiyadagi buyumlar soni")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--screen", help="simulyatsiya oxirida displey rasmini PNG ga saqlash")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = config_mod.load(a.config)
    return run_sim(cfg, a.items, a.seed, a.screen, a.quiet) if a.sim else run_pi(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
