"""3,5" TFT (480x320) uchun holat ekrani: holat, bo'limlar to'lishi, so'nggi buyum, UV-C."""
from __future__ import annotations

from .hal.base import Status

BIN_COLORS = [(242, 196, 40), (150, 160, 235), (110, 205, 140), (95, 175, 235), (240, 125, 135)]
BIN_SHORT = ["Igna", "Qo'lqop", "Shisha", "Plastik", "Bint"]
STATE_COLORS = {"XATO": (230, 70, 70), "OPERATOR": (240, 170, 40), "PAUZA": (240, 170, 40),
                "KUTISH": (150, 150, 150)}


def render(st: Status, size=(480, 320)):
    from PIL import Image, ImageDraw, ImageFont
    w, h = size
    img = Image.new("RGB", size, (16, 20, 26))
    d = ImageDraw.Draw(img)
    def font(names, size):
        for n in names:
            try:
                return ImageFont.truetype(n, size)
            except OSError:
                pass
        return ImageFont.load_default(size)

    big = font(["DejaVuSans-Bold.ttf", "arialbd.ttf"], 26)
    small = font(["DejaVuSans.ttf", "arial.ttf"], 17)

    d.text((14, 10), st.state, font=big, fill=STATE_COLORS.get(st.state, (90, 210, 150)))
    if st.uvc_on:
        d.text((w - 110, 14), "UV-C", font=big, fill=(190, 120, 255))
    d.text((14, 44), st.message or st.last_item or "", font=small, fill=(220, 220, 220))

    top, bar_h, gap = 80, 34, 9
    for i in range(5):
        y = top + i * (bar_h + gap)
        d.text((14, y + 7), f"{i + 1} {BIN_SHORT[i]}", font=small, fill=BIN_COLORS[i])
        x0, x1 = 140, w - 90
        d.rectangle((x0, y, x1, y + bar_h), outline=(70, 75, 85))
        f = st.fill[i]
        if f is not None:
            col = (230, 70, 70) if f >= 0.95 else BIN_COLORS[i]
            d.rectangle((x0 + 2, y + 2, x0 + 2 + int((x1 - x0 - 4) * f), y + bar_h - 2), fill=col)
            d.text((x1 + 8, y + 7), f"{int(f * 100)}%", font=small, fill=(220, 220, 220))
        else:
            d.text((x1 + 8, y + 7), "--", font=small, fill=(120, 120, 120))
    d.text((14, h - 26), f"Jami: {sum(st.counts)}  " + " ".join(str(c) for c in st.counts),
           font=small, fill=(160, 160, 160))
    return img
