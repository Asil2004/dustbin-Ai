# ATCS: Aqlli tibbiy chiqindi saralash stoli, boshqaruv dasturi

Bu dastur **ATCS.01.00.000** stolini (Loyiha: nurse ai) boshqaradi. Pin va kanal raqamlari, deflektor
o'qlari va lenta tezliklari chizmalardan olingan: E1 elektr sxemasi va YaCh saralash mexanizmi.
Ular [config/atcs.toml](config/atcs.toml) faylida turadi.

Boshqaruvchi: **Raspberry Pi 4B** (sxemadagidek) yoki Pi 3 (pinlar bir xil). Kamera: USB webkamera
(standart) yoki Pi Camera Module (`vision.camera`). Dastur Python 3.11+ da yozilgan.
Simulyatsiya va testlar Windows yoki PC da apparatsiz ishlaydi.

## Ish tartibi

| Holat | Nima bo'ladi |
|---|---|
| KUTISH | Lentalar to'xtagan. ToF datchik qo'lni sezsa, qabul eshikchasi (SG90) ochiladi |
| QABUL | Dozator (70 mm/s) va asosiy lenta (200 mm/s) yuradi. Kamera shlyuz oldida buyumni sezadi |
| ANIQLASH | Dozator to'xtaydi, kadr olinadi, model bo'limni aniqlaydi. Ishonch past bo'lsa, 2 marta qayta suratga oladi |
| NAVBAT | Oldingi buyum lentada ≥150 mm uzoqlashishi kutiladi |
| UZATISH | Shlyuz ochiladi va buyum kameradan chiqishi bilan yopiladi. IR datchik buyum asosiy lentaga tushganini tasdiqlaydi |
| PAUZA | Eshik ochiq (hamma narsa to'xtaydi) yoki kerakli bo'lim to'lgan |
| XATO | Buyum shlyuzda tiqilib qolgan. Operator eshikni ochib-yopsa, tizim tiklanadi |
| OPERATOR | Buyum 3 urinishda ham ishonchli aniqlanmadi yoki aniqlanmagan buyum asosiy lentaga tushdi. Stol to'liq to'xtaydi, LEDlar sariq miltillaydi. Operator buyumni olib tashlaydi yoki to'g'rilab qo'yadi, keyin eshikni ochib-yopadi. Buyum joyida qolgan bo'lsa, qayta aniqlanadi |

Asosiy lentadagi har bir buyum enkoder bo'yicha kuzatiladi. Buyum o'z deflektoriga 30 mm qolganda deflektor
55° buriladi va buyum o'qdan 60 mm o'tgach qaytadi. 5-bo'lim lenta oxirida.

**UV-C** faqat stol bo'sh turganda yonadi (60 s dan keyin, 5 daqiqa, soatiga bir marta). Eshik ochilsa, qabul
eshikchasi ochilsa yoki qo'l yaqinlashsa, darhol o'chadi.

**To'lish:** 2–5-bo'limlar JSN-SR04T datchiklari bilan o'lchanadi. 1-bo'lim (sharps 3 L) datchiksiz,
lentadan tushgan ignalar soni bo'yicha hisoblanadi. LED holati: yashil = bo'sh, sariq = 80 % dan oshgan,
qizil = to'lgan.

## Tuzilma

```
atcs/
  controller.py   asosiy holat mashinasi, deflektorlar, UV-C, to'lish
  hal/base.py     apparat interfeyslari
  hal/rpi.py      Raspberry Pi drayverlari (PCA9685, TB6612, enkoderlar, VL53L0X, JSN-SR04T, kamera, TFT)
  hal/sim.py      stol simulyatori
  vision.py       TFLite klassifikatori, ROI kesish
  cameras.py      USB webkamera va Pi kamerasi (bir xil sozlama bilan ochiladi)
  collect.py      o'qitish uchun rasm yig'ish
  livetest.py     modelni kamerada jonli sinash (noutbukda ham)
  display.py      3,5" TFT holat ekrani
  events.py       hodisalar jurnali (logs/events.jsonl)
training/train.py MobileNetV3 o'qitish -> int8 TFLite (Google Colab)
config/atcs.toml  barcha sozlamalar
tests/            28 ta sinov (simulyatsiya, kamera, kamera kesimi, jonli sinov)
```

## Kompyuterda sinash (Windows)

```bash
py -m unittest discover -s tests -v
```

```bash
py -m atcs --sim --items 20 --screen docs/ekran_namuna.png
```

## Raspberry Pi ga o'rnatish

```bash
sudo apt install -y python3-opencv python3-numpy python3-pil
```

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements-pi.txt
```

```bash
.venv/bin/python -m atcs
```

Raspberry Pi OS **Bookworm 64-bit** kerak (32-bit da `ai-edge-litert` yo'q). `raspi-config` orqali I2C va SPI ni yoqing.
Pi kamerasi (`vision.camera = "csi"`) ishlatilsa, `python3-picamera2` ni ham o'rnating.

## Haqiqiy stolda ishga tushirishdan oldin

1. **Model:** `models/atcs.tflite` hali yo'q. Uni yig'ish va o'qitish yo'riqnomasi:
   [docs/kamera_aniqlash.md](docs/kamera_aniqlash.md). Rasm yig'ish uchun `python -m atcs.collect`,
   o'qitish uchun [training/train.py](training/train.py) ishlatiladi.
2. **Kalibrlash:** `belts.*_mm_per_count`, `*_max_mm_s`, `sorting.entry_x`, servo burchaklari va
   `bins.sensor_to_bottom_mm` qiymatlarini stolda o'lchab, sozlamalarga yozish kerak.
3. **Rele:** ko'p 5 V rele modullari past signal bilan yoqiladi. Shunday bo'lsa, `RelayOut(active_high=False)` qiling.
4. **UV-C apparat blokirovkasi:** dasturdan tashqari, gerkonni rele zanjiriga ketma-ket ulash tavsiya etiladi
   (E1 sxemasi, 4-izoh).
5. **Aniqlanmagan buyumlar:** `sorting.unknown_action = "operator"` (standart) bo'lsa, stol to'xtab operatorni
   chaqiradi. `"bin"` bo'lsa, buyum `unknown_bin` ga (5-bo'limga) yuboriladi, lekin bu rejimda ishonchi past
   igna ham bint bo'limiga tushib qolishi mumkin.
6. **Sharps konteyner** almashtirilgach, `Controller.reset_sharps()` chaqiriladi. Buning uchun hozircha
   tugma yoki menyu yo'q.
