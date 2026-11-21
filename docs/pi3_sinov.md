# Raspberry Pi 3 da birinchi sinov: kamera

Maqsad: Pi 3 ga USB webkamerani ulab, kamera nimani ko'rayotganini ekranda ko'rish. Bu sinovda tekshiriladi:
fokus, kesish zonasi (ROI), buyumni sezish va tezlik. Motorlar, servolar va datchiklar kerak emas.

## 1. Tizimni yozish (kompyuterda)

**Raspberry Pi Imager** dasturida:

- Qurilma: **Raspberry Pi 3**.
- Tizim: **Raspberry Pi OS (64-bit)**. Pi'ga monitor ulasangiz, oddiy (desktop) versiyani oling.
  Monitorsiz ishlatsangiz, **Lite (64-bit)** yetarli. 32-bit versiya to'g'ri kelmaydi.
- Sozlamalar (tishli g'ildirak belgisi):
  - hostname: `atcs`;
  - foydalanuvchi va parol;
  - Wi-Fi;
  - **SSH ni yoqing**.

Quvvat bloki **5 V 2,5 A** bo'lsin. Webkamera ham Pi'dan quvvat oladi, kuchsiz blokda Pi qotib qoladi.

## 2. Pi'ga ulanish va o'rnatish

Kompyuterdan, Pi bilan bir xil Wi-Fi tarmog'ida turib:

```bash
ssh <foydalanuvchi>@atcs.local
```

Keyingi buyruqlarni Pi'da birma-bir bajaring:

```bash
sudo apt update && sudo apt install -y git python3-opencv python3-numpy python3-pil python3-venv v4l-utils
```

```bash
git clone https://github.com/Asil2004/dustbin-Ai.git && cd dustbin-Ai
```

## 3. Kamerani tekshirish

Webkamerani Pi'ning USB portiga ulang, keyin:

```bash
v4l2-ctl --list-devices
```

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext | grep -A1 MJPG | grep 1920x1080
```

Ikkinchi buyruq `1920x1080` qatorini chiqarishi kerak. Hech narsa chiqmasa, kamera bu o'lchamni MJPG formatida
bermaydi: unda chiqqan ro'yxatni menga yuboring.

## 4. Kamerani ekranda ko'rish

```bash
python3 -m atcs.livetest --no-model
```

- **Pi'ga monitor ulangan bo'lsa** (desktop tizim), oyna ochiladi. Tugmalar: `q` chiqish, `f` ROI / to'liq kadr,
  `b` fonni yangilash, `s` kadrni `livetest/` ga saqlash.
- **Monitorsiz bo'lsa**, dastur o'zi brauzer rejimiga o'tadi va manzilni yozadi, masalan
  `Brauzerda oching: http://192.168.1.50:8000`. Shu manzilni noutbuk yoki telefon brauzerida oching
  (bir xil Wi-Fi tarmog'ida). Tugmalar sahifaning pastida. To'xtatish: Pi terminalida `Ctrl+C`.

Diqqat: brauzer rejimida tasvirni shu Wi-Fi tarmog'idagi har kim ko'rishi mumkin. Sinovdan keyin to'xtating.

Birinchi qatorda kamera sozlamalari chiqadi:
`USB kamera 1920x1080: fokus=..., ekspozitsiya=..., oq balans=... (qotirildi)`.

## 5. Nimani tekshirish kerak

| Tekshiruv | Qanday | Yaxshi natija |
|---|---|---|
| Tiniqlik | Kamerani 115 mm balandlikka qo'ying, ostiga igna qo'ying, `s` bilan kadr saqlang va kompyuterda kattalashtirib ko'ring | Igna qirralari aniq ko'rinadi. Xira bo'lsa, `config/atcs.toml` dagi `usb_focus` ga qiymat yozing (masalan 0, 50, 100, 150, 200 ni sinab ko'ring) |
| ROI | `f` bilan to'liq kadrga o'ting | Ramka buyum to'xtaydigan joyni qamrab oladi. Kerak bo'lsa `vision.roi` ni o'zgartiring |
| Bo'sh lenta | Hech narsa qo'ymang, `b` bosing | Tepada `bo'sh (o'zgargan 0.00% ...)`, bir necha soniya davomida 0,05 % dan oshmaydi |
| Igna | Eng kichik ignani ramka ichiga qo'ying | `BUYUM BOR`, foiz chegaradan **ikki barobar** yuqori bo'lsa ishonchli |
| Qo'lqop, bint | Katta buyumlarni qo'ying | `BUYUM BOR`, foiz ancha yuqori |
| Tezlik | O'ng yuqoridagi fps | Pi 3 da taxminan 10–15 fps bo'lsa yetarli |
| Quvvat | Boshqa terminalda `vcgencmd get_throttled` | `throttled=0x0`. Boshqa qiymat chiqsa, quvvat bloki kuchsiz |

**Sozlash:** igna foizi chegaradan aniq oshmasa, `presence_min_fraction` ni kamaytiring (masalan 0.0005).
Bo'sh lentada ham `BUYUM BOR` chiqib qolsa (LED miltillashi, soya), uni oshiring yoki `presence_pixel_diff` ni
35–40 qiling. O'zgartirgandan keyin dasturni qayta ishga tushiring.

Bu natijalarni (fps, bo'sh lenta va igna foizlari, saqlangan kadr) menga yuboring. Ular asosida sozlamalarni
aniq qo'yamiz.

## 6. Keyin: model bilan sinov

Model o'qitilgach, uni Pi'ga ko'chirib, aniqlash tezligini o'lchaymiz:

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install ai-edge-litert
```

```bash
.venv/bin/python -m atcs.livetest --model models/atcs.tflite
```

O'ng yuqorida `model ... ms` chiqadi. Bu bitta buyumni aniqlash vaqti, Pi 3 da bir necha yuz ms bo'lishi kutiladi.

## Keyingi qadam: qurilmalarni ulash

Kamera ishlagach, servolar (PCA9685), motorlar (TB6612), enkoderlar va datchiklarni bittalab ulab sinaymiz.
Pi 3 ning pinlari Pi 4 bilan bir xil, ulash sxemasi o'zgarmaydi.
