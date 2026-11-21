# Kamera bilan aniqlash: bosqichma-bosqich

Kamera shlyuz oldida to'xtagan **bitta** buyumni suratga oladi. Model uni 5 sinfdan biriga ajratadi:
`igna`, `qolqop_shlang`, `shisha`, `plastik`, `bint_paxta`. Ishonch 70 % dan past bo'lsa,
stol to'xtab operatorni chaqiradi.

```
[1] Rasm yig'ish        [2] (ixtiyoriy) Roboflow     [3] O'qitish (Colab)       [4] Stolga qo'yish
 atcs.collect     --->   tekshirish, tartiblash --->  training/train.py   --->  models/atcs.tflite
 dataset/<sinf>/*.jpg    "Folder Structure" eksport   atcs.tflite + hisobot     python -m atcs
```

## 0. Kamerani sozlash (stolda, bir marta)

- Kamera lentadan 115 mm balandda, tik pastga qaragan bo'lishi kerak.
- Standart kamera USB webkamera (`vision.camera = "usb"`), 1920×1080 da ishlaydi (`vision.capture_size`).
  4K kamera bo'lsa ham 1080p yetarli, Pi 3 uchun esa kerak. Kamera bu o'lchamni bermasa, dastur xato chiqaradi:
  unda kamera qo'llaydigan 16:9 o'lchamni yozing.
- Ochilgach kamera `usb_warmup_s` soniya o'zi fokus va ekspozitsiyani topadi, keyin ular qotiriladi.
  Qotirilgan qiymatlar ekranga chiqadi (`USB kamera 1920x1080: fokus=..., ekspozitsiya=...`). Rasm xira chiqsa,
  `usb_focus` ga qo'lda qiymat bering. Hamma kamera drayveri ham qotirishni qo'llamaydi: rasmlar orasida
  yorqinlik sakrasa, bu shundan.
- Yorug'lik faqat ichki LED panellardan tushsin.
- `python -m atcs.collect --label igna` ni ishga tushiring va `dataset/roi_tekshiruv.jpg` ni oching.
  Qizil ramka shlyuz oldida buyum to'xtaydigan zonani qamrab olishi kerak. To'g'ri kelmasa,
  `vision.roi` ni o'zgartiring. **ROI o'zgarsa, yig'ilgan rasmlarni qaytadan yig'ish kerak.**

## 1. Rasm yig'ish

```bash
python -m atcs.collect --label igna
```

Buyumni shlyuz oldiga qo'ying va Enter bosing. Keyin buyumni burib yoki boshqasiga almashtirib, yana Enter bosing.
`r` oxirgi rasmni o'chiradi, `q` chiqadi.

**Har sinf uchun kamida 200–300 ta rasm kerak, igna uchun 400 tadan ko'p.** Rasmlar soni emas, xilma-xilligi muhim:

| Sinf | Nimalarni suratga olish kerak |
|---|---|
| igna | Turli o'lchamdagi ochiq ignalar (G18–G30): yolg'iz, plastik qopqoq bilan, lentaning turli joyida, turli burchakda |
| qolqop_shlang | Har xil rangdagi (ko'k, oq, binafsha) qo'lqoplar: g'ijimlangan, yoyilgan, ichi tashqariga ag'darilgan. Sistema shlanglari: o'ralgan va to'g'ri |
| shisha | 1–10 ml ampulalar (butun va singan), flakonlar, yorliqli va yorliqsiz |
| plastik | Shprits korpuslari (ignasiz), 500 ml NaCl flakonlari, qopqoqlar, plastik idishlar |
| bint_paxta | Bint, paxta, salfetkalar: toza va bo'yalgan |

Xavfsizlik uchun faqat **yangi, steril** ignalar va toza materiallardan foydalaning. Qon izlarini qizil bo'yoq bilan taqlid qiling.

**Alohida test to'plami** yig'ing: boshqa kuni va boshqa buyumlar bilan, har sinfdan taxminan 50 ta rasm, `test/` papkasiga.
Bitta buyumning 10 ta surati o'qitishda ham, testda ham bo'lsa, natija haddan tashqari yaxshi ko'rinadi.

**Noutbukda yig'ish:** stoldagi USB webkamerani noutbukka ulab, xuddi shu buyruq bilan yig'sa bo'ladi.
Kamera 115 mm balandda, tik pastga, lentaga o'xshash qora yuza ustida va stoldagi yorug'lik bilan tursa,
bu rasmlar stol uchun ham yaroqli. Noutbukning o'rnatilgan kamerasi birinchi bo'lsa, tashqi kamera uchun
`--camera-index 1` qo'shing. Noutbukning o'z kamerasidagi rasmlar stol uchun yaramaydi.

## 2. Roboflow (ixtiyoriy)

Roboflow rasmlarni ko'zdan kechirish, noto'g'ri sinfdagilarni topish, dataset versiyalarini saqlash
va jamoa bilan bo'lishish uchun qulay.

1. Yangi loyiha oching, turi **Classification** (Object Detection emas).
2. `dataset/` papkasini yuklang. Papka nomlari sinf nomlari bo'lib qoladi.
3. Version yaratishda **augmentatsiyani o'chiring.** `train.py` buni o'zi qiladi. Roboflow nusxalari
   tekshirish to'plamiga tushsa, natija yolg'on yaxshi ko'rinadi.
4. Export formatini **Folder Structure** qilib tanlang. Natijadagi `train/` va `valid/` papkalarini bitta papkaga
   birlashtiring yoki `train/` ni `--data`, `valid/` ni `--test` qilib bering.

Diqqat: bepul rejada loyihalar ochiq (public) bo'lishi mumkin. Joriy shartlarni Roboflow saytida tekshiring.
Rasmlarda bemor ma'lumoti, yorliqdagi ism yoki shtrix-kod bo'lmasin.

Roboflow'ning o'zida o'qitilgan model ham Pi'da ishlaydi. Lekin u uchun Roboflow kutubxonasi va API kalit kerak bo'ladi,
modelni yuklab olishda internet talab qilinadi va u bu loyihadagi `TfliteClassifier` bilan ishlamaydi.
Shuning uchun Roboflow'ni dataset uchun ishlatib, modelni o'zimiz o'qitamiz.

## 3. O'qitish (Google Colab, bepul GPU)

1. `dataset/` ni zip qiling va Colab'ga yoki Google Drive'ga yuklang. `training/train.py` ni ham yuklang.
2. Runtime → Change runtime type → **GPU**.
3. Quyidagilarni yacheykalarda ishga tushiring:

```
!unzip -q dataset.zip -d /content
!python train.py --data /content/dataset --test /content/test --out /content/natija
```

Natijada quyidagilar paydo bo'ladi: `natija/atcs.tflite` (int8, taxminan 3–4 MB), `labels.txt` va `report.json`.

## 4. Natijani o'qish

Hisobotda xatolar jadvali (confusion matrix) bo'ladi. Ikkita raqamga alohida e'tibor bering:

- **`IGNA boshqa bo'limga`** 0 bo'lishi shart. Bu igna ishonch bilan boshqa bo'limga ketgan holatlar soni.
  0 bo'lmasa, skript xato kodi bilan tugaydi va bu modelni stolga qo'yib bo'lmaydi.
- **`operatorga`** stolni operator qanchalik tez-tez chaqirishini ko'rsatadi. Taxminan 5 % dan oshsa,
  ko'proq rasm yig'ish kerak.

Natija yomon bo'lsa, eng ko'p adashtirilgan juftlikni toping (masalan, shisha ↔ plastik) va shu ikki sinfga
ko'proq va xilma-xilroq rasm qo'shing.

## 4a. Noutbukda jonli sinash (Pi'siz)

O'qitilgan modelni stolga qo'yishdan oldin noutbukda, stoldagi webkamera bilan sinab ko'ring:

```bash
py -m pip install ai-edge-litert opencv-python
```

```bash
py -m atcs.livetest --model natija/atcs.tflite
```

Oynada ROI ramkasi, qaror va 5 sinf ehtimoli ko'rinadi. Ramka yashil bo'lsa, buyum bo'limga ketadi,
sariq bo'lsa, stol operatorni chaqiradi (chegara `sorting.min_confidence`). Tugmalar: `f` ROI va to'liq kadr
orasida almashtiradi (kamera hali stolga o'rnatilmagan bo'lsa qulay), `s` kadrni `livetest/` ga saqlaydi,
`q` chiqadi. Boshqa kamera uchun `--camera-index 1` qo'shing. Pi'da ham xuddi shu buyruq ishlaydi,
lekin oyna ochilishi uchun Pi'ga monitor ulangan bo'lishi kerak (Lite tizimida grafik muhit yo'q).

## 5. Stolga qo'yish

```bash
scp natija/atcs.tflite pi@<stol-ip>:~/atcs/models/atcs.tflite
```

```bash
python -m atcs
```

Birinchi ishga tushirishda turli buyumlarni bittadan tashlab, `logs/events.jsonl` dagi `classified` yozuvlarini
kuzating.

## Keyingi yaxshilashlar

- `boshqa` sinfi: ro'yxatda yo'q narsalar (telefon, qo'l, qog'oz). Ular operatorga yuboriladi.
- Ignalar yupqa (0,3–1,2 mm). Ular yetarlicha ko'rinmasa, ROI ni toraytiring yoki `input_size` ni 320 qiling.
  Ikkalasida ham qayta o'qitish kerak.
- Bir kadrda bir nechta buyum tez-tez tushsa, klassifikatsiya o'rniga obyekt aniqlash (YOLO) kerak bo'ladi.
  Shunda Roboflow'da ramka chizib belgilash foydali bo'ladi.
