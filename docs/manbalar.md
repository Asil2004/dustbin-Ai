# O'xshash loyihalardan olingan g'oyalar

GitHub'dagi 5 ta loyihaning kodi o'qib chiqildi. Ularning birortasida ham LICENSE fayli yo'q (MedisortAI
README'da faqat MIT belgisi bor). Shuning uchun **ulardan kod ko'chirilmadi**: g'oyalar olinib, ATCS uchun qaytadan
yozildi. Ularning modellari ham ishlatilmadi: sinflari boshqa (maishiy chiqindi yoki boshqa tibbiy toifalar), eski
formatda (TensorFlow 1) yoki boshqa kamerada o'qitilgan.

| Loyiha | Nima haqida | Biz uchun |
|---|---|---|
| [BowenBZ/Medical_Waste_Sorting_System](https://github.com/BowenBZ/Medical_Waste_Sorting_System) | Tibbiy chiqindi qutisi, Raspberry Pi 3B+, MobileNet, servo + qadamli motor. Sinflar: fon, dori, sharps, kimyo | **Igna ustuvorligi** va **fon sinfi** g'oyalari |
| [mAdLaD09/dustbin (SegreSmart)](https://github.com/mAdLaD09/dustbin) | Ho'l/quruq saralash, Pi 4, USB webkamera, MobileNetV3 TFLite | **Bo'sh lenta sinfi**, **dataset konvertori** (YOLO, guruhli bo'lish, takrorlarni olib tashlash), **avtomatik ishga tushirish** |
| [brendantnn/ai-smartbin](https://github.com/brendantnn/ai-smartbin) | Plastik/qog'oz/metall, Pi 5, ultratovush + webkamera + 2 servo | **Sezishni tasdiqlash** (ketma-ket), **yaroqsiz kadrni tashlash** |
| [Bhuvana-Manogar/MedisortAI](https://github.com/Bhuvana-Manogar/MedisortAI) | Tibbiy chiqindini JSST toifalariga ajratish (faqat dastur) | Qabul qilinmadi (pastda) |
| [RezaAmansyah/Medical-Waste-Classification](https://github.com/RezaAmansyah/Medical-Waste-Classification) | Niqob/qo'lqop/shprits/dori, ResNet50 | Uning **Roboflow dataseti** (CC BY 4.0) bizga qo'shimcha ma'lumot bo'la oladi |

## ATCS ga qo'shilganlar

**1. Igna ustuvorligi** (BowenBZ: "sharps ehtimoli 10 % dan oshsa, sharps"). `sorting.needle_priority = 0.15`.
Model buyumni boshqa sinf deb ishonch bilan aniqlasa ham, igna ehtimoli 15 % dan yuqori bo'lsa, buyum
1-bo'limga (sharps) ketadi. Ignasiz buyum sharps qutisiga tushsa zarari yo'q, igna boshqa qutiga tushsa xavfli.
BowenBZ'dan farqi: noaniq buyumda (hamma sinf ~20 %) qoida ishlamaydi va stol baribir operatorni chaqiradi.
Aks holda har noaniq buyum sharps qutisini to'ldirib yuborardi. Bitta qoida stolda (`atcs/vision.py: decide`),
`livetest` ekranida va o'qitish hisobotida (`train.py --needle-priority`) ishlatiladi. Hisobotda qoidaning
narxi ham chiqadi: `igna qoidasi bilan sharps ga (ignasiz)`.

**2. "bosh" sinfi: bo'sh lenta** (BowenBZ `background`, SegreSmart `empty`). Ixtiyoriy 6-sinf. Soya yoki
yorug'lik o'zgarishi tufayli kamera "buyum bor" desa, model "bo'sh" deydi va stol operatorni chaqirmasdan fonni
yangilab, ishni davom ettiradi. Igna ehtimoli yuqori bo'lsa, "bo'sh" deyilmaydi, operator chaqiriladi.
Yoqish uchun:
1. Bo'sh lentani avtomatik suratga oling (har xil yorug'likda, 300–400 ta):
   `python -m atcs.collect --label bosh --auto 300 --interval 2`
2. `train.py --labels igna,qolqop_shlang,shisha,plastik,bint_paxta,bosh` bilan o'qiting.
3. `config/atcs.toml` da `labels` oxiriga `"bosh"` qo'shing.

**3. Sezishni tasdiqlash** (ai-smartbin: `PRESENT_COUNT = 3`). `timing.presence_confirm = 2`: buyum ketma-ket
2 tekshiruvda sezilgandagina dozator to'xtaydi. Bitta shovqinli kadr stolni to'xtatmaydi.

**4. Yaroqsiz kadrni tashlash** (ai-smartbin: `MIN_BRIGHTNESS`). Pi 3 da USB 2.0 tufayli MJPEG kadr chala kelishi
mumkin ("Corrupt JPEG data"). Endi qora yoki chala kadr aniqlanib, keyingisi olinadi (`cameras.frame_ok`).

**5. Tashqi dataset importi** (SegreSmart `prepare_dataset.py`). `training/import_dataset.py`:
papka va YOLO formatlari, ramkalarni kesish, Roboflow nusxalarini guruhlash, takror rasmlarni olib tashlash,
manifest va litsenziya faylini saqlash. `train.py --extra` tashqi rasmlarni **faqat o'qitishga** qo'shadi.
Tekshirish va test faqat stol kamerasidagi rasmlarda o'tadi. Shunda hisobotdagi raqamlar stolda nima bo'lishini
ko'rsatadi, boshqa kameradagi rasmlar ularni "yaxshilab" ko'rsatmaydi.

**6. Avtomatik ishga tushirish** (SegreSmart `segresmart.service`). `deploy/atcs.service`: Pi yoqilganda stol
dasturi o'zi ishga tushadi, xato bilan to'xtasa 5 soniyadan keyin qayta yonadi.

## Qabul qilinmaganlar va sababi

- **Modelni har kadrda uzluksiz ishlatib ovoz berish** (SegreSmart: 7 kadrdan 5 tasi). Bizda buyum to'xtab
  suratga olinadi va ishonch past bo'lsa 2 marta qayta olinadi. Pi 3 da har kadrga model ishlatish juda sekin.
- **CLAHE, Canny, kontur bilan ROI topish** (MedisortAI). Bizda kamera va zona qotirilgan. Rasmni qo'shimcha
  o'zgartirish o'qitish va ish rejimi orasida farq keltirib chiqarishi mumkin.
- **Ultratovush bilan buyumni sezish** (ai-smartbin). Ingichka ignani ultratovush sezmaydi. Kamera bilan sezish
  aniqroq.
- **Bulut, Firebase, veb-panel** (ai-smartbin, SegreSmart). Tibbiy muassasada stol oflayn ishlashi kerak.
- **Tayyor modellar**. Sinflari mos emas va boshqa kamerada o'qitilgan.

## Tashqi datasetlar

| Dataset | Litsenziya | Bizning sinflarga |
|---|---|---|
| [Roboflow: classification/medical-waste](https://universe.roboflow.com/classification/medical-waste) (4102 rasm, RezaAmansyah ishlatgan) | CC BY 4.0 | `Gloves=qolqop_shlang`, `Syringes=igna` (rasmlarni ko'zdan kechiring: ignasiz shpritslar `plastik` ga), `Masks=bint_paxta` (ixtiyoriy). `Medicines` olinmaydi |
| [Medical Waste 4.0](https://zenodo.org/record/7643417) (Italiya) | Zenodo sahifasidan tekshiring | Qo'lqoplar: `qolqop_shlang`. Igna yo'q |
| Nepal tadqiqoti ([arXiv 2508.07450](https://arxiv.org/pdf/2508.07450), Kaggle) | Kaggle sahifasidan tekshiring | Igna, shprits, qo'lqop bor |

Misol (Roboflow'dan "Folder Structure" formatida yuklab, ochilgandan keyin):

```bash
python training/import_dataset.py --src ~/Downloads/medical-waste --name rf_medwaste --map "Gloves=qolqop_shlang,Syringes=igna" --source https://universe.roboflow.com/classification/medical-waste
```

```bash
python train.py --data dataset --extra dataset_ext --test test
```

Tashqi rasmlar boshqa kamerada, boshqa fonda olingan. Ular faqat o'z rasmlaringizni to'ldiradi. Stol kamerasidagi
rasmlar baribir kerak. Datasetni yuklab olish uchun Roboflow yoki Kaggle hisobi kerak bo'ladi.
