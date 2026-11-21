"""ATCS klassifikatorini o'qitish: MobileNetV3Large (ImageNet) asosida transfer learning -> int8 TFLite.

Google Colab da (TensorFlow oldindan o'rnatilgan, Runtime -> GPU):
  !unzip -q dataset.zip -d /content
  !python train.py --data /content/dataset --out /content/natija
Natija: natija/atcs.tflite  -> Raspberry Pi dagi models/atcs.tflite ga ko'chiriladi.

Dataset tuzilmasi (atcs.collect yoki Roboflow "Folder Structure" eksporti):
  dataset/igna/*.jpg, dataset/qolqop_shlang/*.jpg, dataset/shisha/*.jpg, dataset/plastik/*.jpg, dataset/bint_paxta/*.jpg
Sinflar tartibi config/atcs.toml dagi vision.labels bilan bir xil bo'lishi SHART (indeks + 1 = bo'lim).

Rasmlar ish rejimidagi kabi PIL BILINEAR bilan kvadratga keltiriladi (atcs.vision.prepare), piksel 0..255;
normalizatsiya model ichida (include_preprocessing=True).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_LABELS = ["igna", "qolqop_shlang", "shisha", "plastik", "bint_paxta"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dir(root: Path, labels: list[str], size: int):
    from PIL import Image
    xs, ys, files = [], [], []
    for k, lb in enumerate(labels):
        d = root / lb
        if not d.is_dir():
            raise SystemExit(f"'{d}' papkasi yo'q. Kerakli sinflar: {labels}")
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in EXTS:
                img = Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)
                xs.append(np.asarray(img, dtype=np.uint8)); ys.append(k); files.append(str(f))
    if not xs:
        raise SystemExit(f"{root} da rasm topilmadi")
    return np.stack(xs), np.array(ys, dtype=np.int32), files


def stratified_split(y, frac, seed):
    rng = np.random.default_rng(seed)
    val = []
    for k in np.unique(y):
        idx = np.flatnonzero(y == k); rng.shuffle(idx)
        val.extend(idx[:max(1, int(round(len(idx) * frac)))])
    mask = np.zeros(len(y), bool); mask[val] = True
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def augment_fn(tf):
    def aug(x, y):
        x = tf.cast(x, tf.float32)
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.rot90(x, tf.random.uniform([], 0, 4, tf.int32))      # yuqoridan qaralganda yo'nalish ixtiyoriy
        x = tf.image.random_brightness(x, 20.0)
        x = tf.image.random_contrast(x, 0.85, 1.15)
        x = tf.image.random_saturation(x, 0.85, 1.15)
        return tf.clip_by_value(x, 0.0, 255.0), y
    return aug


def build_model(keras, n, size, dropout=0.3):
    base = keras.applications.MobileNetV3Large(input_shape=(size, size, 3), include_top=False, weights="imagenet",
                                               pooling="avg", include_preprocessing=True)
    base.trainable = False
    inp = keras.Input((size, size, 3))
    x = base(inp, training=False)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n, activation="softmax")(x)
    return keras.Model(inp, out), base


def report(probs, y, labels, threshold, needle=0, title=""):
    pred, conf = probs.argmax(1), probs.max(1)
    n = len(labels)
    cm = np.zeros((n, n), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    sure = conf >= threshold
    r = {
        "title": title,
        "count": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "recall": {labels[k]: float(cm[k, k] / max(1, cm[k].sum())) for k in range(n)},
        "confusion": cm.tolist(),
        "threshold": threshold,
        "to_operator": float((~sure).mean()),
        "wrong_but_sure": int(((pred != y) & sure).sum()),
        # eng xavfli xato: igna ishonch bilan boshqa bo'limga
        "needle_sent_elsewhere": int(((y == needle) & (pred != needle) & sure).sum()),
    }
    w = max(len(s) for s in labels)
    print(f"\n=== {title}: {r['count']} rasm, aniqlik {r['accuracy']:.1%}")
    print(" " * (w + 2) + " ".join(f"{s[:7]:>7}" for s in labels) + "   <- bashorat")
    for k in range(n):
        print(f"{labels[k]:>{w}}  " + " ".join(f"{v:>7}" for v in cm[k]) + f"   recall {r['recall'][labels[k]]:.1%}")
    print(f"Ishonch < {threshold:.2f} -> operatorga: {r['to_operator']:.1%};  "
          f"ishonch bilan xato: {r['wrong_but_sure']};  IGNA boshqa bo'limga: {r['needle_sent_elsewhere']}")
    return r


def tflite_predict(tf, path, x):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    out = []
    for img in x:
        inp = img[None].astype(i["dtype"]) if i["dtype"] != np.int8 else None
        if inp is None:
            s, z = i["quantization"]
            inp = np.clip(np.round(img[None] / s + z), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], inp); it.invoke()
        y = it.get_tensor(o["index"])[0].astype(np.float32)
        s, z = o["quantization"]
        if s:
            y = (y - z) * s
        out.append(y)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset papkasi")
    ap.add_argument("--test", help="alohida test papkasi (boshqa kuni/boshqa buyumlar bilan yig'ilgan) - tavsiya")
    ap.add_argument("--out", default="natija")
    ap.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--finetune-epochs", type=int, default=20)
    ap.add_argument("--unfreeze", type=int, default=60, help="fine-tuning da ochiladigan oxirgi qatlamlar soni")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.70, help="config: sorting.min_confidence")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--float", action="store_true", help="int8 o'rniga float32 TFLite")
    a = ap.parse_args()

    import tensorflow as tf
    import keras
    tf.keras.utils.set_random_seed(a.seed)
    labels = [s.strip() for s in a.labels.split(",")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    x, y, _ = load_dir(Path(a.data), labels, a.size)
    tr, va = stratified_split(y, a.val_split, a.seed)
    print("Rasmlar:", {lb: int((y == k).sum()) for k, lb in enumerate(labels)}, f"o'qitish {len(tr)}, tekshirish {len(va)}")
    cnt = np.bincount(y[tr], minlength=len(labels))
    class_weight = {k: float(len(tr) / (len(labels) * max(1, c))) for k, c in enumerate(cnt)}

    AUTO = tf.data.AUTOTUNE
    train_ds = (tf.data.Dataset.from_tensor_slices((x[tr], y[tr])).shuffle(len(tr), seed=a.seed)
                .map(augment_fn(tf), num_parallel_calls=AUTO).batch(a.batch).prefetch(AUTO))
    val_ds = tf.data.Dataset.from_tensor_slices((x[va].astype(np.float32), y[va])).batch(a.batch)

    model, base = build_model(keras, len(labels), a.size)
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, class_weight=class_weight, callbacks=cb)

    # fine-tuning: oxirgi qatlamlar, BatchNorm muzlatilgan holda
    base.trainable = True
    for layer in base.layers[:-a.unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", ["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=a.finetune_epochs, class_weight=class_weight, callbacks=cb)
    model.save(out / "atcs.keras")

    results = {"labels": labels, "size": a.size}
    results["val_keras"] = report(model.predict(x[va].astype(np.float32), verbose=0), y[va], labels,
                                  a.threshold, title="Tekshirish (Keras)")

    # TFLite eksport
    sm = out / "saved_model"
    model.export(str(sm))
    conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
    if not a.float:
        rng = np.random.default_rng(a.seed)
        rep_idx = rng.choice(tr, size=min(300, len(tr)), replace=False)

        def rep():
            for i in rep_idx:
                yield [x[i:i + 1].astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
    tfl = out / "atcs.tflite"
    tfl.write_bytes(conv.convert())
    (out / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"\nTFLite: {tfl} ({tfl.stat().st_size / 1e6:.1f} MB)")

    results["val_tflite"] = report(tflite_predict(tf, tfl, x[va]), y[va], labels, a.threshold,
                                   title="Tekshirish (TFLite)")
    if a.test:
        xt, yt, _ = load_dir(Path(a.test), labels, a.size)
        results["test_tflite"] = report(tflite_predict(tf, tfl, xt), yt, labels, a.threshold,
                                        title="Alohida test (TFLite)")
    (out / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    final = results.get("test_tflite", results["val_tflite"])
    if final["needle_sent_elsewhere"]:
        print("\nDIQQAT: ignalar ishonch bilan boshqa bo'limga ketmoqda. Ko'proq igna rasmi yig'ing yoki "
              "--threshold ni oshiring. Bu model stolga qo'yilmasin.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
