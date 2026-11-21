"""Buyumni aniqlash: kadr -> (bo'lim raqami 1..5, ishonch 0..1)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Classifier(Protocol):
    def classify(self, frame) -> tuple[int, float]: ...


def crop_roi(frame: np.ndarray, roi) -> np.ndarray:
    """Kadrdan shlyuz oldidagi zonani kesib olish. roi = (x, y, eni, bo'yi) kadr ulushida.
    Rasm yig'ishda ham, ish rejimida ham aynan shu funksiya ishlatiladi."""
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = int(round(x * w)), int(round(y * h))
    x1, y1 = min(w, x0 + int(round(rw * w))), min(h, y0 + int(round(rh * h)))
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise ValueError(f"vision.roi kadrdan tashqarida: {roi}")
    return frame[y0:y1, x0:x1]


def prepare(frame: np.ndarray, roi, size: int, dtype=np.float32, quant=(0.0, 0)) -> np.ndarray:
    """RGB kadr -> modelning kirish tenzori (1, size, size, 3), piksel 0..255."""
    from PIL import Image
    if roi is not None:
        frame = crop_roi(frame, roi)
    x = np.asarray(Image.fromarray(np.ascontiguousarray(frame[..., :3])).resize((size, size), Image.BILINEAR))[None]
    if dtype == np.float32:
        return x.astype(np.float32)
    scale, zero = quant
    if dtype == np.int8 and scale:
        return np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
    return x.astype(dtype)


def _interpreter_class():
    """Pi: ai-edge-litert yoki tflite-runtime. Kompyuter: ular bo'lmasa, to'liq TensorFlow."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except ImportError:
        raise SystemExit("TFLite ishga tushiruvchi topilmadi. O'rnating: pip install ai-edge-litert "
                         "(yoki tensorflow). Python 3.11/3.12 kerak bo'lishi mumkin.") from None


class TfliteClassifier:
    """TensorFlow Lite modeli (Raspberry Pi da ai-edge-litert yoki tflite-runtime).
    Model chiqishi - vision.labels tartibidagi 5 ta sinf ehtimoli; sinf indeksi + 1 = bo'lim.
    Modelga piksel 0..255 oralig'ida beriladi: normalizatsiya model ichida (training/train.py shunday quradi)."""

    def __init__(self, model_path: str | Path, input_size: int, roi=None):
        self._roi = roi
        Interpreter = _interpreter_class()
        self._it = Interpreter(model_path=str(model_path), num_threads=4)
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]
        self._out = self._it.get_output_details()[0]
        self._size = input_size

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        return prepare(frame, self._roi, self._size, self._in["dtype"], self._in.get("quantization", (0.0, 0)))

    def probs(self, frame) -> np.ndarray:
        """Barcha sinflar ehtimoli (vision.labels tartibida)."""
        self._it.set_tensor(self._in["index"], self._prep(frame))
        self._it.invoke()
        y = self._it.get_tensor(self._out["index"])[0].astype(np.float32)
        scale, zero = self._out.get("quantization", (0.0, 0))
        if scale:
            y = (y - zero) * scale
        if y.min() < 0 or abs(float(y.sum()) - 1.0) > 0.05:      # logits -> softmax
            e = np.exp(y - y.max()); y = e / e.sum()
        return y

    def classify(self, frame):
        y = self.probs(frame)
        k = int(np.argmax(y))
        return k + 1, float(y[k])
