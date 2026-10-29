#!/usr/bin/env python3
"""
Убирает одиночный удар по микрофону (глухой «тук» на низах) в заданном месте.
Не ищет сам – время указываешь ты. Длина файла не меняется.

  python3 thump_fix.py input.wav output.wav --at 244.80 [--at 312.4 ...] [--width 0.07] [--cut 130]

В окне шириной --width вокруг каждой точки срезает всё ниже --cut Гц
(женский голос выше ~150 Гц не страдает), края окна – плавные 15 мс.
"""
import argparse
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--at", type=float, action="append", required=True, help="время удара, с")
    ap.add_argument("--width", type=float, default=0.07, help="ширина окна, с")
    ap.add_argument("--cut", type=float, default=130.0, help="срезать ниже, Гц")
    a = ap.parse_args()

    sr, raw = wavfile.read(a.inp)
    dtype = raw.dtype
    y = raw.astype(np.float64)
    if y.ndim == 1:
        y = y[:, None]
    sos = butter(4, a.cut, "high", fs=sr, output="sos")
    fd = int(sr * 0.015)
    for t in a.at:
        c0, c1 = int((t - 1) * sr), int((t + 1) * sr)            # фильтруем с запасом, чтобы не было краевых эффектов
        c0, c1 = max(0, c0), min(len(y), c1)
        hp = sosfiltfilt(sos, y[c0:c1], axis=0)
        w0 = int((t - a.width / 2) * sr) - c0
        w1 = int((t + a.width / 2) * sr) - c0
        mix = np.zeros(c1 - c0)
        mix[w0:w1] = 1
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fd))
        mix[w0 - fd:w0] = ramp; mix[w1:w1 + fd] = ramp[::-1]
        y[c0:c1] = y[c0:c1] * (1 - mix[:, None]) + hp * mix[:, None]
        print(f"удар на {t:.3f} с: срезано ниже {a.cut:.0f} Гц в окне {a.width * 1000:.0f} мс")
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype); y = np.clip(np.round(y), info.min, info.max)
    y = y.astype(dtype)
    if raw.ndim == 1:
        y = y[:, 0]
    wavfile.write(a.out, sr, y)


if __name__ == "__main__":
    main()
