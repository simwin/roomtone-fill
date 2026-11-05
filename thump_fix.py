#!/usr/bin/env python3
"""
Убирает одиночный удар по микрофону (глухой «тук» на низах) в заданном месте.
Не ищет сам – время указываешь ты. Длина файла не меняется.

  python3 thump_fix.py input.wav output.wav --at 244.80 [--at 312.4 ...] [--width 0.07] [--cut 130] [--fade 15] [--click]

В окне шириной --width вокруг каждой точки срезает всё ниже --cut Гц
(женский голос выше ~150 Гц не страдает), края окна – плавные (--fade, по умолчанию 15 мс).
Если на краях слышен «тик» (низ голоса пропадает и возвращается), сделать --fade 40–60.
--click: у удара бывает короткий щелчок в верхах (4–12 кГц) в момент касания – срез низов его не
трогает, он слышен как «тик». С --click скрипт находит самый сильный щелчок в ±40 мс от --at и
прижимает верхи (выше 3 кГц) в окне 30 мс вокруг него до обычного уровня рядом, без провала (края 5 мс).
Проверено 05.11: удар на тестовой записи – --cut 180 --width 0.12 --fade 50 --click.
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
    ap.add_argument("--fade", type=float, default=15.0, help="плавность краёв окна, мс")
    ap.add_argument("--click", action="store_true", help="прижать и щелчок в верхах (выше 3 кГц, 30 мс) до уровня вокруг")
    a = ap.parse_args()

    sr, raw = wavfile.read(a.inp)
    dtype = raw.dtype
    y = raw.astype(np.float64)
    if y.ndim == 1:
        y = y[:, None]
    sos = butter(4, a.cut, "high", fs=sr, output="sos")
    fd = max(1, int(sr * a.fade / 1000))
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
        if a.click:
            seg = y[c0:c1]
            bp = sosfiltfilt(butter(4, [4000, min(12000, sr / 2 * 0.95)], "band", fs=sr, output="sos"), seg.mean(1))
            h = int(sr * 0.002)
            env = np.convolve(bp ** 2, np.ones(h) / h, "same")
            ci = int(t * sr) - c0; r = int(sr * 0.04)
            k = ci - r + int(np.argmax(env[ci - r: ci + r]))           # самый сильный щелчок рядом с ударом
            lp = sosfiltfilt(butter(4, 3000, "low", fs=sr, output="sos"), seg, axis=0)
            hi = seg - lp                                              # верхи выше 3 кГц
            he = np.sqrt(np.convolve((hi ** 2).mean(1), np.ones(h) / h, "same")) + 1e-9
            hw, f2 = int(sr * 0.015), int(sr * 0.005)
            around = np.r_[he[k - 4 * hw:k - hw - f2], he[k + hw + f2:k + 4 * hw]]
            ref = np.median(around)                                    # обычный уровень верхов рядом
            g = np.minimum(1.0, ref / he)                              # прижать щелчок до этого уровня, не ниже
            g = np.convolve(g, np.ones(h) / h, "same")
            m2 = np.zeros(len(seg)); m2[k - hw:k + hw] = 1
            r2 = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f2))
            m2[k - hw - f2:k - hw] = r2; m2[k + hw:k + hw + f2] = r2[::-1]
            gg = 1 - m2 * (1 - g)
            y[c0:c1] = lp + hi * gg[:, None]
            print(f"  щелчок на {(k + c0) / sr:.3f} с: верхи прижаты до уровня вокруг (окно 30 мс)")
        print(f"удар на {t:.3f} с: срезано ниже {a.cut:.0f} Гц в окне {a.width * 1000:.0f} мс, края {a.fade:.0f} мс")
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype); y = np.clip(np.round(y), info.min, info.max)
    y = y.astype(dtype)
    if raw.ndim == 1:
        y = y[:, 0]
    wavfile.write(a.out, sr, y)


if __name__ == "__main__":
    main()
