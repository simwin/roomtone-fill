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
import argparse, glob, os, shutil
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


def put(arr, start, vals):
    """arr[start:start+len(vals)] = vals с обрезкой по краям массива (возле начала/конца файла)."""
    a0, a1 = max(0, start), min(len(arr), start + len(vals))
    if a1 > a0:
        arr[a0:a1] = vals[a0 - start:a1 - start]


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
        n = c1 - c0
        mix = np.zeros(n)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fd))
        put(mix, w0 - fd, ramp); put(mix, w1, ramp[::-1]); mix[max(0, w0):max(0, min(n, w1))] = 1
        y[c0:c1] = y[c0:c1] * (1 - mix[:, None]) + hp * mix[:, None]
        if a.click:
            seg = y[c0:c1]
            if sr < 16000:
                print("  щелчок: частота записи ниже 16 кГц – верхи не трогаю")
                print(f"удар на {t:.3f} с: срезано ниже {a.cut:.0f} Гц в окне {a.width * 1000:.0f} мс, края {a.fade:.0f} мс")
                continue
            bp = sosfiltfilt(butter(4, [4000, min(12000, sr / 2 * 0.95)], "band", fs=sr, output="sos"), seg.mean(1))
            h = int(sr * 0.002)
            env = np.convolve(bp ** 2, np.ones(h) / h, "same")
            ci = int(t * sr) - c0; r = int(sr * 0.04)
            lo_ = max(0, ci - r)
            k = lo_ + int(np.argmax(env[lo_: max(lo_ + 1, min(len(env), ci + r))]))   # самый сильный щелчок рядом с ударом
            lp = sosfiltfilt(butter(4, 3000, "low", fs=sr, output="sos"), seg, axis=0)
            hi = seg - lp                                              # верхи выше 3 кГц
            he = np.sqrt(np.convolve((hi ** 2).mean(1), np.ones(h) / h, "same")) + 1e-9
            hw, f2 = int(sr * 0.015), int(sr * 0.005)
            around = np.r_[he[max(0, k - 4 * hw):max(0, k - hw - f2)], he[k + hw + f2:k + 4 * hw]]
            if around.size == 0:
                print("  щелчок: нет окружения для сравнения (край файла) – верхи не трогаю")
                print(f"удар на {t:.3f} с: срезано ниже {a.cut:.0f} Гц в окне {a.width * 1000:.0f} мс, края {a.fade:.0f} мс")
                continue
            ref = np.median(around)                                    # обычный уровень верхов рядом
            g = np.minimum(1.0, ref / he)                              # прижать щелчок до этого уровня, не ниже
            g = np.convolve(g, np.ones(h) / h, "same")
            m2 = np.zeros(len(seg))
            r2 = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f2))
            put(m2, k - hw - f2, r2); put(m2, k + hw, r2[::-1]); m2[max(0, k - hw):k + hw] = 1
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
    ib, ob = a.inp.rsplit(".", 1)[0], a.out.rsplit(".", 1)[0]     # метки ocenaudio – к новому файлу
    if os.path.abspath(ib) != os.path.abspath(ob):
        moved = []
        for f in sorted(glob.glob(glob.escape(ib) + ".*.srt")):
            shutil.copyfile(f, ob + f[len(ib):]); moved.append(os.path.basename(ob + f[len(ib):]))
        if moved:
            print("Метки ocenaudio перенесены: " + ", ".join(moved))


if __name__ == "__main__":
    main()
