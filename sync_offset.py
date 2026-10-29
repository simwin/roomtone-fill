#!/usr/bin/env python3
"""
Сдвиг между двумя записями одного разговора – по звуку или по движению в кадре.

  python3 sync_offset.py audio A.wav B.wav          # по огибающей звука
  python3 sync_offset.py video A.mov B.mov          # по движению в кадре (для немого видео)
      [--crop-a W:H:X:Y] [--crop-b W:H:X:Y]          # взять только часть кадра (окно собеседницы)
      [--max 180]                                    # искать сдвиг в пределах ±N секунд
      [--parts 8]                                    # проверка по частям записи (0 – не делать)

lag – на сколько секунд позже то же событие находится в B, чем в A:
  событие в A на t с  →  в B на t + lag с.   На таймлайне Resolve, если A начинается в 0: начало B = −lag.
Сила z – во сколько раз пик выше обычного разброса; ориентир: от ~8 – надёжно.

Звонок без наушников: на каждой дорожке есть обе собеседницы (вторая – из динамика с задержкой
связи L), поэтому по ЗВУКУ бывает два пика: Δ − L и Δ + L (Δ – сдвиг часов). Скрипт подписывает пару.
Проверка по частям показывает, держится ли сдвиг всю запись (дрейф часов, плавающая задержка).
"""
import argparse, subprocess, sys
import numpy as np


def movsum(x, k):
    if k <= 1 or len(x) < k:
        return x
    c = np.cumsum(np.concatenate([[0.0], x]))
    y = c[k:] - c[:-k]
    # окно «назад» сдвигает сигнал на (k−1)/2 отсчёта, но одинаково для A и B – на lag не влияет
    return np.concatenate([np.full(k - 1, y[0]), y])


def audio_env(path, sr=8000, hop=8, smooth_ms=5):
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
                        "-f", "s16le", "-"], capture_output=True, check=True)
    x = np.frombuffer(p.stdout, np.int16).astype(np.float32)
    x = np.diff(x, prepend=np.float32(0))          # подавить басы: динамик их не передаёт
    n = len(x) // hop * hop
    e = np.abs(x[:n]).reshape(-1, hop).mean(1)     # огибающая, 1000 точек в секунду
    rate = sr / hop
    k = max(1, int(smooth_ms * rate / 1000))
    e = movsum(e, k) / k
    return np.log1p(e), rate, k


def motion(path, crop=None, fps=30, w=64, h=36, win=0.2):
    vf = (f"crop={crop}," if crop else "") + f"fps={fps},scale={w}:{h}:flags=area,format=gray"
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-an", "-vf", vf, "-f", "rawvideo", "-"],
                          stdout=subprocess.PIPE)
    fsz, out, prev, rest, k = w * h, [], None, b"", 0
    while True:
        buf = pr.stdout.read(fsz * 600)
        if not buf:
            break
        buf = rest + buf
        nf = len(buf) // fsz
        rest = buf[nf * fsz:]
        if nf == 0:
            continue
        f = np.frombuffer(buf[:nf * fsz], np.uint8).reshape(nf, h, w).astype(np.int16)
        if prev is None:
            out.append(np.zeros(1))
        else:
            f = np.concatenate([prev[None], f])
        out.append(np.abs(np.diff(f, axis=0)).mean((1, 2)))
        prev = f[-1]
        k += 1
        if k % 10 == 0:
            print(f"\r  {path}: {sum(map(len, out)) / fps / 60:.1f} мин", end="", file=sys.stderr)
    pr.wait()
    print(file=sys.stderr)
    if pr.returncode:
        sys.exit(f"ffmpeg не смог прочитать {path}")
    d = np.concatenate(out)
    # сумма движения за окно: повторы кадров (звонок шлёт 12–20 к/с) дают нулевые шаги и не мешают
    kw = max(1, int(round(win * fps)))
    return np.log1p(movsum(d, kw)), float(fps), kw


def prep(x, rate, win=3.0):
    """Убрать медленные изменения (экспозиция, общий уровень) и привести к единому масштабу."""
    k = max(1, int(win * rate))
    c = np.cumsum(np.concatenate([[0.0], x]))
    lo = np.clip(np.arange(len(x)) - k // 2, 0, len(x)); hi = np.clip(lo + k, 0, len(x))
    y = x - (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return (y - y.mean()) / (y.std() + 1e-9)


def xcorr(a, b, rate, lo, hi, min_ov):
    """Корреляция для сдвигов lo..hi секунд: c(k) = Σ a[t]·b[t+k] / перекрытие."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(b, n), n)
    k0 = max(int(np.floor(lo * rate)), -(len(a) - 1)); k1 = min(int(np.ceil(hi * rate)), len(b) - 1)
    lags = np.arange(k0, k1 + 1)
    vals = c[lags % n]
    ov = np.minimum(len(a), len(b) - lags) - np.maximum(0, -lags)
    ok = ov >= min_ov
    return lags[ok] / rate, vals[ok] / ov[ok]


def zscore(vals):
    med = np.median(vals)
    return (vals - med) / (1.4826 * np.median(np.abs(vals - med)) + 1e-12)


def peaks(lags, vals, rate, nmax=4, sep=0.08):
    z = zscore(vals)
    out, used = [], np.zeros(len(z), bool)
    for i in np.argsort(-z):
        if used[i]:
            continue
        if (i > 0 and vals[i - 1] > vals[i]) or (i < len(z) - 1 and vals[i + 1] > vals[i]):
            continue                                 # склон соседнего пика, а не вершина
        d = 0.0
        if 0 < i < len(z) - 1:                       # уточнить между отсчётами
            y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
        out.append((lags[i] + d / rate, z[i]))
        used[max(0, i - int(sep * rate)):i + int(sep * rate) + 1] = True
        if len(out) >= nmax:
            break
    return out


def pair(pk, ratio=0.5):
    """Пара «напрямую / из динамика»: два сравнимых пика на расстоянии 0,08–2 с."""
    if len(pk) < 2:
        return None
    best = pk[0]
    for lag, z in pk[1:]:
        if z >= ratio * best[1] and 0.08 < abs(lag - best[0]) < 2.0:
            return tuple(sorted([best[0], lag]))
    return None


def by_parts(a, b, rate, center, parts, audio, search=2.0):
    seg = len(a) // parts
    print(f"\nПо частям (±{search:g} с вокруг найденного):")
    print("  отрезок A, мин      lag, с    сила z" + ("     пара (Δ, L)" if audio else ""))
    for i in range(parts):
        s, e = i * seg, (i + 1) * seg
        lags, vals = _local(a[s:e], b, rate, s, center, search)
        if lags is None:
            print(f"  {s / rate / 60:5.1f}–{e / rate / 60:5.1f}       нет перекрытия с B")
            continue
        pk = peaks(lags, vals, rate)
        txt = ""
        if audio:
            pr = pair(pk)
            txt = f"     Δ {(pr[0] + pr[1]) / 2:+.3f}, L {(pr[1] - pr[0]) / 2:.3f}" if pr else \
                  "     (в отрезке слышен в основном один голос)"
        print(f"  {s / rate / 60:5.1f}–{e / rate / 60:5.1f}      {pk[0][0]:+9.3f}   {pk[0][1]:6.1f}{txt}")


def _local(sa, b, rate, s, center, search):
    """Сдвиг отрезка sa (начинается в A на s отсчётов) по окну B вокруг center."""
    off, w = int(round(center * rate)), int(round(search * rate))
    b0, b1 = max(0, s + off - w), min(len(b), s + len(sa) + off + w)
    if b1 - b0 < len(sa) // 2:
        return None, None
    lo = (s + off - w - b0) / rate; hi = (s + off + w - b0) / rate
    lags, vals = xcorr(sa, b[b0:b1], rate, lo, hi, len(sa) // 2)
    if len(lags) < 5:
        return None, None
    return lags + (b0 - s) / rate, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audio", "video"])
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--max", type=float, default=180)
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--crop-a"); ap.add_argument("--crop-b")
    a = ap.parse_args()
    audio = a.mode == "audio"
    if audio:
        xa, r, _ = audio_env(a.a); xb, _, _ = audio_env(a.b)
    else:
        xa, r, _ = motion(a.a, a.crop_a); xb, _, _ = motion(a.b, a.crop_b)
    pa, pb = prep(xa, r), prep(xb, r)
    lags, vals = xcorr(pa, pb, r, -a.max, a.max, min(len(pa), len(pb)) // 2)
    pk = peaks(lags, vals, r)
    print(f"A = {a.a}  ({len(xa) / r:.2f} с)\nB = {a.b}  ({len(xb) / r:.2f} с)")
    print("  lag, с     сила z")
    for lag, z in pk:
        print(f"  {lag:+9.3f}   {z:6.1f}")
    best = pk[0]
    print(f"\nЛучшее: lag = {best[0]:+.3f} с → на таймлайне начало B = {-best[0]:+.3f} с от начала A")
    pr = pair(pk) if audio else None
    if pr:
        print(f"Пара «голос напрямую / из динамика»: {pr[0]:+.3f} и {pr[1]:+.3f} с\n"
              f"  сдвиг часов Δ = {(pr[0] + pr[1]) / 2:+.3f} с, задержка связи L = {(pr[1] - pr[0]) / 2:.3f} с")
    if a.parts > 1:
        center = (pr[0] + pr[1]) / 2 if pr else best[0]
        by_parts(pa, pb, r, center, a.parts, audio)


if __name__ == "__main__":
    main()
