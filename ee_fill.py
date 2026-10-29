#!/usr/bin/env python3
"""
Заделывает «Э» фоном комнаты, не меняя длину файла.

  python3 ee_fill.py input.wav regions.txt output.wav [--search 10] [--xfade 15]

regions.txt – AudacityRegions.txt из Auphonic (start end label, в секундах).
Для каждого региона ищет рядом (±search секунд) самый тихий ровный кусок
той же длины, вне других регионов, и вставляет его с короткими
равномощными кроссфейдами на краях. Длина выходного файла совпадает
с исходной до сэмпла. Формат (частота, каналы, 16/24/32 бит) сохраняется.
Рядом пишется output.labels.txt – метки для Audacity/ocenaudio, чтобы
пробежаться по местам заплаток.
"""
import argparse, sys
import numpy as np
from scipy.io import wavfile


def frame_db(mono, hop):
    n = len(mono) // hop
    r = np.sqrt((mono[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(r)


def tile_patch(x, db, banned, hop, sr, c, search, need, floor, xf_ms=40):
    """Склейка фона из тихих кусков: берём самые длинные куски, какие есть рядом,
    выравниваем их по громкости и сшиваем длинными кроссфейдами."""
    xf = int(sr * xf_ms / 1000)
    t = np.linspace(0, np.pi / 2, xf)[:, None]
    for chunk_s in (0.5, 0.4, 0.3, 0.25, 0.2):
        nf = int(chunk_s * 100)
        if nf * hop <= 2 * xf:
            continue
        lo = max(0, c - int(search * 100)); hi = min(len(db) - nf, c + int(search * 100))
        cand = []
        for k in range(lo, hi):
            if banned[k:k + nf].any():
                continue
            w = db[k:k + nf]
            if w.max() <= floor + 6 and w.max() - np.median(w) <= 5:
                cand.append((abs(k - c), k))
        if not cand:
            continue
        cand.sort()
        step = nf * hop - xf
        picked, used = [], np.zeros(len(db), bool)
        for _, k in cand:
            if used[k:k + nf].any():
                continue
            picked.append(k); used[k:k + nf] = True
            if len(picked) * step + xf >= need:
                break
        if len(picked) * step + xf < need and len(picked) < 2:
            continue                                 # кусок один и короткий – пробуем мельче
        while len(picked) * step + xf < need:      # мало кусков – повторяем по кругу
            picked += picked[: max(1, len(picked))]
        chunks = [x[k * hop: (k + nf) * hop].copy() for k in picked]
        rms = [np.sqrt((ch ** 2).mean()) + 1e-9 for ch in chunks]
        target = float(np.median(rms))
        chunks = [ch * (target / r) for ch, r in zip(chunks, rms)]   # один уровень у всех кусков
        out = chunks[0]
        for ch in chunks[1:]:
            out = np.concatenate([out[:-xf], out[-xf:] * np.cos(t) + ch[:xf] * np.sin(t), ch[xf:]])
        p_db = float(np.mean([db[k:k + nf].mean() for k in picked]))
        return out[:need].copy(), f"{len(picked)}×{chunk_s:.2f}с", p_db
    return None, "", 0.0


def far_patch(x, db, banned, hop, c, need, floor):
    """Целый кусок фона нужной длины из ближайшей длинной паузы в любом месте файла.
    Громкость подгоняется под фон рядом с «Э»."""
    nf = int(np.ceil(need / hop))
    gfloor = np.percentile(db, 10)
    best = None
    for k in range(0, len(db) - nf, 2):
        if banned[k:k + nf].any():
            continue
        w = db[k:k + nf]
        if w.max() <= gfloor + 3 and w.max() - np.median(w) <= 5:   # тихо и ровно, без призвуков
            d = abs(k - c)
            if best is None or d < best[0]:
                best = (d, k)
    if best is None:
        return None
    k = best[1]
    patch = x[k * hop: k * hop + need].copy()
    # уровень фона рядом: тихие кадры в ±10 с, не занятые «Э»
    lo, hi = max(0, c - 1000), min(len(db), c + 1000)
    near = db[lo:hi][(~banned[lo:hi]) & (db[lo:hi] <= floor + 4)]
    if len(near):
        tgt = 10 * np.log10(np.mean(10 ** (near / 10)))
        cur = 10 * np.log10(np.mean(10 ** (db[k:k + nf] / 10)))
        patch *= 10 ** (np.clip(tgt - cur, -4, 4) / 20)
        p_db = tgt
    else:
        p_db = db[k:k + nf].mean()
    return patch, f"{k / 100:7.2f}*", p_db


def extend_tail(db, s, e, max_ms=80):
    """Дотянуть границы региона, пока звук затухает (Auphonic иногда режет «Э» раньше конца)."""
    c = int(s * 100)
    floor = np.percentile(db[max(0, c - 1000): c + 1000], 3)
    sm = np.convolve(db, np.ones(3) / 3, mode="same")     # сгладить неровное затухание
    steps = max_ms // 10
    a, b = int(s * 100), int(e * 100)
    k = b
    while k < min(len(db) - 1, b + steps) and sm[k] > floor + 6 and sm[k] <= sm[k - 1] + 2:
        k += 1
    e2 = k / 100 if k > b else e
    k = a - 1                                    # влево – не больше 20 мс: дальше может быть конец слова
    while k > max(0, a - 1 - 2) and sm[k] > floor + 6 and sm[k] <= sm[k + 1] + 2:
        k -= 1
    s2 = (k + 1) / 100 if k < a - 1 else s
    return (min(s, s2), max(e, e2))


def fill(x, sr, regs, scale, search=10.0, xfade=15.0, extend=True, verbose=True, avoid=()):
    """x – float, (сэмплы, каналы). Возвращает (y, список (s, e) фактических регионов).
    avoid – дополнительные места (s, e), откуда нельзя брать фон (например, найденные вдохи)."""
    n_total = len(x)
    mono = x.mean(1)
    hop = int(sr * 0.01)                      # кадр 10 мс
    db = frame_db(mono / scale, hop)

    regs = sorted(regs)
    if extend:
        regs = [extend_tail(db, s, e) for s, e in regs]

    # кадры, занятые «Э», не годятся как источник фона
    banned = np.zeros(len(db), bool)
    for s, e in list(regs) + list(avoid):
        banned[max(0, int(s * 100) - 5): int(e * 100) + 5] = True

    xf = int(sr * xfade / 1000)
    t = np.linspace(0, np.pi / 2, xf)[:, None] if xf > 0 else None
    fade_in, fade_out = (np.sin(t), np.cos(t)) if xf > 0 else (None, None)

    y = x.copy()
    labels = []
    if verbose: print(f"{'№':>3} {'Э, с':>17} {'длит':>5} {'Э дБ':>6}  {'фон из, с':>9} {'фон дБ':>7}  (дБ – от полной шкалы)")
    for i, (s, e) in enumerate(regs, 1):
        a0, a1 = int(round(s * sr)), int(round(e * sr))
        L = a1 - a0
        need = L + 2 * xf                      # с запасом под кроссфейды
        nf = int(np.ceil(need / hop))
        c = int(s * 100)
        lo = max(0, c - int(search * 100)); hi = min(len(db) - nf, c + int(search * 100))
        best, best_k = None, None
        for flat_db in (5, 8, None):              # сначала ищем ровный кусок, без щелчков
            for k in range(lo, hi):
                if banned[k:k + nf].any():
                    continue
                w = db[k:k + nf]
                if flat_db is not None and w.max() - np.median(w) > flat_db:
                    continue
                score = np.percentile(w, 90) + 0.5 * w.mean()
                if best is None or score < best:
                    best, best_k = score, k
            if best_k is not None:
                break
        # фон комнаты рядом: 3% самых тихих кадров (10-й перцентиль в плотной речи завышен)
        floor = np.percentile(db[max(0, c - 1000): c + 1000], 3)
        single = None
        if best_k is not None:
            single = (x[best_k * hop: best_k * hop + need].copy(),
                      f"{best_k * hop / sr:9.2f}", db[best_k: best_k + nf].mean())
        far = far_patch(x, db, banned, hop, c, need, floor) if single is None or \
            db[best_k: best_k + nf].mean() > floor + 4 else None
        tiled = tile_patch(x, db, banned, hop, sr, c, search, need, floor) if far is None else (None, "", 0.0)
        # порядок: целая пауза рядом → целая пауза подальше (с подгонкой громкости) → склейка
        if single is not None and db[best_k: best_k + nf].mean() <= floor + 4:
            patch, src_note, p_db = single
        elif far is not None:
            patch, src_note, p_db = far
        elif tiled[0] is not None and (single is None or tiled[2] < single[2] - 3):
            patch, src_note, p_db = tiled
        elif single is not None and single[2] <= floor + 8:
            patch, src_note, p_db = single
        else:                                   # подходящей тишины нет – не трогаем, чем вставить речь
            if verbose: print(f"{i:>3}: не нашёл фон, пропускаю")
            continue

        # кладём заплатку на место [a0-xf, a1+xf], края – кроссфейдом
        b0, b1 = a0 - xf, a1 + xf
        if b0 < 0 or b1 > n_total:
            b0, b1 = a0, a1; patch = patch[xf: xf + L]
            y[b0:b1] = patch
        else:
            seg = y[b0:b1].copy()
            seg[xf:-xf] = patch[xf:-xf]
            seg[:xf] = y[b0:a0] * fade_out + patch[:xf] * fade_in
            seg[-xf:] = y[a1:b1] * fade_in + patch[-xf:] * fade_out
            y[b0:b1] = seg

        e_db = db[int(s * 100): int(e * 100)].mean()
        if verbose: print(f"{i:>3} {s:8.3f}–{e:8.3f} {e - s:5.2f} {e_db:6.1f}  {src_note} {p_db:7.1f}")
        labels.append(f"{s:.3f}\t{e:.3f}\tЭ{i}")

    return y, [tuple(map(float, l.split("\t")[:2])) for l in labels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("regions"); ap.add_argument("out")
    ap.add_argument("--search", type=float, default=10.0, help="радиус поиска фона, с")
    ap.add_argument("--xfade", type=float, default=15.0, help="кроссфейд на краях, мс")
    ap.add_argument("--no-extend", action="store_true", help="не дотягивать границы по затуханию")
    a = ap.parse_args()

    sr, raw = wavfile.read(a.inp)
    dtype = raw.dtype
    x = raw.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    scale = float(np.iinfo(dtype).max + 1) if np.issubdtype(dtype, np.integer) else 1.0
    regs = []
    for line in open(a.regions, encoding="utf-8"):
        p = line.split()
        if len(p) >= 2:
            regs.append((float(p[0]), float(p[1])))

    y, done = fill(x, sr, regs, scale, a.search, a.xfade, not a.no_extend)
    write_wav(a.out, sr, y, dtype, raw.ndim)
    lab = a.out.rsplit(".", 1)[0] + ".labels.txt"
    open(lab, "w", encoding="utf-8").write("".join(f"{s:.3f}\t{e:.3f}\tЭ{i}\n" for i, (s, e) in enumerate(done, 1)))
    print(f"\nГотово: {a.out} ({len(y) / sr:.3f} с, как и исходник), метки: {lab}")


def write_wav(path, sr, y, dtype, ndim):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        y = np.clip(np.round(y), info.min, info.max)
    y = y.astype(dtype)
    if ndim == 1:
        y = y[:, 0]
    wavfile.write(path, sr, y)


if __name__ == "__main__":
    main()
