#!/usr/bin/env python3
"""
Находит вдохи в речи и заменяет их фоном комнаты. Длина файла не меняется.
Нужен ee_fill.py в той же папке.

  python3 breath_soften.py input.wav output.wav               найти и убрать
  python3 breath_soften.py input.wav x.wav --list-only        только найти, записать метки
  python3 breath_soften.py input.wav output.wav --labels F    обработать по готовым меткам
  python3 breath_soften.py input.wav output.wav --soft        не заменять, а приглушать

Как отличается вдох (версия 3, 29.10.2026 – проверено на 5-минутной тестовой записи):
  • голос – по периодичности в полосе 70–1000 Гц (окно 60 мс): у вдохов и тишины
    0,27–0,35, у далёкого голоса собеседницы ~0,64, у ведущей ~0,88. Кепстральный пик
    (CPP) добавочно: у чистой близкой речи ~21, но далёкий голос с эхом он НЕ видит –
    одного его мало (на этом v3-черновик чуть не вырезал речь собеседницы);
  • вдох – шум в полосе 1–8 кГц на 10+ дБ выше фона этой полосы, без гармоник;
  • от шипящих («с», «ш») отличается спектром (4–8 кГц не выше 1–2 кГц на 5+ дБ)
    и местом: согласная прижата к голосу, вдох стоит в паузе;
  • найденное место расширяется, пока идёт шум без гармоник, но не через шипящую
    и не через всплеск (начало слова на «с…», «т…» не съедается);
  • от голоса всегда остаётся зазор: 30 мс после речи, 50 мс до речи.
НЕ обрабатываются и пишутся в output.check.txt для проверки на слух: шумовые места
громче обычной речи записи без 3 дБ, длиннее 1,2 с или с 3+ голосовыми кадрами подряд.
(Смешок звонкий – его отсекает периодичность. «Соседняя речь» там, где говорит только
собеседница, – её тихий далёкий голос, поэтому сравнивать с ней вдох строже нельзя.)

Метки output.breaths.txt: начало, конец, «вдохN замена». Лишнюю строку можно удалить
и запустить с --labels; строку из .check.txt можно перенести в .breaths.txt.
"""
import argparse, importlib.util, os
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import percentile_filter

_spec = importlib.util.spec_from_file_location(
    "ee_fill", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ee_fill.py"))
ee_fill = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ee_fill)

HOP_S = 0.01


def features(m, sr):
    """Кадры по 10 мс: громкость, полосы, кепстральный пик."""
    hop, win = int(sr * HOP_S), int(sr * 0.04)
    win2 = int(sr * 0.06)                                   # длинное окно для периодичности
    n = (len(m) - win2) // hop
    w = np.hanning(win); w2 = np.hanning(win2)
    nfft = 4096; nfft2 = 8192
    fr2 = np.fft.rfftfreq(nfft2, 1 / sr)
    vband = (fr2 >= 70) & (fr2 < 1000)                      # полоса голоса: основной тон и первая форманта
    lagl, lagh = int(sr / 450), int(sr / 100)
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    bands = {"br": (1000, 8000), "mid": (1000, 2000), "hi": (4000, 8000)}
    sel = {k: (fr >= a) & (fr < b) for k, (a, b) in bands.items()}
    qlo, qhi = int(sr / 450), int(sr / 120)                 # высота голоса 120–450 Гц
    qq = np.arange(int(sr * 0.001), int(sr * 0.02))
    Areg = np.vstack([qq, np.ones_like(qq)]).T
    out = {k: np.empty(n) for k in ["db", "cpp", "per"] + list(bands)}
    for s in range(0, n, 2000):
        e = min(n, s + 2000)
        idx = np.arange(win)[None, :] + hop * np.arange(s, e)[:, None]
        f = m[idx] * w
        out["db"][s:e] = 10 * np.log10((f ** 2).mean(1) + 1e-14)
        S = np.abs(np.fft.rfft(f, nfft, axis=1))
        P = S ** 2 + 1e-14
        for k in bands:
            out[k][s:e] = 10 * np.log10(P[:, sel[k]].mean(1))
        C = 20 * np.log10(np.abs(np.fft.irfft(20 * np.log10(S + 1e-9), axis=1))[:, : nfft // 2] + 1e-9)
        seg = C[:, qlo:qhi]
        pk = seg.argmax(1) + qlo
        coef = np.linalg.lstsq(Areg, C[:, qq].T, rcond=None)[0]
        out["cpp"][s:e] = seg.max(1) - (coef[0] * pk + coef[1])
        # периодичность в полосе 70–1000 Гц: далёкий голос с эхом её сохраняет, вдох и тишина – нет
        idx2 = np.arange(win2)[None, :] + hop * np.arange(s, e)[:, None]
        P2 = np.abs(np.fft.rfft(m[idx2] * w2, nfft2, axis=1)) ** 2
        ac = np.fft.irfft(P2 * vband, axis=1)
        out["per"][s:e] = ac[:, lagl:lagh].max(1) / (ac[:, 0] + 1e-12)
    out["db_fl"] = percentile_filter(out["db"], 3, size=2001, mode="nearest")   # фон (3% самых тихих в ±10 с)
    out["br_fl"] = percentile_filter(out["br"], 3, size=2001, mode="nearest")
    out["sp"] = percentile_filter(out["db"], 95, size=2001, mode="nearest")     # громкость речи
    return out


def voice_mask(F):
    """Голос: периодичность в полосе 70–1000 Гц (ловит и далёкий голос собеседницы) или чёткие гармоники.
    Серия кадров со слабой периодичностью (0,6–0,75) – голос, только если в ней есть кадр выше 0,75
    или она длится 100+ мс. У речи (и далёкой) почти каждая серия доходит до 0,82–0,95; в конце
    громкого вдоха бывает серия 60 мс с максимумом 0,67 – это ещё вдох (5,84–5,91 теста)."""
    db, cpp, per = F["db"], F["cpp"], F["per"]
    raw = ((per > 0.6) & (db > F["db_fl"] + 8)) | ((cpp > 18) & (db > F["db_fl"] + 12))
    strong = ((per > 0.75) & (db > F["db_fl"] + 8)) | ((cpp > 18) & (db > F["db_fl"] + 12))
    out = np.zeros_like(raw)
    d = np.diff(np.r_[0, raw.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= 10 or strong[a:b].any():
            out[a:b] = True
    return out


def detect(F):
    db, cpp, br, mid, hi, per = F["db"], F["cpp"], F["br"], F["mid"], F["hi"], F["per"]
    n = len(db)
    voiced = voice_mask(F)
    sib = (hi - mid > 5) & (br > F["br_fl"] + 10)                 # шипящая/свистящая
    noise = (br > F["br_fl"] + 10) & (cpp < 17) & (per < 0.5) & ~voiced
    vidx = np.flatnonzero(voiced)

    # 1) шумовые события
    ev, i = [], 0
    while i < n:
        if not noise[i]:
            i += 1; continue
        j = i
        while j < n and (noise[j] or noise[j:j + 4].any()):
            j += 1
        if j - i >= 12:
            ev.append([i, j])
        i = j

    # 2) расширение: пока шум без гармоник, но не через шипящую и не через всплеск
    soft = (br > F["br_fl"] + 5) & (cpp < 17.5) & ~voiced      # расширяться можно до настоящего голоса
    for e in ev:
        a, b = e
        e += [a, b]                              # запомнить зародыш: по нему судим, вдох это или согласная
        med = np.percentile(br[a:b], 90) + 2      # вдох к концу нарастает – мерим по громкой части
        k = a
        while k > max(0, a - 30) and soft[k - 1] and not sib[k - 1] and br[k - 1] < med + 6:
            k -= 1
        e[0] = k
        k = b
        while k < min(n, b + 30) and soft[k] and not sib[k] and br[k] < med + 6:
            k += 1
        e[1] = k
    ev.sort()
    merged = []
    for a, b, sa, sb in ev:
        if merged and a <= merged[-1][1] + 2 and not voiced[merged[-1][1]:a].any():
            m = merged[-1]
            merged[-1] = [m[0], max(m[1], b), min(m[2], sa), max(m[3], sb)]
        else:
            merged.append([a, b, sa, sb])

    # 3) классификация
    keep, check = [], []
    sp_glob = np.percentile(F["sp"], 90)               # обычная громкость речи в записи
    for a, b, sa, sb in merged:
        # контекст и спектр – по зародышу (до расширения), обработка – по расширенному месту
        left = vidx[vidx < sa]; right = vidx[vidx >= sb]
        gl = sa - left[-1] if len(left) else 999
        gr = right[0] - sb if len(right) else 999
        dur = sb - sa
        P = lambda x: 10 * np.log10(np.mean(10 ** (x[sa:sb] / 10)))
        hm = P(hi) - P(mid)
        lvl = np.percentile(db[a:b], 90)
        why = None
        if gl < 3 and gr < 3:
            continue                                   # внутри слова – согласная
        guard_l = guard_r = 0
        if min(gl, gr) < 3 and dur < 20:
            # короткое и прижато к слову: в первых 60 мс у слова может звучать согласная («х», «ть») – её не трогаем
            if gl < 3: guard_l = 6
            if gr < 3: guard_r = 6
        if hm > 5:
            continue                                   # «с», «ш» – свистящая
        if np.median(per[sa:sb]) > 0.42:
            continue                                   # в основном периодично – похоже на голос
        vr = np.diff(np.r_[0, voiced[a:b].astype(int), 0])   # голос внутри расширенного места
        if len(vr) and (np.flatnonzero(vr == -1) - np.flatnonzero(vr == 1)).max(initial=0) >= 3:
            why = "внутри 3+ кадра голоса – проверь"
        sp_ref = max(F["sp"][a], sp_glob)              # где говорит только собеседница, местная «речь» тихая
        if why is None and np.median(per[a:b]) > 0.42:
            why = "под вдохом может звучать голос – проверь"
        if why is None and lvl > sp_ref - 3:
            why = "громко, как речь – проверь"
        elif why is None and b - a > 120:
            why = "длинно – проверь"
        # зазор от голоса: 40 мс после речи, 50 мс до речи (относительно расширенного места)
        L2 = vidx[vidx < a]; R2 = vidx[vidx >= b]
        a2 = max(a, L2[-1] + 4) if len(L2) and a - L2[-1] < 4 else a
        b2 = min(b, R2[0] - 5) if len(R2) and R2[0] - b < 5 else b
        # зажато голосом с обеих сторон – это может быть стык согласных или шёпот между словами:
        # трогаем, только если явно длинное (0,25+ с) и явно без периодичности (медиана ≤ 0,38)
        if len(L2) and len(R2) and a - L2[-1] < 6 and R2[0] - b < 6 and \
                (b - a < 25 or np.median(per[a:b]) > 0.38):
            continue
        if guard_l and len(L2): a2 = max(a2, L2[-1] + 1 + guard_l)
        if guard_r and len(R2): b2 = min(b2, R2[0] - guard_r)
        if b2 - a2 < 10:
            continue
        (check if why else keep).append((a2 * HOP_S, b2 * HOP_S, why or ""))
    return keep, check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--labels", help="взять места из готового файла меток вместо поиска")
    ap.add_argument("--list-only", action="store_true", help="только найти и записать метки")
    ap.add_argument("--soft", action="store_true", help="не заменять фоном, а приглушать")
    ap.add_argument("--db", type=float, default=None, help="для --soft: на сколько приглушить, дБ")
    a = ap.parse_args()

    sr, raw = wavfile.read(a.inp)
    dtype = raw.dtype
    x = raw.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    scale = float(np.iinfo(dtype).max + 1) if np.issubdtype(dtype, np.integer) else 1.0

    base = a.out.rsplit(".", 1)[0]
    if a.labels:
        regs = [tuple(map(float, l.split()[:2])) for l in open(a.labels, encoding="utf-8") if l.strip()]
        check = []
    else:
        F = features(x.mean(1) / scale, sr)
        keep, check = detect(F)
        regs = [(s, e) for s, e, _ in keep]
        open(base + ".check.txt", "w", encoding="utf-8").write(
            "".join(f"{s:.3f}\t{e:.3f}\tпроверь{i} {w}\n" for i, (s, e, w) in enumerate(check, 1)))
    open(base + ".breaths.txt", "w", encoding="utf-8").write(
        "".join(f"{s:.3f}\t{e:.3f}\tвдох{i} {'тише' if a.soft else 'замена'}\n" for i, (s, e) in enumerate(regs, 1)))
    print(f"Вдохов: {len(regs)} → {base}.breaths.txt;  на проверку: {len(check)} → {base}.check.txt")
    for i, (s, e) in enumerate(regs, 1):
        print(f"{i:>3} {s:8.2f}–{e:8.2f}  {e - s:.2f} с")
    for s, e, w in check:
        print(f"  ? {s:8.2f}–{e:8.2f}  {e - s:.2f} с  {w}")
    if a.list_only:
        return

    avoid = [(s, e) for s, e, _ in check]
    if not a.soft:
        y, _ = ee_fill.fill(x, sr, regs, scale, extend=False, verbose=False, avoid=avoid)
    else:
        y = x.copy()
        hop = int(sr * HOP_S); mono = x.mean(1) / scale; nfr = len(mono) // hop
        fdb = 10 * np.log10((mono[: nfr * hop].reshape(nfr, hop) ** 2).mean(1) + 1e-14)
        fd = int(sr * 0.03)
        for s, e in regs:
            c0, c1 = int(s * 100), int(e * 100)
            fl = np.percentile(fdb[max(0, c0 - 1000): c0 + 1000], 3)
            att = a.db if a.db is not None else float(np.clip(np.percentile(fdb[c0:max(c1, c0 + 1)], 90) - (fl + 3), 6, 30))
            g = 10 ** (-att / 20)
            i0, i1 = int(s * sr), int(e * sr)
            env = np.ones(len(y)); env[i0:i1] = g
            ramp = 1 - (1 - g) * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fd)))
            env[max(0, i0 - fd):i0] = ramp[-(i0 - max(0, i0 - fd)):]
            env[i1:i1 + fd] = ramp[::-1][: len(env[i1:i1 + fd])]
            y *= env[:, None]
    ee_fill.write_wav(a.out, sr, y, dtype, raw.ndim)
    print(f"Готово: {a.out} (длина {len(y) / sr:.3f} с, как у исходника)")


if __name__ == "__main__":
    main()
