#!/usr/bin/env python3
"""
Области «fix» из ocenaudio → файл меток для breath_soften.py --labels.

В ocenaudio: выделить место, Ctrl+K, назвать область fix. Потом в окне «Свойства аудио»
(Ctrl+I) → «Маркеры» экспортировать области в CSV (колонки: начало, конец, название, комментарий).

  python3 ocen_fix.py regions.csv fix.txt [--name fix] [--exclude fix1.txt fix2.txt ...]

--exclude – файлы меток прошлых прогонов: места с тем же началом (±10 мс) пропускаются,
чтобы не заделывать одно и то же дважды. Затем:

  python3 breath_soften.py track.wav track_fixed.wav --labels fix.txt
"""
import argparse, csv, io


def secs(x):
    x = x.strip()
    parts = x.split(":")
    v = 0.0
    for p in parts:
        v = v * 60 + float(p.replace(",", "."))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("out")
    ap.add_argument("--name", default="fix", help="название областей, которые брать (без учёта регистра)")
    ap.add_argument("--exclude", nargs="*", default=[], help="файлы меток уже сделанных мест")
    a = ap.parse_args()

    old = []
    for f in a.exclude:
        for l in open(f, encoding="utf-8"):
            if l.strip():
                old.append(float(l.split()[0]))
    text = open(a.csv, encoding="utf-8-sig").read()                 # utf-8-sig – на случай BOM
    first = text.splitlines()[0] if text.strip() else ""
    delim = ";" if first.count(";") > first.count(",") else ","
    rows, bad = [], 0
    for r in csv.reader(io.StringIO(text), delimiter=delim):     # StringIO – комментарии в кавычках с переводом строки
        if len(r) >= 3 and r[2].strip().lower() == a.name.lower():
            try:
                s, e = secs(r[0]), secs(r[1])
            except ValueError:
                bad += 1
                continue
            if e > s:
                rows.append((s, e))
            else:
                bad += 1                                               # точечный маркер – длины нет
    if bad:
        print(f"пропущено строк «{a.name}» без начала/конца или нулевой длины: {bad}")
    rows.sort()
    new = [(s, e) for s, e in rows if not any(abs(s - o) < 0.01 for o in old)]
    open(a.out, "w", encoding="utf-8").write(
        "".join(f"{s:.3f}\t{e:.3f}\tвдох{i} замена\n" for i, (s, e) in enumerate(new, 1)))
    print(f"«{a.name}» в CSV: {len(rows)}, уже сделано: {len(rows) - len(new)}, новых: {len(new)} → {a.out}")
    for s, e in new:
        print(f"  {s:8.2f}–{e:8.2f}  ({e - s:.2f} с)")


if __name__ == "__main__":
    main()
