# roomtone-fill

Remove filler sounds ("uh", "eh") and breaths from speech **without changing the timing** – each
removed spot is filled with room tone of exactly the same length. The output file is identical in
length to the input down to the sample, so it drops straight back under the original video,
including multicam edits.

Detection of fillers is done by [Auphonic](https://auphonic.com) (Automatic Cutting → Cut Fillers,
Mode: *Export Uncut Audio*, output: *Cut List* in `.AudacityRegions.txt` format).
These scripts do the filling, plus breath detection and two small helpers.

> Code comments and console output are in Russian; this README describes the algorithms in English.

## Tools

| Script | What it does |
|---|---|
| `ee_fill.py` | Fills every region of an Audacity cut list with room tone of the same length |
| `breath_soften.py` | Finds breaths in speech and fills them the same way (uses `ee_fill.py`) |
| `thump_fix.py` | Removes a single low-frequency thump (hand hitting the mic) at a given time |
| `sync_offset.py` | Finds the offset between two recordings of the same conversation – by audio, or by motion in the frame for a camera without sound |

```
pip install numpy scipy          # sync_offset.py also needs ffmpeg in PATH

python3 ee_fill.py track.wav track.AudacityRegions.txt track_clean.wav
python3 breath_soften.py track_clean.wav track_final.wav
python3 thump_fix.py track_final.wav track_final2.wav --at 304.805
python3 sync_offset.py audio camA.wav camB.wav
python3 sync_offset.py video camA.mov screen_recording.mov
```

Every script writes Audacity label files next to the output, so all patched spots can be reviewed
by ear (File → Import → Labels; Alt+→ / Alt+← jumps between labels).

## How room tone is chosen (`ee_fill.py`)

The signal is analysed in 10 ms frames (RMS level in dB).

**Preparation**

* **Region extension.** The cut list sometimes ends a filler before its decay has finished. The
  region is extended to the right by up to 80 ms while the level keeps falling and stays more than
  6 dB above the local floor; to the left by at most 20 ms (further left is usually the end of the
  previous word).
* **Banned frames.** All filler regions (with a 50 ms margin) and all detected breaths are marked
  as unusable – room tone is never taken from another filler or breath.
* **Local floor** = 3rd percentile of frame levels within ±10 s. (The 10th percentile turned out
  too high in dense speech.)

**Source selection, in order of preference**

1. **Nearby, in one piece.** Within ±10 s, look for a continuous window of the needed length
   (region + crossfade margins) that contains no banned frames. The window must be *flat*:
   max − median ≤ 5 dB (relaxed to 8 dB, then to no limit, if nothing is found) – this rejects
   windows containing clicks, lip smacks or the tail of a word. Among valid windows the one with
   the lowest `P90 + 0.5·mean` wins (quiet and even). It is used if its mean level is within
   **4 dB** of the local floor.
2. **A longer pause anywhere in the file.** If the nearby choice is too loud, search the whole file
   for the closest window that is quiet in absolute terms (max ≤ global 10th percentile + 3 dB)
   and flat (max − median ≤ 5 dB). Its gain is matched to the room-tone level around the filler
   (energy average of nearby quiet frames), with the correction **clamped to ±4 dB** – larger
   corrections would amplify or flatten the noise character audibly.
3. **Tiling.** If there is no long enough pause at all, stitch several short quiet chunks from
   ±10 s (trying 0.5, 0.4, 0.3, 0.25, 0.2 s), normalise them to their median RMS and join them
   with 40 ms equal-power crossfades, repeating them cyclically if needed. Used only if the result
   is at least 3 dB quieter than the best single nearby window.
4. **Nearby but louder.** Accept the nearby window from step 1 if it is within 8 dB of the floor.
5. **Otherwise skip.** If nothing suitable exists, the filler is left untouched and reported –
   inserting a piece of speech or a loud noise is worse than a remaining "uh".

**Insertion.** The patch replaces `[start − 15 ms, end + 15 ms]`; both edges use 15 ms
equal-power (sin/cos) crossfades. Sample rate, channels and bit depth are preserved.

**In practice** (one 42-minute interview track, quiet room, 244 fillers from Auphonic): 94
patches came from step 1, 150 from step 2, none needed tiling, none were skipped. Steps 2 and the
flatness requirement were what removed the last audible seams: before them, the "quietest nearby"
window sometimes contained a faint click or a rise in background noise.

## How breaths are detected (`breath_soften.py`)

Frames of 10 ms with three features:

* **Voicing** – normalised autocorrelation peak in the 70–1000 Hz band (60 ms window). Breaths and
  silence give ~0.27–0.35, a distant voice heard from a laptop speaker ~0.64, close speech ~0.88.
  The cepstral peak (CPP) is used as an extra cue, but it misses distant reverberant voices, so it
  is never used alone.
* **Breath noise** – energy in 1–8 kHz at least 10 dB above the local floor of that band, with no
  voicing.
* **Not a sibilant** – sibilants have 4–8 kHz clearly above 1–2 kHz and sit right against voiced
  frames; breaths sit in pauses.

A detected spot grows while unvoiced noise continues, but never through a sibilant or a transient
(so a word starting with "s" or "t" is not eaten), and keeps a gap of 30 ms after and 50 ms before
speech. Spots that are louder than typical speech minus 3 dB, longer than 1.2 s, or contain 3+
voiced frames are **not touched** – they go to `*.check.txt` for listening. Accepted breaths are
filled with `ee_fill.fill()` (no region extension; doubtful spots are banned as sources).

## `sync_offset.py`

Cross-correlation (FFT) of either the audio envelope (1 kHz, pre-emphasised, 5 ms smoothing) or a
motion signal (mean absolute frame difference at 64×36, summed over 0.2 s so that duplicated
frames of a 12–20 fps call stream don't matter). Prints the best offsets with a robust z-score and
re-checks the offset in 8 parts of the recording to reveal clock drift or changing latency.

For a video call recorded without headphones each track contains both voices – one direct, one
through the speaker – so the audio correlation shows two peaks, Δ − L and Δ + L (Δ = clock offset,
L = call latency); the script reports this pair.

## License

MIT – see `LICENSE`.
