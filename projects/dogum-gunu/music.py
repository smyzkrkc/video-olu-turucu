"""Sinematik doğum günü müziği (sentez): 3/4, 90 BPM.

Bar 1-9  : ambient piyano arpej + pad (Am F C G Am F C G G7)
Bar 10-17: yumuşak piyano + celesta ile "Happy Birthday" melodisi
Bar 18   : çözülme akoru + reverb kuyruğu
"""
import sys
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt

SR = 48000
BPM = 90
BEAT = 60 / BPM          # 0.667 s
BAR = 3 * BEAT           # 2.0 s
TOTAL = 38.0

rng = np.random.default_rng(7)
N = int(SR * TOTAL)
dry = np.zeros((N, 2))
fx = np.zeros((N, 2))    # impact / whoosh (daha az reverb)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def nm(s):
    """'C4', 'F#3' -> midi"""
    base = NOTE[s[0]]
    i = 1
    if s[1] == "#":
        base += 1
        i = 2
    return 12 * (int(s[i:]) + 1) + base


def add(buf, sig, t, pan=0.0, gain=1.0):
    i = int(t * SR)
    if i < 0:
        sig, i = sig[-i:], 0
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    buf[i:i + len(sig), 0] += sig * l * gain
    buf[i:i + len(sig), 1] += sig * r * gain


def piano(f, dur, vel=0.5):
    n = int(SR * (dur + 1.5))
    t = np.arange(n) / SR
    out = np.zeros(n)
    B = 0.0004  # inharmonisite
    for k in range(1, 9):
        fk = f * k * np.sqrt(1 + B * k * k)
        if fk > SR / 2.2:
            break
        amp = (1 / k ** 1.3) * (1 + 0.6 * vel * (k > 2))
        decay = 1.2 + 0.9 * k + f / 400
        out += amp * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28)) * np.exp(-decay * t)
    att = np.minimum(1, t / 0.004)
    # bırakma
    rel = np.where(t > dur, np.exp(-(t - dur) * 6), 1.0)
    return out * att * rel * vel * 0.35


def celesta(f, vel=0.3):
    n = int(SR * 2.5)
    t = np.arange(n) / SR
    s = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * f * 4.01 * t) * np.exp(-8 * t))
    return s * np.exp(-2.2 * t) * np.minimum(1, t / 0.002) * vel * 0.3


def pad(freqs, dur, vel=0.12):
    n = int(SR * dur)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for f in freqs:
        for d in (-0.08, 0.0, 0.07):
            ff = midi(f) * 2 ** (d / 12)
            for k, a in ((1, 1.0), (2, 0.35), (3, 0.15), (4, 0.06)):
                out += a * np.sin(2 * np.pi * ff * k * t + rng.uniform(0, 6.28))
    env = np.minimum(1, t / 1.2) * np.minimum(1, (dur - t) / 1.0)
    lfo = 1 + 0.08 * np.sin(2 * np.pi * 0.25 * t)
    return out * env * lfo * vel / len(freqs) / 3


def bass(f, dur, vel=0.22):
    n = int(SR * dur)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * midi(f) * t) + 0.2 * np.sin(4 * np.pi * midi(f) * t)
    env = np.minimum(1, t / 0.05) * np.minimum(1, (dur - t) / 0.4)
    return s * env * vel


def boom(vel=0.6):
    n = int(SR * 3)
    t = np.arange(n) / SR
    f = 30 + 45 * np.exp(-t * 6)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 1.6)
    noise = rng.standard_normal(n) * np.exp(-t * 25) * 0.15
    return (s + noise) * vel


def whoosh(dur=1.3, vel=0.18):
    """Reverse yükselen nefes; t=dur anında zirve yapar."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    sos = butter(2, [400, 5000], btype="band", fs=SR, output="sos")
    x = sosfilt(sos, x)
    env = (t / dur) ** 3
    return x * env * vel


CH = {
    "Am": ["A2", "A3", "C4", "E4"], "F": ["F2", "F3", "A3", "C4"],
    "C": ["C3", "G3", "C4", "E4"], "G": ["G2", "G3", "B3", "D4"],
    "G7": ["G2", "G3", "B3", "F4"], "F2": ["F2", "C4", "F4", "A4"],
    "Cg": ["G2", "G3", "C4", "E4"],
}

intro = ["Am", "F", "C", "G", "Am", "F", "C", "G", "G7"]
for b, c in enumerate(intro):
    t0 = b * BAR
    notes = [nm(x) for x in CH[c]]
    # pad ve bas
    add(dry, pad(notes[1:], BAR + 1.2, 0.10 if b < 2 else 0.14), t0 - 0.2, 0, 1)
    if b >= 2:
        add(dry, bass(notes[0], BAR), t0, 0, 0.9)
    # arpej (8'likler): kök-5-oktav-3
    arp = [notes[1], notes[2], notes[3], notes[2] + 12, notes[3], notes[2]]
    for i, n_ in enumerate(arp):
        if b < 1 and i % 2:
            continue
        v = 0.30 + 0.08 * (i == 0) + 0.02 * b
        add(dry, piano(midi(n_), BEAT * 0.9, v), t0 + i * BEAT / 2, pan=-0.3 + 0.12 * i, gain=1)

# Happy Birthday — öncü (pickup) bar 9'un son vuruşu
mel_start = 9 * BAR  # 18 s
melody = [  # (nota, vuruş uzunluğu)
    ("G4", .75), ("G4", .25),
    ("A4", 1), ("G4", 1), ("C5", 1), ("B4", 2), ("G4", .75), ("G4", .25),
    ("A4", 1), ("G4", 1), ("D5", 1), ("C5", 2), ("G4", .75), ("G4", .25),
    ("G5", 1), ("E5", 1), ("C5", 1), ("B4", 1), ("A4", 1), ("F5", .75), ("F5", .25),
    ("E5", 1), ("C5", 1), ("D5", 1), ("C5", 3),
]
# melodi bar 10 = 20 s'de başlar; öncü bir vuruş önce (19.33 s)
t = mel_start + BAR - BEAT
for n_, d in melody:
    f = midi(nm(n_))
    add(dry, piano(f, d * BEAT * 0.95, 0.55), t, pan=0.05, gain=1)
    add(dry, celesta(f * 2, 0.22), t, pan=0.25, gain=1)
    t += d * BEAT

mel_chords = ["C", "G", "G", "C", "C", "F2", "Cg", "C"]
for b, c in enumerate(mel_chords):
    t0 = mel_start + BAR * (b + 1)
    notes = [nm(x) for x in CH[c]]
    add(dry, pad(notes[1:], BAR + 1.0, 0.16), t0 - 0.2, 0, 1)
    add(dry, bass(notes[0], BAR, 0.24), t0, 0, 1)
    # vals eşlik: vuruş 2 ve 3'te akor
    for k in (1, 2):
        for n_ in notes[1:]:
            add(dry, piano(midi(n_), BEAT * 0.6, 0.16), t0 + k * BEAT, pan=-0.35, gain=1)

# final çözülme
tf = mel_start + BAR * 9  # 36 s
for n_ in ["C2", "C3", "G3", "C4", "E4", "G4", "C5"]:
    add(dry, piano(midi(nm(n_)), 2.0, 0.35), tf, pan=0, gain=1)
add(dry, pad([nm("C4"), nm("E4"), nm("G4")], 2.0, 0.12), tf, 0, 1)

# sahne geçişleri: impact + whoosh
for tc in [4.0, 10.0, 14.0, 18.0, 22.0, 28.0]:
    add(fx, whoosh(), tc - 1.3, pan=0, gain=1)
for tc, v in [(4.0, 0.55), (20.0, 0.4), (28.0, 0.6)]:
    add(fx, boom(v), tc, 0, 1)


def reverb(x, sec=3.2, wet=0.35):
    n = int(SR * sec)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-t * 6.9 / sec)
        sos = butter(1, 6000, fs=SR, output="sos")
        ir = sosfilt(sos, ir)
        ir[: int(0.02 * SR)] *= np.linspace(0, 1, int(0.02 * SR))
        ir /= np.sqrt(np.sum(ir ** 2))
        out[:, ch] = fftconvolve(x[:, ch], ir)[: len(x)]
    return x * (1 - wet) + out * wet * 1.6


mix = reverb(dry, 3.5, 0.42) + reverb(fx, 1.8, 0.2)
# yumuşak doygunluk + fade
mix = np.tanh(mix * 1.4) / 1.4
tt = np.arange(N) / SR
mix *= np.minimum(1, tt / 0.8)[:, None]
mix *= np.clip((TOTAL - tt) / 2.0, 0, 1)[:, None]
mix /= np.max(np.abs(mix)) + 1e-9
mix *= 0.89

out = sys.argv[1] if len(sys.argv) > 1 else "music.wav"
pcm = (mix * 32767).astype("<i2")
import wave
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", out)
