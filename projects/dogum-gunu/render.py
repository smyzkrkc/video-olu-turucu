"""Dikey (1080x1920) sinematik doğum günü videosu.

Kullanım:
  python3 render.py <foto_klasörü> <font_klasörü> <music.wav> <çıktı.mp4> [--preview t1,t2,...]

Fotoğraflar klasörde 1.jpg ... 5.jpg olarak beklenir.
"""
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

W, H, FPS = 1080, 1920, 30
DUR = 38.0
XF = 0.8  # çapraz geçiş süresi

PHOTOS, FONTS, MUSIC, OUT = sys.argv[1:5]
SERIF = os.path.join(FONTS, "CormorantGaramond-Italic.ttf")
SANS = "/usr/share/fonts/opentype/inter/Inter-Light.otf"

CREAM = (244, 237, 228)


# ---------------------------------------------------------------- renk
def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def enhance_and_grade(img, wb=0.35, exposure=1.0, warmth=1.0):
    x = np.asarray(img, dtype=np.float32) / 255.0
    # otomatik seviye (yumuşak)
    lo = np.percentile(x, 0.4, axis=(0, 1))
    hi = np.percentile(x, 99.6, axis=(0, 1))
    stretched = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    x = x * 0.4 + stretched * 0.6
    # kısmi gri-dünya beyaz dengesi
    mean = x.reshape(-1, 3).mean(0)
    gain = mean.mean() / (mean + 1e-6)
    x = x * (1 + wb * (gain - 1))
    x = np.clip(x * exposure, 0, 1)
    # filmik S-eğrisi: siyahları hafif kaldır, vurguları yuvarla
    x = 0.5 + (x - 0.5) * 1.10
    x = np.clip(x, 0, 1)
    x = x * x * (3 - 2 * x) * 0.35 + x * 0.65
    x = 0.028 + x * (0.965 - 0.028)
    lum = (x @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
    # split-tone: gölgeler teal, vurgular sıcak
    sh = 1 - smoothstep(0.0, 0.45, lum)
    hl = smoothstep(0.5, 1.0, lum)
    x = x + sh * np.array([-0.022, 0.006, 0.026]) + hl * np.array([0.03, 0.012, -0.022]) * warmth
    # doygunluk (hafif azalt)
    x = lum + (x - lum) * 0.9
    # bloom
    glow = np.clip(lum - 0.72, 0, 1) * np.array([1.0, 0.86, 0.7])
    gimg = Image.fromarray((np.clip(glow, 0, 1) * 255).astype(np.uint8))
    gimg = gimg.filter(ImageFilter.GaussianBlur(max(img.size) / 60))
    x = x + np.asarray(gimg, dtype=np.float32) / 255.0 * 0.35
    out = Image.fromarray((np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8))
    return out.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))


def load(n, **kw):
    im = ImageOps.exif_transpose(Image.open(os.path.join(PHOTOS, f"{n}.jpg"))).convert("RGB")
    return enhance_and_grade(im, **kw)


IMG = {
    1: load(1, wb=0.25, warmth=0.6),
    2: load(2, wb=0.3, exposure=0.9),
    3: load(3, wb=0.2, warmth=0.7),
    4: load(4, wb=0.35),
    5: load(5, wb=0.3, warmth=0.6),
}
# yatay fotoğraf için bulanık arka plan
_bg = IMG[5].copy()
_s = max(W / _bg.width, H / _bg.height)
_bg = _bg.resize((int(_bg.width * _s) + 2, int(_bg.height * _s) + 2), Image.BICUBIC)
_bg = _bg.crop(((_bg.width - W) // 2, (_bg.height - H) // 2, (_bg.width - W) // 2 + W, (_bg.height - H) // 2 + H))
BG5 = np.asarray(_bg.filter(ImageFilter.GaussianBlur(45)), dtype=np.float32) / 255.0 * 0.45


# ---------------------------------------------------------------- hareket
def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def ease_out(t):
    t = np.clip(t, 0, 1)
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


def cover(img, zoom, cx, cy):
    sw, sh = img.size
    s = max(W / sw, H / sh) * zoom
    hw, hh = W / 2 / s, H / 2 / s
    cx = np.clip(cx * sw, hw, sw - hw)
    cy = np.clip(cy * sh, hh, sh - hh)
    data = (1 / s, 0, cx - hw, 0, 1 / s, cy - hh)
    out = img.transform((W, H), Image.AFFINE, data, resample=Image.BICUBIC)
    return np.asarray(out, dtype=np.float32) / 255.0


def framed(img, scale, bg):
    """Yatay fotoğraf: bulanık arka plan üzerinde ortalanmış kare."""
    fw = W * 0.88 * scale
    s = fw / img.width
    fh = img.height * s
    x0, y0 = (W - fw) / 2, (H - fh) / 2 - 40
    data = (1 / s, 0, -x0 / s, 0, 1 / s, -y0 / s)
    fg = img.transform((W, H), Image.AFFINE, data, resample=Image.BICUBIC)
    fg = np.asarray(fg, dtype=np.float32) / 255.0
    yy, xx = np.ogrid[:H, :W]
    # yumuşak kenarlı maske + gölge
    m = (np.clip(xx - x0 + 1, 0, 1) * np.clip(x0 + fw - xx, 0, 1) *
         np.clip(yy - y0 + 1, 0, 1) * np.clip(y0 + fh - yy, 0, 1))[..., None]
    dx = np.maximum(np.maximum(x0 - xx, xx - (x0 + fw)), 0)
    dy = np.maximum(np.maximum(y0 + 30 - yy, yy - (y0 + fh + 30)), 0)
    shadow = np.exp(-(dx ** 2 + dy ** 2) / (2 * 40 ** 2))[..., None] * 0.6
    base = bg * (1 - shadow)
    return base * (1 - m) + fg * m


def final_bg(u):
    arr = cover(IMG[1], lerp(1.5, 1.68, ease(u)), 0.47, lerp(0.47, 0.44, ease(u)))
    r = float(9 * ease((u - 0.12) / 0.3))
    if r > 0.4:
        im = Image.fromarray((arr * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r))
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


# sahneler: (başlangıç, bitiş, çizim fonksiyonu(yerel_t 0..1))
def sc_kb(n, z0, z1, c0, c1):
    return lambda u: cover(IMG[n], lerp(z0, z1, ease(u)),
                           lerp(c0[0], c1[0], ease(u)), lerp(c0[1], c1[1], ease(u)))


SCENES = [
    (4.0, 10.4, sc_kb(1, 1.0, 1.12, (0.5, 0.52), (0.47, 0.47))),
    (9.6, 14.4, sc_kb(3, 1.08, 1.16, (0.36, 0.42), (0.64, 0.40))),
    (13.6, 18.4, sc_kb(2, 1.16, 1.0, (0.42, 0.44), (0.46, 0.50))),
    (17.6, 22.4, lambda u: framed(IMG[5], lerp(1.0, 1.06, ease(u)), BG5)),
    (21.6, 28.4, sc_kb(4, 1.0, 1.1, (0.5, 0.46), (0.5, 0.40))),
    (27.6, DUR, lambda u: final_bg(u)),
]


# ---------------------------------------------------------------- yazı
def font(path, size, var=None):
    f = ImageFont.truetype(path, size)
    if var:
        try:
            f.set_variation_by_name(var)
        except Exception:
            pass
    return f


def text_sprite(text, fnt, tracking=0.0, color=CREAM):
    """Harf aralıklı metni RGBA sprite olarak çizer."""
    pad = 40
    widths = [fnt.getlength(c) for c in text]
    tw = sum(widths) + tracking * fnt.size * (len(text) - 1)
    asc, desc = fnt.getmetrics()
    im = Image.new("RGBA", (int(tw) + 2 * pad, asc + desc + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    x = pad
    for c, w in zip(text, widths):
        d.text((x, pad), c, font=fnt, fill=color + (255,))
        x += w + tracking * fnt.size
    # yumuşak gölge
    a = im.getchannel("A").filter(ImageFilter.GaussianBlur(max(6, fnt.size // 10)))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    sh.putalpha(a.point(lambda v: int(v * 0.55)))
    return Image.alpha_composite(sh, im)


def line_sprite(w, alpha=180):
    im = Image.new("RGBA", (w + 80, 42), (0, 0, 0, 0))
    ImageDraw.Draw(im).line([(40, 21), (40 + w, 21)], fill=CREAM + (alpha,), width=2)
    return im


SERIF_M = font(SERIF, 76, "Light")
SERIF_L = font(SERIF, 150, "Light")
SANS_S = font(SANS, 26)

LOW = 1330  # alt yazı hattı (Reels arayüzünün üstünde)
TEXTS = [
    # (giriş, çıkış, sprite, merkez_y)
    (0.7, 3.5, text_sprite("Bazı insanlar", SERIF_M), 900),
    (1.5, 3.5, text_sprite("hayatı güzelleştirir.", SERIF_M), 1000),
    (5.0, 9.3, text_sprite("BİR HİKÂYE", SANS_S, 0.45), LOW - 80),
    (5.4, 9.3, text_sprite("seninle başlayan", SERIF_M), LOW),
    (10.6, 13.6, text_sprite("aynı yöne bakan iki kalp", SERIF_M), LOW),
    (14.6, 17.6, text_sprite("kahkahalarla dolu anlar", SERIF_M), LOW),
    (18.6, 21.6, text_sprite("her yolda yan yana", SERIF_M), LOW + 120),
    (22.6, 27.3, text_sprite("ve daha nice güzel yıllara", SERIF_M), LOW),
    (29.2, 36.6, text_sprite("MUTLU YILLAR", SANS_S, 0.6), 1080),
    (29.8, 36.6, text_sprite("İyi ki", SERIF_L), 1210),
    (30.4, 36.6, text_sprite("doğdun", SERIF_L), 1360),
    (31.2, 36.6, line_sprite(120), 1490),
]

# metin okunurluğu için alt gradyan
_yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
GRAD_LOW = 1 - 0.6 * smoothstep(0.48, 0.78, _yy) * (1 - 0.4 * smoothstep(0.85, 1.0, _yy))
GRAD_FINAL = 0.62 - 0.3 * smoothstep(0.45, 0.8, _yy)


def composite_text(frame, t):
    for t0, t1, spr, cy in TEXTS:
        if t < t0 or t > t1 + 0.01:
            continue
        a_in = ease_out((t - t0) / 0.9)
        a_out = 1 - ease((t - (t1 - 0.6)) / 0.6)
        alpha = a_in * a_out
        if alpha <= 0.003:
            continue
        blur = float((1 - a_in) * 7)
        s = spr.filter(ImageFilter.GaussianBlur(blur)) if blur > 0.3 else spr
        dy = (1 - a_in) * 22
        arr = np.asarray(s, dtype=np.float32) / 255.0
        x0 = (W - s.width) // 2
        y0 = int(cy - s.height / 2 + dy)
        ys, xs = max(0, y0), max(0, x0)
        ye, xe = min(H, y0 + s.height), min(W, x0 + s.width)
        sub = arr[ys - y0:ye - y0, xs - x0:xe - x0]
        a = sub[..., 3:4] * alpha
        # hafif gölge için alttaki alanı biraz koyulaştır
        frame[ys:ye, xs:xe] = frame[ys:ye, xs:xe] * (1 - a) + sub[..., :3] * a
    return frame


# ---------------------------------------------------------------- doku
_rng = np.random.default_rng(3)
GRAIN = [np.repeat(np.repeat(_rng.standard_normal((H // 2, W // 2)).astype(np.float32), 2, 0), 2, 1)[..., None]
         for _ in range(12)]
_yv, _xv = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.sqrt(((_xv - W / 2) / (W * 0.62)) ** 2 + ((_yv - H / 2) / (H * 0.62)) ** 2)
VIGNETTE = (1 - 0.42 * smoothstep(0.45, 1.25, _r))[..., None]
CUTS = [10.0, 14.0, 18.0, 22.0, 28.0]


def light_leak(t):
    """Geçişlerde sıcak ışık sızıntısı (screen)."""
    for i, c in enumerate(CUTS + [4.25]):
        d = (t - c) / 0.55
        if abs(d) < 1.6:
            k = np.exp(-d * d * 1.6) * (0.32 if c != 4.25 else 0.25)
            cx = W * (-0.05 + 1.1 * (i % 2)) + d * 260 * (1 - 2 * (i % 2))
            cy = H * (0.25 + 0.5 * ((i * 0.61) % 1))
            g = np.exp(-(((_xv - cx) / (W * 0.45)) ** 2 + ((_yv - cy) / (H * 0.3)) ** 2))[..., None]
            return g * k * np.array([1.0, 0.55, 0.25], dtype=np.float32)
    return None


# ---------------------------------------------------------------- kare
def render(i):
    t = i / FPS
    frame = np.zeros((H, W, 3), dtype=np.float32)
    for a, b, fn in SCENES:
        if a <= t <= b:
            u = (t - a) / (b - a)
            img = fn(u)
            alpha = ease((t - a) / (XF if a > 4.0 else 0.6))
            frame = frame * (1 - alpha) + img * alpha
    # okunurluk gradyanları
    if 4.0 <= t <= 28.4:
        frame *= GRAD_LOW
    if t >= 27.6:
        frame *= 1 - (1 - GRAD_FINAL) * ease((t - 27.6) / 2.0)
    leak = light_leak(t)
    if leak is not None:
        frame = 1 - (1 - frame) * (1 - leak)
    frame = composite_text(frame, t)
    frame *= VIGNETTE
    lum = frame.mean(axis=2, keepdims=True)
    frame += GRAIN[i % len(GRAIN)] * (0.012 + 0.03 * lum * (1 - lum))
    # final karartma
    if t > DUR - 1.6:
        frame *= 1 - ease((t - (DUR - 1.6)) / 1.6)
    return (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    if "--preview" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]
        for t in ts:
            Image.fromarray(render(int(t * FPS))).save(f"{OUT}_{t:05.1f}.jpg", quality=90)
        return
    n = int(DUR * FPS)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", MUSIC,
           "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-maxrate", "14M", "-bufsize", "28M", "-pix_fmt", "yuv420p",
           "-profile:v", "high", "-tune", "grain",
           "-af", "loudnorm=I=-14:TP=-1.0:LRA=11", "-c:a", "aac", "-b:a", "256k", "-ar", "48000",
           "-shortest", "-movflags", "+faststart", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for k, fr in enumerate(pool.imap(render, range(n), chunksize=4)):
            p.stdin.write(fr.tobytes())
            if k % 150 == 0:
                print(f"{k}/{n}", flush=True)
    p.stdin.close()
    p.wait()
    print("bitti:", OUT)


if __name__ == "__main__":
    main()
