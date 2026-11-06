"""Ekran kaydındaki Reels'ten Instagram arayüzünü kırpar, eldivenli elin tuttuğu
telefon ekranındaki yazıları yeni metinle değiştirir (sessiz çıktı).

Kullanım: python3 edit.py <girdi.mp4> <font.ttf> <çıktı.mp4> [--preview t1,t2]
"""
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

SRC, FONT, OUT = sys.argv[1:4]
FPS = 30
CROP = (30, 310, 964, 1970)  # x0, y0, x1, y1 (1180x2556 kayıtta temiz alan)
W, H = 1080, 1920
SEARCH_Y = (900, 1700)       # telefonun arandığı dikey bant

# (başlangıç, bitiş, metin) — kaynak zamanına göre
WORDS = [
    (4.4, 6.05, "HAYALİNDEKİ"),
    (6.05, 7.05, "TIRNAKLAR"),
    (7.05, 8.2, "İÇİN"),
    (8.2, 99, "@sevgiguldenbeauty"),
]
# kurgu: (kaynak_başlangıç, kaynak_bitiş) parçaları; bitiş==başlangıç ise donmuş kare
FREEZE = 0.8
SEGMENTS = [(0.5, 4.7), (4.95, 8.45), (8.45, 8.45), (1.9, 2.9)]


def read_frame(t):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", SRC, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(2556, 1180, 3)


def screen_mask(fr):
    f = fr.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    m = (b > g + 4) & (r > g + 4) & (r > 12) & (r < 140)
    m[: SEARCH_Y[0]] = False
    m[SEARCH_Y[1]:] = False
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < 3000:
        return None
    m = lab == k
    return ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=3))


def corners(m):
    ys, xs = np.nonzero(m)
    s, d = xs + ys, xs - ys
    return np.array([[xs[s.argmin()], ys[s.argmin()]], [xs[d.argmax()], ys[d.argmax()]],
                     [xs[s.argmax()], ys[s.argmax()]], [xs[d.argmin()], ys[d.argmin()]]], float)


def persp_coeffs(dst, src):
    """PIL PERSPECTIVE: çıktı (dst) noktasını kaynak (src) noktasına eşler."""
    A, B = [], []
    for (x, y), (u, v) in zip(dst, src):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        B += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(B, float))


SW, SH = 1000, 450  # sanal ekran çözünürlüğü


def text_alpha(text, k):
    """Yazı maskesi; k (0..1) kelimenin beliriş animasyonu."""
    im = Image.new("L", (SW, SH), 0)
    size = 200
    while True:
        fnt = ImageFont.truetype(FONT, size)
        fnt.set_variation_by_name(b"Medium Italic")
        if fnt.getlength(text) < SW * 0.78 or size < 40:
            break
        size -= 6
    fnt = ImageFont.truetype(FONT, int(size * (0.93 + 0.07 * k)))
    fnt.set_variation_by_name(b"Medium Italic")
    ImageDraw.Draw(im).text((SW / 2, SH / 2 + (1 - k) * 12), text, font=fnt, fill=255, anchor="mm")
    return im


def text_at(t):
    for a, b, w in WORDS:
        if a <= t < b:
            return w, float(np.clip((t - a) / 0.15, 0, 1))
    return None, 0.0


TEXT_RGB = np.array([248, 234, 244], np.float32)
_noise = np.random.default_rng(1)


def replace_text(fr, m, c, word, k):
    """Ekranın gerçek dokusunu korur: eski harfleri çevreden doldurur, yeni yazıyı ekler."""
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, fr.shape[0])
    x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, fr.shape[1])
    sub = fr[y0:y1, x0:x1].astype(np.float32)
    ms = m[y0:y1, x0:x1]
    lum = sub.mean(2)
    base = np.median(lum[ms])
    textlike = (sub[..., 2] > sub[..., 0] - 18) & (lum > base + 12)
    old = ndimage.binary_dilation(ms & textlike, iterations=7) & ms
    valid = (ms & ~old).astype(np.float32)
    num = ndimage.gaussian_filter(sub * valid[..., None], (14, 14, 0))
    den = ndimage.gaussian_filter(valid, 14)[..., None] + 1e-4
    fill = num / den
    fill += _noise.normal(0, 2.0, fill.shape)  # kayıt grenini taklit et
    sub[old] = fill[old]
    co = persp_coeffs(c - [x0, y0], [(0, 0), (SW, 0), (SW, SH), (0, SH)])
    a = text_alpha(word, k).transform((x1 - x0, y1 - y0), Image.PERSPECTIVE, tuple(co), Image.BICUBIC)
    a = np.asarray(a.filter(ImageFilter.GaussianBlur(1.1)), np.float32)[..., None] / 255
    a *= ndimage.binary_erosion(ms, iterations=1)[..., None] * k
    sub = sub * (1 - a) + TEXT_RGB * a
    # parmak çevresinde orijinal kareyi koru (yumuşak geçişle)
    orig = fr[y0:y1, x0:x1].astype(np.float32)
    skin = (orig[..., 0] > orig[..., 2] + 25) & (lum > base + 15)
    prot = ndimage.gaussian_filter(ndimage.binary_dilation(skin, iterations=5).astype(np.float32), 3)
    keep = np.clip(prot * 1.6, 0, 1)[..., None]
    sub = sub * (1 - keep) + orig * keep
    fr[y0:y1, x0:x1] = np.clip(sub, 0, 255).astype(np.uint8)
    return fr


def timeline():
    out = []
    for a, b in SEGMENTS:
        n = int(round((FREEZE if a == b else b - a) * FPS))
        out += [a if a == b else a + i / FPS for i in range(n)]
    return out


def main():
    times = timeline()
    if "--preview" in sys.argv:
        times = [float(x) for x in sys.argv[sys.argv.index("--preview") + 1].split(",")]
    enc = None
    if "--preview" not in sys.argv:
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
    prev_c = None
    cache = {}
    for i, t in enumerate(times):
        fr = cache.get(t)
        if fr is None:
            fr = read_frame(t)
            cache = {t: fr}
        fr = fr.copy()
        word, k = text_at(t) if t >= 4.0 else (None, 0.0)
        m = screen_mask(fr) if word else None
        if m is not None:
            c = corners(m)
            if prev_c is not None and np.abs(c - prev_c).max() < 60:
                c = prev_c * 0.5 + c * 0.5  # titreşim yumuşatma
            prev_c = c
            # başparmağa değen eski yazı pikselleri de dörtgen içinde kapsansın
            poly = Image.new("L", (1180, 2556), 0)
            ImageDraw.Draw(poly).polygon([tuple(p) for p in c], fill=1)
            poly = np.asarray(poly, bool)
            f16 = fr.astype(np.int16)
            # beyaz-pembe yazı: mavi kanal yüksek (tende mavi, kırmızının çok altında)
            text = (f16.min(2) > 185) & (f16[..., 2] >= f16[..., 1]) & (f16[..., 2] > f16[..., 0] - 18) & poly
            m = ndimage.binary_fill_holes(ndimage.binary_closing(m | text, iterations=2)) & poly
            fr = replace_text(fr, m, c, word, k)
        else:
            prev_c = None
        x0, y0, x1, y1 = CROP
        img = Image.fromarray(fr[y0:y1, x0:x1]).resize((W, H), Image.LANCZOS)
        if enc:
            enc.stdin.write(img.tobytes())
        else:
            img.save(f"{OUT}_{t:.2f}.jpg", quality=88)
    if enc:
        enc.stdin.close()
        enc.wait()


if __name__ == "__main__":
    main()
