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

WORDS = ["HAYALİNDEKİ", "TIRNAKLAR", "İÇİN", "DOĞRU", "ADRES", "@sevgiguldenbeauty"]
# orijinalde parmağın ekranı yukarı kaydırdığı anlar (kaynak saniyesi)
SWIPES = [5.95, 6.45, 6.95, 7.48, 8.01]
SWIPE_DUR = 0.28
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


def font_for(text):
    size = 200
    while True:
        fnt = ImageFont.truetype(FONT, size)
        fnt.set_variation_by_name(b"Medium Italic")
        if fnt.getlength(text) < SW * 0.78 or size < 40:
            return fnt
        size -= 6


def text_alpha(t):
    """Kaydırma anlarıyla senkron, yukarı kayan yazı maskesi."""
    im = Image.new("L", (SW, SH), 0)
    d = ImageDraw.Draw(im)
    i = sum(t >= s for s in SWIPES)
    p = 1.0
    if i > 0:
        x = np.clip((t - SWIPES[i - 1]) / SWIPE_DUR, 0, 1)
        p = 1 - (1 - x) ** 3
    d.text((SW / 2, SH / 2 + (1 - p) * SH), WORDS[i], font=font_for(WORDS[i]), fill=255, anchor="mm")
    if p < 1:
        d.text((SW / 2, SH / 2 - p * SH), WORDS[i - 1], font=font_for(WORDS[i - 1]), fill=255, anchor="mm")
    return im


TEXT_RGB = np.array([248, 234, 244], np.float32)
_noise = np.random.default_rng(1)


def replace_text(fr, m, c, t):
    """Ekranın gerçek dokusunu korur: eski harfleri çevreden doldurur, yeni yazıyı ekler."""
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, fr.shape[0])
    x0, x1 = max(xs.min() - 20, 0), min(xs.max() + 20, fr.shape[1])
    orig = fr[y0:y1, x0:x1].astype(np.float32)
    sub = orig.copy()
    ms = m[y0:y1, x0:x1]
    lum = sub.mean(2)
    base = np.median(lum[ms])
    # pembe-beyaz yazı: kırmızı ve mavi yeşilden yüksek (gümüş çerçeve nötr gri)
    textlike = (sub[..., 2] - sub[..., 1] > 6) & (sub[..., 0] - sub[..., 1] > 6) & (lum > base + 5)
    old = ndimage.binary_dilation(ms & textlike, iterations=7) & ms
    valid = (ms & ~old & (sub[..., 2] > sub[..., 1]) & (lum < base + 25)).astype(np.float32)
    num = ndimage.gaussian_filter(sub * valid[..., None], (24, 24, 0))
    den = ndimage.gaussian_filter(valid, 24)[..., None] + 1e-4
    fill = num / den + _noise.normal(0, 2.0, sub.shape)  # kayıt grenini taklit et
    sub[old] = fill[old]
    co = persp_coeffs(c - [x0, y0], [(0, 0), (SW, 0), (SW, SH), (0, SH)])
    a = text_alpha(t).transform((x1 - x0, y1 - y0), Image.PERSPECTIVE, tuple(co), Image.BICUBIC)
    a = np.asarray(a.filter(ImageFilter.GaussianBlur(1.1)), np.float32)[..., None] / 255
    a *= ndimage.binary_erosion(ms, iterations=1)[..., None]
    sub = sub * (1 - a) + TEXT_RGB * a
    # sadece ten piksellerini (parmak) orijinal haliyle geri koy
    # tende yeşil >= mavi; bordo zemin ve pembe yazı kenarlarında mavi > yeşil
    skin = (np.clip((orig[..., 1] - orig[..., 2] + 4) / 8, 0, 1) *
            np.clip((orig[..., 0] - orig[..., 2] - 15) / 15, 0, 1) * (lum > base + 10))
    keep = ndimage.gaussian_filter(ndimage.grey_dilation(skin, size=3), 1.2)[..., None]
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
        m = screen_mask(fr) if t >= 4.9 else None
        if m is not None or (prev_c is not None and t >= 4.9):
            if m is None:
                c = prev_c  # hızlı kaydırmada tespit kaçarsa son konumu kullan
            else:
                c = corners(m)
                if prev_c is not None and np.abs(c - prev_c).max() < 60:
                    c = prev_c * 0.5 + c * 0.5  # titreşim yumuşatma
            prev_c = c
            poly = Image.new("L", (1180, 2556), 0)
            # silme alanı: kenarlardan çıkan/giren eski yazı için dörtgeni dikeyde uzat
            down = ((c[3] - c[0]) + (c[2] - c[1])) / 2
            ce = c.copy()
            ce[:2] -= down * 0.06
            ce[2:] += down * 0.18
            ImageDraw.Draw(poly).polygon([tuple(p) for p in ce], fill=1)
            poly = np.asarray(poly, bool)
            m = poly  # tüm ekran dörtgeni; parmaklar replace_text içinde korunur
            fr = replace_text(fr, m, c, t)
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
