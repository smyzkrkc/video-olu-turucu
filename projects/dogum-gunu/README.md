# Dikey sinematik doğum günü videosu

5 fotoğraftan 1080x1920 / 30 fps / 38 sn Reels-TikTok formatında video üretir.
Tamamen yerel çalışır (Python + NumPy + Pillow + SciPy + FFmpeg), API anahtarı gerekmez.

- **music.py** — 3/4, 90 BPM sentez müzik: ambient piyano arpejleri + pad, ardından piyano/celesta ile "Happy Birthday" melodisi, sahne geçişlerinde whoosh/impact ve reverb.
- **render.py** — otomatik seviye + beyaz dengesi, filmik S-eğrisi, teal/sıcak split-tone, bloom, Ken Burns hareketleri, ışık sızıntılı çapraz geçişler, vinyet, film greni ve odak-açılır minimalist yazılar (Cormorant Garamond Italic + Inter Light).

```bash
# font (bir kere)
mkdir -p fonts && curl -sSfL -o fonts/CormorantGaramond-Italic.ttf \
  "https://raw.githubusercontent.com/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond-Italic%5Bwght%5D.ttf"

python3 music.py music.wav
python3 render.py <foto_klasörü> fonts music.wav dogum_gunu.mp4
# hızlı önizleme kareleri:
python3 render.py <foto_klasörü> fonts music.wav onizleme --preview 2.5,7,12,33
```

Yazılar `render.py` içindeki `TEXTS` listesinden, sahne süreleri/hareketleri `SCENES` listesinden değiştirilir.
Müşteri fotoğrafları ve çıktı videolar gizlilik için repoya eklenmez.
