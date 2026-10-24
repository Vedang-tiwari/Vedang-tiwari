#!/usr/bin/env python3
"""
Step 3a — Prep photo for ASCII conversion
Prepares the source photo for clean, high-contrast monochrome ASCII art:
- Isolates the subject and contrasts features (eyes, mustache, hair, suit)
- Composites onto pure white (255) so background maps to spaces in the ASCII ramp
- Uses rembg + OpenCV CLAHE / bilateral if available, with robust Pillow+NumPy fallback

Output: source-prepped.png (grayscale)
Usage:
    python scripts/prep_photo.py [source-photo.png] [source-prepped.png]
"""
import os
import sys
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "source-photo.png")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "source-prepped.png")

if not os.path.exists(INP):
    alts = [
        os.path.join(HERE, "..", "source-photo.jpg"),
        os.path.join(HERE, "..", "source-photo.jpeg"),
        "source-photo.png",
        "source-photo.jpg",
    ]
    for alt in alts:
        if os.path.exists(alt):
            INP = alt
            break

if not os.path.exists(INP):
    print(f"[-] Source photo not found: {INP}")
    sys.exit(1)


def prep_with_rembg(inp_path, out_path):
    from rembg import remove
    import cv2

    img = Image.open(inp_path).convert("RGBA")
    cut = remove(img)
    rgb = np.array(cut.convert("RGB"))
    alpha = np.array(cut.split()[-1])

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = gray
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, 9, 40, 9)

    lo, hi = np.percentile(smooth[alpha > 128], [2, 92])
    tone = np.clip((smooth.astype(np.float32) - lo) / (hi - lo), 0, 1)

    fine = cv2.GaussianBlur(smooth, (0, 0), 1.5).astype(np.float32)
    coarse = cv2.GaussianBlur(smooth, (0, 0), 6).astype(np.float32)
    lines = np.clip((coarse - fine) / 40.0, 0, 1)
    out = np.clip(tone - 0.6 * lines, 0, 1) * 255.0

    mask = cv2.GaussianBlur(alpha.astype(np.float32) / 255.0, (0, 0), 1.0)
    out = out * mask + 255.0 * (1.0 - mask)

    ys, xs = np.where(alpha > 20)
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) + 60
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    canvas = np.full((side, side), 255, np.uint8)
    x0, y0 = cx - side // 2, cy - side // 2
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, out.shape[1]), min(y0 + side, out.shape[0])
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = out[sy0:sy1, sx0:sx1].astype(np.uint8)

    Image.fromarray(canvas, mode="L").save(out_path)
    print(f"[+] Processed with rembg+OpenCV -> {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")


def prep_with_pillow(inp_path, out_path):
    img = Image.open(inp_path).convert("RGB")
    w, h = img.size

    gray = ImageOps.grayscale(img)
    gray_contrasted = ImageOps.autocontrast(gray, cutoff=2)
    crisp = gray_contrasted.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    arr = np.array(crisp, dtype=np.float32)

    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w / 2.0, h * 0.45
    dist = np.sqrt(((X - center_x) / (w * 0.44)) ** 2 + ((Y - center_y) / (h * 0.52)) ** 2)

    mask = np.clip(1.0 - (dist - 0.70) / 0.35, 0.0, 1.0)
    toned = (arr - 30) / (220 - 30) * 255.0
    toned = np.clip(toned, 0, 255)

    final = toned * mask + 255.0 * (1.0 - mask)
    final = np.clip(final, 0, 255).astype(np.uint8)

    prepped = Image.fromarray(final, mode="L")
    prepped.save(out_path)
    print(f"[+] Processed with Pillow+NumPy -> {out_path} ({prepped.width}x{prepped.height})")


def main():
    try:
        import cv2
        import rembg
        prep_with_rembg(INP, OUT)
    except (ImportError, Exception) as e:
        print(f"[*] Standard pipeline (rembg/opencv: {e}) -> using high-quality Pillow+NumPy pipeline...")
        prep_with_pillow(INP, OUT)


if __name__ == "__main__":
    main()
