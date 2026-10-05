## You can't run it on GitHub Pages or inside the tier list.
## Uploading it to GitHub only stores the file. 
## You have to run it on your own computer in a terminal or command prompt.

''' 
How to run it ;

1. First of all, you should install Python on your computer first ( there are tons of videos on YouTube about it, so do it
2. Now install the OpenCV library on your pc so you can open your terminal and type this ---- pip install pillow numpy opencv-python
3. Now you can use this to crop the images before uploading to the browser 

'''

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
WORK = 256          # size used for analysis (fast, and enough for framing)
FACE_WEIGHT = 8.0   # faces count much more than general saliency

_cascades = [
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
    cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
]


def find_faces(rgb):
    """Return face boxes (x, y, w, h) in the coordinates of `rgb`."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    min_side = max(24, min(gray.shape) // 15)
    boxes = []
    for c in _cascades:
        if c.empty():
            continue
        found = c.detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
        boxes.extend([tuple(map(int, b)) for b in found])
    return boxes


def saliency(rgb):
    """Spectral-residual saliency map, values 0..1."""
    small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (64, 64)).astype(np.float32)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2)
    sal = cv2.resize(sal, (rgb.shape[1], rgb.shape[0]))
    sal -= sal.min()
    return sal / sal.max() if sal.max() > 0 else sal


def best_window(weights):
    """Pick the square window position (left, top, side) with the most weight."""
    h, w = weights.shape
    side = min(w, h)
    if w == h:
        return 0, 0, side
    axis = 1 if w > h else 0                       # slide along the long side
    profile = weights.sum(axis=0 if w > h else 1)  # weight per column or row
    csum = np.concatenate([[0], np.cumsum(profile)])
    n = len(profile) - side + 1
    scores = csum[side:side + n] - csum[:n]
    # tiny pull toward the center so ties and empty maps give a centered crop
    centre = (len(profile) - side) / 2
    scores = scores - 1e-3 * np.abs(np.arange(n) - centre) * (scores.max() + 1e-9)
    pos = int(np.argmax(scores))
    return (pos, 0, side) if axis == 1 else (0, pos, side)


def smart_crop(im):
    """Return (cropped PIL image, description of what it focused on)."""
    rgb_full = np.array(im.convert("RGB"))
    k = WORK / max(rgb_full.shape[:2])
    rgb = cv2.resize(rgb_full, None, fx=min(1, k), fy=min(1, k), interpolation=cv2.INTER_AREA)

    weights = saliency(rgb)
    note = "main subject"
    faces = find_faces(rgb_full)
    if faces:
        s = rgb.shape[1] / rgb_full.shape[1]
        for x, y, fw, fh in faces:
            pad = 0.2
            x0, y0 = max(0, int((x - pad * fw) * s)), max(0, int((y - pad * fh) * s))
            x1, y1 = int((x + fw * (1 + pad)) * s), int((y + fh * (1 + pad)) * s)
            weights[y0:y1, x0:x1] += FACE_WEIGHT
        note = f"{len(faces)} face(s)"
    elif weights.max() < 0.05:
        note = "nothing stood out, centered"

    left, top, side = best_window(weights)
    scale = rgb_full.shape[1] / rgb.shape[1]
    box = [int(round(v * scale)) for v in (left, top, left + side, top + side)]
    box[2] = min(box[2], im.width)
    box[3] = min(box[3], im.height)
    return im.crop(tuple(box)), note


def process(path, out_dir, size):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        cropped, note = smart_crop(im)
        if size:
            cropped = cropped.resize((size, size), Image.LANCZOS)
        out = out_dir / (path.stem + ".png")
        cropped.save(out)
    return out, note


def main():
    ap = argparse.ArgumentParser(description="Smart square crop around faces and subjects.")
    ap.add_argument("input", help="folder with images")
    ap.add_argument("output", nargs="?", help="folder for cropped copies")
    ap.add_argument("--size", type=int, default=400, help="final size in px (0 = keep)")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_dir():
        raise SystemExit(f"Not a folder: {src}")
    dst = Path(args.output) if args.output else src / "smart_cropped"
    dst.mkdir(parents=True, pip install pillow numpy opencv-pythonexist_ok=True)

    files = [p for p in sorted(src.iterdir()) if p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit("No images found.")

    for p in files:
        try:
            out, note = process(p, dst, args.size)
            print(f"{p.name} -> {out.name}  (focused on: {note})")
        except Exception as e:
            print(f"skipped {p.name}: {e}")
    print(f"Done. Output in {dst}")


if __name__ == "__main__":
    main()
