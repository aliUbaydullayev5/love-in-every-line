#!/usr/bin/env python3
"""Офлайн-рендер strokes.json в PNG (как в браузере) + метрики и сравнение.

  render_preview.py strokes.json out.png [--scale 1.5] [--ref --demo]
  render_preview.py --metrics strokes.json --demo
  render_preview.py --side-by-side a.json b.json out.png --demo
Источник: --demo (skimage astronaut) или --photo путь.
"""
import argparse, gzip, json, sys, time
from pathlib import Path
import cv2
import numpy as np

PAPER = np.array([0xFB, 0xF6, 0xEA], np.float32) / 255  # RGB
SS = 2  # supersampling


def hex_rgb(c):
    c = c.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return np.array([int(c[i : i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


def render(data, scale=1.0):
    """Возвращает RGB float32 [0..1] размером (H*scale, W*scale). Alpha-over в порядке order, затем multiply на бумагу."""
    k = scale * SS
    w, h = int(round(data["width"] * k)), int(round(data["height"] * k))
    A = np.zeros((h, w), np.float32)
    C = np.zeros((h, w, 3), np.float32)  # premultiplied
    for s in sorted(data["strokes"], key=lambda s: s["order"]):
        p = np.asarray(s["pts"], np.float32).reshape(-1, 2) * k
        pad = int(s["w"] * k) + 3
        x0, y0 = max(int(p[:, 0].min()) - pad, 0), max(int(p[:, 1].min()) - pad, 0)
        x1, y1 = min(int(p[:, 0].max()) + pad + 1, w), min(int(p[:, 1].max()) + pad + 1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        wf = s["w"] * k
        th = max(1, round(wf))
        cv2.polylines(m, [np.round(p - (x0, y0)).astype(np.int32).reshape(-1, 1, 2)], False, 255, th, cv2.LINE_AA)
        a = m.astype(np.float32) / 255 * (s["o"] * min(1.0, wf / th))
        a3 = a[..., None]
        C[y0:y1, x0:x1] = hex_rgb(s["color"]) * a3 + C[y0:y1, x0:x1] * (1 - a3)
        A[y0:y1, x0:x1] = a + A[y0:y1, x0:x1] * (1 - a)
    out = PAPER * (C + (1 - A[..., None]))  # canvas multiply paper
    out = cv2.resize(out, (int(round(data["width"] * scale)), int(round(data["height"] * scale))), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0, 1)


def gray(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def load_source(args, wh):
    if args.photo:
        bgr = cv2.imread(args.photo, cv2.IMREAD_COLOR)
        if bgr is None:
            sys.exit(f"не удалось прочитать {args.photo}")
    elif args.demo:
        from skimage import data
        bgr = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    else:
        sys.exit("нужен --demo или --photo")
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    return cv2.resize(g, wh, interpolation=cv2.INTER_AREA)


def ssim(a, b):
    c1, c2 = 0.01**2, 0.03**2
    f = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))).mean())


def metrics(data, path, src):
    t = time.time()
    r = gray(render(data))
    rt = time.time() - t
    ba, bb = (cv2.GaussianBlur(x, (0, 0), 4) for x in (r, src))
    raw = Path(path).read_bytes()
    return dict(
        strokes=len(data["strokes"]),
        json_MB=round(len(raw) / 1e6, 3),
        gzip_MB=round(len(gzip.compress(raw, 6)) / 1e6, 3),
        tone_pearson=round(float(np.corrcoef(ba.ravel(), bb.ravel())[0, 1]), 4),
        ssim=round(ssim(ba, bb), 4),
        mean_abs_tone_err=round(float(np.abs(ba - bb).mean()), 4),
        render_s=round(rt, 2),
    )


def label(img, text):
    img = (img * 255).round().astype(np.uint8)  # putText в cv2 5 не принимает float
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (255, 255, 255), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return img.astype(np.float32) / 255


def save(rgb, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--ref", action="store_true", help="добавить исходник слева (нужен --demo/--photo)")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--photo")
    a = ap.parse_args()
    load = lambda p: json.loads(Path(p).read_text())

    if a.metrics:
        d = load(a.files[0])
        src = load_source(a, (d["width"], d["height"]))
        for k, v in metrics(d, a.files[0], src).items():
            print(f"{k}: {v}")
    elif a.side_by_side:
        if len(a.files) != 3:
            sys.exit("нужно: a.json b.json out.png")
        da, db = load(a.files[0]), load(a.files[1])
        src = load_source(a, (da["width"], da["height"]))
        ra, rb = render(da, a.scale), render(db, a.scale)
        sg = cv2.resize(np.repeat(src[..., None], 3, 2), (ra.shape[1], ra.shape[0]), interpolation=cv2.INTER_AREA)
        panels = [label(x, t) for x, t in ((sg, "source (gray)"), (ra, Path(a.files[0]).name), (rb, Path(a.files[1]).name))]
        save(np.concatenate(panels, 1), a.files[2])
    else:
        if len(a.files) != 2:
            sys.exit("нужно: strokes.json out.png")
        d = load(a.files[0])
        img = render(d, a.scale)
        if a.ref:
            src = load_source(a, (d["width"], d["height"]))
            sg = cv2.resize(np.repeat(src[..., None], 3, 2), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            img = np.concatenate([sg, img], 1)
        save(img, a.files[1])


if __name__ == "__main__":
    main()
