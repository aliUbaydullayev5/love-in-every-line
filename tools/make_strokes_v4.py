#!/usr/bin/env python3
"""Экспериментальные профили A/B/C одного генератора (поверх make_strokes.py).

  tools/.venv/bin/python tools/make_strokes_v4.py assets/girlfriend.jpg --profile B --out tools/out/variants/B.json

Отличия от v3:
  * Регионы — мягкие веса (кожа / ткань / волосы), без булевых зон и прямоугольных швов.
  * Платок = ткань: направления по складкам (средний масштаб), без «волосяной» текстуры.
    Волосы включаются только вручную: красная краска в assets/<имя>.regions.png (синяя = ткань).
  * Кожа: «лепка» объёма — тон относительно медианы кожи, а не абсолютная яркость (лицо не бледное).
  * Цель тона = смесь тона и pencil-sketch (dodge) — линии и мягкая тень в одном слое.
  * Увеличенная зона глаз под оправу очков; усиленный локальный контраст в деталях лица.
  * Штрихи: разброс угла/длины/нажима, нажим по длине (B, C), мягкий базовый слой (B).
"""
import argparse, json, math, sys, time
from pathlib import Path
import cv2
import numpy as np
from make_strokes import W, read, to_w, landmarks, crop_box, segment, face_maps, skeleton_paths, smooth_path, simplify, pack

DMAX = 0.74
HAND = math.radians(-38)

# pass: (cat, угол, шаг, ширина, alpha, порог дефицита, мин. D, макс. длина, область)
PROFILES = {
    "D2": dict(  # D + возвращённая глубина теней (контраст B), остальное как в D
        title="Universal Tonal v2", gain=2.8, base=0.17, lam=0.30, fab=0.78, fade=0.4, cap=0.9, k=0.8, pressure=True, cont=1.0, fab_cont=0.4, new=True,
        mod=dict(skin=(1.15, 0.85, 0.95), fabric=(1.1, 0.95, 1.0), detail=(0.6, 1.1, 0.75)),
        passes=[
            ("hatch", 0, 5.0, 1.7, 0.07, 0.03, 0.05, 130, "all"),
            ("hatch", 0, 3.8, 1.2, 0.10, 0.03, 0.05, 70, "skin"),
            ("hatch", 0, 1.9, 0.5, 0.32, 0.03, 0.05, 20, "face"),
            ("shadow", 36, 1.7, 0.6, 0.5, 0.04, 0.2, 14, "face"),
            ("hatch", 0, 3.6, 0.6, 0.26, 0.03, 0.06, 100, "all"),
            ("hatch", 12, 3.2, 0.7, 0.32, 0.04, 0.15, 90, "all"),
            ("shadow", -7, 2.9, 0.8, 0.42, 0.05, 0.27, 70, "all"),
            ("cross", 55, 3.2, 0.8, 0.36, 0.05, 0.4, 50, "all"),
            ("shadow", 22, 2.4, 0.9, 0.52, 0.06, 0.55, 40, "all"),
            ("shadow", 18, 1.5, 0.55, 0.65, 0.05, 0.5, 10, "face"),
        ]),
    "D": dict(  # универсальный: baseline B + естественность, глаза, ткань, полутона (без ручных масок)
        title="Universal Tonal", gain=2.8, base=0.17, lam=0.28, fab=0.62, fade=0.35, pressure=True, cont=1.0, fab_cont=0.4, new=True,
        mod=dict(skin=(1.15, 0.85, 0.95), fabric=(1.2, 0.9, 0.95), detail=(0.6, 1.1, 0.75)),
        passes=[
            ("hatch", 0, 5.0, 1.7, 0.07, 0.03, 0.05, 130, "all"),  # мягкая растушёвка
            ("hatch", 0, 3.8, 1.2, 0.10, 0.03, 0.05, 70, "skin"),  # широкие полутона кожи
            ("hatch", 0, 1.9, 0.5, 0.32, 0.03, 0.05, 20, "face"),
            ("shadow", 36, 1.7, 0.6, 0.48, 0.04, 0.2, 14, "face"),
            ("hatch", 0, 3.7, 0.6, 0.24, 0.03, 0.06, 100, "all"),
            ("hatch", 12, 3.3, 0.7, 0.30, 0.04, 0.15, 90, "all"),
            ("shadow", -7, 3.0, 0.75, 0.36, 0.05, 0.28, 70, "all"),
            ("cross", 55, 3.4, 0.75, 0.32, 0.05, 0.42, 50, "all"),
            ("shadow", 20, 2.2, 0.8, 0.5, 0.06, 0.52, 36, "focus"),  # тёмные акценты только в зоне лица
            ("shadow", 18, 1.5, 0.55, 0.62, 0.05, 0.5, 10, "face"),
        ]),
    "A": dict(  # Fine Graphite: тонко, мягко, чисто
        title="Fine Graphite", gain=1.9, base=0.11, lam=0.20, fab=0.55, fade=0.45, pressure=False, cont=0.9, fab_cont=0.45,
        mod=dict(skin=(1.1, 0.85, 0.85), fabric=(1.15, 0.9, 0.9), detail=(0.6, 1.1, 0.75)),
        passes=[
            ("hatch", 0, 1.9, 0.45, 0.30, 0.03, 0.05, 20, "face"),
            ("shadow", 38, 1.7, 0.5, 0.45, 0.04, 0.2, 14, "face"),
            ("hatch", 0, 3.4, 0.5, 0.20, 0.025, 0.05, 90, "all"),
            ("hatch", 10, 3.0, 0.55, 0.26, 0.03, 0.14, 80, "all"),
            ("shadow", -8, 2.8, 0.6, 0.34, 0.04, 0.27, 60, "all"),
            ("cross", 52, 3.2, 0.6, 0.30, 0.045, 0.42, 45, "all"),
            ("shadow", 18, 1.5, 0.5, 0.6, 0.05, 0.5, 10, "face"),
        ]),
    "B": dict(  # Tonal Pencil: объём, слои, тёмные акценты
        title="Tonal Pencil", gain=2.7, base=0.15, lam=0.30, fab=0.8, fade=0.35, pressure=True, cont=1.0, fab_cont=0.6,
        mod=dict(skin=(1.15, 0.8, 0.9), fabric=(1.0, 1.0, 1.0), detail=(0.6, 1.15, 0.75)),
        passes=[
            ("hatch", 0, 5.0, 1.7, 0.075, 0.03, 0.05, 130, "all"),  # мягкий базовый слой («растушёвка»)
            ("hatch", 0, 1.9, 0.5, 0.34, 0.03, 0.05, 20, "face"),
            ("shadow", 36, 1.7, 0.6, 0.5, 0.04, 0.2, 14, "face"),
            ("hatch", 0, 3.6, 0.6, 0.26, 0.03, 0.06, 100, "all"),
            ("hatch", 12, 3.2, 0.7, 0.32, 0.04, 0.15, 90, "all"),
            ("shadow", -7, 2.9, 0.8, 0.42, 0.05, 0.27, 70, "all"),
            ("cross", 55, 3.2, 0.8, 0.38, 0.05, 0.4, 50, "all"),
            ("shadow", 22, 2.4, 0.9, 0.55, 0.06, 0.55, 40, "all"),
            ("shadow", 18, 1.5, 0.55, 0.65, 0.05, 0.5, 10, "face"),
        ]),
    "C": dict(  # Artistic Portrait: лицо главное, одежда лёгкая
        title="Artistic Portrait", gain=2.4, base=0.14, lam=0.26, fab=0.5, fade=0.3, pressure=True, cont=1.0, fab_cont=0.35,
        mod=dict(skin=(1.1, 0.85, 0.9), fabric=(1.5, 0.8, 0.9), detail=(0.55, 1.2, 0.7)),
        passes=[
            ("hatch", 0, 1.8, 0.5, 0.34, 0.03, 0.05, 22, "face"),
            ("shadow", 36, 1.6, 0.6, 0.52, 0.04, 0.2, 16, "face"),
            ("hatch", 0, 3.8, 0.55, 0.22, 0.03, 0.06, 110, "all"),
            ("hatch", 9, 3.4, 0.65, 0.3, 0.04, 0.16, 90, "all"),
            ("shadow", -6, 3.0, 0.75, 0.4, 0.05, 0.3, 65, "all"),
            ("cross", 50, 3.6, 0.75, 0.34, 0.05, 0.45, 45, "all"),
            ("shadow", 18, 1.4, 0.55, 0.68, 0.05, 0.5, 10, "face"),
        ]),
}


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--profile", choices=list(PROFILES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-maps", help="сохранить карты (цель тона, направления, регионы) в .npz и выйти")
    ap.add_argument("--regions", help="PNG-маска: красный = волосы, синий = ткань (по умолчанию assets/<имя>.regions.png, если есть)")
    a = ap.parse_args()
    pf = PROFILES[a.profile]

    orig = read(a.photo, False)
    img = to_w(orig)
    P = landmarks(img)
    if P is None and not pf.get("new"):
        sys.exit("Ошибка: лицо не найдено.")
    if P is not None:
        x0, y0, x1, y1 = crop_box(P, img.shape, "bust")
        s = orig.shape[1] / W
        sl = (slice(int(y0 * s), int(y1 * s)), slice(int(x0 * s), int(x1 * s)))
        img = to_w(orig[sl])
        P = landmarks(img)
    else:
        print("лицо не найдено — режим без детализации лица (общий портрет/сюжет)")
        sl = (slice(None), slice(None))
    has_face = P is not None
    H = img.shape[0]
    gray8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(Path(a.out).with_name(Path(a.out).stem + "_crop.png")), img)  # кадр для сравнения (вне public/)

    subj, how = segment(img, P)
    if subj is None:
        if not pf.get("new"):
            sys.exit("Ошибка: не удалось выделить человека.")
        yy0, xx0 = np.mgrid[0:H, 0:W].astype(np.float32)  # fallback: мягкая овальная виньетка
        subj = np.clip(1.6 - np.sqrt(((xx0 - W / 2) / (W * 0.62)) ** 2 + ((yy0 - H * 0.5) / (H * 0.62)) ** 2), 0, 1) ** 0.7
    sub = subj > 0.5
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # ---------- регионы: мягкие веса ----------
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    if has_face:
        fm = face_maps(P, H, eye_grow=4)  # зона глаз с запасом под оправу очков
        detail, oval, eyes, brows, lips, fw = (fm[k] for k in ("detail", "oval", "eyes", "brows", "lips", "fw"))
        cx, cy = float(P[:, 0].mean()), float(P[:, 1].mean())
        chin, fh = float(P[:, 1].max()), float(np.ptp(P[:, 1]))
        core = (cv2.erode(oval.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ((eyes + brows + lips) < 0.5)
        cr, cb = np.median(ycc[..., 1][core]), np.median(ycc[..., 2][core])
        chroma = abs(cr - 128) + abs(cb - 128)
        skin_col = (np.abs(ycc[..., 1] - cr) < 12) & (np.abs(ycc[..., 2] - cb) < 13)
        if pf.get("new") and chroma < 8:  # ч/б или почти без цвета: по цвету кожу не ищем, только овал
            skin_col = np.zeros((H, W), bool)
            print("снимок почти без цвета — кожа определяется только по овалу лица")
        near_face = cv2.dilate(oval.astype(np.uint8), np.ones((int(fw * 0.25) | 1,) * 2, np.uint8)) > 0
    else:
        z = np.zeros((H, W), np.float32)
        detail = oval = eyes = brows = lips = z
        fw, cx, cy, chin, fh = W * 0.3, W / 2, H * 0.35, H * 0.65, H * 0.3
        core = np.zeros((H, W), bool)
        skin_col = np.zeros((H, W), bool)
        near_face = np.zeros((H, W), bool)
    Ws = cv2.GaussianBlur(np.maximum(oval, (skin_col & near_face & sub).astype(np.float32)), (0, 0), max(2.0, fw * 0.025))
    Wh = np.zeros((H, W), np.float32)
    rp = Path(a.regions) if a.regions else Path(a.photo).with_suffix(".regions.png")
    if rp.is_file() and not pf.get("new"):  # ручная маска: красный = волосы, синий = ткань
        rm = to_w(cv2.imdecode(np.fromfile(str(rp), np.uint8), cv2.IMREAD_COLOR)[sl]) if True else None
        Wh = cv2.GaussianBlur(((rm[..., 2] > 180) & (rm[..., 1] < 90) & (rm[..., 0] < 90)).astype(np.float32), (0, 0), 3)
        print(f"ручная маска регионов: {rp} (волосы {Wh.mean() * 100:.1f}% кадра)")
    Ws = np.clip(Ws, 0, 1) * (1 - Wh)
    Wf = np.clip(1 - Ws - Wh, 0, 1)
    subs = np.clip(subj * 1.1, 0, 1)
    fade = np.clip(1.0 - (yy - (chin + fh * 0.5)) / max(H - (chin + fh * 0.5), 1), pf["fade"], 1.0)

    # ---------- целевая темнота D ----------
    sm = cv2.bilateralFilter(gray8, 7, 28, 5)
    sm_f = cv2.bilateralFilter(cv2.bilateralFilter(gray8, 9, 40, 7), 9, 40, 7)  # ткань: меньше шума, складки сохраняются
    tone = cv2.GaussianBlur(cv2.createCLAHE(1.5, (6, 6)).apply(sm).astype(np.float32) / 255, (0, 0), 0.9)
    tone_f = cv2.GaussianBlur(cv2.createCLAHE(1.8, (6, 6)).apply(sm_f).astype(np.float32) / 255, (0, 0), 0.9)
    tone_s = cv2.GaussianBlur(sm.astype(np.float32) / 255, (0, 0), 2.2)
    tone_d = cv2.GaussianBlur(cv2.createCLAHE(3.0, (4, 4)).apply(sm).astype(np.float32) / 255, (0, 0), 0.7)
    med = float(np.median(tone_s[core])) if core.any() else 0.5
    D_skin = np.clip(pf["base"] + pf["gain"] * (med - tone_s), 0, 1)  # лепка объёма: тень = темнее медианы кожи
    D_fab = np.clip((1 - tone_f) ** 1.15, 0, 1) * pf["fab"] * fade
    if pf.get("new"):
        hp = tone - cv2.GaussianBlur(tone, (0, 0), 3.0)  # тонкие светлые полоски (<0) и тёмные складки (>0)
        low = cv2.GaussianBlur(tone_f, (0, 0), 2.0)
        cap = pf.get("cap", 0.55)
        D_fab = cap * np.tanh(np.clip((1 - low) ** 1.15, 0, 1) * pf["fab"] / cap) * fade  # мягкий предел: нет «чёрных плит»
        D_fab = np.clip(D_fab - 2.2 * hp * pf["fab"], 0, 1)  # полоски ткани остаются бумагой, складки темнее
    dodge = np.clip(gray8.astype(np.float32) * 255 / (255 - cv2.GaussianBlur(255 - gray8.astype(np.float32), (0, 0), 6) + 1), 0, 255) / 255
    D_sketch = np.clip((1 - dodge) * 1.3 - 0.04, 0, 1)  # pencil-sketch: линии формы + мягкая тень
    D = Ws * D_skin + Wf * D_fab + Wh * np.clip((1 - tone) ** 1.2, 0, 1) * 0.8
    D = (1 - pf["lam"]) * D + pf["lam"] * D_sketch * (0.5 + 0.5 * (1 - Ws))
    D_det = np.clip((1 - tone_d) ** 1.25, 0, 1)
    D = D * (1 - detail) + np.maximum(D, D_det * 0.85) * detail  # глаза/оправа/губы: усиленный контраст
    if pf.get("new"):
        kk = pf.get("k", 1.25)
        D = (1 - np.exp(-kk * D)) / (1 - np.exp(-kk))  # плечо кривой: полутона лица заметнее, чёрные сжаты
    D = np.where(D < 0.05, 0, D) * DMAX * subs

    # ---------- направления ----------
    g = cv2.GaussianBlur(gray8.astype(np.float32), (0, 0), 1.2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    ang2, cohs = [], []
    for sig in (2.5, 7, 28):
        sxx, syy, sxy = (cv2.GaussianBlur(v, (0, 0), sig) for v in (gx * gx, gy * gy, gx * gy))
        t = 0.5 * np.arctan2(2 * sxy, sxx - syy) + math.pi / 2
        coh = np.sqrt((sxx - syy) ** 2 + 4 * sxy**2) / (sxx + syy + 1e-3)
        ang2.append(np.stack([np.cos(2 * t), np.sin(2 * t)]))
        cohs.append(0.15 + np.clip(coh, 0, 1))
    if pf.get("new"):  # «волосоподобная» текстура: мелкая, когерентная, с высокой энергией градиента (волосы ИЛИ ткань) — без масок
        E = cv2.GaussianBlur(np.hypot(gx, gy) ** 2, (0, 0), 5)
        En = np.clip(E / (np.percentile(E[sub], 90) + 1e-6), 0, 1) if sub.any() else E
        st = np.clip((En - 0.25) / 0.45, 0, 1) * np.clip((cohs[0] - 0.15) / 0.5, 0, 1)
        head = np.exp(-(((xx - cx) / (fw * 1.7)) ** 2 + ((yy - cy + fh * 0.2) / (fh * 1.8)) ** 2))
        Wh = np.clip(st * head * (1 - Ws), 0, 1)
        Wf = np.clip(Wf - Wh, 0, 1)
    wt = np.zeros((3, H, W), np.float32) + 0.02
    for mk, w3 in ((Ws, (0.05, 0.25, 0.7)), (Wf, (0.35, 0.5, 0.15) if pf.get("new") else (0.15, 0.6, 0.25)), (Wh, (0.7, 0.3, 0.0))):
        for k in range(3):
            wt[k] += mk * w3[k]
    wt[0] += detail * 0.8
    V = sum(wt[k] * cohs[k] * ang2[k] for k in range(3))
    hand = np.array([math.cos(2 * HAND), math.sin(2 * HAND)], np.float32).reshape(2, 1, 1) * (Ws * 0.5 * (1 - detail))
    V = V / (np.linalg.norm(V, axis=0) + 1e-6) + hand
    if pf.get("new") and has_face:
        # оси глаз и бровей из landmarks — только как ориентир направления штрихов (линии берутся из фото)
        from mediapipe.tasks.python import vision
        Cn = vision.FaceLandmarksConnections
        for conns, grow, wgt in ((Cn.FACE_LANDMARKS_LEFT_EYE, 2, 1.0), (Cn.FACE_LANDMARKS_RIGHT_EYE, 2, 1.0), (Cn.FACE_LANDMARKS_LEFT_EYEBROW, 2, 0.8), (Cn.FACE_LANDMARKS_RIGHT_EYEBROW, 2, 0.8)):
            idx = sorted({i for c in conns for i in (c.start, c.end)})
            vx, vy, _, _ = cv2.fitLine(P[idx].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            em = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(em, cv2.convexHull(P[idx].astype(np.int32)), 1)
            em = cv2.GaussianBlur(cv2.dilate(em, np.ones((grow * 2 + 1,) * 2, np.uint8)).astype(np.float32), (0, 0), 2.0) * wgt
            ax = np.array([(vx * vx - vy * vy), 2 * vx * vy], np.float32).reshape(2, 1, 1)  # cos2a, sin2a оси
            V = V * (1 - em) + ax * em * 1.6
    theta = 0.5 * np.arctan2(V[1], V[0])
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.0)
    mag_n = mag / (np.percentile(mag, 99) + 1e-6)

    if a.dump_maps:
        np.savez_compressed(a.dump_maps, D=D, theta=theta, detail=detail, subj=subj, Ws=Ws, Wf=Wf, img=img, tone=tone, gray=gray8, DMAX=DMAX)
        print(f"карты сохранены: {a.dump_maps}")
        return
    rng = np.random.default_rng({"A": 21, "B": 22, "C": 23, "D": 24, "D2": 25}[a.profile])
    strokes, order = [], 0

    def add(cat, pts, w, o, color, pressure=False):
        nonlocal order
        pts = np.asarray(pts, np.float32)
        if len(pts) < 2:
            return
        parts = [(pts, w, o)]
        if pressure and len(pts) >= 10:  # нажим по длине: лёгкий старт, плотная середина, лёгкий конец
            n = len(pts)
            i1, i2 = max(n // 4, 2), n - max(n // 4, 2)
            parts = [(pts[: i1 + 1], w * 0.8, o * 0.55), (pts[i1 : i2 + 1], w, o), (pts[i2:], w * 0.8, o * 0.55)]
        for pp, ww, oo in parts:
            if len(pp) < 2:
                continue
            order += 1
            strokes.append(dict(id=len(strokes), cat=cat, order=order, w=round(float(ww), 2), o=round(float(oo), 3), color=color, dur=1.0, pts=pack(pp)))

    def strength(p):
        ix = np.clip(np.array(p)[:, 0].astype(int), 0, W - 1)
        iy = np.clip(np.array(p)[:, 1].astype(int), 0, H - 1)
        return float(np.clip(mag_n[iy, ix].mean(), 0, 1))

    def mid(p):
        q = p[len(p) // 2]
        return min(int(q[1]), H - 1), min(int(q[0]), W - 1)

    def emit_contour(path, cat, w, base, spread, maxseg):
        pts = smooth_path(path, 2.0)
        st = strength(path)
        i, n = 0, len(pts)
        while i < n - 1:
            seg = int(rng.integers(max(6, maxseg // 3), maxseg))
            chunk = simplify(pts[i : i + seg + 1], 0.4)
            if len(chunk) >= 2:
                o = base * (0.4 + 0.9 * st) * (1 - spread * rng.random())
                add(cat, chunk, w * rng.uniform(0.8, 1.2), min(o, 0.75), "#2a2a2e")
            i += seg + int(rng.integers(0, 3))

    def edge_paths(sigma, lo, hi, minlen):
        e = cv2.Canny(cv2.GaussianBlur(cv2.createCLAHE(1.5, (6, 6)).apply(sm), (0, 0), sigma), lo, hi)
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return e, [p for p in skeleton_paths(e) if len(p) >= minlen]

    medv = float(np.median(sm))
    sub_near = cv2.dilate(sub.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    # 1) основные формы (силуэт, крупные складки). В ткани — только заметные края, мягко.
    coarse_e, coarse = edge_paths(2.0, int(max(14, 0.30 * medv)), int(max(40, 0.80 * medv)), 36)
    coarse.sort(key=lambda p: -len(p))
    for p in coarse:
        iy, ix = mid(p)
        st = strength(p)
        if not sub_near[iy, ix] or st < 0.2:
            continue
        if Wf[iy, ix] > 0.5 and st < pf["fab_cont"]:
            continue
        emit_contour(p, "contour", 0.75, 0.30 * pf["cont"], 0.5, 60)
    n_form = len(strokes)

    # 2) ключевые контуры лица: глаза, оправа, брови, нос, губы
    near_coarse = cv2.dilate(coarse_e, np.ones((5, 5), np.uint8))
    _, fine = edge_paths(0.9, int(max(18, 0.4 * medv)), int(max(50, 1.0 * medv)), 8)
    fine = [p for p in fine if detail[mid(p)] > 0.3 and (detail[mid(p)] > 0.6 or not near_coarse[mid(p)])] if has_face else []
    fine.sort(key=lambda p: -detail[mid(p)])
    for p in fine:
        dt = detail[mid(p)]
        emit_contour(p, "feature", 0.55, (0.3 + 0.35 * dt) * pf["cont"], 0.35, 26)
    n_face_c = len(strokes) - n_form

    # 3) тональные проходы
    Cacc = np.zeros((H, W), np.float32)

    def deposit(pts, w, o):
        pi = np.round(pts).astype(np.int32)
        bx0, by0 = max(pi[:, 0].min() - 3, 0), max(pi[:, 1].min() - 3, 0)
        bx1, by1 = min(pi[:, 0].max() + 4, W), min(pi[:, 1].max() + 4, H)
        if bx1 <= bx0 or by1 <= by0:
            return
        m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        cv2.polylines(m, [(pi - (bx0, by0)).reshape(-1, 1, 2)], False, 255, max(1, round(w)), cv2.LINE_AA)
        roi = Cacc[by0:by1, bx0:bx1]
        roi[:] = 1 - (1 - roi) * (1 - (m.astype(np.float32) / 255) * (o * min(w, 1.0)))

    cell = 1.5
    mods = pf["mod"]
    if pf.get("new"):
        nrng = np.random.default_rng(99)
        ang_noise = cv2.GaussianBlur(nrng.standard_normal((H, W)).astype(np.float32), (0, 0), 22)
        ang_noise = ang_noise / (ang_noise.std() + 1e-6) * math.radians(9)  # плавный, пространственно когерентный разброс угла
        sp_noise = cv2.GaussianBlur(nrng.standard_normal((H, W)).astype(np.float32), (0, 0), 30)
        sp_noise = sp_noise / (sp_noise.std() + 1e-6)
    else:
        ang_noise = sp_noise = None

    def run_pass(cat, rot, sp, lw, op, thr, dmin, maxlen, area):
        t1 = time.time()
        gate = {"face": (detail > 0.35) & sub, "skin": (Ws > 0.4) & sub, "focus": (detail > 0.12) & sub}.get(area, sub)
        gate_f = gate.astype(np.float32)
        deficit = D - cv2.GaussianBlur(Cacc, (0, 0), 1.5)
        occ = np.zeros((int(H / cell) + 2, int(W / cell) + 2), np.uint8)
        step = max(1.2, sp * 0.7)
        ys, xs = np.mgrid[0:H:step, 0:W:step]
        seeds = np.stack([xs.ravel(), ys.ravel()], 1) + rng.uniform(-step * 0.45, step * 0.45, (xs.size, 2))
        ix_, iy_ = np.clip(seeds[:, 0].astype(int), 0, W - 1), np.clip(seeds[:, 1].astype(int), 0, H - 1)
        seeds = seeds[(gate_f[iy_, ix_] > 0.5) & (D[iy_, ix_] >= dmin * 0.8) & (deficit[iy_, ix_] >= thr * 0.7)]
        rng.shuffle(seeds)
        layer = []

        def trace(x, y, sgn, sd, mx, rot_r):
            pts, px, py, pa = [], x, y, None
            for _ in range(int(mx / 1.5)):
                iy, ix = int(min(max(py, 0), H - 1)), int(min(max(px, 0), W - 1))
                if not (0 <= px < W and 0 <= py < H) or deficit[iy, ix] < thr * 0.3 or D[iy, ix] < dmin * 0.7 or (area == "face" and gate_f[iy, ix] < 0.4):
                    break
                ang = theta[iy, ix] + rot_r + (ang_noise[iy, ix] if ang_noise is not None else 0.0)
                if pa is not None:
                    d = (ang - pa + math.pi / 2) % math.pi - math.pi / 2
                    if abs(d) > 0.55:
                        break
                    ang = pa + 0.65 * d
                pa = ang
                px, py = px + sgn * 1.5 * math.cos(ang), py + sgn * 1.5 * math.sin(ang)
                oy, ox = int(py / cell), int(px / cell)
                if not (0 <= oy < occ.shape[0] and 0 <= ox < occ.shape[1]):
                    break
                if len(pts) > 1 and occ[max(oy - sd, 0) : oy + sd + 1, max(ox - sd, 0) : ox + sd + 1].any():
                    break
                pts.append((px, py))
            return pts

        for sx, sy in seeds:
            iy, ix = min(int(sy), H - 1), min(int(sx), W - 1)
            dt, ws_, wf_ = float(detail[iy, ix]), float(Ws[iy, ix]), float(Wf[iy, ix])
            wsum = max(dt + ws_ * (1 - dt) + wf_ * (1 - dt), 1e-3)
            kd, ks, kf = dt / wsum, ws_ * (1 - dt) / wsum, wf_ * (1 - dt) / wsum
            spm = kd * mods["detail"][0] + ks * mods["skin"][0] + kf * mods["fabric"][0]
            om = kd * mods["detail"][1] + ks * mods["skin"][1] + kf * mods["fabric"][1]
            wm = kd * mods["detail"][2] + ks * mods["skin"][2] + kf * mods["fabric"][2]
            if sp_noise is not None:
                spm *= 1 + 0.22 * float(sp_noise[iy, ix])
            sd = max(1, round(sp * spm / cell * 0.75))
            oy, ox = int(sy / cell), int(sx / cell)
            if occ[max(oy - sd, 0) : oy + sd + 1, max(ox - sd, 0) : ox + sd + 1].any():
                continue
            rot_r = math.radians(rot + rng.normal(0, 4.5))
            mx = maxlen * (0.45 + 0.9 * rng.random()) * (1 - 0.3 * dt)
            f = trace(sx, sy, 1, sd, mx, rot_r)
            b = trace(sx, sy, -1, sd, mx, rot_r)
            pts = b[::-1] + [(sx, sy)] + f
            if len(pts) < 4:
                continue
            for px, py in pts:
                occ[int(py / cell), int(px / cell)] = 1
            pts = simplify(smooth_path(pts, 1.2), 0.3)
            if len(pts) < 2:
                continue
            if sp_noise is not None and len(pts) >= 4:  # лёгкое «дрожание руки» вдоль штриха
                pts = pts + smooth_path(nrng.normal(0, 0.45, (len(pts), 2)), 1.5) * (1 - 0.7 * dt)
            o = float(np.clip(op * om * (0.7 + 0.6 * D[iy, ix] / DMAX) * rng.uniform(0.75, 1.25), 0.03, 0.85))
            w = lw * wm * rng.uniform(0.7, 1.3)
            deposit(pts, w, o)
            layer.append((pts, w, o))
        layer.sort(key=lambda t: (int(t[0][0][1] // 40), t[0][0][0] if int(t[0][0][1] // 40) % 2 == 0 else -t[0][0][0]))
        for pts, w, o in layer:
            add(cat, pts, w, o, "#1d1d21" if cat == "shadow" else "#2d2d32", pressure=pf["pressure"] and len(pts) >= 10)
        print(f"  {area:4s} {cat:6s} {rot:+4.0f}° sp{sp}: {len(layer):5d} ({time.time() - t1:.1f}с)")

    n0 = len(strokes)
    for ps in pf["passes"]:
        run_pass(*ps)
    print(f"[{a.profile}] {pf['title']}: контуры {n_form}+{n_face_c}, тон {len(strokes) - n0}")

    if len(strokes) < 1500:
        sys.exit("Ошибка: слишком мало штрихов.")
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(dict(version=1, width=W, height=H, strokes=strokes), separators=(",", ":")))
    print(f"[{a.profile}] штрихов: {len(strokes)} | {dst.stat().st_size / 1e6:.2f} МБ | {time.time() - t0:.1f}с -> {dst}")


if __name__ == "__main__":
    main()
