#!/usr/bin/env python3
"""Фото -> public/strokes.json (v3: портретный кадр, сегментация, лицо в приоритете, разные штрихи для кожи/волос/одежды).

  npm run strokes -- photo.jpg                 # лицо и плечи (по умолчанию)
  npm run strokes -- photo.jpg --crop head     # только голова;  --crop none — без кадрирования
  npm run strokes -- --demo                    # тест: портрет NASA из scikit-image (public domain)

Конвейер:
  1. MediaPipe FaceLandmarker -> портретный кадр (4:5) вокруг лица.
  2. MediaPipe ImageSegmenter (или grabCut, или без сегментации) -> маска человека; фон остаётся чистой бумагой.
  3. Регионы: кожа / волосы / одежда / детали лица (глаза, брови, губы, нос). Landmarks — только маски и
     ориентиры плотности, линии лица берутся из краёв самого фото.
  4. Для каждого региона свои направления, плотность, нажим и порядок проходов; штрих ставится
     только там, где накопленная темнота (с учётом alpha-композитинга) меньше целевой.
Порядок рисования: контуры формы -> контуры лица -> глаза/нос/губы -> волосы -> кожа -> одежда -> финальные акценты.
"""
import argparse, json, math, os, sys, time, urllib.request
from pathlib import Path
import cv2
import numpy as np

W = 900  # рабочая ширина (координаты холста)
MODELS = Path(__file__).parent / "models"
URL = "https://storage.googleapis.com/mediapipe-models/"
FACE_MODEL = ("face_landmarker.task", URL + "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
SEG_MODEL = ("selfie_segmenter.tflite", URL + "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite")
DMAX = 0.72  # максимальная «воспринимаемая» темнота карандаша
HAND = math.radians(-38)  # естественный наклон руки при штриховке кожи


# ------------------------------------------------------------------ загрузка / детекторы
def model(spec):
    path = MODELS / spec[0]
    if not path.exists():
        if os.environ.get("LIE_OFFLINE"):  # сервер/Docker: никаких загрузок во время запроса
            raise FileNotFoundError(f"модель {spec[0]} отсутствует в {MODELS} (LIE_OFFLINE=1)")
        MODELS.mkdir(exist_ok=True)
        urllib.request.urlretrieve(spec[1], path)
    return str(path)


EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def read(path, demo):
    if demo:
        from skimage import data
        return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    p = Path(path)
    if not p.is_file():  # то же имя с другим расширением (girlfriend.jpeg/.png/.webp)
        alt = [p.with_suffix(e) for e in EXTS if p.with_suffix(e).is_file()]
        if not alt:
            sys.exit(f"Ошибка: файл '{path}' не найден.\nПоложи фото в {p.parent}/ под именем {p.name} (или .jpeg/.png/.webp) и повтори.")
        p = alt[0]
    if p.suffix.lower() not in EXTS:
        sys.exit(f"Ошибка: формат '{p.suffix}' не поддерживается. Нужен один из: {', '.join(EXTS)}.")
    if p.stat().st_size == 0:
        sys.exit(f"Ошибка: файл '{p}' пустой.")
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)  # учитывает EXIF-поворот телефона
    if img is None:
        sys.exit(f"Ошибка: '{p}' повреждён или не является изображением.")
    h, w = img.shape[:2]
    if min(h, w) < 400:
        sys.exit(f"Ошибка: разрешение {w}x{h} слишком мало (минимум 400px по меньшей стороне, лучше от 1000px).")
    if min(h, w) < 900:
        print(f"предупреждение: разрешение {w}x{h} невысокое — мелкие детали лица будут хуже (лучше от 1000px)")
    print(f"фото: {p} ({w}x{h})")
    return img


def to_w(img):
    s = W / img.shape[1]
    return cv2.resize(img, (W, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def landmarks(img):
    """478 точек лица в пикселях или None (нет пакета/модели/лица)."""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=model(FACE_MODEL)), num_faces=1, min_face_detection_confidence=0.3)
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with vision.FaceLandmarker.create_from_options(opts) as lm:
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception as e:
        print(f"landmarks недоступны ({type(e).__name__}: {e})")
        return None
    if not res.face_landmarks:
        print("лицо не найдено")
        return None
    h, w = img.shape[:2]
    return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], np.float32)


def crop_box(P, shape, mode):
    """Портретный кадр 4:5 вокруг лица. Возвращает (x0,y0,x1,y1) в пикселях изображения."""
    h, w = shape[:2]
    fw = float(np.ptp(P[:, 0]))
    cx, cy = float(P[:, 0].mean()), float(P[:, 1].mean())
    cw = fw * (2.1 if mode == "head" else 3.1)
    ch = cw * 1.25
    cw, ch = min(cw, w), min(ch, h)
    if cw / ch > 0.8:
        cw = ch * 0.8
    else:
        ch = cw / 0.8
    x0 = min(max(cx - cw / 2, 0), w - cw)
    y0 = min(max(cy - ch * (0.40 if mode == "head" else 0.36), 0), h - ch)
    return int(x0), int(y0), int(x0 + cw), int(y0 + ch)


def segment(img, P):
    """Маска человека 0..1 и название метода. Безопасный fallback: grabCut по лицу, затем без сегментации."""
    H = img.shape[0]
    m = None
    how = ""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        o = vision.ImageSegmenterOptions(base_options=mpp.BaseOptions(model_asset_path=model(SEG_MODEL)), output_confidence_masks=True, output_category_mask=False)
        with vision.ImageSegmenter.create_from_options(o) as s:
            r = s.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        raw = r.confidence_masks[0].numpy_view()
        m = cv2.resize(np.ascontiguousarray(raw.reshape(raw.shape[0], raw.shape[1])), (W, H), interpolation=cv2.INTER_LINEAR)
        how = "MediaPipe selfie segmenter"
    except Exception as e:
        print(f"сегментатор недоступен ({type(e).__name__}) — пробую grabCut")
    ok = lambda mm: mm is not None and 0.12 < (mm > 0.5).mean() < 0.95
    if not ok(m) and P is not None:
        fw = float(np.ptp(P[:, 0]))
        cx, cy = float(P[:, 0].mean()), float(P[:, 1].mean())
        small = cv2.resize(img, (W // 2, H // 2))
        gc = np.full(small.shape[:2], cv2.GC_PR_BGD, np.uint8)
        gc[: int(H * 0.02), :] = gc[:, :4] = gc[:, -4:] = cv2.GC_BGD
        x0, x1 = int((cx - fw * 1.7) / 2), int((cx + fw * 1.7) / 2)
        gc[max(int((cy - fw * 1.1) / 2), 0) :, max(x0, 0) : x1] = cv2.GC_PR_FGD
        gc[max(int((cy - fw * 0.5) / 2), 0) : int((cy + fw * 0.5) / 2), max(int((cx - fw * 0.5) / 2), 0) : int((cx + fw * 0.5) / 2)] = cv2.GC_FGD
        try:
            cv2.grabCut(small, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
            m = cv2.resize(((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32), (W, H))
            how = "grabCut"
        except cv2.error:
            m = None
    if not ok(m):
        print("сегментация неуверенная — фон не вычищается (fallback: овальная виньетка)")
        return None, "нет"
    m = (m > 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 2:  # оставляем крупные компоненты (человек), выбрасываем мусор
        keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.04 * W * H]
        m = np.isin(lab, keep).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 3), how


def face_maps(P, H, eye_grow=2):
    """Маски по landmarks: detail (мягкая карта детализации), oval, eyes, brows, lips."""
    from mediapipe.tasks.python import vision
    C = vision.FaceLandmarksConnections

    def mask(conns, grow=0):
        idx = sorted({i for c in conns for i in (c.start, c.end)})
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(P[idx].astype(np.int32)), 1)
        if grow:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2))
        return m.astype(np.float32)

    fw = float(np.ptp(P[:, 0]))
    g = max(2, int(fw * 0.04))
    oval = mask(C.FACE_LANDMARKS_FACE_OVAL, 0)
    eyes = np.maximum(mask(C.FACE_LANDMARKS_LEFT_EYE, g * eye_grow), mask(C.FACE_LANDMARKS_RIGHT_EYE, g * eye_grow))
    brows = np.maximum(mask(C.FACE_LANDMARKS_LEFT_EYEBROW, g), mask(C.FACE_LANDMARKS_RIGHT_EYEBROW, g))
    lips = mask(C.FACE_LANDMARKS_LIPS, g)
    nose = mask(C.FACE_LANDMARKS_NOSE, g // 2)
    detail = np.maximum.reduce([oval * 0.35, nose * 0.7, lips * 0.95, brows * 0.9, eyes * 1.0])
    return dict(detail=cv2.GaussianBlur(detail, (0, 0), max(2.0, fw * 0.025)), oval=oval, eyes=eyes, brows=brows, lips=lips, fw=fw)


# ------------------------------------------------------------------ контуры / утилиты
def skeleton_paths(edges):
    sk = cv2.ximgproc.thinning(edges)
    ys, xs = np.nonzero(sk)
    pix = set(zip(xs.tolist(), ys.tolist()))
    nb = lambda p: [(p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy) and (p[0] + dx, p[1] + dy) in pix]
    deg = {p: len(nb(p)) for p in pix}
    seen, paths = set(), []

    def walk(start):
        path, cur = [start], start
        seen.add(start)
        while True:
            nxt = [q for q in nb(cur) if q not in seen]
            if not nxt:
                return path
            nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            cur = nxt[0]
            seen.add(cur)
            path.append(cur)

    for p in sorted(pix, key=lambda p: (deg[p] != 1, p)):
        if p not in seen:
            paths.append(walk(p))
    return paths


def smooth_path(pts, sigma=2.0):
    a = np.array(pts, np.float32)
    if len(a) < 7:
        return a
    k = int(sigma * 3) * 2 + 1
    pad = np.pad(a, ((k, k), (0, 0)), mode="edge")
    return cv2.GaussianBlur(pad.reshape(-1, 1, 2), (1, k), sigma).reshape(-1, 2)[k:-k]


def simplify(pts, eps):
    return cv2.approxPolyDP(np.asarray(pts, np.float32).reshape(-1, 1, 2), eps, False).reshape(-1, 2)


def pack(pts):
    out = []
    for v in np.round(np.asarray(pts, np.float64).reshape(-1) * 2).astype(int).tolist():
        out.append(v // 2 if v % 2 == 0 else v / 2)
    return out


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("photo", nargs="?")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="public/strokes.json")
    ap.add_argument("--crop", choices=["bust", "head", "none"], default="bust", help="портретный кадр: лицо+плечи / голова / без")
    ap.add_argument("--no-segment", action="store_true", help="не вычищать фон")
    ap.add_argument("--allow-no-face", action="store_true", help="продолжить, даже если лицо не найдено (хуже качество)")
    ap.add_argument("--density", type=float, default=1.0, help="<1 меньше штрихов, >1 больше")
    a = ap.parse_args()
    if not a.photo and not a.demo:
        ap.error("укажите фото или --demo")

    orig = read(a.photo, a.demo)
    img = to_w(orig)
    P = landmarks(img)
    if P is not None and a.crop != "none":
        x0, y0, x1, y1 = crop_box(P, img.shape, a.crop)
        s = orig.shape[1] / W
        img = to_w(orig[int(y0 * s) : int(y1 * s), int(x0 * s) : int(x1 * s)])
        P = landmarks(img)
        print(f"кадр: {a.crop}, {x1 - x0}x{y1 - y0}px из {W}x{orig.shape[0] * W // orig.shape[1]}")
    H = img.shape[0]
    gray8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    has_face = P is not None
    print("лицо:", f"найдено, landmarks {len(P)}" if has_face else "НЕ найдено — общий режим без детализации лица")
    if not has_face and not a.allow_no_face and not a.demo:
        sys.exit("Ошибка: лицо не найдено (или не установлен mediapipe). Нужен портрет анфас/в 3/4, лицо крупно, хорошее освещение.\n"
                 "Проверь: npm run setup:py. Если уверен в фото — добавь флаг --allow-no-face (качество будет ниже).")

    # --- маска человека ---
    subj, how = (None, "выкл") if a.no_segment else segment(img, P)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if subj is None:  # fallback: мягкая овальная виньетка вокруг центра
        subj = np.clip(1.6 - np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.5) / (H * 0.62)) ** 2), 0, 1) ** 0.7
    print(f"сегментация: {how}, человек = {(subj > 0.5).mean() * 100:.0f}% кадра")
    sub = subj > 0.5

    # --- регионы ---
    zeros = np.zeros((H, W), np.float32)
    if has_face:
        fm = face_maps(P, H)
        detail, oval, eyes, brows, lips, fw = (fm[k] for k in ("detail", "oval", "eyes", "brows", "lips", "fw"))
        cx, cy = float(P[:, 0].mean()), float(P[:, 1].mean())
        chin = float(P[:, 1].max())
        fh = float(np.ptp(P[:, 1]))
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        core = cv2.erode(oval.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        core &= (eyes + brows + lips) < 0.5
        cr, cb = np.median(ycc[..., 1][core]), np.median(ycc[..., 2][core])
        skin_col = (np.abs(ycc[..., 1] - cr) < 13) & (np.abs(ycc[..., 2] - cb) < 14)
        neck = skin_col & sub & (yy > cy) & (np.abs(xx - cx) < fw * 0.95)
        skin = ((cv2.dilate(oval, np.ones((5, 5), np.uint8)) > 0) | neck) & sub
        head_zone = (np.abs(xx - cx) < fw * 1.45) & (yy < chin + fh * 0.5) & (((xx - cx) / (fw * 1.35)) ** 2 + ((yy - cy + fh * 0.1) / (fh * 1.55)) ** 2 < 1.6)
        hair = sub & ~skin & head_zone & (yy < chin - fh * 0.1)  # длинные волосы ниже подбородка попадут в «одежду» (ограничение)
        cloth = sub & ~skin & ~hair
        fade = np.clip(1.0 - (yy - (chin + fh * 0.55)) / max(H - (chin + fh * 0.55), 1), 0.4, 1.0)  # одежда растворяется книзу
    else:
        detail = oval = eyes = brows = lips = zeros
        hair, skin, cloth = sub, np.zeros((H, W), bool), np.zeros((H, W), bool)
        fade, fw, cy, chin, fh = np.ones((H, W), np.float32), W * 0.3, H * 0.4, H * 0.7, H * 0.3

    # --- тон ---
    sm = cv2.bilateralFilter(gray8, 7, 28, 5)
    eq = cv2.createCLAHE(1.5, (6, 6)).apply(sm)
    eq_d = cv2.createCLAHE(3.0, (4, 4)).apply(sm)  # усиленный локальный контраст для глаз/губ
    tone = cv2.GaussianBlur(eq.astype(np.float32) / 255, (0, 0), 0.9)
    tone_d = cv2.GaussianBlur(eq_d.astype(np.float32) / 255, (0, 0), 0.7)
    D0 = np.clip((1 - tone) ** 1.15, 0, 1)
    Dd = np.clip((1 - tone_d) ** 1.25, 0, 1)
    D = D0 * (1 - detail) + Dd * detail
    D = np.where(D < 0.06, 0, D)
    scale = np.where(skin, 0.95, np.where(hair, 0.85, np.where(cloth, 0.8 * fade, 0.5))).astype(np.float32)
    scale = cv2.GaussianBlur(scale, (0, 0), 4)
    D = D * scale * DMAX * np.clip(subj * 1.1, 0, 1)
    D = np.where(detail > 0.5, np.maximum(D, Dd * 0.78 * DMAX * np.clip(subj, 0, 1)), D)  # глаза/губы не «съедаются» масштабом

    # --- поле направлений: 3 масштаба, разные веса по регионам ---
    g = cv2.GaussianBlur(gray8.astype(np.float32), (0, 0), 1.2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    ang2, cohs = [], []
    for sig in (2.5, 7, 28):
        sxx, syy, sxy = (cv2.GaussianBlur(v, (0, 0), sig) for v in (gx * gx, gy * gy, gx * gy))
        t = 0.5 * np.arctan2(2 * sxy, sxx - syy) + math.pi / 2
        coh = np.sqrt((sxx - syy) ** 2 + 4 * sxy**2) / (sxx + syy + 1e-3)
        ang2.append(np.stack([np.cos(2 * t), np.sin(2 * t)]))
        cohs.append(0.15 + np.clip(coh, 0, 1))
    wt = np.zeros((3, H, W), np.float32)
    for mk, w3 in ((skin, (0.05, 0.25, 0.7)), (hair, (0.7, 0.3, 0.0)), (cloth, (0.15, 0.6, 0.25))):
        for k in range(3):
            wt[k] += mk * w3[k]
    wt[0] += detail * 0.8
    wt += 0.02
    V = sum(wt[k] * cohs[k] * ang2[k] for k in range(3))
    hand = np.array([math.cos(2 * HAND), math.sin(2 * HAND)], np.float32).reshape(2, 1, 1) * (skin * 0.55 * (1 - detail)).astype(np.float32)
    V = V / (np.linalg.norm(V, axis=0) + 1e-6) + hand  # на коже примешиваем естественный наклон руки
    theta = 0.5 * np.arctan2(V[1], V[0])

    mag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.0)
    mag_n = mag / (np.percentile(mag, 99) + 1e-6)

    rng = np.random.default_rng(11)
    strokes, order = [], 0

    def add(cat, pts, w, o, color):
        nonlocal order
        if len(pts) < 2:
            return
        order += 1
        strokes.append(dict(id=len(strokes), cat=cat, order=order, w=round(float(w), 2), o=round(float(o), 3), color=color, dur=1.0, pts=pack(pts)))

    def strength(p):
        ix = np.clip(np.array(p)[:, 0].astype(int), 0, W - 1)
        iy = np.clip(np.array(p)[:, 1].astype(int), 0, H - 1)
        return float(np.clip(mag_n[iy, ix].mean(), 0, 1))

    def emit_contour(path, cat, w, base, spread, maxseg):
        pts = smooth_path(path, 2.0)
        s = strength(path)
        i, n = 0, len(pts)
        while i < n - 1:
            seg = int(rng.integers(max(6, maxseg // 3), maxseg))
            chunk = simplify(pts[i : i + seg + 1], 0.4)
            if len(chunk) >= 2:
                o = base * (0.4 + 0.9 * s) * (1 - spread * rng.random())
                add(cat, chunk, w * (0.8 + 0.4 * rng.random()), min(o, 0.7), "#2a2a2e")
            i += seg + int(rng.integers(0, 3))

    def edge_paths(sigma, lo, hi, minlen):
        e = cv2.Canny(cv2.GaussianBlur(eq, (0, 0), sigma), lo, hi)
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return e, [p for p in skeleton_paths(e) if len(p) >= minlen]

    def mid(p):
        q = p[len(p) // 2]
        return min(int(q[1]), H - 1), min(int(q[0]), W - 1)

    med = float(np.median(eq))
    sub_near = cv2.dilate(sub.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    # 1) основные формы: силуэт и крупные контуры человека (только внутри/у границы человека)
    coarse_e, coarse = edge_paths(2.2, int(max(14, 0.30 * med)), int(max(40, 0.80 * med)), 40)
    coarse.sort(key=lambda p: -len(p))
    for p in coarse:
        iy, ix = mid(p)
        if not sub_near[iy, ix] or strength(p) < 0.22 or (cloth[iy, ix] and strength(p) < 0.4):
            continue
        emit_contour(p, "contour", 0.75, 0.30, 0.5, 60)
    n_form = len(strokes)

    # 2) ключевые контуры лица: тонкие края внутри зон детализации (глаза, брови, нос, губы, овал)
    if has_face:
        near_coarse = cv2.dilate(coarse_e, np.ones((5, 5), np.uint8))
        _, fine = edge_paths(0.9, int(max(18, 0.4 * med)), int(max(50, 1.0 * med)), 8)
        fine = [p for p in fine if detail[mid(p)] > 0.3 and (detail[mid(p)] > 0.6 or not near_coarse[mid(p)])]
        fine.sort(key=lambda p: -detail[mid(p)])
        for p in fine:
            dt = detail[mid(p)]
            emit_contour(p, "feature", 0.55, 0.26 + 0.3 * dt, 0.4, 26)
    n_face_c = len(strokes) - n_form

    # --- тональные проходы ---
    C = np.zeros((H, W), np.float32)

    def deposit(pts, w, o):
        pi = np.round(pts).astype(np.int32)
        x0, y0 = max(pi[:, 0].min() - 2, 0), max(pi[:, 1].min() - 2, 0)
        x1, y1 = min(pi[:, 0].max() + 3, W), min(pi[:, 1].max() + 3, H)
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.polylines(m, [(pi - (x0, y0)).reshape(-1, 1, 2)], False, 255, max(1, round(w)), cv2.LINE_AA)
        roi = C[y0:y1, x0:x1]
        roi[:] = 1 - (1 - roi) * (1 - (m.astype(np.float32) / 255) * (o * min(w, 1.0)))

    dens = a.density
    cell = 1.5

    def run_pass(name, gate, cat, rot, sp, lw, op, thr, dmin, maxlen, jitter=4.0, color=None):
        t1 = time.time()
        gate_f = gate.astype(np.float32)
        deficit = D - cv2.GaussianBlur(C, (0, 0), 2.0)
        occ = np.zeros((int(H / cell) + 2, int(W / cell) + 2), np.uint8)
        step = max(1.2, sp * 0.7 / dens)
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
                if not (0 <= px < W and 0 <= py < H) or deficit[iy, ix] < thr * 0.3 or D[iy, ix] < dmin * 0.7 or gate_f[iy, ix] < 0.4:
                    break
                ang = theta[iy, ix] + rot_r
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
            dt = float(detail[iy, ix])
            sd = max(1, round(sp * (1 - 0.35 * dt) / dens / cell * 0.75))
            oy, ox = int(sy / cell), int(sx / cell)
            if occ[max(oy - sd, 0) : oy + sd + 1, max(ox - sd, 0) : ox + sd + 1].any():
                continue
            rot_r = math.radians(rot + rng.normal(0, jitter))  # лёгкий разброс угла — не «станок»
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
            o = float(np.clip(op * (0.7 + 0.6 * D[iy, ix] / DMAX) * rng.uniform(0.75, 1.25), 0.04, 0.8))
            w = lw * rng.uniform(0.7, 1.3) * (1 - 0.2 * dt)
            deposit(pts, w, o)
            layer.append((pts, w, o))
        layer.sort(key=lambda t: (int(t[0][0][1] // 40), t[0][0][0] if int(t[0][0][1] // 40) % 2 == 0 else -t[0][0][0]))  # змейкой
        for pts, w, o in layer:
            add(cat, pts, w, o, color or ("#1d1d21" if cat == "shadow" else "#2d2d32"))
        print(f"  {name:10s} {cat:6s} {rot:+4.0f}°: {len(layer):5d} штрихов ({time.time() - t1:.1f}с)")

    face_core = (detail > 0.5) & sub
    skin_soft = skin & (detail <= 0.5)
    stages = []
    if has_face:
        stages += [
            ("лицо", face_core, "hatch", 0, 1.9, 0.5, 0.34, 0.03, 0.05, 18),
            ("лицо", face_core, "shadow", 35, 1.7, 0.6, 0.5, 0.04, 0.22, 14),
        ]
    stages += [
        ("волосы", hair, "hatch", 0, 3.8, 0.6, 0.26, 0.03, 0.05, 130),
        ("волосы", hair, "hatch", 8, 3.3, 0.7, 0.32, 0.04, 0.14, 110),
        ("волосы", hair, "shadow", -6, 2.9, 0.8, 0.42, 0.05, 0.26, 90),
        ("волосы", hair, "cross", 38, 3.2, 0.8, 0.38, 0.05, 0.38, 50),
        ("кожа", skin_soft, "hatch", 0, 4.8, 0.6, 0.16, 0.025, 0.05, 55),
        ("кожа", skin_soft, "hatch", 10, 4.0, 0.7, 0.21, 0.03, 0.14, 45),
        ("кожа", skin_soft, "shadow", -8, 3.5, 0.8, 0.3, 0.04, 0.26, 40),
        ("одежда", cloth, "hatch", 0, 5.6, 0.6, 0.17, 0.03, 0.07, 100),
        ("одежда", cloth, "shadow", 6, 4.2, 0.8, 0.28, 0.04, 0.2, 80),
        ("одежда", cloth, "cross", 55, 4.6, 0.8, 0.28, 0.045, 0.34, 55),
    ]
    if has_face:
        stages += [("акценты", (detail > 0.55) & sub, "shadow", 20, 1.5, 0.5, 0.6, 0.05, 0.45, 10)]
    n_before = len(strokes)
    for st in stages:
        run_pass(*st)
    n_tone = len(strokes) - n_before

    out = dict(version=1, width=W, height=H, strokes=strokes)
    bad = [st["id"] for st in strokes if any(not (-5 <= v <= max(W, H) + 5) for v in st["pts"])]
    if len(strokes) < 1500 or bad:
        sys.exit(f"Ошибка: обработка дала некорректный результат (штрихов {len(strokes)}, вне холста {len(bad)}). strokes.json НЕ перезаписан.")
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, separators=(",", ":")))
    tmp.replace(dst)  # атомарно: при сбое прежний strokes.json остаётся
    Path("tools/out").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("tools/out/last_source_crop.png", gray8)  # кадр для сравнения; ВНЕ public/, в сборку не попадает
    print(f"контуры формы: {n_form}, контуры лица: {n_face_c}, тон: {n_tone}")
    print(f"landmarks: {'да' if has_face else 'нет'} | штрихов: {len(strokes)} | {Path(a.out).stat().st_size / 1e6:.2f} МБ | {time.time() - t0:.1f}с -> {a.out}")


if __name__ == "__main__":
    main()
