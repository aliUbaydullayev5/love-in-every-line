#!/usr/bin/env python3
"""Подход E: error-driven подбор штрихов (coarse-to-fine) по целевой карте темноты.

Независимая реализация идей (не копия кода):
  * greedy-цикл «остаток ошибки -> кандидат -> вычесть вклад» (идея pintr BSD-3 и DrawingBotV3 GPLv3: только идея);
  * направление штриха — поле вдоль изолиний яркости (structure tensor, карта `theta` из make_strokes_v4.py);
  * coarse-to-fine: сначала широкие длинные слабые штрихи (σ=3), затем всё тоньше и короче;
  * штрих принимается, только если реально уменьшает взвешенную ошибку (важность лица выше).

  python tools/engine_e.py MAPS.npz --target photo|sketch --out out.json [--sketch-model tools/models/informative_anime_512.onnx]

MAPS.npz: tools/make_strokes_v4.py --dump-maps (карта цели D, направления theta, детали лица, маска человека).
"""
import argparse, json, math, resource, sys, time
from pathlib import Path
import cv2
import numpy as np
from make_strokes import pack, simplify, smooth_path

# σ сглаживания, ширина, alpha, макс. длина, порог ошибки, число seed-ов за раунд, раундов, шанс «поперечного» кандидата
PASSES = [
    dict(sigma=3.0, w=2.2, o=0.10, L=110, thr=0.05, seeds=1400, rounds=3, cross=0.0, minL=26),
    dict(sigma=2.2, w=1.2, o=0.16, L=80, thr=0.04, seeds=2600, rounds=4, cross=0.05, minL=18),
    dict(sigma=1.6, w=0.8, o=0.26, L=55, thr=0.035, seeds=3600, rounds=4, cross=0.15, minL=13),
    dict(sigma=1.2, w=0.9, o=0.60, L=60, thr=0.03, seeds=2500, rounds=3, cross=0.0, minL=14, struct=True),  # только F: тёмные структуры
    dict(sigma=1.2, w=0.6, o=0.38, L=35, thr=0.03, seeds=4200, rounds=3, cross=0.25, minL=10),
    dict(sigma=0.9, w=0.5, o=0.50, L=20, thr=0.03, seeds=4200, rounds=2, cross=0.2, minL=8),
]


def sketch_target(img, model_path):
    """ML-эскиз (Informative Drawings, MIT, ONNX, CPU) -> карта тёмных линий 0..1 в размере кадра."""
    import onnxruntime as ort
    H, W = img.shape[:2]
    s = 512 / max(H, W)
    nh, nw = round(H * s), round(W * s)
    rgb = cv2.cvtColor(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    pad = cv2.copyMakeBorder(rgb, 0, 512 - nh, 0, 512 - nw, cv2.BORDER_REPLICATE)  # без искажения пропорций
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    t = time.time()
    y = sess.run(None, {sess.get_inputs()[0].name: np.transpose(pad.astype(np.float32) / 255, (2, 0, 1))[None]})[0]
    print(f"  ML-эскиз: инференс {time.time() - t:.2f}с")
    out = np.clip(y[0, 0], 0, 1)[:nh, :nw]
    return 1 - cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)


# MediaPipe FaceMesh: индексы контуров губ (внешний/внутренний, верх/низ)
LIP_OU = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIP_OL = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIP_IU = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIP_IL = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


def _resample(poly, step=1.0):
    poly = np.asarray(poly, np.float32)
    seg = np.hypot(*np.diff(poly, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    t = np.arange(0, cum[-1], step)
    return np.stack([np.interp(t, cum, poly[:, 0]), np.interp(t, cum, poly[:, 1])], 1)


def _snap(path, gmag, r=3):
    """Подгоняет контур по нормали к максимуму градиента исходного фото (±r px). Возвращает путь и силу края 0..1."""
    tg = np.gradient(cv2.GaussianBlur(path, (0, 0), 2.0).astype(np.float32), axis=0)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6
    offs = np.arange(-r, r + 1, dtype=np.float32)
    H, W = gmag.shape
    samp = np.zeros((len(path), len(offs)), np.float32)
    for j, o in enumerate(offs):
        q = path + nrm * o
        samp[:, j] = cv2.remap(gmag, q[:, 0:1].reshape(1, -1).astype(np.float32), q[:, 1:2].reshape(1, -1).astype(np.float32), cv2.INTER_LINEAR).ravel()
    best = offs[samp.argmax(1)]
    best = cv2.GaussianBlur(best.reshape(-1, 1), (0, 0), 3.0).ravel()  # гладкий сдвиг, без зигзагов
    snapped = path + nrm * best[:, None]
    return snapped, samp.max(1)


def lip_stage(img, gray, C, strokes, rng, sketch_label=""):
    """Дополнительный слой реальных штрихов губ. Возвращает число добавленных штрихов (0 = этап пропущен)."""
    from make_strokes import landmarks
    H, W = gray.shape
    P = landmarks(img)
    if P is None:
        print("  губы: лицо не найдено — этап пропущен")
        return 0
    poly = lambda ids: P[ids]
    outer = np.vstack([poly(LIP_OU), poly(LIP_OL)[::-1][1:-1]])
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [np.round(outer).astype(np.int32)], 1)
    face_w = float(np.ptp(P[:, 0]))
    lip_w = float(np.hypot(*(P[291] - P[61])))
    if not (0.25 < lip_w / face_w < 0.75) or mask.sum() < 0.0003 * H * W:
        print("  губы: геометрия неправдоподобна — этап пропущен")
        return 0
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.8)
    ring = (cv2.dilate(mask, np.ones((17, 17), np.uint8)) > 0) & (cv2.dilate(mask, np.ones((5, 5), np.uint8)) == 0)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    inside = mask > 0
    d_cr = abs(float(ycc[..., 1][inside].mean() - ycc[..., 1][ring].mean()))
    d_y = abs(float(g[inside].mean() - g[ring].mean())) * 255
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    ref = float(np.percentile(gm[ring | inside], 95)) + 1e-6
    gmn = np.clip(gm / ref, 0, 1).astype(np.float32)
    if d_cr < 4 and d_y < 10:  # границы губ в фото не подтверждаются
        print(f"  губы: контраст с кожей мал (ΔCr={d_cr:.1f}, ΔY={d_y:.1f}) — этап пропущен")
        return 0
    # контуры: подгонка по градиенту фото
    out_closed = _resample(np.vstack([outer, outer[:1]]), 1.0)
    out_snap, out_s = _snap(out_closed, gmn)
    sup = float(np.median(out_s))
    if sup < 0.12:
        print(f"  губы: край не подтверждён изображением (опора {sup:.2f}) — этап пропущен")
        return 0
    last = max((st["order"] for st in strokes), default=0)
    added = []

    def emit(path, cat, w, o, color="#26262a"):
        nonlocal last
        pts = simplify(smooth_path(path, 1.0), 0.25)
        if len(pts) < 2:
            return
        last += 1
        added.append(dict(id=len(strokes) + len(added), cat=cat, order=last, w=round(float(w), 2), o=round(float(o), 3), color=color, dur=1.0, pts=pack(pts)))

    # 1) мягкий контур: только там, где фото действительно показывает край; непрозрачность по силе края (не обводка)
    i = 0
    while i < len(out_snap) - 4:
        n = int(rng.integers(10, 22))
        chunk, s_c = out_snap[i : i + n + 1], float(out_s[i : i + n + 1].mean())
        if s_c > 0.2 * max(sup, 0.3) and len(chunk) > 3:
            emit(chunk, "feature", 0.5 * rng.uniform(0.85, 1.15), min(0.1 + 0.42 * s_c / max(sup, 0.3), 0.34))
        i += n + int(rng.integers(1, 4))
    # 2) линия между губами: по внутреннему контуру, непрозрачность по реальной тёмности щели в фото
    lipref = float(np.median(g[inside]))
    open_gap = float(np.mean(np.hypot(*(poly(LIP_IU) - poly(LIP_IL)).T)))
    paths = [LIP_IU] + ([LIP_IL] if open_gap > 3.0 else [])
    for k, ids in enumerate(paths):
        ip, _ = _snap(_resample(poly(ids), 1.0), gmn, 2)
        iy = np.clip(np.round(ip[:, 1]).astype(int), 0, H - 1)
        ix = np.clip(np.round(ip[:, 0]).astype(int), 0, W - 1)
        dark = np.clip((lipref - g[iy, ix]) / 0.28, 0, 1)
        i = 0
        while i < len(ip) - 4:
            n = int(rng.integers(8, 18))
            d_c = float(dark[i : i + n + 1].mean())
            if d_c > 0.12:
                emit(ip[i : i + n + 1], "shadow", 0.55, min(0.14 + 0.42 * d_c, 0.5 if k == 0 else 0.3), "#1d1d21")
            i += n + int(rng.integers(0, 3))
    # 3) светотень и объём: подбор коротких штрихов вдоль формы губ к тону самого фото внутри губ
    lo, hi = np.percentile(g[inside], [4, 96])
    Tl = np.clip((hi - g) / (hi - lo + 1e-4), 0, 1) ** 1.1 * 0.5 * (mask > 0)  # локальный контраст внутри губ, потолок 0.5 (мягко)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dx, dy = cv2.Sobel(cv2.GaussianBlur(dist, (0, 0), 1.5), cv2.CV_32F, 1, 0), cv2.Sobel(cv2.GaussianBlur(dist, (0, 0), 1.5), cv2.CV_32F, 0, 1)
    th = np.arctan2(dx, -dy)  # касательная к контуру губ (перпендикулярна градиенту дистанции)
    gate = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    # учёт уже нарисованного E-sketch и добавленных штрихов
    Cl = C.copy()
    for st in added:
        pp = np.round(np.array(st["pts"], np.float32).reshape(-1, 2)).astype(np.int32)
        m_ = np.zeros((H, W), np.uint8)
        cv2.polylines(m_, [pp.reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_AA)
        Cl = 1 - (1 - Cl) * (1 - m_.astype(np.float32) / 255 * st["o"])
    Tb, n_sh = cv2.GaussianBlur(Tl, (0, 0), 1.0), 0
    occ = np.zeros((H, W), np.uint8)  # занятость: не ставим штрихи друг на друга (иначе в уголке рта копится тёмный сгусток)
    for rd in range(3):
        Cb = cv2.GaussianBlur(Cl, (0, 0), 1.0)
        err = (Tb - Cb) * gate
        wgt = np.clip(err - 0.03, 0, None) ** 2
        tot = float(wgt.sum())
        if tot < 1e-4:
            break
        flat = wgt.ravel() / tot
        seeds = rng.choice(flat.size, size=min(260, int((flat > 0).sum())), p=flat)
        for sd in seeds:
            sy, sx = divmod(int(sd), W)
            if occ[sy, sx]:
                continue
            pts_ = []
            for sgn in (1, -1):
                px, py, pa, half = float(sx), float(sy), None, []
                for _ in range(10):
                    ix, iy_ = int(px), int(py)
                    if not (0 <= ix < W and 0 <= iy_ < H) or not gate[iy_, ix] or err[iy_, ix] < 0.015 or (len(half) > 1 and occ[iy_, ix]):
                        break
                    ang = th[iy_, ix]
                    if pa is not None:
                        d = (ang - pa + math.pi / 2) % math.pi - math.pi / 2
                        if abs(d) > 0.6:
                            break
                        ang = pa + 0.7 * d
                    pa = ang
                    px, py = px + sgn * 1.3 * math.cos(ang), py + sgn * 1.3 * math.sin(ang)
                    half.append((px, py))
                pts_.append(half)
            pts = pts_[1][::-1] + [(float(sx), float(sy))] + pts_[0]
            if len(pts) < 5:
                continue
            o = float(np.clip(0.2 + 0.5 * Tb[sy, sx], 0.12, 0.5)) * rng.uniform(0.8, 1.15)
            pp = np.round(np.array(pts)).astype(np.int32)
            F = np.zeros((H, W), np.uint8)
            cv2.polylines(F, [pp.reshape(-1, 1, 2)], False, 255, 1, cv2.LINE_AA)
            Cl = 1 - (1 - Cl) * (1 - F.astype(np.float32) / 255 * o)
            cv2.polylines(occ, [pp.reshape(-1, 1, 2)], False, 1, 3)
            emit(np.array(pts, np.float32), "hatch", 0.5, o, "#2d2d32")
            n_sh += 1
    strokes.extend(added)
    print(f"  губы: добавлено {len(added)} штрихов (контур+линия+светотень {n_sh}); ΔCr={d_cr:.1f}, опора края {sup:.2f}, зазор {open_gap:.1f}px")
    return len(added)


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("maps")
    ap.add_argument("--target", choices=["photo", "sketch", "hybrid"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sketch-model", default="tools/models/informative_anime_512.onnx")
    ap.add_argument("--lips", action="store_true", help="дополнительный слой штрихов губ (по landmarks, с проверкой по фото)")
    ap.add_argument("--max-strokes", type=int, default=26000)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    z = np.load(a.maps)
    D, theta, detail, subj, img = z["D"], z["theta"], z["detail"], z["subj"], z["img"]
    DMAX = float(z["DMAX"])
    H, W = D.shape
    sub = np.clip(subj * 1.1, 0, 1).astype(np.float32)
    if a.target == "photo":
        T = D.astype(np.float32)
    elif a.target == "sketch":  # ML-эскиз как «художественная подготовка»: линии из модели + мягкий тон из фото
        S = sketch_target(img, a.sketch_model)
        lines = np.clip((S - 0.12) * 1.25, 0, 1) * 0.8
        T = np.clip(np.maximum(lines, 0) + 0.7 * cv2.GaussianBlur(D.astype(np.float32), (0, 0), 6), 0, DMAX).astype(np.float32) * sub
    hy = a.target == "hybrid"
    if hy:
        # --- F: художественная цель (ML-эскиз + мягкий тон) + цель точности (линии/детали исходного фото) ---
        gray = z["gray"].astype(np.float32) / 255
        S = sketch_target(img, a.sketch_model)
        lines = np.clip((S - 0.12) * 1.25, 0, 1) * 0.8
        T_art = np.clip(lines + 0.7 * cv2.GaussianBlur(D.astype(np.float32), (0, 0), 6), 0, DMAX)
        b = lambda x, sg: cv2.GaussianBlur(x, (0, 0), sg)
        dd = np.maximum(-(b(gray, 1.0) - b(gray, 1.9)), -0.8 * (b(gray, 2.0) - b(gray, 3.6)))  # тёмные линии двух масштабов (DoG)
        Lf = np.clip(dd / (np.percentile(dd[sub > 0.5], 97) + 1e-6), 0, 1) ** 0.8
        gm = np.hypot(cv2.Sobel(b(gray, 1.0), cv2.CV_32F, 1, 0), cv2.Sobel(b(gray, 1.0), cv2.CV_32F, 0, 1))
        edge_n = np.clip(gm / (np.percentile(gm[sub > 0.5], 99) + 1e-6), 0, 1)  # сила контура (для штрафа за искажение)
        edge_s = b(edge_n, 2.0)
        # визуальная значимость: spectral residual (без нейросети)
        sm = cv2.resize(gray, (256, int(256 * H / W)), interpolation=cv2.INTER_AREA)
        F_ = np.fft.fft2(sm)
        logA = np.log(np.abs(F_) + 1e-6)
        sal = np.abs(np.fft.ifft2(np.exp(logA - cv2.blur(logA, (3, 3)) + 1j * np.angle(F_)))) ** 2
        sal = cv2.resize(b(sal.astype(np.float32), 2.5), (W, H))
        sal = np.clip(sal / (np.percentile(sal, 99) + 1e-6), 0, 1)
        raw = 2.0 * detail + 2.0 * edge_s / (edge_s.max() + 1e-6) * 2 + 1.0 * sal  # лицо, контуры, значимость; лицо не обязательно
        imp_n = np.clip(raw / (np.percentile(raw[sub > 0.5], 95) + 1e-6), 0, 1).astype(np.float32)
        wf = imp_n ** 0.7
        T = np.maximum(T_art, wf * Lf * 0.8 * DMAX)  # важные линии фото не теряются в ML-стилизации
        wd = np.clip(detail * 1.6, 0, 1)
        T = T * (1 - wd) + np.maximum(T, 0.85 * D) * wd  # лицо/глаза/губы/оправа: тон оригинала
        T = np.clip(T, 0, DMAX).astype(np.float32) * sub
        # --- сохранение значимых тёмных структур (оправа, провода, контуры предметов, тонкие тёмные детали) ---
        g8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
        R = np.zeros_like(gray)
        for k in (5, 9, 15, 21):  # black-hat на нескольких масштабах: тёмнее окружения (замыкание минус изображение)
            bh = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32) / 255
            R = np.maximum(R, bh)
        dark_abs = b(gray, 1.0) < 0.62 * float(np.median(gray[sub > 0.5]))  # структура должна быть реально тёмной, а не просто темнее блика
        Mb = ((R > 0.15) & dark_abs & (sub > 0.5)).astype(np.uint8)
        Mb = cv2.morphologyEx(Mb, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # непрерывность: закрываем мелкие разрывы
        nC, lab, st, _ = cv2.connectedComponentsWithStats(Mb, connectivity=8)
        keep = np.zeros(nC, bool)
        for i in range(1, nC):  # устойчивые компоненты: не пыль, не большие тёмные площади (это тон, а не структура)
            ar = st[i, cv2.CC_STAT_AREA]
            if 30 <= ar <= 0.03 * H * W and float(R[lab == i].mean()) >= 0.19:
                keep[i] = True
        M = b(keep[lab].astype(np.float32), 0.8)
        Sd = np.clip(R / 0.45, 0, 1) ** 0.7
        T = np.maximum(T, DMAX * 0.95 * M * Sd * sub)  # толщина и контраст структуры восстанавливаются целью
        imp = (1 + 3.0 * imp_n + 3.0 * M).astype(np.float32) * sub  # штрихи, восстанавливающие структуру, ценнее
        # направления: вдоль хребта структуры; на линзах/радужке — локальная структура фото, а не ось глаза
        gxx, gyy = cv2.Sobel(b(gray, 1.0), cv2.CV_32F, 1, 0), cv2.Sobel(b(gray, 1.0), cv2.CV_32F, 0, 1)
        sxx, syy, sxy = b(gxx * gxx, 2.0), b(gyy * gyy, 2.0), b(gxx * gyy, 2.0)
        th_loc = 0.5 * np.arctan2(2 * sxy, sxx - syy) + math.pi / 2
        coh = np.sqrt((sxx - syy) ** 2 + 4 * sxy**2) / (sxx + syy + 1e-4)
        Rx, Ry = cv2.Sobel(b(R, 1.5), cv2.CV_32F, 1, 0), cv2.Sobel(b(R, 1.5), cv2.CV_32F, 0, 1)
        rxx, ryy, rxy = b(Rx * Rx, 1.5), b(Ry * Ry, 1.5), b(Rx * Ry, 1.5)
        th_R = 0.5 * np.arctan2(2 * rxy, rxx - ryy) + math.pi / 2
        eye = np.clip(detail * 1.5, 0, 1)
        w_R = M
        w_loc = eye * (1 - M) * np.clip(coh * 2, 0, 1)
        w_base = np.clip(1 - w_R - w_loc, 0.05, 1) * (1 - 0.8 * eye * (1 - M))  # ось глаза почти не влияет на линзу
        Vx = w_base * np.cos(2 * theta) + w_R * np.cos(2 * th_R) + w_loc * np.cos(2 * th_loc)
        Vy = w_base * np.sin(2 * theta) + w_R * np.sin(2 * th_R) + w_loc * np.sin(2 * th_loc)
        theta = (0.5 * np.arctan2(Vy, Vx)).astype(np.float32)
        print(f"  тёмные структуры: {int(keep.sum())} компонентов, покрытие {(M > 0.5).mean() * 100:.1f}% кадра")
        print(f"  важность: лицо {'да' if detail.max() > 0 else 'нет (универсальные карты)'}, средняя imp_n {imp_n[sub > 0.5].mean():.2f}")
    else:
        imp = (1 + 2.0 * detail).astype(np.float32) * sub  # лицо важнее

    rng = np.random.default_rng(a.seed)
    C = np.zeros((H, W), np.float32)  # накопленное покрытие (alpha-over)
    strokes, order = [], 0
    n_eval = 0

    def trace(x, y, sgn, L, rot, err, thr):
        pts, px, py, pa = [], x, y, None
        for _ in range(int(L / 1.5)):
            ix, iy = int(px), int(py)
            if not (0 <= ix < W and 0 <= iy < H) or err[iy, ix] < thr * 0.35 or sub[iy, ix] < 0.3:
                break
            ang = theta[iy, ix] + rot
            if pa is not None:
                d = (ang - pa + math.pi / 2) % math.pi - math.pi / 2
                if abs(d) > 0.55:
                    break
                ang = pa + 0.65 * d
            pa = ang
            px, py = px + sgn * 1.5 * math.cos(ang), py + sgn * 1.5 * math.sin(ang)
            pts.append((px, py))
        return pts

    for pi_, ps in enumerate(PASSES):
        sg = ps["sigma"]
        Tb = cv2.GaussianBlur(T, (0, 0), sg)
        t1, acc = time.time(), 0
        if ps.get("struct") and not hy:
            continue
        for rd in range(ps["rounds"]):
            if len(strokes) >= a.max_strokes:
                break
            Cb = cv2.GaussianBlur(C, (0, 0), sg)
            err = Tb - Cb
            wgt = np.clip(err - ps["thr"], 0, None) ** 2 * imp
            if ps.get("struct"):
                wgt = wgt * M  # проход структур работает только по найденным тёмным структурам
            tot = float(wgt.sum())
            if tot < 1e-3:
                break
            flat = wgt.ravel() / tot
            seeds = rng.choice(flat.size, size=min(ps["seeds"], int((flat > 0).sum())), p=flat)
            batch = []
            for sd in seeds:
                sy, sx = divmod(int(sd), W)
                dt = float(detail[sy, sx])
                L = ps["L"] * rng.uniform(0.5, 1.2) * (1 - 0.3 * dt)
                base = 0.0 if rng.random() > ps["cross"] * (0.0 if hy and dt > 0.5 else 1.0) else (math.pi / 2 + rng.uniform(-0.5, 0.5))  # иногда поперёк потока
                jit = 0.12 if hy and dt > 0.5 else 0.35  # F: в глазах/губах направления согласованы
                best = None
                if hy and detail[sy, sx] > 0.97 and M[sy, sx] < 0.3 and coh[sy, sx] < 0.15:
                    continue  # бесструктурная линза/радужка: случайные штрихи не нужны
                for var in range(3):  # 3 кандидата: небольшие повороты потока
                    rot = base + (0.0 if var == 0 else rng.uniform(-jit, jit))
                    f = trace(sx, sy, 1, L, rot, err, ps["thr"])
                    b = trace(sx, sy, -1, L, rot, err, ps["thr"])
                    pts = b[::-1] + [(float(sx), float(sy))] + f
                    if len(pts) < 4:
                        continue
                    pts = simplify(smooth_path(pts, 1.2), 0.3)
                    if len(pts) < 2:
                        continue
                    pi2 = np.round(pts).astype(np.int32)
                    pad = int(3 * sg) + 3
                    x0, y0 = max(pi2[:, 0].min() - pad, 0), max(pi2[:, 1].min() - pad, 0)
                    x1, y1 = min(pi2[:, 0].max() + pad + 1, W), min(pi2[:, 1].max() + pad + 1, H)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    F = np.zeros((y1 - y0, x1 - x0), np.uint8)
                    w = ps["w"] * rng.uniform(0.75, 1.25) * ((1 + 0.6 * M[sy, sx]) if hy else 1.0)
                    cv2.polylines(F, [(pi2 - (x0, y0)).reshape(-1, 1, 2)], False, 255, max(1, round(w)), cv2.LINE_AA)
                    Fs = F.astype(np.float32) / 255 * min(w, 1.0)
                    o = float(np.clip(ps["o"] * rng.uniform(0.8, 1.3) * (0.7 + 0.6 * Tb[sy, sx] / DMAX) * ((1 + 0.8 * M[sy, sx]) if hy else 1.0), 0.03, 0.85))
                    cb, tb, im = Cb[y0:y1, x0:x1], Tb[y0:y1, x0:x1], imp[y0:y1, x0:x1]
                    new = cb + (1 - cb) * o * cv2.GaussianBlur(Fs, (0, 0), sg)  # предсказанное сглаженное покрытие
                    gain = float((im * ((tb - cb) ** 2 - (tb - new) ** 2)).sum())
                    n_eval += 1
                    if hy:
                        inn = float(imp_n[sy, sx])
                        parr = pts.astype(np.float32)
                        plen = float(cv2.arcLength(parr.reshape(-1, 1, 2), False))
                        if plen < max(7.0, ps["minL"] * (1 - 0.4 * inn)):  # штраф за короткие случайные линии
                            continue
                        ov = float((Fs * (C[y0:y1, x0:x1] > 0.4)).sum() / (Fs.sum() + 1e-6))  # повторяющиеся штрихи
                        ii = np.clip(np.round(parr).astype(int), 0, [W - 1, H - 1])
                        seg = np.diff(parr, axis=0)
                        mid = ii[:-1]
                        d_ang = np.abs(np.sin(np.arctan2(seg[:, 1], seg[:, 0]) - theta[mid[:, 1], mid[:, 0]]))
                        en = float(edge_n[ii[:, 1], ii[:, 0]].mean())
                        err2 = float((im * (tb - cb) ** 2).sum()) + 1e-9
                        ov = ov * (1 - float(M[sy, sx]))  # на тёмных структурах слои допустимы
                        gain = gain * (1 - 0.5 * min(ov, 1.0)) * (1 - min(0.9, 2.5 * en * float(d_ang.mean())))  # искажение сильных контуров
                        if gain / err2 < 0.012 * (1.0 + 1.0 * (1 - inn)):  # шум: штрих должен заметно уменьшать локальную ошибку
                            continue
                    if best is None or gain > best[0]:
                        best = (gain, pts, w, o, (x0, y0, x1, y1), Fs, new)
                if best is None or (not hy and best[0] <= ps["thr"] * 0.6 * max(len(best[1]), 1) * 0.02) or (hy and best[0] <= 0):
                    continue
                gain, pts, w, o, (x0, y0, x1, y1), Fs, new = best
                C[y0:y1, x0:x1] = 1 - (1 - C[y0:y1, x0:x1]) * (1 - Fs * o)
                Cb[y0:y1, x0:x1] = new  # локально обновляем сглаженную карту
                err[y0:y1, x0:x1] = Tb[y0:y1, x0:x1] - new
                batch.append((pts, w, o))
            batch.sort(key=lambda t: (int(t[0][0][1] // 40), t[0][0][0] if int(t[0][0][1] // 40) % 2 == 0 else -t[0][0][0]))  # полосами, змейкой
            for pts, w, o in batch:
                parts = [(pts, w, o)]
                if len(pts) >= 10:  # нажим по длине
                    n = len(pts)
                    i1, i2 = max(n // 4, 2), n - max(n // 4, 2)
                    parts = [(pts[: i1 + 1], w * 0.8, o * 0.55), (pts[i1 : i2 + 1], w, o), (pts[i2:], w * 0.8, o * 0.55)]
                for pp, ww, oo in parts:
                    if len(pp) < 2:
                        continue
                    order += 1
                    strokes.append(dict(id=len(strokes), cat="shadow" if oo > 0.35 else "hatch", order=order, w=round(float(ww), 2), o=round(float(oo), 3),
                                        color="#1d1d21" if oo > 0.35 else "#2d2d32", dur=1.0, pts=pack(pp)))
            acc += len(batch)
        print(f"  проход {pi_ + 1} σ={sg}: принято {acc} штрихов ({time.time() - t1:.1f}с)")

    if a.lips:
        lip_stage(img, z["gray"], C, strokes, np.random.default_rng(a.seed + 100))
    if len(strokes) < 1500:
        sys.exit("Ошибка: слишком мало штрихов.")
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(dict(version=1, width=W, height=H, strokes=strokes), separators=(",", ":")))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    fin = float(np.mean((np.clip(cv2.GaussianBlur(C, (0, 0), 2) - cv2.GaussianBlur(T, (0, 0), 2), -1, 1) ** 2)[sub > 0.5]))
    print(f"[{'F' if hy else 'E-' + a.target}] штрихов: {len(strokes)} | {dst.stat().st_size / 1e6:.2f} МБ | {time.time() - t0:.1f}с | пик RAM {rss:.0f} МБ | оценок кандидатов {n_eval} | MSE(человек) {fin:.4f}")


if __name__ == "__main__":
    main()
