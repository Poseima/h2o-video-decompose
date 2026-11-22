"""Native-pixel, paired local appearance measurements. All distances are in source pixels."""

import itertools
import json
import math
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np

# Measurement resolution floors, not aesthetic targets. Temporal scatter raises each floor.
# value: minimum absolute delta, fractional delta, unit, label, controlled trial.
SPECS = {
    "grain_scale": (
        0.85,
        0.22,
        "px",
        "非周期纹理特征波长",
        "只改噪声纹理的 Scale，复测特征波长。",
    ),
    "grain_axis": (
        16,
        0,
        "°",
        "非周期细纹轴向",
        "只旋转拉丝或纤维的映射坐标，复测主轴。",
    ),
    "shadow_softness": (
        1.2,
        0.30,
        "px",
        "暗斑边界过渡宽度",
        "固定遮挡物间距，只改光源尺寸，复测半影过渡。",
    ),
    "chroma_width": (
        0.8,
        0.32,
        "px",
        "颜色交界过渡宽度",
        "只改两种材质之间遮罩的羽化宽度，复测色彩过渡。",
    ),
    "period": (1.1, 0.18, "px", "纹理周期", "只调整纹理映射缩放，复测周期和角度。"),
    "angle": (14, 0, "°", "纹理轴向", "只旋转映射坐标，复测轴向，保持尺度。"),
    "periodicity": (
        0.055,
        0.28,
        "比值",
        "规则纹理集中度",
        "只改变规则纹理与随机纹理的混合比例。",
    ),
    "fine": (
        2,
        0.33,
        "灰阶",
        "表面细纹强度",
        "只改底色纹理对比度，检查轮廓清晰度是否稳定。",
    ),
    "clustering": (
        0.12,
        0.42,
        "变异系数",
        "纹理聚集度",
        "只改磨损遮罩的低频调制，复测分区密度。",
    ),
    "edge_width": (
        0.72,
        0.27,
        "px",
        "轮廓过渡宽度",
        "固定材质，只改焦距平面的虚化强度；对照其他深度的边缘。",
    ),
    "edge_jitter": (
        0.32,
        0.42,
        "px",
        "轮廓横向起伏",
        "只改边沿遮罩的位移幅度，同时复测边缘过渡宽度。",
    ),
    "edge_breaks": (0.12, 0.35, "占比", "边缘间断率", "只改边缘磨损遮罩的覆盖率。"),
    "color_fringe": (
        0.55,
        0.38,
        "px",
        "色边宽度",
        "只调整边沿颜色遮罩宽度；全画面同向偏移时另测色散。",
    ),
    "tone": (12, 0.10, "灰阶", "局部明度", "固定灯光和曝光，只改表面底色后复测。"),
    "color_a": (
        7,
        0.0,
        "Lab a*",
        "局部红绿偏色",
        "固定光源，只沿红绿方向微调底色复测。",
    ),
    "color_b": (
        7,
        0.0,
        "Lab b*",
        "局部黄蓝偏色",
        "固定光源，只沿黄蓝方向微调底色复测。",
    ),
    "gradient": (9, 0.25, "灰阶", "表面明暗渐变", "锁定底色，只改变主光方位复测。"),
    "bright_contrast": (
        14,
        0.28,
        "灰阶",
        "局部亮斑对比",
        "固定底色，只改灯光功率，复测亮斑和中间调。",
    ),
    "highlight_width": (
        1.4,
        0.24,
        "px",
        "亮斑半高宽",
        "固定粗糙度，只改变灯尺寸，同时复测阴影软边；若阴影已吻合，另做只改粗糙度的试验。",
    ),
    "shadow_width": (
        1.4,
        0.24,
        "px",
        "暗带半深宽",
        "先核对遮挡间距，再只改主光尺寸复测暗带。",
    ),
    "attachment": (
        0.24,
        0,
        "相关差",
        "纹理相对运动",
        "固定相机运动，只切换物体与屏幕坐标，比较运动补偿前后的细纹相关。",
    ),
}
WHY = {
    "highlight_width": "局部亮斑可能来自反射、灯尺寸或底色亮斑",
    "shadow_width": "暗带可能来自投影、接触间隙或底色",
    "edge_width": "局部清晰度受对焦、运动、倒角及编码影响",
    "attachment": "正值偏向跟随表面，负值偏向锁在画幅；需有可测的表面位移",
}
WHY.update(
    {
        "period": "重复图样的间距有偏差，候选原因是 UV 比例、纹理频率或投影缩放",
        "angle": "条纹主轴有偏差，候选原因是 UV 旋转或物体朝向",
        "periodicity": "规则细节所占能量改变，可能来自纹理叠加强度或采样损失",
        "fine": "细纹对比有偏差，候选原因包括底色纹理、凹凸强度与成像软化",
        "grain_scale": "非周期纹理的频谱重心不同，颗粒尺度和光学软化都可能影响读数",
        "grain_axis": "细纹能量的主要方向不同，可能来自拉丝方向或运动拖影",
        "clustering": "小块之间的细节疏密不同，候选原因是遮罩调制或混入物体边界",
        "edge_jitter": "轮廓沿线的位置波动不同，候选原因是边缘遮罩或几何起伏",
        "edge_breaks": "轮廓可连续测到的长度不同，候选原因是磨损、遮挡或纹理干扰",
        "color_fringe": "不同色通道的边沿位置分离，可能来自色散或边缘着色",
        "chroma_width": "材质色彩交界的羽化或色度采样会改变过渡宽度",
        "gradient": "局部明暗跨度不同，可能来自布光、表面朝向或底色渐变",
        "tone": "表面底色、曝光与入射光强都能改变局部明度",
        "color_a": "底色、光源颜色或调色可能改变红绿分量",
        "color_b": "底色、光源颜色或调色可能改变黄蓝分量",
        "bright_contrast": "局部亮斑突出程度不同，可能来自灯光、反射或底色",
        "shadow_softness": "暗斑边界可能来自投影或底色遮罩，阈值宽度随形状变化",
    }
)


def gray(im):
    return cv.cvtColor(im, cv.COLOR_BGR2GRAY).astype(np.float32)


def region(im, b):
    x, y, w, h = b
    return im[y : y + h, x : x + w]


def box_center(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def coverage(a, b):
    x, y, w, h = a
    X, Y, W, H = b
    overlap = max(0, min(x + w, X + W) - max(x, X)) * max(
        0, min(y + h, Y + H) - max(y, Y)
    )
    return overlap / min(w * h, W * H)


def candidates(im, replica):
    h, w = im.shape[:2]
    g = gray(im)
    short = min(h, w)
    if short < 32:
        return []
    side = max(32, min(144, short // 5))
    residual = cv.absdiff(im, replica).mean(axis=2).astype(np.float32)
    integral = cv.integral(np.minimum(residual, 30))
    scored = []
    for n in sorted({side, min(short, side * 2)}):
        # Tile the full field, including borders. Choose different material/edge roles per sector.
        for y in np.linspace(
            0, h - n, min(18, math.ceil(2 * (h - n) / n) + 1), dtype=int
        ):
            for x in np.linspace(
                0, w - n, min(24, math.ceil(2 * (w - n) / n) + 1), dtype=int
            ):
                p = cv.resize(
                    g[y : y + n, x : x + n], (32, 32), interpolation=cv.INTER_AREA
                )
                smooth = cv.GaussianBlur(p, (0, 0), 2)
                tex = float(np.std(p - smooth))
                spread = float(p.std())
                light = float(np.percentile(smooth, 94) - np.median(smooth))
                dark = float(np.median(smooth) - np.percentile(smooth, 6))
                edge = float(
                    np.percentile(
                        np.abs(cv.Sobel(smooth, cv.CV_32F, 1, 0))
                        + np.abs(cv.Sobel(smooth, cv.CV_32F, 0, 1)),
                        90,
                    )
                )
                deviation = (
                    integral[y + n, x + n]
                    - integral[y, x + n]
                    - integral[y + n, x]
                    + integral[y, x]
                ) / (n * n)
                scores = (tex, edge, light, dark, -spread, deviation)
                sector = (
                    min(3, int(4 * (x + n / 2) / w)),
                    min(3, int(4 * (y + n / 2) / h)),
                    n,
                )
                scored.append((sector, (int(x), int(y), n, n), scores))
    winners = {}
    for sector, b, scores in scored:
        for j, score in enumerate(scores):
            k = sector, j
            if k not in winners or score > winners[k][0]:
                winners[k] = (score, b)
    selected = {b for _, b in winners.values()}
    # Add small windows on extrema between grid centers, including peripheral subjects.
    tiny = max(32, side // 2)
    scale = min(1.0, 480 / max(w, h))
    thumb = cv.resize(g, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    diff = cv.resize(
        residual, (thumb.shape[1], thumb.shape[0]), interpolation=cv.INTER_AREA
    )
    features = (
        cv.GaussianBlur(diff, (0, 0), max(1, tiny * scale / 4)),
        abs(thumb - cv.GaussianBlur(thumb, (0, 0), 2)),
    )
    for proxy in features:
        for patch_row in np.array_split(np.arange(proxy.shape[0]), 4):
            for patch_col in np.array_split(np.arange(proxy.shape[1]), 4):
                part = proxy[
                    patch_row[0] : patch_row[-1] + 1, patch_col[0] : patch_col[-1] + 1
                ]
                Y, X = np.unravel_index(np.argmax(part), part.shape)
                if part[Y, X] < 2:
                    continue
                x = int(np.clip((patch_col[0] + X) / scale - tiny / 2, 0, w - tiny))
                y = int(np.clip((patch_row[0] + Y) / scale - tiny / 2, 0, h - tiny))
                selected.add((x, y, tiny, tiny))
    return sorted(selected)


def match(a, b, box):
    """Match low-pass context; cross-check reverse location and affine-free local residual."""
    h, w = a.shape
    x, y, n, _ = box
    pad = n // 3
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + n + pad), min(h, y + n + pad)
    radius = min(40, max(6, n // 3))

    def seek(src, dst, rect):
        X, Y, W, H = rect
        l, t = max(0, X - radius), max(0, Y - radius)
        r, bot = min(w, X + W + radius), min(h, Y + H + radius)
        template = src[Y : Y + H, X : X + W]
        if template.std() < 2.5:
            return X, Y, 0.0
        scores = cv.matchTemplate(dst[t:bot, l:r], template, cv.TM_CCOEFF_NORMED)
        _, best, _, (u, v) = cv.minMaxLoc(scores)
        here = float(scores[Y - t, X - l])
        if here >= best - 0.018:
            return X, Y, here
        return l + u, t + v, best

    rect = (x0, y0, x1 - x0, y1 - y0)
    X, Y, score = seek(a, b, rect)
    dx, dy = X - x0, Y - y0
    revx, revy, _ = seek(b, a, (X, Y, rect[2], rect[3]))
    err = math.hypot(revx - x0, revy - y0)
    d = float(np.mean(np.abs(region(a, rect) - region(b, (X, Y, *rect[2:])))))
    # Constant regions have no resolvable translation. Admit only close, same-coordinate contexts.
    if score < 0.57 and (d > 4 or np.std(region(a, rect)) > 7):
        return None
    if err > max(2.5, n * 0.045) or d > 33:
        return None
    return (
        (x + dx, y + dy, n, n),
        float(np.clip(max(score, 0.64) - err / n - d / 320, 0.3, 1)),
        err,
    )


def spectrum(g):
    h, w = g.shape
    band = g - cv.GaussianBlur(g, (0, 0), max(2, min(h, w) / 10))
    p = np.abs(np.fft.fft2(band * np.outer(np.hanning(h), np.hanning(w)))) ** 2
    yy, xx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    f = np.hypot(xx, yy)
    mask = (f >= 2.8 / min(w, h)) & (f < 0.44)
    p *= mask
    iy, ix = np.unravel_index(np.argmax(p), p.shape)
    strength = float(p[iy, ix] / max(1.0, p.sum()))
    result = {"periodicity": strength}
    if strength > 0.055 and band.std() > 1.5:
        result["period"] = 1 / float(f[iy, ix])
        result["angle"] = (math.degrees(math.atan2(yy[iy, ix], xx[iy, ix])) + 90) % 180
    elif band.std() > 2 and p.sum() > 0:
        # Radial median frequency uses spectral energy, independent of random phase.
        frequencies = np.linspace(0, 0.45, 50)
        power, _ = np.histogram(f, bins=frequencies, weights=p)
        k = int(np.searchsorted(np.cumsum(power), power.sum() / 2))
        result["grain_scale"] = 1 / float((frequencies[k] + frequencies[k + 1]) / 2)
        moment = (p * np.exp(2j * np.arctan2(yy, xx))).sum() / p.sum()
        if abs(moment) > 0.6:
            result["grain_axis"] = (math.degrees(np.angle(moment)) / 2 + 90) % 180
    return result


def edges(im, g):
    # Long straight local fragments support normal profiles even on gently curved outlines.
    view = np.uint8(np.clip(g, 0, 255))
    lines = cv.createLineSegmentDetector(cv.LSD_REFINE_NONE).detect(view)[0]
    if lines is None:
        return {}
    segments = sorted(
        lines.reshape(-1, 4), key=lambda l: -math.hypot(l[2] - l[0], l[3] - l[1])
    )
    values = []
    t = np.linspace(-8, 8, 65, dtype=np.float32)
    for x, y, X, Y in segments[:12]:
        dx, dy = X - x, Y - y
        length = math.hypot(dx, dy)
        if length < max(14, min(g.shape) * 0.24):
            continue
        u = np.linspace(0.1, 0.9, max(12, int(0.8 * length)), dtype=np.float32)[:, None]
        mx = x + dx * u - dy / length * t
        my = y + dy * u + dx / length * t
        keep = (
            (mx.min(1) >= 0)
            & (my.min(1) >= 0)
            & (mx.max(1) < g.shape[1] - 1)
            & (my.max(1) < g.shape[0] - 1)
        )
        if keep.sum() < 10:
            continue
        rows = cv.remap(im, mx[keep], my[keep], cv.INTER_LINEAR).astype(np.float32)
        lum = rows @ np.array([0.114, 0.587, 0.299], dtype=np.float32)

        def crossing(r, frac):
            low = r[:, :6].mean(1)
            scale = r[:, -6:].mean(1) - low
            q = (r - low[:, None]) / np.where(abs(scale) > 1, scale, 1)[:, None]
            cross = (q[:, :-1] <= frac) & (q[:, 1:] > frac)
            k = np.argmin(np.where(cross, np.abs(t[:-1]), 100), axis=1)
            i = np.arange(len(r))
            z = q[i, k + 1] - q[i, k]
            pos = t[k] + 0.25 * (frac - q[i, k]) / np.maximum(z, 0.0001)
            ok = cross.any(1) & (abs(scale) > 20)
            return pos, ok

        a, ka = crossing(lum, 0.1)
        b, kb = crossing(lum, 0.9)
        m, km = crossing(lum, 0.5)
        good = ka & kb & km & (b > a) & (b - a < 14)
        if good.sum() < 10 or good.mean() < 0.55:
            continue
        axis = u[keep, 0][good]
        offsets = m[good]
        residual = offsets - np.polyval(np.polyfit(axis, offsets, 2), axis)
        chans = [crossing(rows[:, :, k], 0.5) for k in range(3)]
        allvalid = good & np.logical_and.reduce([ok for _, ok in chans])
        fringe = (
            np.median(np.ptp([v[allvalid] for v, _ in chans], axis=0))
            if allvalid.sum() >= 8
            else np.nan
        )
        values.append(
            (np.median((b - a)[good]), np.std(residual), 1 - good.mean(), fringe)
        )
    if not values:
        return {}
    return {
        key: float(np.median(z[np.isfinite(z)]))
        for key, z in zip(
            ("edge_width", "edge_jitter", "edge_breaks", "color_fringe"),
            np.asarray(values).T,
        )
        if np.isfinite(z).any()
    }


def patch_metrics(im):
    g = gray(im)
    n = min(g.shape)
    smooth = cv.GaussianBlur(g, (0, 0), max(2, n / 14))
    strong = cv.Canny(g.astype("uint8"), 60, 150)
    interior = cv.dilate(strong, np.ones((5, 5), np.uint8)) == 0
    fine = g - cv.GaussianBlur(g, (0, 0), 1.5)
    lab = cv.cvtColor(im.astype("float32") / 255, cv.COLOR_BGR2Lab)
    result = {
        "tone": float(np.median(g)),
        "color_a": float(np.median(lab[:, :, 1])),
        "color_b": float(np.median(lab[:, :, 2])),
    }
    if interior.mean() > 0.30:
        result["fine"] = float(np.sqrt(np.mean(fine[interior] ** 2)))
        cells = [
            np.abs(z).mean()
            for row in np.array_split(fine, 4)
            for z in np.array_split(row, 4, axis=1)
        ]
        if np.mean(cells) > 2:
            result["clustering"] = float(np.std(cells) / np.mean(cells))
    frequency = spectrum(g)
    if interior.mean() < 0.65:
        frequency.pop("grain_scale", None)
        frequency.pop("grain_axis", None)
    result.update(frequency)
    result["gradient"] = float(np.percentile(smooth, 90) - np.percentile(smooth, 10))
    result.update(edges(im, g))
    # Isoluminant boundaries still have measurable chromatic transitions.
    chroma = max((lab[:, :, 1], lab[:, :, 2]), key=lambda z: z.std())
    if chroma.std() > 5 and g.std() < chroma.std():
        channel = np.uint8(np.clip(128 + chroma * 1.8, 0, 255))
        width = edges(
            cv.cvtColor(channel, cv.COLOR_GRAY2BGR), channel.astype(np.float32)
        ).get("edge_width")
        if width is not None:
            result["chroma_width"] = width
    for key, signal in (("highlight_width", smooth), ("shadow_width", -smooth)):
        bg = np.median(signal)
        peak = np.percentile(signal, 99)
        if peak - bg < 22:
            if key == "highlight_width":
                result["bright_contrast"] = 0.0
            continue
        mask = np.uint8(signal > (peak + bg) / 2)
        count, labels, stats, centers = cv.connectedComponentsWithStats(mask)
        widths = []
        for index, (x, y, w, h, area) in enumerate(stats[1:count], 1):
            if (
                x > 0
                and y > 0
                and x + w < n
                and y + h < n
                and 6 <= area < 0.42 * n * n
                and min(w, h) / max(w, h) > 0.10
            ):
                widths.append((2 * math.sqrt(area / math.pi), index))
        if key == "highlight_width":
            result["bright_contrast"] = float(peak - bg) if widths else 0.0
        if widths:
            width, index = max(widths)
            result[key] = width
            result[key + "_center"] = centers[index]
            if key == "shadow_width":
                core = np.uint8(signal > bg + 0.8 * (peak - bg))
                skirt = np.uint8(signal > bg + 0.2 * (peak - bg))
                _, sk, st, _ = cv.connectedComponentsWithStats(skirt)
                pos = np.unravel_index(
                    np.argmax(np.where(labels == index, signal, -1000)), signal.shape
                )
                tag = sk[pos]
                if tag:
                    x, y, w, h, outer = st[tag]
                    inner = np.count_nonzero((sk == tag) & core.astype(bool))
                    if x > 0 and y > 0 and x + w < n and y + h < n and inner > 4:
                        result["shadow_softness"] = (
                            math.sqrt(outer) - math.sqrt(inner)
                        ) / math.sqrt(math.pi)
    return result


def blur_metric(im):
    g = gray(im)
    return cv.GaussianBlur(g, (0, 0), 3.0)


def wrapped(a, b, key):
    return ((a - b + 90) % 180 - 90) if key in ("angle", "grain_axis") else a - b


def correlation(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    return float(np.dot(a, b) / max(1.0, np.linalg.norm(a) * np.linalg.norm(b)))


def attachment(image, other, box, next_box):
    if math.dist(box[:2], next_box[:2]) < 1.8:
        return None

    def hp(im, b):
        p = gray(region(im, b))
        return p - cv.GaussianBlur(p, (0, 0), 1.2)

    a = hp(image, box)
    if a.std() < 3:
        return None
    return correlation(a, hp(other, next_box)) - correlation(a, hp(other, box))


def compression_control(pairs, mid):
    """Test whether diffuse, mostly high-frequency residuals resemble re-encoding the source."""
    a, b = pairs[mid]
    raw = gray(a) - gray(b)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    energy = rms(raw)
    tiles = [
        rms(t)
        for row in np.array_split(raw, 8)
        for t in np.array_split(row, 10, axis=1)
    ]
    if energy < 1 or np.mean(np.asarray(tiles) > 1.5) < 0.5:
        return None
    low = cv.GaussianBlur(raw, (0, 0), 3)
    if rms(low) > 0.36 * energy or np.percentile(abs(low), 99) > 8:
        return None
    # Use native pixels, bounding the control encode cost on large frames.
    h, w = a.shape[:2]
    cw = min(640, w) // 2 * 2
    ch = min(480, h) // 2 * 2
    x = (w - cw) // 2
    y = (h - ch) // 2
    clips = [l[y : y + ch, x : x + cw] for l, _ in pairs]
    raw_control = b"".join(z.tobytes() for z in clips)
    args = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "1"]
    try:
        encoded = subprocess.run(
            args
            + [
                "-f",
                "rawvideo",
                "-s",
                f"{cw}x{ch}",
                "-pix_fmt",
                "bgr24",
                "-r",
                "25",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-crf",
                "28",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "1",
                "-f",
                "h264",
                "pipe:1",
            ],
            input=raw_control,
            capture_output=True,
            check=True,
        ).stdout
        decoded = subprocess.run(
            args
            + [
                "-f",
                "h264",
                "-i",
                "pipe:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ],
            input=encoded,
            capture_output=True,
            check=True,
        ).stdout
        control = np.frombuffer(decoded, np.uint8).reshape(len(clips), ch, cw, 3)[mid]
    except (ValueError, OSError, subprocess.CalledProcessError):
        return None
    reference = clips[mid].astype(np.float32)
    observed = reference - b[y : y + ch, x : x + cw]
    simulated = reference - control
    similarity = correlation(simulated, observed)
    if similarity >= 0.45 and 0.28 <= rms(observed) / max(0.1, rms(simulated)) <= 3.5:
        return {
            "residual_correlation": round(similarity, 3),
            "rms": round(energy, 2),
            "lowpass_rms": round(rms(low), 2),
        }
    return None


def object_proposals(image):
    h, w = image.shape[:2]
    scale = min(1.0, 480 / max(h, w))
    small = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
    outlines = cv.Canny(cv.GaussianBlur(small, (5, 5), 1), 45, 110)
    contours, _ = cv.findContours(
        cv.morphologyEx(outlines, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE,
    )
    regions = []
    for contour in contours:
        x, y, bw, bh = cv.boundingRect(contour)
        if (
            0.008 * small.shape[0] * small.shape[1]
            < cv.contourArea(contour)
            < 0.75 * small.shape[0] * small.shape[1]
            and cv.contourArea(contour) > 0.4 * bw * bh
        ):
            regions.append(
                {"box": [v / scale for v in (x, y, bw, bh)], "source": "closed_contour"}
            )
    return regions


def findings_for_window(pairs, mid, times, out=None, prefix="local", objects=None):
    """Synchronized native-resolution pairs within one shot; neighbors establish local uncertainty."""
    a, b = pairs[mid]
    h, w = a.shape[:2]
    if any(l.shape != a.shape or r.shape != a.shape for l, r in pairs):
        return {
            "status": "insufficient",
            "reason": "原生画幅不同，像素尺度不可直接比较",
            "findings": [],
        }
    if min(h, w) < 32:
        return {"status": "insufficient", "reason": "空间采样不足", "findings": []}
    if len(pairs) < 2:
        return {"status": "insufficient", "reason": "缺少同镜邻帧", "findings": []}
    if all(np.array_equal(l, r) for l, r in pairs):
        return {"status": "matched", "findings": [], "reason": "采样帧逐像素一致"}
    codec = compression_control(pairs, mid)
    if codec:
        return {
            "status": "insufficient",
            "reason": "跨区域残差与原片重编码实验相似，材质和编码影响暂时无法分离。",
            "encoding_control": codec,
            "findings": [],
        }
    low = [(blur_metric(l), blur_metric(r)) for l, r in pairs]
    results = []
    rejected = []
    valid = 0
    weakly_tracked = 0
    moved = []
    sectors = set()
    measured = set()
    supported = set()
    objects = objects or object_proposals(a)
    for box in candidates(a, b):
        aligned = match(*low[mid], box)
        if aligned is None:
            rejected.append(box)
            continue
        rb, confidence, error = aligned
        valid += 1
        center = box_center(box)
        sector = (min(3, int(4 * center[0] / w)), min(3, int(4 * center[1] / h)))
        measured.add(sector)
        enclosing = [
            o
            for o in objects
            if coverage(box, o["box"]) >= 0.85
            and o["box"][2] >= box[2]
            and o["box"][3] >= box[3]
        ]
        obj = (
            min(enclosing, key=lambda o: o["box"][2] * o["box"][3])
            if enclosing
            else None
        )
        displacement = math.dist(box[:2], rb[:2])
        if displacement > max(2.0, box[2] * 0.05):
            moved.append(
                {
                    "roi": box,
                    "render_roi": rb,
                    "displacement_px": round(displacement, 2),
                }
            )
        A = region(a, box)
        B = region(b, rb)
        if np.mean(cv.absdiff(A, B)) < 0.55:
            continue
        ag, bg = gray(A), gray(B)
        residual = ag - bg
        occupied = float(np.mean(np.abs(residual) > 2))
        # A blur can remove one harmonic and reveal another without changing texture scale.
        # Compare optical transfer of aligned luminance in both directions.
        initial = float(np.mean(abs(residual[4:-4, 4:-4])))
        blur_fit = 0.0
        if initial > 1.0:
            for k in (0.9, 1.8, 3.2):
                for sharp, soft in ((ag, bg), (bg, ag)):
                    err = float(
                        np.mean(
                            abs((cv.GaussianBlur(sharp, (0, 0), k) - soft)[4:-4, 4:-4])
                        )
                    )
                    blur_fit = max(blur_fit, 1 - err / initial)
        data = [(mid, patch_metrics(A), patch_metrics(B))]
        for i, (left, right) in enumerate(pairs):
            if i == mid:
                continue
            lk = match(low[mid][0], low[i][0], box)
            rk = match(low[mid][1], low[i][1], rb)
            if not lk or not rk:
                continue
            cross = match(*low[i], lk[0])
            if not cross or math.dist(cross[0][:2], rk[0][:2]) > max(3, box[2] * 0.05):
                continue
            lm, rm = (
                patch_metrics(region(left, lk[0])),
                patch_metrics(region(right, rk[0])),
            )
            for im, oth, cur, nxt, m in [
                (a, left, box, lk[0], lm),
                (b, right, rb, rk[0], rm),
            ]:
                att = (
                    attachment(im, oth, cur, nxt)
                    if min(lk[1], rk[1]) > 0.88
                    and math.dist(
                        np.subtract(lk[0][:2], box[:2]), np.subtract(rk[0][:2], rb[:2])
                    )
                    < 1.5
                    else None
                )
                if att is not None:
                    m["attachment"] = att
            data.append((i, lm, rm))
        if len(data) < 2:
            weakly_tracked += 1
        noise_floor = 0.25 * np.median(np.abs(ag - cv.medianBlur(ag, 3)))
        for key, (floor, frac, unit, label, trial) in SPECS.items():
            rows = [(i, l[key], r[key]) for i, l, r in data if key in l and key in r]
            if key in ("highlight_width", "shadow_width", "shadow_softness"):
                landmark = (
                    "highlight_width_center"
                    if key == "highlight_width"
                    else "shadow_width_center"
                )
                rows = [
                    (i, l[key], r[key])
                    for i, l, r in data
                    if key in l
                    and key in r
                    and math.dist(l[landmark], r[landmark]) < max(3.0, box[2] * 0.09)
                ]
            if (
                key in ("period", "angle", "periodicity", "grain_axis", "grain_scale")
                and blur_fit > 0.32
            ):
                continue
            if len(rows) < 2 or (key != "attachment" and rows[0][0] != mid):
                continue
            supported.add(key)
            left = np.array([v for _, v, _ in rows])
            right = np.array([v for _, _, v in rows])
            ref = float(left[0] + np.median(wrapped(left, left[0], key)))
            differences = wrapped(right, left, key)
            delta = float(np.median(differences))
            scatter = float(np.median(abs(wrapped(left, ref, key))))
            # Video quantization noise and registration uncertainty bound what is observable.
            noise = noise_floor
            tol = max(floor, frac * abs(ref), 3 * scatter)
            if key in ("fine", "tone", "gradient"):
                tol += float(noise)
            if unit == "px":
                tol += 0.35 * error
            agree = np.mean(np.sign(differences) == np.sign(delta))
            if abs(delta) <= tol or agree < 0.75:
                continue
            observed = sorted(times[i] for i, _, _ in rows)
            salience = 1 + min(
                0.55,
                float(
                    np.std(
                        low[mid][0][box[1] : box[1] + box[3], box[0] : box[0] + box[2]]
                    )
                )
                / 90,
            )
            if obj and obj.get("layer") == "FG_HERO":
                salience *= 1.3
            priority = (
                math.log1p(abs(delta) / tol)
                * confidence
                * agree
                * salience
                * (1 + min(0.5, observed[-1] - observed[0]))
            )
            specificity = {
                "angle": 2.6,
                "period": 2.6,
                "highlight_width": 3.0,
                "shadow_width": 2.0,
                "edge_width": 1.7,
            }
            priority *= specificity.get(key, 1.0) * (0.55 + 0.65 * occupied)
            if key == "edge_width" and blur_fit > 0.32:
                priority *= 1 + 2 * blur_fit
            if key == "edge_width":
                sectors.add(sector)
            results.append(
                {
                    "metric": key,
                    "label": label,
                    "unit": unit,
                    "direction": "增加" if delta > 0 else "减少",
                    "reference_range": [
                        round(float(left.min()), 3),
                        round(float(left.max()), 3),
                    ],
                    "render_value": round(
                        float(
                            (ref + delta) % 180
                            if key in ("angle", "grain_axis")
                            else np.median(right)
                        ),
                        3,
                    ),
                    "difference": round(delta, 3),
                    "resolution_floor": round(tol, 3),
                    "roi": box,
                    "render_roi": rb,
                    "relative_to_frame": [round(box[2] / w, 4), round(box[3] / h, 4)],
                    "reference_roi_fraction": round(ref / box[2], 4)
                    if unit == "px"
                    else None,
                    "object_extent": obj,
                    "reference_fraction_of_object_width": round(ref / obj["box"][2], 5)
                    if obj and unit == "px"
                    else None,
                    "match_error_px": round(error, 2),
                    "confidence": round(confidence * agree, 3),
                    "time_s": times[mid],
                    "observed_interval_s": [observed[0], observed[-1]],
                    "samples": len(rows),
                    "priority": round(priority, 4),
                    "cause": WHY.get(key, label + "的局部图像统计发生可复现变化"),
                    "trial": trial,
                }
            )
    ranked = []
    for f in sorted(results, key=lambda v: -v["priority"]):
        if any(
            f["metric"] == p["metric"] and coverage(f["roi"], p["roi"]) > 0.28
            for p in ranked
        ):
            continue
        if sum(p["metric"] == f["metric"] for p in ranked) >= 3:
            continue
        ranked.append(f)
        if len(ranked) >= 12:
            break
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        scale = min(1.0, 760 / w)
        locator = cv.resize(a, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        for i, f in enumerate(ranked):
            name = f"{prefix}-{i + 1:02}.png"
            if not cv.imwrite(
                str(out / name),
                np.hstack([region(a, f["roi"]), region(b, f["render_roi"])]),
            ):
                raise OSError(name)
            f["crop"] = name
            x, y, bw, bh = [round(v * scale) for v in f["roi"]]
            cv.rectangle(locator, (x, y), (x + bw - 1, y + bh - 1), (0, 210, 255), 1)
            cv.putText(
                locator,
                str(i + 1),
                (x + 1, y + 13),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
            )
            cv.putText(
                locator,
                str(i + 1),
                (x + 1, y + 13),
                cv.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 230, 255),
                1,
            )
        if ranked and not cv.imwrite(str(out / f"{prefix}-positions.jpg"), locator):
            raise OSError("无法写入全帧位置图")
    return {
        "status": (
            "differences"
            if ranked
            else "limited"
            if rejected or weakly_tracked or valid == 0
            else "no_stable_difference"
        ),
        "locator": f"{prefix}-positions.jpg" if out and ranked else None,
        "time_s": times[mid],
        "native_size": [w, h],
        "matched_regions": valid,
        "unmatched_regions": len(rejected),
        "insufficient_tracks": weakly_tracked,
        "unresolved_features": [
            label
            for key, label in (
                ("highlight_width", "高光形状"),
                ("shadow_softness", "暗斑软边"),
                ("attachment", "纹理相对运动"),
            )
            if key not in supported
        ],
        "geometry_first": rejected[:8],
        "displaced_regions": moved[:8],
        "clarity_coverage": {
            "sectors_with_edge_width_difference": len(sectors),
            "comparable_sectors": len(measured),
            "grid": [4, 4],
        },
        "findings": ranked,
    }


class Video:
    def __init__(self, path):
        from decompose import load_source

        self.path = Path(path)
        frames, stream = load_source(self.path)
        self.times = np.array([f["time_s"] for f in frames], dtype=float)
        if not np.isfinite(self.times).all() or (np.diff(self.times) < 0).any():
            raise ValueError("缺少可靠的逐帧 PTS")
        self.transfer = stream.get("color_transfer", "unknown")
        self.rotation = next(
            (
                int(x.get("rotation", 0)) % 360
                for x in stream.get("side_data_list", [])
                if "rotation" in x
            ),
            0,
        )
        fps = stream["avg_frame_rate"].split("/")
        self.step = (
            float(np.median(np.diff(self.times)))
            if len(frames) > 1
            else int(fps[1]) / max(1, int(fps[0]))
        )
        self.cap = cv.VideoCapture(str(path))
        self.cap.set(cv.CAP_PROP_ORIENTATION_AUTO, 0)
        if not self.cap.isOpened():
            raise ValueError(f"无法解码 {path}")

    def read(self, i):
        if int(self.cap.get(cv.CAP_PROP_POS_FRAMES)) != i:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ok, im = self.cap.read()
        if not ok:
            raise ValueError(f"无法读取帧 {i}，{self.path}")
        return im

    def nearest(self, t):
        j = int(np.searchsorted(self.times, t))
        return min(
            {min(j, len(self.times) - 1), max(j - 1, 0)},
            key=lambda i: abs(self.times[i] - t),
        )


def similar_shot(a, b):
    x = cv.resize(a, (48, 32), interpolation=cv.INTER_AREA)
    y = cv.resize(b, (48, 32), interpolation=cv.INTER_AREA)
    hist = lambda v: cv.calcHist([v], [0, 1, 2], None, [6, 6, 6], [0, 256] * 3)
    distance = cv.compareHist(hist(x), hist(y), cv.HISTCMP_BHATTACHARYYA)
    difference = cv.absdiff(x, y).mean()
    return difference < 24 or (distance < 0.65 and difference < 60)


def analyze_at(
    source,
    render,
    index,
    render_index,
    out,
    prefix,
    bounds,
    objects=None,
    time_offset=0.0,
):
    t = float(source.times[index])
    if abs(t + time_offset - render.times[render_index]) > max(
        0.002, 0.55 * max(source.step, render.step)
    ):
        return {
            "status": "insufficient",
            "reason": "源片与渲染 PTS 未对齐",
            "findings": [],
            "time_s": t,
        }
    if (
        source.rotation != render.rotation
        or source.transfer in ("arib-std-b67", "smpte2084")
        or render.transfer in ("arib-std-b67", "smpte2084")
    ):
        return {
            "status": "insufficient",
            "reason": "显示旋转不同或使用 HDR 传递函数，需先统一为同一显示空间。",
            "findings": [],
            "time_s": t,
        }
    anchor = (source.read(index), render.read(render_index))
    rows = []
    seen = set()
    offset = max(2, round(0.16 / max(0.001, source.step)))
    for j in sorted(
        {
            max(bounds[0], index - offset),
            index - 1,
            index,
            index + 1,
            min(bounds[1] - 1, index + offset),
        },
        key=lambda j: (abs(j - index), j),
    ):
        if not bounds[0] <= j < bounds[1]:
            continue
        k = (
            render_index
            if j == index
            else render.nearest(source.times[j] + time_offset)
        )
        if k in seen or abs(render.times[k] - (source.times[j] + time_offset)) > max(
            0.002, 0.55 * max(source.step, render.step)
        ):
            continue
        p = anchor if j == index else (source.read(j), render.read(k))
        if j != index and not all(similar_shot(a, b) for a, b in zip(anchor, p)):
            continue
        rows.append((j, k, p))
        seen.add(k)
    rows.sort(key=lambda row: row[0])
    mid = next(i for i, row in enumerate(rows) if row[0] == index)
    result = findings_for_window(
        [r[2] for r in rows],
        mid,
        [float(source.times[r[0]]) for r in rows],
        out,
        prefix,
        objects,
    )
    result.update(
        source_frame=index,
        render_frame=render_index,
        render_time_s=float(render.times[render_index]),
        render_time_offset_s=time_offset,
        time_s=t,
        frame_pairs=[[j, k] for j, k, _ in rows],
    )
    return result


def compare_metadata(ref, rep, out, *, reference_video=None, render_video=None):
    from decompose import verify_source

    def path(info, override):
        identity = info.get("source", {})
        p = override if override is not None else Path(identity.get("path", ""))
        if not p.is_file() or p.stat().st_size != identity.get(
            "bytes", p.stat().st_size
        ):
            raise ValueError("原生视频不可用或大小与登记值不符")
        if override is not None or identity.get("sha256"):
            verify_source(p, identity)
        return p

    from contextlib import ExitStack

    try:
        with ExitStack() as stack:
            a = Video(path(ref, reference_video))
            stack.callback(a.cap.release)
            b = Video(path(rep, render_video))
            stack.callback(b.cap.release)
            i, j = int(ref["target_frame"]), int(rep["target_frame"])
            if not (0 <= i < len(a.times) and 0 <= j < len(b.times)):
                raise ValueError("帧号超出视频")
            objects = []
            aw, ah = ref.get("analysis_size") or (0, 0)
            sw, sh = (
                a.cap.get(cv.CAP_PROP_FRAME_WIDTH),
                a.cap.get(cv.CAP_PROP_FRAME_HEIGHT),
            )
            if aw and ah:
                for obj in (ref.get("objects_3d_bom") or {}).get("objects", []):
                    if len(obj.get("bbox_px", [])) != 4:
                        continue
                    x, y, X, Y = obj["bbox_px"]
                    if X > x and Y > y:
                        objects.append(
                            {
                                "box": [
                                    x * sw / aw,
                                    y * sh / ah,
                                    (X - x) * sw / aw,
                                    (Y - y) * sh / ah,
                                ],
                                "source": "scene3d",
                                "id": obj.get("id"),
                                "layer": obj.get("depth_layer"),
                            }
                        )
            return analyze_at(
                a,
                b,
                i,
                j,
                out,
                "lookdiff",
                (0, len(a.times)),
                objects,
                float(b.times[j] - a.times[i]),
            )
    except (OSError, ValueError, TypeError, KeyError, cv.error) as exc:
        return {"status": "insufficient", "reason": str(exc), "findings": []}
    except subprocess.CalledProcessError:
        return {
            "status": "insufficient",
            "reason": "ffprobe 无法读取来源",
            "findings": [],
        }


def markdown(report):
    lines = [
        "",
        "## 局部质感差异",
        "",
        f"状态 `{report['status']}`。" + report.get("reason", ""),
    ]
    meanings = {
        "matched": "采样帧一致",
        "insufficient": "证据不足",
        "limited": "部分选区证据不足",
        "differences": "发现可复测差异",
        "no_stable_difference": "采样范围内未发现稳定偏差",
        "overview": "跨镜头排序",
    }
    lines[-1] = f"{meanings.get(report['status'], report['status'])}。" + report.get(
        "reason", ""
    )
    if "time_s" in report:
        lines[-1] = f"{report['time_s']:.3f} 秒。" + lines[-1]
    if report.get("locator"):
        lines.append(f"[查看全帧取样位置]({report['locator']})")
    if report.get("displaced_regions"):
        largest = max(report["displaced_regions"], key=lambda x: x["displacement_px"])
        lines.append(
            f"配准时有区域平移至多 {largest['displacement_px']} px，先核对构图、物体位置及遮挡。"
        )
    if report.get("unmatched_regions"):
        lines.append(
            f"有 {report['unmatched_regions']} 个候选区域配准不可靠，先核对几何或遮挡。"
        )
    if report.get("insufficient_tracks"):
        lines.append(f"{report['insufficient_tracks']} 个选区未能在邻帧可靠复测。")
    scope = report.get("clarity_coverage", {})
    if scope.get("sectors_with_edge_width_difference"):
        lines.append(
            f"清晰度差异落在 4×4 网格中的 {scope['sectors_with_edge_width_difference']} 格，本次成功对位 {scope['comparable_sectors']} 格。"
        )
    if report.get("unresolved_features"):
        lines.append(
            "存在残差的可比区域中，以下属性缺少可追踪证据："
            + "、".join(report["unresolved_features"])
            + "。"
        )
    for f in report["findings"]:
        scale = (
            f"相当于候选物体宽度的 {100 * f['reference_fraction_of_object_width']:.2f}%。"
            if f.get("reference_fraction_of_object_width") is not None
            else ""
        )
        crop = f" [原尺寸对照]({f['crop']})" if "crop" in f else ""
        lines.extend(
            [
                "",
                f"- {f['time_s']:.3f} 秒，区域 {list(f['roi'])}，{f['samples']} 个采样时刻覆盖 {f['observed_interval_s'][0]:.3f}–{f['observed_interval_s'][1]:.3f} 秒。**{f['label']}{f['direction']}**。原片 {f['reference_range']}，渲染 {f['render_value']} {f['unit']}，检测下限 {f['resolution_floor']} {f['unit']}，配准往返误差 {f['match_error_px']} px。{scale}{f['cause']}。{f['trial']}{crop}",
            ]
        )
    return lines


def compare_movies(ref, rep, out):
    from contextlib import ExitStack

    from decompose import raw_frames

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        a = Video(ref)
        stack.callback(a.cap.release)
        b = Video(rep)
        stack.callback(b.cap.release)
        prev = None
        cuts = [0]
        for i, im in enumerate(raw_frames(Path(ref), (64, 40))):
            if prev is not None and not similar_shot(prev, im):
                cuts.append(i)
            prev = im
        if prev is None or i + 1 != len(a.times):
            raise ValueError("FFmpeg 解码帧数和逐帧 PTS 数量不一致")
        cuts.append(len(a.times))
        windows = []
        for start, stop in itertools.pairwise(cuts):
            first = start
            while first < stop:
                end = min(
                    stop,
                    max(first + 1, int(a.times.searchsorted(a.times[first] + 0.4))),
                )
                i = (first + end - 1) // 2
                r = analyze_at(
                    a, b, i, b.nearest(a.times[i]), out, f"look-{i:07}", (start, stop)
                )
                r["shot_frames"] = [start, stop]
                windows.append(r)
                first = end
    ranked = sorted(
        [f for r in windows for f in r["findings"]], key=lambda f: -f["priority"]
    )
    data = {
        "source": str(ref),
        "render": str(rep),
        "shot_count": len(cuts) - 1,
        "windows": windows,
        "ranked_findings": ranked,
    }
    (out / "lookdiff.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 跨片局部质感诊断",
        f"\n{len(cuts) - 1} 个候选镜头，{len(windows)} 个测量窗口。",
    ]
    lines += markdown({"status": "overview", "findings": ranked[:8]})
    for r in windows:
        lines += markdown(r)
    (out / "lookdiff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data
