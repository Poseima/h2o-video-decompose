"""Local temporal evidence in display coordinates. No object labels or project assets."""

import cv2 as cv
import numpy as np

GRID = 8


def profile(before, after, flow, reverse, xx, yy):
    """For each patch, trace pixels to the other frame. Keep border reveals separate."""
    h, w = xx.shape
    x, y = xx + flow[:, :, 0], yy + flow[:, :, 1]
    valid = (x >= 1) & (x < w - 2) & (y >= 1) & (y < h - 2)
    sample = cv.remap(after, x, y, cv.INTER_LINEAR)
    error = np.max(cv.absdiff(before, sample), axis=2)
    cycle = np.linalg.norm(flow + cv.remap(reverse, x, y, cv.INTER_LINEAR), axis=2)
    edges = cv.Canny(before, 45, 115)
    targets = cv.dilate(cv.Canny(after, 45, 115), np.ones((5, 5), np.uint8))
    broken = (edges > 0) & (cv.remap(targets, x, y, cv.INTER_NEAREST) == 0)

    def pool(mask):
        return cv.resize(
            mask.astype(np.float32), (GRID, GRID), interpolation=cv.INTER_AREA
        )

    support = pool(valid)

    def fraction(mask):
        return pool(mask & valid) / np.maximum(support, 0.01)

    return np.stack(
        [
            fraction(cycle > 2.5),
            fraction(error > 36),
            pool(broken & valid) / np.maximum(0.04, pool((edges > 0) & valid)),
            support,
            pool(edges > 0),
        ],
        axis=-1,
    ).round(3)


def local_events(ref, rep, cadence):
    """Compare both directional profiles and follow a candidate in nearby exposures."""

    def arrays(rows):
        return np.array([r["patch_flow"] for r in rows], dtype=np.float32)

    a, b = arrays(ref), arrays(rep)
    ta, tb = (np.array([r["t"] for r in rows]) for rows in (ref, rep))
    result = {}

    def window(times, left, right):
        return slice(
            np.searchsorted(times, left), np.searchsorted(times, right, side="right")
        )

    for k, row in enumerate(rep):
        t, dt = row["t"], row["dt"]
        aligned = a[
            window(ta, t - max(cadence) * 0.55 - 1e-5, t + max(cadence) * 0.55 + 1e-5)
        ]
        j = min(len(ta) - 1, int(np.searchsorted(ta, t)))
        if not len(aligned) or row["structure"] > 0.25 or ref[j]["structure"] > 0.25:
            continue
        if abs(row["brightness"]) > 15 or abs(ref[j]["brightness"]) > 15:
            continue
        cur = b[k]
        expected = aligned.max(axis=0)
        excess = cur - expected
        span = max(0.16, 2 * dt)
        past = b[window(tb, t - span, t - 0.4 * dt)]
        future = b[window(tb, t + 0.4 * dt, t + span)]
        if not len(past) or not len(future):
            continue
        neighbors = np.concatenate((past, future))
        baseline = np.median(neighbors, axis=0)
        prev, post = past.max(axis=0), future.max(axis=0)

        def spread(m):
            return np.stack(
                [cv.dilate(layer, np.ones((3, 3), np.uint8)) for layer in m]
            )

        supported = cur[..., 3] > 0.55
        residual = excess[..., 1]
        edges = excess[..., 2]
        motion = np.linalg.norm(row["motion"][:2])
        # A sudden residual with lost contours can establish an actual replacement.
        strong = (
            (residual > 0.29)
            & (edges > 0.23)
            & (cur[..., 1] - baseline[..., 1] > 0.22)
            & (
                cur[..., 1] - np.minimum(spread(prev[..., 1]), spread(post[..., 1]))
                > 0.11
            )
        )
        # Tracking loss alone warrants review only. Require spatial & temporal contrast.
        ambiguous = (
            (excess[..., 0] > 0.47)
            & (cur[..., 0] > 0.62)
            & (cur[..., 0] - np.median(cur[..., 0], axis=(1, 2))[:, None, None] > 0.32)
            & ((cur[..., 0] - baseline[..., 0] > 0.16) | (edges > 0.18))
        )
        # A cycle error in a uniform panel has no trackable visual feature.
        # Permit an independent tracking warning on a texture-rich, persistent patch.
        texture = cur[..., 4] > 0.035
        next_change = np.maximum(prev[..., 1], post[..., 1])
        corroborated = (
            (residual > 0.07)
            | ((next_change > 0.14) & (cur[..., 1] > 0.05))
            | (
                (cur[..., 0] > 0.85)
                & (np.minimum(prev[..., 0], post[..., 0]) > 0.65)
                & (cur[..., 4] > 0.09)
            )
        )
        flags = (strong | (ambiguous & texture & corroborated)) & supported
        if flags.any(axis=0).mean() > 0.38:
            continue
        count, labels, stats, _ = cv.connectedComponentsWithStats(
            flags.any(axis=0).astype(np.uint8), 8
        )
        for n in range(1, count):
            x, y, w, h, area = map(int, stats[n])
            cells = labels == n
            interior = x > 0 and y > 0 and x + w < GRID and y + h < GRID
            direct = (
                strong[:, cells].any()
                and interior
                and ref[j]["support"] >= 3
                and motion < 0.015
                and row["support"] >= 3
                and area <= GRID * GRID // 5
                and not np.any(abs(np.array(row["regions"])[:, 1:4]) > 32)
            )
            # Strong motion near either boundary admits a physical occlusion explanation.
            entry = float(cur[1, :, :, 1][cells].max())
            leave = float(cur[0, :, :, 1][cells].max())
            direction = (
                "显露"
                if entry > leave + 0.10
                else "消失"
                if leave > entry + 0.10
                else "外观或位置变化"
            )
            bbox = [x / GRID, y / GRID, w / GRID, h / GRID]
            kind = "局部轮廓突变" if direct else "局部入画或遮挡待复核"
            result.setdefault(k, []).append(
                {
                    "kind": kind,
                    "status": "需返修" if direct else "待复核",
                    "region": f"局部框 {x},{y},{w},{h}",
                    "bbox": bbox,
                    "direction": direction,
                    "tracking_excess": round(float(excess[..., 0][:, cells].max()), 3),
                    "residual_excess": round(float(residual[:, cells].max()), 3),
                }
            )
    return result


BORDER_BOXES = [(x, y, 0.2, 0.5) for x in (0.0, 0.8) for y in (0.0, 0.5)] + [
    (x, y, 0.5, 0.2) for y in (0.0, 0.8) for x in (0.0, 0.5)
]


def border_palette(image):
    """Chroma occupancy at eight border segments, independent of individual models."""
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    rows = []
    for x, y, bw, bh in BORDER_BOXES:
        patch = hsv[int(y * h) : round((y + bh) * h), int(x * w) : round((x + bw) * w)]
        hue = patch[:, :, 0].astype(int) // 15
        use = (patch[:, :, 1] > 65) & (patch[:, :, 2] > 45)
        rows.append(np.bincount(hue[use], minlength=12) / max(1, use.size))
    return np.asarray(rows, dtype=np.float16)


def border_onsets(rows):
    times = np.asarray([row["t"] for row in rows])
    values = np.stack([row["border_palette"] for row in rows]).astype(float)
    events = []
    for i, t in enumerate(times):
        lo, hi = np.searchsorted(times, [t - 0.35, t + 0.35])
        if i - lo < 3 or hi - i < 3:
            continue
        a, b = values[lo : i + 1], values[i:hi]
        # Detect a sustained smooth rise through the middle of an entry.
        baseline = a[:2].mean(axis=0)
        delta = b[-2:].mean(axis=0) - baseline
        middle = baseline + delta * 0.5
        sign = np.sign(delta)
        rises = (
            (abs(delta) > 0.16)
            & ((a[-2] - middle) * sign < 0)
            & ((a[-1] - middle) * sign >= 0)
        )
        for zone, channel in zip(*np.where(rises)):
            curve = values[lo:hi, zone, channel]
            steps = np.diff(curve) * sign[zone, channel]
            change = abs(delta[zone, channel])
            if steps.max() > 0.48 * change or steps.clip(max=0).sum() < -0.17 * change:
                continue
            if any(
                r["structure"] > 0.22 or abs(r["brightness"]) > 15 for r in rows[lo:hi]
            ):
                continue
            if any(
                e[1:3] == (zone, channel) and t - times[e[0]] < 0.32
                for e in events[-32:]
            ):
                continue
            events.append((i, zone, channel, float(delta[zone, channel])))
    return events


def timing_events(ref, rep):
    """Comparable border reveals with a measurable onset offset. Ambiguous matches abstain."""
    reference = border_onsets(ref)
    result = {}
    for k, zone, channel, amplitude in border_onsets(rep):
        t = rep[k]["t"]
        choices = [
            j
            for j, z, c, a in reference
            if z == zone
            and c == channel
            and abs(ref[j]["t"] - t) < 0.48
            and 0.45 < a / amplitude < 2.2
        ]
        if len(choices) != 1:
            continue
        delta = t - ref[choices[0]]["t"]
        if abs(delta) <= max(0.09, 2.5 * max(rep[k]["dt"], ref[choices[0]]["dt"])):
            continue
        result.setdefault(k, []).append(
            {
                "kind": "连续入画时机偏差",
                "status": "待复核",
                "region": f"边缘区域 {zone + 1} 色带 {channel + 1}",
                "bbox": list(BORDER_BOXES[zone]),
                "direction": "同色区域显露" if amplitude > 0 else "同色区域被遮挡",
                "reference_event_time": ref[choices[0]]["t"],
                "offset_seconds": round(delta, 4),
            }
        )
    return result
