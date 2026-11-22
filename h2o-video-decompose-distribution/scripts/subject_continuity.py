"""Short-term appearance changes and trajectories of visible colour components."""

import cv2 as cv
import numpy as np


def patch_signature(residual, flow, before, aligned):
    # The finer lattice preserves small features that occupy a fraction of a v2 cell.
    h, w = residual.shape[:2]
    size = (max(4, round(w / 12)), max(4, round(h / 12)))

    def pool(a):
        return cv.resize(a.astype(np.float32), size, interpolation=cv.INTER_AREA)

    diff = np.max(abs(residual), axis=2)
    edge = cv.Canny(before, 25, 70) > 0
    missing = cv.dilate(cv.Canny(aligned, 25, 70), np.ones((3, 3), np.uint8)) == 0
    yy, xx = np.indices((h, w))
    valid = (
        (xx + flow[:, :, 0] >= 1)
        & (xx + flow[:, :, 0] < w - 2)
        & (yy + flow[:, :, 1] >= 1)
        & (yy + flow[:, :, 1] < h - 2)
    )
    # Variance distinguishes lettering sweeping over a patch from the interior of a surface.
    luminance = cv.cvtColor(before, cv.COLOR_BGR2GRAY).astype(np.float32)
    deviation = np.sqrt(
        np.maximum(0, pool(luminance * luminance) - pool(luminance) ** 2)
    )
    return np.dstack(
        [
            pool((diff > 14) & valid),
            pool(np.minimum(diff, 64) * valid) / 64,
            pool(edge & missing & valid),
            pool(flow[:, :, 0]) / w * size[0],
            pool(flow[:, :, 1]) / h * size[1],
            pool(np.max(before, axis=2)),
            pool(np.mean(residual, axis=2) * valid),
            pool(before),
            deviation,
            pool(residual * valid[:, :, None]),
        ]
    ).astype(np.float16)


def components(image):
    small = cv.resize(
        image,
        (max(8, image.shape[1] // 3), max(8, image.shape[0] // 3)),
        interpolation=cv.INTER_AREA,
    )
    hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    hue, sat, value = cv.split(hsv)
    visible = (sat > 55) & (value > 32)
    result = []
    for band in range(8):
        distance = abs(hue.astype(np.int16) - round(180 * band / 8))
        mask = visible & (np.minimum(distance, 180 - distance) < 15)
        n, _, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for x, y, bw, bh, area in stats[1:n]:
            if area < max(6, h * w * 0.00045) or area > h * w * 0.32:
                continue
            result.append(
                (band, np.array([x / w, y / h, bw / w, bh / h]), area / (h * w))
            )
    return result


def center(box):
    return box[:2] + box[2:] * 0.5


def trace(rows):
    active = []
    tracks = []
    for i, row in enumerate(rows):
        if row["structure"] > 0.24 or abs(row["brightness"]) > 18:
            active = []
        blobs = row["blobs"]
        pairs = []
        for track in active:
            _, band, box, area = track[-1]
            pred = center(box)
            if len(track) > 1:
                pred = pred + (pred - center(track[-2][2]))
            for j, (other, rect, mass) in enumerate(blobs):
                if band != other:
                    continue
                dist = float(np.linalg.norm(pred - center(rect)))
                if (
                    dist < min(0.15, 0.04 + max(box[2:]) * 0.5)
                    and 0.15 < mass / area < 7
                ):
                    pairs.append(
                        (dist + 0.017 * abs(np.log(mass / area)), id(track), track, j)
                    )
        used = set()
        assigned = set()
        new = []
        for _, key, track, j in sorted(pairs, key=lambda v: v[:2]):
            if key in used or j in assigned:
                continue
            track.append((i, *blobs[j]))
            new.append(track)
            used.add(key)
            assigned.add(j)
        for j, blob in enumerate(blobs):
            if j not in assigned:
                track = [(i, *blob)]
                new.append(track)
                tracks.append(track)
        active = new
    return tracks


def entries(tracks, rows):
    result = []
    for tr in tracks:
        i, band, box, area = tr[0]
        if i == 0:
            continue
        seg = [r for r in tr if rows[r[0]]["t"] - rows[i]["t"] <= 0.45]
        if len(seg) < 4 or rows[seg[-1][0]]["t"] - rows[i]["t"] < 0.12:
            continue
        edges = [box[0], box[1], 1 - box[0] - box[2], 1 - box[1] - box[3]]
        side = int(np.argmin(edges))
        axis = side % 2
        if edges[side] > 0.055:
            continue
        peak = max(r[3] for r in seg)
        delta = center(seg[-1][2]) - center(box)
        if (
            peak < max(0.01, 2.3 * area)
            or delta[axis] * (1 if side < 2 else -1) < 0.025
        ):
            continue
        result.append((i, band, side, box, peak, delta))
    return result


def subject_events(ref, rep):
    out = {}
    times = np.array([r["t"] for r in ref])

    def add(i, kind, box, **details):
        out.setdefault(i, []).append(
            dict(
                kind=kind,
                status="待复核",
                bbox=box.tolist(),
                region="主体框 " + ",".join(f"{x:.3f}" for x in box),
                **details,
            )
        )

    tracks = trace(rep)
    other = entries(trace(ref), ref)
    for i, band, side, box, area, direction in entries(tracks, rep):
        matches = []
        for j, b, s, rect, mass, vec in other:
            if b != band or s != side or abs(times[j] - rep[i]["t"]) > 0.55:
                continue
            dist = np.linalg.norm(center(rect) - center(box))
            if dist > 0.24 or np.dot(direction, vec) <= 0 or not 0.5 < mass / area < 2:
                continue
            if np.linalg.norm(direction - vec) > 0.08 + 0.55 * np.linalg.norm(
                direction
            ):
                continue
            matches.append(
                (
                    dist
                    + 0.15 * abs(np.log(mass / area))
                    + 0.25 * np.linalg.norm(direction - vec),
                    j,
                    rect,
                )
            )
        if not matches:
            continue
        matches.sort(key=lambda a: a[0])
        if len(matches) > 1 and matches[1][0] - matches[0][0] < 0.035:
            continue
        _, j, rect = matches[0]
        offset = rep[i]["t"] - times[j]
        dt = max(rep[i]["dt"], ref[j]["dt"])
        if abs(offset) <= max(2 * dt, 0.085):
            continue
        add(
            i,
            "主体入画时机偏差",
            box,
            reference_bbox=rect.tolist(),
            reference_event_time=ref[j]["t"],
            reference_event_frame=ref[j]["frame"],
            offset_seconds=round(offset, 4),
            offset_interval=[
                round(offset - ref[j]["dt"], 4),
                round(offset + rep[i]["dt"], 4),
            ],
            direction="提前入画" if offset < 0 else "延后入画",
        )
    for i, r in enumerate(rep):
        if (
            i < 2
            or i + 2 >= len(rep)
            or r["structure"] > 0.17
            or abs(r["brightness"]) > 10
        ):
            continue
        j = int(np.argmin(abs(times - r["t"])))
        cur = r["signature"].astype(np.float32)
        h, w = cur.shape[:2]
        around = [
            p["signature"].astype(np.float32)
            for p in rep[i - 2 : i] + rep[i + 1 : i + 3]
        ]
        yy, xx = np.indices((h, w), dtype=np.float32)
        # Follow each cell across neighbouring exposures using its local displacement.
        near = np.maximum.reduce(
            [
                cv.remap(
                    a[:, :, :3],
                    xx + step * cur[:, :, 3],
                    yy + step * cur[:, :, 4],
                    cv.INTER_LINEAR,
                    borderMode=cv.BORDER_REPLICATE,
                )
                for step, a in zip((-2, -1, 1, 2), around)
            ]
        )
        refsig = cv.resize(ref[j]["signature"].astype(np.float32)[:, :, :3], (w, h))
        expected = cv.dilate(refsig, np.ones((3, 3), np.uint8))
        surprise = cur[:, :, :3] - np.maximum(near, expected)
        mask = (
            (surprise[:, :, 0] > 0.19)
            & (surprise[:, :, 1] > 0.068)
            & ((surprise[:, :, 2] > 0.007) | (surprise[:, :, 1] > 0.14))
        )
        n, _, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for x, y, bw, bh, area in stats[1:n]:
            if (
                area < 3
                or bw < 2
                or bh < 2
                or area > h * w * 0.10
                or x == 0
                or y == 0
                or x + bw == w
                or y + bh == h
            ):
                continue
            box = np.array([x / w, y / h, bw / w, bh / h])
            add(
                i,
                "局部外观跨帧跳变",
                box,
                direction="轮廓或材质变化",
                reference_bbox=box.tolist(),
                reference_event_time=ref[j]["t"],
                reference_event_frame=ref[j]["frame"],
            )

        # Neutral brightness changes preserve opponent colour. A material swap in
        # deep shadow can change chroma substantially with almost no luminance step.
        def chroma_step(sig):
            delta = sig[:, :, 11:14].astype(np.float32)
            delta -= delta.mean(axis=2, keepdims=True)
            return np.linalg.norm(delta, axis=2) / (
                16 + sig[:, :, 5].astype(np.float32)
            )

        shift = chroma_step(cur)
        usual = np.median([chroma_step(a) for a in around], axis=0)
        original = cv.resize(chroma_step(ref[j]["signature"]), (w, h))
        changed = (
            (shift > 0.105)
            & (shift - usual > 0.065)
            & (shift - original > 0.06)
            & (cur[:, :, 10] < 18)
        )
        n, _, stats, _ = cv.connectedComponentsWithStats(changed.astype(np.uint8), 8)
        for x, y, bw, bh, area in stats[1:n]:
            if (
                area < 5
                or area > w * h * 0.06
                or min(bw, bh) < 2
                or area / (bw * bh) < 0.40
            ):
                continue
            box = np.array([x / w, y / h, bw / w, bh / h])
            add(
                i,
                "局部外观跨帧跳变",
                box,
                direction="连续表面的相对色度突变",
                reference_bbox=box.tolist(),
                reference_event_time=ref[j]["t"],
                reference_event_frame=ref[j]["frame"],
            )
        # Interior appearance can change on a dim, flat part without losing any edge.
        # Require a signed colour step, previously steady shading and a coherent cluster.
        shade = np.stack(
            [
                p["signature"][:, :, 6].astype(np.float32)
                for p in rep[i - 2 : i] + rep[i + 1 : i + 3]
            ]
        )
        excursion = abs(cur[:, :, 6]) - np.median(abs(shade), axis=0)
        ref_step = cv.resize(
            abs(ref[j]["signature"][:, :, 6].astype(np.float32)), (w, h)
        )
        flat = (
            (cur[:, :, 5] < 105)
            & (cur[:, :, 10] < 16)
            & (excursion > 8)
            & (abs(cur[:, :, 6]) - ref_step > 5)
            & (cur[:, :, 0] > 0.15)
        )
        for polarity in (-1, 1):
            valid = flat & (polarity * cur[:, :, 6] > 0)
            n, _, stats, _ = cv.connectedComponentsWithStats(valid.astype(np.uint8), 8)
            for x, y, bw, bh, area in stats[1:n]:
                if (
                    not 5 <= area <= h * w * 0.07
                    or max(bw, bh) > 2.2 * min(bw, bh)
                    or bw < 2
                    or bh < 2
                    or area / (bw * bh) < 0.45
                ):
                    continue
                if (
                    np.median(np.linalg.norm(cur[y : y + bh, x : x + bw, 3:5], axis=2))
                    > 0.6
                ):
                    continue
                box = np.array([x / w, y / h, bw / w, bh / h])
                add(
                    i,
                    "局部外观跨帧跳变",
                    box,
                    direction="持续部件的明暗或材质阶跃",
                    reference_bbox=box.tolist(),
                    reference_event_time=ref[j]["t"],
                    reference_event_frame=ref[j]["frame"],
                )
    # A tracked fragment terminating in an interior cell with an unusual outgoing
    # residual warrants checking. Gradual contraction and departure at the border do not.
    for tr in tracks:
        i, _, box, area = tr[-1]
        if len(tr) < 5 or i + 2 >= len(rep) or rows_duration(tr, rep) < 0.12:
            continue
        if min(*box[:2], *(1 - box[:2] - box[2:])) < 0.035 or area < 0.0012:
            continue
        if area < 0.72 * np.median([t[3] for t in tr[-5:]]):
            continue
        r = rep[i + 1]
        j = int(np.argmin(abs(times - r["t"])))
        if max(r["structure"], ref[j]["structure"]) > 0.17:
            continue
        sig = r["signature"]
        h, w = sig.shape[:2]
        cx, cy = np.clip((center(box) * [w, h]).astype(int), [0, 0], [w - 1, h - 1])
        ys = slice(max(0, cy - 1), cy + 2)
        xs = slice(max(0, cx - 1), cx + 2)

        scores = [
            float(
                cv.resize(row["signature"][:, :, 0].astype(np.float32), (w, h))[
                    ys, xs
                ].mean()
            )
            for row in (r, rep[i], rep[i + 2], ref[j])
        ]
        if scores[0] - max(scores[1:]) < 0.16:
            continue
        add(
            i + 1,
            "遮挡处可见轨迹中断",
            box,
            direction="末次可追踪区域",
            last_visible_time=rep[i]["t"],
            last_visible_frame=rep[i]["frame"],
        )
    # Several short, dim fragments lost together can bound the last visible area
    # without claiming that the video proves the object's identity.
    fragments = {}
    for tr in tracks:
        i, band, box, area = tr[-1]
        if len(tr) < 2 or i + 2 >= len(rep) or not 0.0005 < area < 0.01:
            continue
        if min(*box[:2], *(1 - box[:2] - box[2:])) < 0.045:
            continue
        fragments.setdefault((i, band), []).append(box)
    for (i, _), boxes in fragments.items():
        r = rep[i + 1]
        j = int(np.argmin(abs(times - r["t"])))
        if len(boxes) < 2 or max(r["structure"], ref[j]["structure"]) > 0.17:
            continue
        sig = r["signature"]
        h, w = sig.shape[:2]
        for k, first in enumerate(boxes):
            for second in boxes[k + 1 :]:
                distance = np.linalg.norm(center(first) - center(second))
                if not 0.025 < distance < 0.14:
                    continue
                low = np.minimum(first[:2], second[:2])
                high = np.maximum(first[:2] + first[2:], second[:2] + second[2:])
                x, y = np.floor(low * [w, h]).astype(int)
                X, Y = np.ceil(high * [w, h]).astype(int)
                patch = sig[y:Y, x:X].astype(np.float32)
                peer = cv.resize(
                    ref[j]["signature"].astype(np.float32)[:, :, 0], (w, h)
                )[y:Y, x:X]
                if (
                    patch[:, :, 5].mean() > 70
                    or patch[:, :, 0].mean() < 0.07
                    or patch[:, :, 0].mean() - peer.mean() < 0.035
                ):
                    continue
                add(
                    i + 1,
                    "遮挡处可见轨迹中断",
                    np.r_[low, high - low],
                    direction="低可见度碎片同时失去追踪，主体身份待核",
                    last_visible_frame=rep[i]["frame"],
                    last_visible_time=rep[i]["t"],
                )
                break
            else:
                continue
            break
    # Re-encoding can alter flow even when corresponding visible patches agree.
    for i, items in out.items():
        row = rep[i]
        j = int(np.argmin(abs(times - row["t"])))
        kept = []
        for item in items:
            if item["kind"] == "主体入画时机偏差":
                kept.append(item)
                continue
            box = np.array(item["bbox"])
            differences = []
            for a, b in ((i, j), (min(i + 1, len(rep) - 1), min(j + 1, len(ref) - 1))):
                view = rep[a]["signature"][:, :, 7:10].astype(np.float32)
                h, w = view.shape[:2]
                peer = cv.resize(
                    ref[b]["signature"][:, :, 7:10].astype(np.float32), (w, h)
                )
                x, y = np.floor(box[:2] * [w, h]).astype(int)
                X, Y = np.ceil((box[:2] + box[2:]) * [w, h]).astype(int)
                differences.append(np.mean(abs(view[y:Y, x:X] - peer[y:Y, x:X])))
            if max(differences) > 9:
                kept.append(item)
        unique = {(item["kind"], item["region"]): item for item in kept}
        out[i] = list(unique.values())
    return out


def rows_duration(track, rows):
    return rows[track[-1][0]]["t"] - rows[track[0][0]]["t"]
