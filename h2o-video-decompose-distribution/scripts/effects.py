"""Reference-led, subject-specific effect checks. Times are relative display seconds."""

import bisect
import hashlib
import itertools
import json
import math
import re
import subprocess
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import cv2 as cv
import numpy as np
from decompose import load_source, raw_frames, write_image, write_json
from measure_core import flow, project, sample, solve_pair

PASS, FAIL, REVIEW = "通过", "需返修", "待复核"
MODES = {"motion", "perspective", "trail", "focus", "visibility", "impact", "manual"}
EPS = 1e-6


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def timeline(path):
    frames, stream = load_source(path)
    pts = np.array([f["time_s"] for f in frames], dtype=float)
    if not len(pts) or not np.isfinite(pts).all() or (np.diff(pts) <= 0).any():
        raise ValueError("缺少可靠的显示时间")
    last = frames[-1]["duration_s"] or (
        float(np.median(np.diff(pts))) if len(pts) > 1 else 0
    )
    rotation = next(
        (i["rotation"] for i in stream.get("side_data_list", []) if "rotation" in i), 0
    )
    if not last or rotation % 90:
        raise ValueError("无法确定显示时长或画面方向")
    return (
        pts,
        float(pts[-1] + last),
        (stream["width"], stream["height"]),
        int(rotation // 90),
        float(frames[0]["pts_s"]),
        bool(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=index",
                    "-of",
                    "csv=p=0",
                    str(path),
                ]
            ).strip()
        ),
    )


def track_box(spec, t):
    if isinstance(spec, list):
        return spec
    keys = spec["keys"]
    times, boxes = zip(*keys)
    return [float(np.interp(t, times, dim)) for dim in zip(*boxes)]


def validate(items):
    seen = set()
    if not isinstance(items, list) or not items:
        raise ValueError("效果清单为空")
    for e in items:
        if not re.fullmatch(r"[\w-]+", e["id"]) or e["id"] in seen:
            raise ValueError("效果标识须唯一且适合用作文件名")
        seen.add(e["id"])
        if not all(
            isinstance(e.get(k), str) and e[k].strip()
            for k in ("subject", "criterion", "evidence")
        ):
            raise ValueError("每项须写明主体、通过条件及原片依据")
        a, b = e["range"]
        if not 0 <= a < b or not np.isfinite(b) or e["mode"] not in MODES:
            raise ValueError("效果类型或时间范围无效")
        regions = e["regions"]
        if not regions or (e["mode"] == "focus" and len(regions) < 2):
            raise ValueError("须指定主体区域，景深至少指定两个层次")
        if e["mode"] == "impact" and len(regions) != 1:
            raise ValueError("单一击点须对应一个具名的动作区域")
        for region in regions:
            if not region["name"].strip():
                raise ValueError("区域须具名")
            for spec in [region["ref"], region.get("rep", region["ref"])]:
                keys = spec["keys"] if isinstance(spec, dict) else [(a, spec)]
                if not keys or any(x[0] >= y[0] for x, y in itertools.pairwise(keys)):
                    raise ValueError("区域关键帧须按时间递增")
                for t, box in keys:
                    if (
                        not math.isfinite(t)
                        or len(box) != 4
                        or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)
                    ):
                        raise ValueError("区域须用归一化左上、右下坐标")


class Frames:
    def __init__(self, path, meta, a, b):
        pts, _dur, size, self.turns = meta[:4]
        self.pts = pts
        self.first = max(0, bisect.bisect_right(pts, a + EPS) - 1)
        self.end = min(len(pts), bisect.bisect_left(pts, b - EPS))
        self.index = self.first - 1
        self.data = raw_frames(path, size, self.first, self.end)
        self.image = None

    def at(self, t):
        target = min(self.end - 1, bisect.bisect_right(self.pts, t + EPS) - 1)
        while self.index < target:
            frame = next(self.data, None)
            if frame is None:
                raise ValueError("视频帧未完整解码")
            self.image = np.ascontiguousarray(np.rot90(frame, self.turns))
            self.index += 1
        return self.image


def crop(im, box):
    h, w = im.shape[:2]
    x, y = np.floor(np.array(box[:2]) * [w, h]).astype(int)
    r, b = np.ceil(np.array(box[2:]) * [w, h]).astype(int)
    return im[y:b, x:r]


def pair_patches(images, region, t):
    boxes = [track_box(region.get(k, region["ref"]), t) for k in ("ref", "rep")]
    patches = [crop(im, b) for im, b in zip(images, boxes)]
    h, w = patches[0].shape[:2]
    if min(n for patch in patches for n in patch.shape[:2]) < 16:
        return None
    scale = min(1, 420 / max(h, w), patches[1].shape[0] / h, patches[1].shape[1] / w)
    size = (max(8, round(w * scale)), max(8, round(h * scale)))
    return [
        cv.resize(cv.cvtColor(p, cv.COLOR_BGR2GRAY), size, interpolation=cv.INTER_AREA)
        for p in patches
    ]


def matching(a, b):
    """Return evidence of correspondence; spread and independent check points matter."""
    if min(a.std(), b.std()) < 5:
        return None

    def corr(x, y):
        return float(cv.matchTemplate(x, y, cv.TM_CCOEFF_NORMED)[0, 0])

    small = [cv.GaussianBlur(p, (0, 0), 2.2) for p in (a, b)]
    if corr(*small) > 0.96 and np.mean(cv.absdiff(*small)) < 7:
        return np.eye(3), 0.35
    sift = cv.SIFT_create(nfeatures=420, contrastThreshold=0.012)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if min(len(ka), len(kb)) < 12:
        return None
    matcher = cv.BFMatcher()

    def matches(x, y):
        return {
            (m.queryIdx, m.trainIdx)
            for m, n in matcher.knnMatch(x, y, k=2)
            if m.distance < 0.78 * n.distance
        }

    both = matches(da, db) & {(j, i) for i, j in matches(db, da)}
    if len(both) < 12:
        return None
    pts = np.float32([[ka[i].pt, kb[j].pt] for i, j in sorted(both)])
    u, v = pts[:, 0], pts[:, 1]
    H, mask = cv.findHomography(u, v, cv.RANSAC, 2.0)
    if H is None or mask.sum() < 12 or mask.mean() < 0.65:
        return None
    good = mask.ravel().astype(bool)
    for points in (u[good], v[good]):
        if cv.contourArea(cv.convexHull(points)) < a.size * 0.12:
            return None
    fit = solve_pair(u[good], v[good])
    if not fit or fit["holdout_p95_px"] > 2.5:
        return None
    return np.asarray(fit["H"]), max(0.35, fit["holdout_p95_px"])


def edges(im):
    gx = cv.Sobel(im, cv.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv.Sobel(im, cv.CV_32F, 0, 1, ksize=3, scale=0.125)
    points = sample(im, gx, gy, np.ones_like(im))
    if len(points) < 7:
        return None
    x, y = points[:, :2].astype(int).T
    theta = np.arctan2(gy[y, x], gx[y, x])
    widths = points[:, 2]
    # Compare width along several edge normals, including the two diagonal axes.
    by_axis = []
    for angle in np.arange(4) * np.pi / 4:
        group = widths[np.abs(np.cos(theta - angle)) >= np.cos(np.pi / 8)]
        by_axis.append(float(np.median(group)) if len(group) >= 3 else None)
    return float(np.median(widths)), by_axis


def motion(prev, cur):
    if prev is None or prev.shape != cur.shape:
        return None
    pts = cv.goodFeaturesToTrack(prev, 150, 0.012, 5)
    if pts is None:
        return None
    a, b = flow(prev, cur, pts[:, 0])
    if len(a) < 8:
        return None
    mat, ins = cv.estimateAffinePartial2D(
        a, b, method=cv.RANSAC, ransacReprojThreshold=1.7
    )
    if mat is None or ins.sum() < 8:
        return None
    if cv.contourArea(cv.convexHull(a[ins.ravel().astype(bool)])) < cur.size * 0.09:
        return None
    center = np.array(cur.shape[::-1]) * 0.5
    return np.r_[
        mat[:, :2] @ center + mat[:, 2] - center,
        math.atan2(mat[1, 0], mat[0, 0]) * max(center),
        np.log(max(np.linalg.norm(mat[:, 0]), 1e-8)) * max(center),
    ]


def landmark_score(pair, prior, seeds, tracks, boxes):
    """Both versions have independently identified corners at the first display time."""
    shapes = []
    for k, (im, seed, box) in enumerate(zip(pair, seeds, boxes)):
        if prior is None and tracks[k] is None:
            points = np.float32(seed) * np.float32(im.shape[::-1])
            if (
                points.shape != (4, 2)
                or not np.isfinite(points).all()
                or not np.all((np.asarray(seed) >= 0) & (np.asarray(seed) <= 1))
            ):
                raise ValueError("首帧四角须依左上、右上、右下、左下标注在对应区域内")
        elif prior is None or tracks[k] is None or len(tracks[k]) != 4:
            tracks[k] = []
            continue
        else:
            _, points = flow(prior[k], im, np.float32(tracks[k]).reshape(4, 2))
        tracks[k] = points
        if len(points) != 4 or not cv.isContourConvex(np.float32(points)):
            tracks[k] = []
            continue
        confidence = cv.cornerMinEigenVal(im, 5)
        if any(
            confidence[
                max(0, int(y) - 4) : int(y) + 5, max(0, int(x) - 4) : int(x) + 5
            ].max(initial=0)
            < 1e-6
            for x, y in points
        ):
            tracks[k] = []
            continue
        shape = points / im.shape[::-1] * (np.asarray(box[2:]) - box[:2])
        shape -= shape.mean(axis=0)
        shapes.append((shape / np.linalg.norm(shape)).ravel().tolist())
    if len(shapes) != 2:
        return None
    return shapes, max(0.014, 2 / min(pair[0].shape))


def region_score(mode, pair, prev, base, boxes, identity):
    a, b = pair
    w, h = a.shape[::-1]
    if mode == "visibility":
        if not base or not matching(*base):
            return None
        # Fixed region, measured relative to its visible initial state.
        vals = []
        for image, background in zip(pair, base):
            delta = image.astype(np.float32) - background
            cells = (
                cv.resize(delta, (3, 3), interpolation=cv.INTER_AREA).ravel().tolist()
            )
            vals.append([float(np.abs(delta).mean()), *cells])
        return vals, 6.0
    match = matching(a, b)
    if match is None:
        if (
            not identity
            or mode not in ("motion", "trail", "focus")
            or min(a.std(), b.std()) < 5
        ):
            return None
        noise = 0.6
    else:
        H, noise = match
    if mode == "motion":
        if prev is None:
            return None
        m = [motion(x, y) for x, y in zip(prev, pair)]
        if any(x is None for x in m):
            return None
        return [x.tolist() for x in m], max(0.9, 0.0025 * max(w, h), noise)
    if mode == "perspective":
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        q = project(H, corners)
        if not np.isfinite(q).all():
            return None
        # Recover image-space shape before centering. Keep differential foreshortening.
        shapes = []
        for points, box in zip((corners, q), boxes):
            shape = points / [w, h] * (np.array(box[2:]) - box[:2])
            shape -= shape.mean(axis=0)
            shapes.append((shape / max(np.linalg.norm(shape), 1e-8)).ravel().tolist())
        return shapes, max(0.018, 3 * noise / min(w, h))
    profiles = [edges(x) for x in pair]
    if any(x is None for x in profiles):
        return None
    if mode == "focus":
        return [x[0] for x in profiles], max(0.8, 0.003 * min(w, h))
    # Narrow and broad derivatives along four normals respond differently to a
    # directional streak. This also works when ESF lacks one orientation of edge.
    directional = []
    for im, profile in zip(pair, profiles):
        g = im.astype(np.float32)
        gradients = []
        for radius in (0.65, 2.8):
            layer = cv.GaussianBlur(g, (0, 0), radius)
            gradients.append(
                (cv.Sobel(layer, cv.CV_32F, 1, 0), cv.Sobel(layer, cv.CV_32F, 0, 1))
            )
        axes = []
        for theta in np.arange(4) * np.pi / 4:
            energies = [
                np.mean(np.abs(x * np.cos(theta) + y * np.sin(theta)))
                for x, y in gradients
            ]
            if min(energies) < 1.8:
                return None
            axes.append(float(np.log(energies[0] / energies[1])))
        # Dimensionless log ratios, followed by a separately normalized edge width.
        # Relative angular power in one frequency band isolates directional spread.
        win = np.outer(np.hanning(h), np.hanning(w))
        power = np.abs(np.fft.rfft2((g - g.mean()) * win)) ** 2
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.rfftfreq(w)[None, :]
        radius = np.hypot(fx, fy)
        angle = np.arctan2(fy, fx)
        band = (radius > 0.08) & (radius < 0.28)
        power = power / (np.mean(power[band]) + 1e-9)
        angular = [
            np.log1p(np.mean(power[band & (np.abs(np.cos(angle - theta)) > 0.94)]))
            for theta in np.arange(4) * np.pi / 4
        ]
        directional.append(axes + angular + [profile[0] / max(8, 0.025 * min(w, h))])
    return directional, 0.18


def difference(values):
    return float(np.max(np.abs(np.asarray(values[0]) - values[1])))


def sound_peak(path, a, b, meta):
    if not meta[5]:
        return None
    start = a + meta[4]
    # Keep the original audio/video offset, including a late first audio packet.
    af = f"atrim=start={start:.9f}:end={b + meta[4]:.9f},asetpts=PTS-({start:.9f})/TB,aresample=16000:async=1:first_pts=0"
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-copyts",
        "-seek_timestamp",
        "1",
        "-ss",
        str(max(0, start - 0.5)),
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        af,
        "-t",
        str(b - a),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    r = subprocess.run(cmd, check=False, capture_output=True)
    x = np.frombuffer(r.stdout, dtype="<f4")
    if r.returncode or len(x) < 320:
        return None
    rms = np.sqrt(np.mean(x[: len(x) // 80 * 80].reshape(-1, 80) ** 2, axis=1))
    rise = np.maximum(0, np.diff(rms, prepend=0))
    i = int(rise.argmax())
    other = rise.copy()
    other[max(0, i - 5) : i + 6] = 0
    if rise[i] < max(0.012, 2 * other.max()):
        return None
    return a + i * 0.005


def evidence(paths, metas, a, b, folder, number):
    """Preserve the frame already on screen at a and the offset of each audio track."""
    name = f"{number:03}-同步对照.mp4"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-copyts"]
    filters = []
    audio = []
    for i, (path, meta) in enumerate(zip(paths, metas)):
        times = meta[0]
        first = max(0, bisect.bisect_right(times, a + EPS) - 1)
        held = meta[4] + times[first]
        start = a + meta[4]
        stop = b + meta[4]
        cmd += ["-seek_timestamp", "1", "-ss", str(max(0, held - 0.5)), "-i", str(path)]
        filters.append(
            f"[{i}:v:0]trim=start={held - EPS:.9f}:end={stop:.9f},settb=AVTB,setpts='max(0,PTS-({start:.9f})/TB)',scale=480:360:force_original_aspect_ratio=decrease,pad=480:360:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]"
        )
        if meta[5]:
            filters.append(
                f"[{i}:a:0]atrim=start={start:.9f}:end={stop:.9f},asetpts=PTS-({start:.9f})/TB[a{i}]"
            )
            audio += ["-map", f"[a{i}]"]
    filters.append("[v0][v1]hstack[v]")
    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        *audio,
        "-t",
        str(b - a),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "vfr",
        "-enc_time_base:v",
        "1:90000",
        "-c:a",
        "aac",
        str(folder / name),
    ]
    subprocess.run(cmd, check=True)
    return name


def assess(paths, metas, e, folder, num):
    a, b = e["range"]
    mode = e["mode"]
    end = min(b, *[m[1] for m in metas])
    rows = []
    worst = -1
    snapshot = None
    peak_rows = []
    prev = [None] * len(e["regions"])
    base = prev.copy()
    prev_vel = prev.copy()
    landmarks = [[None, None] for _ in prev]
    histories = [[] for _ in prev]
    residual_history = [[] for _ in prev]
    if a < end:
        with ExitStack() as stack:
            movies = [Frames(p, m, a, end) for p, m in zip(paths, metas)]
            for movie in movies:
                stack.callback(movie.data.close)
            times = sorted(
                {
                    a,
                    *[
                        round(float(t), 6)
                        for movie in movies
                        for t in movie.pts[movie.first : movie.end]
                        if a < t < end - EPS
                    ],
                }
            )
            prev_indices = None
            for i, t in enumerate(times):
                images = [m.at(t) for m in movies]
                indices = [m.index for m in movies]
                parts = []
                measurements = []
                rank = 0
                reasons = []
                for j, reg in enumerate(e["regions"]):
                    boxes = [
                        track_box(reg.get(k, reg["ref"]), t) for k in ("ref", "rep")
                    ]
                    pair = pair_patches(images, reg, t)
                    if pair is None:
                        reasons.append("区域分辨率不足")
                        continue
                    # Resample previous crop only when its analysis scale is unchanged.
                    prior = (
                        prev[j]
                        if prev[j] and prev[j][0].shape == pair[0].shape
                        else None
                    )
                    if i == 0:
                        base[j] = pair
                    if (
                        mode == "impact"
                        and prior
                        and (reg.get("identity_evidence") or matching(*base[j]))
                    ):
                        peak_rows.append(
                            (
                                t,
                                [
                                    float(cv.absdiff(x, y).mean())
                                    for x, y in zip(prior, pair)
                                ],
                            )
                        )
                    if mode not in ("manual", "impact"):
                        if mode == "perspective" and "corners_ref" in reg:
                            result = landmark_score(
                                pair,
                                prior,
                                [reg["corners_ref"], reg["corners_rep"]],
                                landmarks[j],
                                boxes,
                            )
                        else:
                            result = (
                                region_score(
                                    mode,
                                    pair,
                                    prior,
                                    base[j],
                                    boxes,
                                    reg.get("identity_evidence"),
                                )
                                if mode != "visibility"
                                or base[j][0].shape == pair[0].shape
                                else None
                            )
                        if result:
                            vals, tol = result
                            if mode == "focus":
                                # All layers expressed as fraction of the current display width.
                                scale = (boxes[0][2] - boxes[0][0]) / pair[0].shape[1]
                                vals = [v * scale for v in vals]
                                tol *= scale
                            # Estimate ordinary frame-to-frame variation on this very subject.
                            # Limit its influence so an intentional fast move cannot hide a jump.
                            ref = np.atleast_1d(vals[0])
                            history = histories[j]
                            if history and history[-1].shape != ref.shape:
                                history.clear()
                            if len(history) >= 4:
                                noise_floor = float(
                                    np.median(np.abs(np.diff(history, axis=0)))
                                )
                                tol += min(tol, noise_floor * 0.5)
                            history.append(ref)
                            del history[:-8]
                            err = difference(vals)
                            if mode == "motion":
                                if prev_indices and any(
                                    x == y for x, y in zip(indices, prev_indices)
                                ):
                                    reasons.append("两版曝光时刻不同，需复核运动")
                                if prev_vel[j] is not None:
                                    acc = np.asarray(vals) - prev_vel[j]
                                    err = max(err, difference(acc) * 0.65)
                                residual = np.asarray(vals[1]) - vals[0]
                                track = residual_history[j]
                                track.append((t, residual))
                                while track and track[0][0] < t - 0.45:
                                    track.pop(0)
                                if len(track) >= 5:
                                    trajectory = np.stack([v for _, v in track])
                                    flips = (trajectory[1:] * trajectory[:-1] < 0).sum(
                                        axis=0
                                    )
                                    oscillation = np.median(np.abs(trajectory), axis=0)
                                    if (
                                        np.any(flips >= 3)
                                        and np.max(np.abs(residual)) > 0.3 * tol
                                    ):
                                        err = max(
                                            err,
                                            1.65
                                            * float(np.max(oscillation[flips >= 3])),
                                        )
                                prev_vel[j] = np.array(vals)
                            parts.append((vals, tol, err))
                            measurements.append(
                                {
                                    "region": reg["name"],
                                    "values": vals,
                                    "tolerance": tol,
                                }
                            )
                        elif mode != "motion" or i > 0:
                            reasons.append("主体配对、纹理或测量依据不足")
                            prev_vel[j] = None
                            residual_history[j].clear()
                    prev[j] = pair
                if mode == "focus" and len(parts) >= 2:
                    for vals, tol, _ in parts[1:]:
                        relative = np.asarray(parts[0][0]) - vals
                        rank = max(rank, difference(relative) / (tol + parts[0][1]))
                if parts:
                    rank = max(rank, max(err / tol for _, tol, err in parts))
                if mode in ("manual", "impact"):
                    reasons.append("需核对专项目标")
                baseline = mode == "motion" and i == 0
                status = REVIEW if reasons or not parts and not baseline else PASS
                if status == PASS and rank > 1:
                    status = FAIL
                stop = times[i + 1] if i + 1 < len(times) else end
                rows.append(
                    {
                        "t": t,
                        "end": stop,
                        "frames": indices,
                        "status": status,
                        "ratio": round(rank, 3),
                        "metrics": measurements,
                        "reason": "；".join(dict.fromkeys(reasons)),
                    }
                )
                if rank > worst:
                    worst = rank
                    box = e["regions"][0]
                    crops = [
                        crop(im, track_box(box.get(k, box["ref"]), t))
                        for im, k in zip(images, ("ref", "rep"))
                    ]
                    height = min(480, *[im.shape[0] for im in crops])
                    snapshot = np.hstack(
                        [
                            cv.resize(
                                im,
                                (
                                    max(1, round(im.shape[1] * height / im.shape[0])),
                                    height,
                                ),
                            )
                            for im in crops
                        ]
                    )
                prev_indices = indices
    if mode == "impact" and rows and peak_rows:
        audio = [sound_peak(path, a, end, meta) for path, meta in zip(paths, metas)]
        offsets = []
        visual = []
        for j, stamp in enumerate(audio):
            ranked = sorted(peak_rows, key=lambda r: r[1][j], reverse=True)
            other = max(
                (v[j] for t, v in ranked if abs(t - ranked[0][0]) > 0.065), default=0
            )
            if stamp is not None and ranked[0][1][j] > max(5, other * 1.8):
                # A one-frame impact has both entry and exit edges. Anchor its entry.
                onset = min(
                    t
                    for t, v in ranked
                    if abs(t - ranked[0][0]) <= 0.065 and v[j] >= 0.8 * ranked[0][1][j]
                )
                visual.append(onset)
                offsets.append(stamp - onset)
        if len(offsets) == 2:
            tol = max(
                0.015, *[np.median(np.diff(m[0])) for m in metas if len(m[0]) > 1]
            )
            status = FAIL if difference(offsets) > tol + EPS else PASS
            for row in rows:
                in_event = (
                    min(*visual, *audio) - EPS <= row["t"] <= max(*visual, *audio) + tol
                )
                row.update(
                    status=status if in_event else PASS,
                    reason="单一瞬态相对可辨画面冲击的时间差",
                    metrics=[
                        {
                            "audio_s": audio,
                            "visual_s": visual,
                            "offset_s": offsets,
                            "tolerance": float(tol),
                        }
                    ],
                )
    if mode == "motion" and len(rows) < 2:
        for row in rows:
            row.update(status=REVIEW, reason="运动至少需要一对相邻帧")
    if end < b - EPS:
        rows.append(
            {
                "t": max(a, end),
                "end": b,
                "status": REVIEW,
                "reason": "超出可配对显示范围",
                "ratio": 0,
                "metrics": [],
            }
        )
    states = {r["status"] for r in rows}
    status = (
        FAIL if FAIL in states else REVIEW if REVIEW in states or not rows else PASS
    )
    intervals = []
    for row in rows:
        if row["status"] == PASS:
            continue
        if (
            intervals
            and intervals[-1]["status"] == row["status"]
            and abs(intervals[-1]["end"] - row["t"]) < EPS
            and intervals[-1]["reason"] == row["reason"]
        ):
            event = intervals[-1]
            event["end"] = row["end"]
            if row["ratio"] > event["ratio"]:
                event.update(worst=row["t"], ratio=row["ratio"])
        else:
            intervals.append(
                {
                    "start": row["t"],
                    "end": row["end"],
                    "status": row["status"],
                    "reason": row["reason"],
                    "worst": row["t"],
                    "ratio": row["ratio"],
                }
            )
    if e.get("review_required") and status == PASS:
        status = REVIEW
    clip = evidence(paths, metas, a, end, folder, num) if a < end else None
    still = None
    if snapshot is not None:
        still = f"{num:03}-局部.jpg"
        write_image(folder / still, snapshot)
    data = f"{num:03}-逐帧.json"
    write_json(folder / data, rows)
    return dict(
        **e,
        status=status,
        automatic=status,
        has_audio=all(m[5] for m in metas),
        method=mode,
        paired_s=max(0, end - a),
        measured_s=sum(
            r["end"] - r["t"] for r in rows if r["status"] != REVIEW and r["metrics"]
        ),
        pairs=sum("frames" in r for r in rows),
        events=intervals,
        clip=clip,
        still=still,
        data=data,
    )


def measurements(folder, signature, paths, items):
    # Publish all evidence together. A failed decode leaves the last report usable.
    group = folder / ("证据-" + signature[:16])
    if not group.exists():
        with TemporaryDirectory(dir=folder) as tmp:
            stage = Path(tmp) / group.name
            stage.mkdir()
            metas = [timeline(p) for p in paths]
            results = [
                assess(paths, metas, e, stage, i) for i, e in enumerate(items, 1)
            ]
            write_json(stage / "自动测量.json", results)
            try:
                stage.rename(group)
            except OSError:
                if not group.is_dir():
                    raise
    results = json.loads((group / "自动测量.json").read_text(encoding="utf-8-sig"))
    for row in results:
        for key in ("clip", "still", "data"):
            if row[key]:
                file = group / row[key]
                if not file.is_file() or not file.stat().st_size:
                    raise ValueError(f"验收证据缺失：{file}，请重新输出至新目录")
                row[key] = str(file.relative_to(folder))
    return results


def publish(path, content):
    with NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(content)
    Path(f.name).replace(path)


def run(args):
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    items = plan["effects"]
    validate(items)
    folder = args.out
    folder.mkdir(parents=True, exist_ok=True)
    paths = [args.reference, args.replica]
    fingerprints = {
        k: digest(p)
        for k, p in zip(("reference", "replica", "plan"), [*paths, args.plan])
    }
    signature = hashlib.sha256(
        (digest(Path(__file__)) + "".join(fingerprints.values())).encode()
    ).hexdigest()
    previous = {}
    if args.previous:
        prior = json.loads(args.previous.read_text(encoding="utf-8-sig"))
        original = prior.get("fingerprints", {})
        if (
            any(original.get(k) != fingerprints[k] for k in ("reference", "plan"))
            or original.get("replica") == fingerprints["replica"]
        ):
            raise ValueError("修前报告须使用相同原片和清单，以及不同的导出")
        previous = {e["id"]: e["status"] for e in prior["effects"]}
    decisions = {}
    if args.decisions:
        doc = json.loads(args.decisions.read_text(encoding="utf-8-sig"))
        if doc["review_id"] != signature:
            raise ValueError("复核单与本次导出或清单不符")
        decisions = doc["decisions"]
        if set(decisions) - {e["id"] for e in items}:
            raise ValueError("复核单包含未知效果")
    results = measurements(folder, signature, paths, items)
    for e, r in zip(items, results, strict=True):
        if e["id"] in decisions:
            decision = decisions[e["id"]]
            if (
                decision["status"] not in (PASS, FAIL, REVIEW)
                or not decision["reason"].strip()
                or not decision["evidence"].strip()
            ):
                raise ValueError("人工判断须附结论、理由与证据")
            proof = (args.decisions.parent / decision["evidence"]).resolve()
            if not proof.is_file():
                raise ValueError(f"人工证据文件不可读：{proof}")
            if decision["status"] == PASS and (
                (e["mode"] == "impact" and not r["has_audio"])
                or not r["clip"]
                or r["paired_s"] + EPS < e["range"][1] - e["range"][0]
            ):
                raise ValueError("缺测范围须补测")
            r.update(status=decision["status"], review=decision, method="人工复核")
        r["fixed"] = previous.get(e["id"]) == FAIL and r["status"] == PASS
    states = {r["status"] for r in results}
    status = FAIL if FAIL in states else REVIEW if REVIEW in states else PASS
    result_doc = {
        "status": status,
        "review_id": signature,
        "fingerprints": fingerprints,
        "reference": str(paths[0]),
        "replica": str(paths[1]),
        "effects": results,
    }
    lines = [
        "# 逐项效果验收",
        f"本次清单共 {len(results)} 项，结论为**{status}**。[原速与四倍慢放](同步审片.html)",
    ]
    fixed = [r["subject"] for r in results if r["fixed"]]
    if args.previous:
        lines.append("修前问题已复测通过：" + ("、".join(fixed) or "0 项"))
    for state in (FAIL, REVIEW, PASS):
        group = [r for r in results if r["status"] == state]
        lines.append(f"## {state}（{len(group)} 项）")
        for r in group:
            a, b = r["range"]
            links = " · ".join(
                f"[{k}]({r[key]})"
                for k, key in [
                    ("原速对照", "clip"),
                    ("局部", "still"),
                    ("读数", "data"),
                ]
                if r[key]
            )
            why = r.get("review", {}).get("reason") or r["criterion"]
            lines.append(
                f"- **{a:.3f}–{b:.3f} 秒 · {r['subject']}**　{why}。已配对 {r['paired_s']:.3f} 秒，自动取得有效读数 {r['measured_s']:.3f} 秒。{links}"
            )
            if r["events"]:
                event = max(r["events"], key=lambda v: v["ratio"])
                lines.append(
                    f"  自动候选 {len(r['events'])} 段，代表时刻 {event['worst']:.3f} 秒。{event['reason']}"
                )
    publish(folder / "逐项效果验收.md", "\n\n".join(lines) + "\n")
    print(f"{status}，报告 {folder / '逐项效果验收.md'}")
    from html import escape

    clips = []
    for r in results:
        if r["clip"]:
            clips.append(
                f"<section><h2>{escape(r['subject'])} · {r['range'][0]:.3f}–{r['range'][1]:.3f} 秒</h2><p>{escape(r['criterion'])}</p><video controls preload='none' src='{r['clip']}'></video><p><button onclick='this.parentNode.previousElementSibling.playbackRate=1'>原速</button> <button onclick='this.parentNode.previousElementSibling.playbackRate=.25'>四倍慢放</button></p></section>"
            )
    publish(
        folder / "同步审片.html",
        "<!doctype html><meta charset=utf-8><title>同步审片</title><style>body{max-width:960px;margin:auto;font:16px sans-serif}video{width:100%}</style><h1>左原片 · 右导出</h1>"
        + "".join(clips),
    )
    publish(
        folder / "effects.json",
        json.dumps(result_doc, ensure_ascii=False, indent=2, allow_nan=False),
    )
    return {PASS: 0, FAIL: 2, REVIEW: 3}[status]
