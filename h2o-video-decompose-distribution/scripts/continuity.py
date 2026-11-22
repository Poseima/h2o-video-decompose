"""Reference-based temporal review. All times are relative to the first displayed frame."""

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np

PASS, FAIL, REVIEW = "通过", "需返修", "待复核"
from local_continuity import border_palette, local_events, profile, timing_events
from subject_continuity import components, patch_signature, subject_events

VERSION = 3
SIDE = 384


def measure_pair(previous, image, dense):
    gray = cv.cvtColor(previous, cv.COLOR_BGR2GRAY)
    nxt = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    fw = dense.calc(gray, nxt, None)
    bw = dense.calc(nxt, gray, None)
    x, y = xx + fw[:, :, 0], yy + fw[:, :, 1]
    valid = (x >= 1) & (x < w - 2) & (y >= 1) & (y < h - 2)

    def warp(a):
        return cv.remap(a, x, y, cv.INTER_LINEAR)

    err = np.linalg.norm(fw + warp(bw), axis=2)
    color = warp(image).astype(np.float32) - previous
    dy = cv.cvtColor(color, cv.COLOR_BGR2GRAY)
    mask = (np.abs(dy) > 20) & valid
    regions = []
    for row in range(3):
        for col in range(3):
            tile = np.s_[
                row * h // 3 : (row + 1) * h // 3, col * w // 3 : (col + 1) * w // 3
            ]
            v = valid[tile]
            regions.append(
                [
                    float(np.mean(mask[tile][v])),
                    *np.median(color[tile][v], axis=0).tolist(),
                    float(np.mean(err[tile][v] > 2)),
                ]
                if v.any()
                else [0.0, 0.0, 0.0, 0.0, 0.0]
            )
    # Textured tracks, evenly distributed so a single foreground object cannot dominate.
    points = cv.goodFeaturesToTrack(gray, 600, 0.012, 5)
    src, dst = [], []
    bins = {}
    for point in [] if points is None else points[:, 0]:
        px, py = point.astype(int)
        cell = (3 * px // w, 3 * py // h)
        if valid[py, px] and err[py, px] < 1.8 and bins.get(cell, 0) < 16:
            bins[cell] = 1 + bins.get(cell, 0)
            src.append(point)
            dst.append(point + fw[py, px])
    motion = np.zeros(4)
    support = 0
    if len(src) >= 8:
        mat, inliers = cv.estimateAffinePartial2D(
            np.float32(src),
            np.float32(dst),
            method=cv.RANSAC,
            ransacReprojThreshold=1.8,
        )
        if mat is not None:
            center = np.array([w / 2, h / 2])
            shift = (mat[:, :2] @ center + mat[:, 2] - center) / max(w, h)
            motion = [
                *shift,
                math.atan2(mat[1, 0], mat[0, 0]),
                math.log(max(1e-8, np.linalg.norm(mat[:, 0]))),
            ]
            good = np.asarray(src)[inliers.ravel().astype(bool)]
            support = len({(int(3 * a / w), int(3 * b / h)) for a, b in good})

    def avg(a):
        return float(np.mean(a[valid])) if valid.any() else 1.0

    return {
        "signature": patch_signature(color, fw, previous, warp(image)),
        "blobs": components(image),
        "patch_flow": np.stack(
            [
                profile(previous, image, fw, bw, xx, yy),
                profile(image, previous, bw, fw, xx, yy),
            ]
        ).astype(np.float16),
        "motion": np.round(motion, 6).tolist(),
        "support": support,
        "valid": round(float(valid.mean()), 4),
        "mismatch": round(avg(err > 2), 4),
        "structure": round(avg(mask), 4),
        "brightness": round(avg(dy), 2),
        "regions": np.round(regions, 3).tolist(),
        "activity": round(float(cv.absdiff(gray, nxt).mean()), 3),
    }


def rotation(stream):
    return (
        int(
            next(
                (
                    r["rotation"]
                    for r in stream.get("side_data_list", [])
                    if "rotation" in r
                ),
                0,
            )
        )
        % 360
    )


def read_movie(path, start, end, load, decode):
    frames, stream = load(path)
    if any(r["time_s"] is None for r in frames):
        raise ValueError("缺少显示时间戳")
    times = np.array([r["time_s"] for r in frames])
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("有效显示时间戳不足或次序异常")
    duration = times[-1] + (frames[-1]["duration_s"] or np.median(np.diff(times)))
    first = max(0, int(np.searchsorted(times, start - 0.5, "left")) - 2)
    last = min(len(times), int(np.searchsorted(times, end + 0.5, "left")) + 2)
    w, h = stream["width"], stream["height"]
    ratio = SIDE / max(w, h)
    size = tuple(max(8, round(v * ratio)) for v in (w, h))
    dense = cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    rows = []
    previous = None
    thumbs = []
    count = 0
    for i, image in enumerate(decode(path, size, first, last), first):
        image = np.ascontiguousarray(np.rot90(image, rotation(stream) // 90))
        thumbs.append(
            cv.resize(
                cv.cvtColor(image, cv.COLOR_BGR2GRAY),
                (16, 16),
                interpolation=cv.INTER_AREA,
            )
        )
        if previous is not None:
            row = measure_pair(previous, image, dense)
            row["border_palette"] = border_palette(image)
            row.update(frame=i, t=float(times[i]), dt=float(times[i] - times[i - 1]))
            rows.append(row)
        previous = image
        count += 1
    if count != last - first or not rows:
        raise ValueError("所选区间没有完整解码")
    return (
        rows,
        np.array(thumbs[1:]),
        float(duration),
        len(times),
        float(np.median(np.diff(times))),
    )


def aligned_visual(a, b):
    a = a.astype(float).ravel()
    b = b.astype(float).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / max(1.0, np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate(ref, rep, thumbnails, cadence, start, end):
    times = np.array([r["t"] for r in ref])
    replica_times = np.array([r["t"] for r in rep])
    defects = []
    checks = []

    def add(kind, status, region="全画面"):
        if b["t"] < start - 1e-7 or b["t"] - b["dt"] >= end + 1e-7:
            return
        defects.append(
            {
                "frame": b["frame"],
                "reference_frame": a["frame"],
                "time": b["t"],
                "kind": kind,
                "status": status,
                "region": region,
            }
        )

    def velocity(row):
        return np.array(row["motion"]) / (30 * row["dt"])

    def interval_velocity(rows, stamps, left, right):
        begin = int(np.searchsorted(stamps, left, side="right"))
        end = min(len(rows), int(np.searchsorted(stamps, right)) + 1)
        total = np.zeros(4)
        length = 0.0
        for row in rows[begin:end]:
            overlap = max(0.0, min(row["t"], right) - max(row["t"] - row["dt"], left))
            total += overlap * velocity(row)
            length += overlap
        return total / max(1e-9, length)

    localized = local_events(ref, rep, cadence)
    for k, rows in timing_events(ref, rep).items():
        localized.setdefault(k, []).extend(rows)
    for k, rows in subject_events(ref, rep).items():
        localized.setdefault(k, []).extend(rows)
    history = []
    for k, b in enumerate(rep):
        t = b["t"]
        right = int(np.searchsorted(times, t))
        j = min(
            {max(0, right - 1), min(len(times) - 1, right)},
            key=lambda idx: (abs(times[idx] - t), idx),
        )
        a = ref[j]
        # Window overlaps include the incoming and outgoing adjacent-frame pairs.
        if t - b["dt"] >= end + 1e-7:
            continue
        similarity = aligned_visual(thumbnails[0][j], thumbnails[1][k])
        if t >= start - 1e-7:
            checks.append(
                {
                    "frame": b["frame"],
                    "reference_frame": a["frame"],
                    "pts": t,
                    "reference_pts": a["t"],
                    "spatial_correlation": round(similarity, 3),
                }
            )
        if abs(times[j] - t) > max(cadence) * 0.61 + 1e-5:
            add("时间覆盖不足", REVIEW)
            continue
        # Compare native pairs at dissimilar cadences over their actual exposure interval.
        lo = np.searchsorted(times, t - b["dt"] + 1e-6)
        hi = np.searchsorted(times, t + 1e-6)
        group = ref[lo:hi] if lo < hi else [a]
        slow = (
            abs(cadence[0] - cadence[1]) > 0.12 * min(cadence)
            or abs(b["dt"] - a["dt"]) > 0.2 * a["dt"]
        )
        expected = max(group, key=lambda r: r["structure"]) if slow else a
        if slow and a["structure"] > expected["structure"]:
            expected = a
        nearby = ref[max(0, j - 5) : min(j + 6, len(ref))]
        local = np.array([r["structure"] for r in nearby])
        noise = float(np.median(abs(local - np.median(local))))
        excess = b["structure"] - expected["structure"]
        structural = (
            excess > max(0.12, 6 * noise)
            and b["mismatch"] > expected["mismatch"] + 0.19
        )
        if structural:
            add("场景或图层突变", FAIL)
        ar = np.asarray(expected["regions"])
        br = np.asarray(b["regions"])
        tile = int(np.argmax(br[:, 0] - ar[:, 0]))
        region = f"{tile // 3 + 1}行{tile % 3 + 1}列"
        regional_jump = br[tile, 0] - ar[tile, 0]
        unreliable_tracks = br[tile, 4] - ar[tile, 4] > 0.20
        if not structural and (
            regional_jump > 0.36 or regional_jump > 0.18 and unreliable_tracks
        ):
            add("局部结构变化", REVIEW, region)
            if start - 1e-7 <= t:
                defects[-1]["bbox"] = [(tile % 3) / 3, (tile // 3) / 3, 1 / 3, 1 / 3]
                defects[-1]["residual_excess"] = float(regional_jump)
        if t >= start - 1e-7:
            for finding in localized.get(k, []):
                defects.append(
                    dict(
                        frame=b["frame"], reference_frame=a["frame"], time=t, **finding
                    )
                )
        colors = br[:, 1:4] - ar[:, 1:4]
        tile = int(np.argmax(np.max(abs(colors), axis=1)))
        light_gap = abs(b["brightness"] - expected["brightness"])
        opposite = (
            b["brightness"] * expected["brightness"] <= 0
            and max(abs(b["brightness"]), abs(expected["brightness"])) > 14
        )
        flash = light_gap > 12 and (
            opposite
            or light_gap
            > max(12, 3 * np.median([abs(r["brightness"]) for r in nearby]))
        )
        if flash or np.max(abs(colors)) > 30:
            add(
                "亮度或颜色阶跃",
                FAIL if flash or np.max(abs(colors)) > 48 else REVIEW,
                "全画面" if flash else f"{tile // 3 + 1}行{tile % 3 + 1}列",
            )
        event = max(expected["structure"], b["structure"]) > 0.18 or flash
        if not event and (
            min(a["support"], b["support"]) < 3 or min(a["valid"], b["valid"]) < 0.45
        ):
            add("跟踪依据不足", REVIEW)
        if similarity < 0.05 and not event:
            add("跨片画面对应关系待确认", REVIEW)
        if (
            b["activity"] < 0.40
            and a["activity"] > 2.0
            and np.linalg.norm(a["motion"][:2]) > 0.0015
        ):
            ordinary_hold = b["dt"] < 0.95 * a["dt"] and lo == hi
            if not ordinary_hold:
                add("重复帧或停顿", REVIEW if slow else FAIL)
        if slow:
            # Integrate velocities over a common display interval, including partial
            # native pairs, to avoid treating frame-rate conversion as camera shake.
            left = t - max(*cadence, a["dt"], b["dt"])
            delta = interval_velocity(rep, replica_times, left, t) - interval_velocity(
                ref, times, left, t
            )
        else:
            delta = velocity(b) - velocity(a)
        if event or min(a["support"], b["support"]) < 3:
            history.clear()
            continue
        recent = np.asarray([velocity(r) for r in nearby])
        scale = np.maximum(
            [0.006, 0.006, 0.012, 0.012],
            4 * np.median(abs(recent - np.median(recent, axis=0)), axis=0),
        )
        if slow:
            scale *= 1.5
        if history:
            acceleration = (delta - history[-1]) / scale
            if np.max(abs(acceleration)) > 1.8:
                add("运镜急停、加速或跳位", REVIEW if slow else FAIL)
        history.append(delta)
        history = history[-7:]
        if len(history) >= 4 and np.max(abs(np.mean(history[-4:], axis=0)) / scale) > 2:
            add("运镜速度或方向持续偏离", REVIEW)
        if len(history) >= 4:
            offsets = np.array(history) / scale
            alternating = np.sum(
                (offsets[1:] * offsets[:-1] < 0)
                & (abs(offsets[1:] - offsets[:-1]) > 0.65),
                axis=0,
            )
            if max(alternating) >= 3:
                add("连续往返抖动", REVIEW if slow else FAIL)
    # Audit reference events too. A missing cut or flash must not disappear when the export has fewer frames.
    for r in ref:
        if r["t"] < start or r["t"] - r["dt"] >= end:
            continue
        idx = int(
            np.clip(np.searchsorted(replica_times, r["t"] - 1e-6), 0, len(rep) - 1)
        )
        candidates = [rep[idx]]
        if abs(cadence[0] - cadence[1]) > 0.12 * min(cadence) and idx:
            candidates.append(rep[idx - 1])
        a = r
        b = candidates[0]
        if (
            r["structure"] > 0.27
            and all(r["structure"] - o["structure"] > 0.22 for o in candidates)
            and r["mismatch"] > 0.35
        ):
            add("原片事件缺失或错时", REVIEW)
    # Look for a persistent content offset, even when timestamp origins match.
    # Use several frames so a single occlusion or moving foreground cannot set the lag.
    if len(checks) >= 8:
        sample = checks[:: max(1, len(checks) // 48)]
        ref_indices = np.array(
            [row["reference_frame"] - ref[0]["frame"] for row in sample]
        )
        cmp_indices = np.array([row["frame"] - rep[0]["frame"] for row in sample])
        max_lag = min(15, round(0.5 / cadence[0]), len(ref) // 3)
        scored = []
        for lag in range(-max_lag, max_lag + 1):
            indices = ref_indices + lag
            valid = (indices >= 0) & (indices < len(ref))
            if valid.sum() < max(6, len(sample) * 0.65):
                continue
            gains = [
                aligned_visual(thumbnails[0][i], thumbnails[1][k])
                - c["spatial_correlation"]
                for i, k, c, v in zip(indices, cmp_indices, sample, valid)
                if v
            ]
            scored.append((float(np.median(gains)), lag))
        if scored and max(scored)[0] > 0.09:
            _gain, lag = max(scored)
            b = rep[cmp_indices[len(sample) // 2]]
            a = ref[ref_indices[len(sample) // 2]]
            add(f"内容时间对齐存疑（候选偏移 {lag} 原片帧）", REVIEW)
            for item in defects:
                item["status"] = REVIEW
    return checks, defects


def evidence(reference, replica, out, intervals, findings):
    folder = out / "evidence"
    folder.mkdir(exist_ok=True)
    for n, (start, end) in enumerate(intervals):
        # Use a display-time grid only for playback evidence, never for detection.
        args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        for movie in (reference, replica):
            args += ["-ss", str(start), "-t", str(end - start), "-i", str(movie)]
        boxes = []
        reference_boxes = []
        for item in findings:
            when = item.get("reference_event_time")
            if "reference_bbox" in item and start <= when <= end:
                x, y, w, h = item["reference_bbox"]
                t = when - start
                reference_boxes.append(
                    f"drawbox=x=iw*{x}:y=ih*{y}:w=iw*{w}:h=ih*{h}:t=2:color=cyan:enable='between(t,{max(0, t - 0.035):.4f},{t + 0.04:.4f})'"
                )
        reference_overlay = ",".join(reference_boxes + [""])
        for item in findings:
            if "bbox" in item and start <= item["time"] <= end:
                x, y, w, h = item["bbox"]
                t = item["time"] - start
                boxes.append(
                    f"drawbox=x=iw*{x}:y=ih*{y}:w=iw*{w}:h=ih*{h}:t=2:color=yellow:enable='between(t,{max(0, t - 0.035):.4f},{t + 0.04:.4f})'"
                )
        overlay = ",".join(boxes)
        if overlay:
            overlay += ","
        args += [
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            (
                f"[0:v]fps=30,{reference_overlay}scale=480:480:force_original_aspect_ratio=decrease,pad=480:480:(ow-iw)/2:(oh-ih)/2,setsar=1[a];"
                f"[1:v]fps=30,{overlay}scale=480:480:force_original_aspect_ratio=decrease,pad=480:480:(ow-iw)/2:(oh-ih)/2,setsar=1[b];"
                "[a][b]hstack=shortest=1[v]"
            ),
            "-map",
            "[v]",
            "-an",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-threads",
            "2",
        ]
        normal = folder / f"{n:03}-normal.mp4"
        slow = folder / f"{n:03}-slow.mp4"
        with tempfile.TemporaryDirectory(dir=folder) as graph_dir:
            pos = args.index("-filter_complex")
            graph = Path(graph_dir) / "evidence.ffgraph"
            graph.write_text(args[pos + 1], encoding="utf-8")
            if len(args[pos + 1]) > 48000:
                version = subprocess.check_output(
                    ["ffmpeg", "-version"], text=True
                ).split()[2]
                option = (
                    "-/filter_complex"
                    if version[:1].isdigit() and int(version.split(".")[0]) >= 7
                    else "-filter_complex_script"
                )
                args[pos : pos + 2] = [option, str(graph)]
            subprocess.run(args + [str(normal)], check=True)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(normal),
                "-vf",
                "setpts=4*PTS",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                str(slow),
            ],
            check=True,
        )
        yield {
            "start": start,
            "end": end,
            "normal": str(normal.relative_to(out)),
            "slow": str(slow.relative_to(out)),
        }


def write_verdict(path, report):
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    Path(handle.name).replace(path)


def event_intervals(findings, pairs):
    # Coalesce adjacent observations while retaining frame-level audit records.
    by_frame = {r["frame"]: r for r in pairs}
    groups = {}
    for item in sorted(findings, key=lambda row: row["frame"]):
        key = (item["kind"], item["region"], item["status"])
        intervals = groups.setdefault(key, [])
        r = by_frame[item["frame"]]
        if intervals and r["frame"] <= intervals[-1]["last_frame"] + 1:
            intervals[-1].update(last_frame=r["frame"], end=r["t"])
        else:
            intervals.append(
                {
                    "first_frame": r["frame"],
                    "last_frame": r["frame"],
                    "start": r["t"] - r["dt"],
                    "end": r["t"],
                }
            )
    return sorted(
        [
            dict(
                kind=k[0], region=k[1], status=k[2], duration=i["end"] - i["start"], **i
            )
            for k, items in groups.items()
            for i in items
        ],
        key=lambda item: item["start"],
    )


def run(args, load, decode, identify):
    cv.setNumThreads(3)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "version": VERSION,
        "status": REVIEW,
        "coverage": {},
        "findings": [],
        "evidence": [],
        "complete": False,
    }
    dest = args.out / "continuity.json"
    write_verdict(dest, report)
    code = 3
    try:
        identity = [identify(p) for p in (args.reference, args.replica)]
        signature = json.dumps(
            [identity, args.start, args.end, VERSION], sort_keys=True
        )
        report["review_id"] = hashlib.sha256(signature.encode()).hexdigest()
        report["inputs"] = identity
        start = args.start if args.start is not None else 0.0
        end = args.end if args.end is not None else float("inf")
        if (
            not math.isfinite(start)
            or start < 0
            or end <= start
            or (args.end is not None and not math.isfinite(end))
        ):
            raise ValueError("时间范围无效")
        movies = [
            read_movie(p, start, end, load, decode)
            for p in (args.reference, args.replica)
        ]
        ref, rep = [m[0] for m in movies]
        duration = [m[2] for m in movies]
        cadence = [m[4] for m in movies]
        stop = min(end, *duration)
        report["coverage"] = {
            "requested": [start, args.end],
            "compared_seconds": [start, stop],
            "duration_seconds": duration,
            "frames_in_files": [m[3] for m in movies],
        }
        missing = (end != float("inf") and end > min(duration) + 0.001) or (
            args.end is None and abs(duration[0] - duration[1]) > 0.001
        )
        checks, defects = evaluate(
            ref, rep, [m[1] for m in movies], cadence, start, stop
        )
        if not checks or start >= stop:
            raise ValueError("没有可对照的相邻帧")

        def relevant(rows):
            return [
                r
                for r in rows
                if r["t"] >= start - 1e-7 and r["t"] - r["dt"] < stop + 1e-7
            ]

        selected = [relevant(ref), relevant(rep)]
        report["coverage"]["native_pairs"] = [len(group) for group in selected]
        report["coverage"]["replica_pairs_compared"] = len(checks)
        report["measurements"] = "pairs.json"
        report["complete"] = not missing and len(checks) == len(selected[1])
        if not report["complete"]:
            report["coverage"]["reason"] = "审查范围或片尾有未覆盖时间"
        # Persist measurements before any reviewer decisions, including indices of every native pair.
        (args.out / "pairs.json").write_text(
            json.dumps(
                {
                    "reference": [
                        {
                            k: v
                            for k, v in r.items()
                            if k
                            not in (
                                "patch_flow",
                                "border_palette",
                                "signature",
                                "blobs",
                            )
                        }
                        for r in selected[0]
                    ],
                    "replica": [
                        {
                            k: v
                            for k, v in r.items()
                            if k
                            not in (
                                "patch_flow",
                                "border_palette",
                                "signature",
                                "blobs",
                            )
                        }
                        for r in selected[1]
                    ],
                    "matches": checks,
                },
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        decisions = {}
        if args.decisions:
            review = json.loads(args.decisions.read_text(encoding="utf-8-sig"))
            if review["review_id"] != report["review_id"]:
                raise ValueError("复核记录与输入文件或区间不符")
            decisions = review["decisions"]
        intervals = []
        for row in defects:
            key = (
                f"{row['frame']}:{row['reference_frame']}:{row['kind']}:{row['region']}"
            )
            row["id"] = key
            row["detected"] = row["status"]
            if key in decisions:
                decision = decisions[key]
                if (
                    decision.get("status") not in (PASS, FAIL)
                    or not str(decision.get("reason", "")).strip()
                ):
                    raise ValueError("人工复核须写明结论及判据")
                row["status"] = decision["status"]
                row["review_reason"] = decision["reason"]
            peer_time = row.get("reference_event_time", row["time"])
            intervals.append(
                (
                    max(0, min(row["time"], peer_time) - 0.35),
                    min(min(duration), max(row["time"], peer_time) + 0.45),
                )
            )
        report["findings"] = defects
        report["events"] = event_intervals(defects, rep)
        if set(decisions) - {r["id"] for r in defects}:
            raise ValueError("复核记录存在不匹配的候选")
        merged = []
        for left, right in sorted(intervals):
            if merged and left <= merged[-1][1] + 0.12:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        report["evidence"] = list(
            evidence(args.reference, args.replica, args.out, merged, defects)
        )
        status = {r["status"] for r in defects}
        report["status"] = (
            FAIL
            if FAIL in status
            else REVIEW
            if REVIEW in status or not report["complete"]
            else PASS
        )
        code = {PASS: 0, FAIL: 2, REVIEW: 3}[report["status"]]
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        subprocess.SubprocessError,
        cv.error,
    ) as exc:
        report["status"] = REVIEW
        report["complete"] = False
        report["error"] = str(exc)
    write_verdict(dest, report)
    print(
        f"continuity {report['status']} | {len(report['findings'])} findings | {dest}"
    )
    return code
