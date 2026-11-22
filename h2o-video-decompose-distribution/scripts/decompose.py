#!/usr/bin/env python3
"""Inventory source frames or measure a reviewed interval."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

from measure_core import flow, ncc_matches, normalize, pick, project, sample, solve_pair, stats


def write_image(path, image):
    """Encode with OpenCV and let Python open the Unicode filesystem path."""
    path = Path(path)
    success, encoded = cv.imencode(path.suffix, image)
    if not success:
        raise OSError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_source(source):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,time_base,duration,color_transfer:stream_side_data=rotation:"
        "frame=best_effort_timestamp_time,duration_time,pkt_duration_time,key_frame,pict_type",
        "-of", "json", str(source),
    ]
    data = json.loads(subprocess.check_output(command))
    decoded = data.get("frames", [])
    if not decoded:
        raise ValueError("No decodable video frames on video stream 0.")
    first = decoded[0].get("best_effort_timestamp_time")
    origin = float(first) if first is not None else None
    frames = []
    for index, frame in enumerate(decoded):
        pts = frame.get("best_effort_timestamp_time")
        duration = frame.get("duration_time") or frame.get("pkt_duration_time")
        frames.append(dict(frame=index, pts_s=float(pts) if pts is not None else None,
                           time_s=round(float(pts)-origin, 9) if pts is not None and origin is not None else None,
                           duration_s=float(duration) if duration is not None else None,
                           key_frame=frame.get("key_frame"), picture_type=frame.get("pict_type")))
    stream = data["streams"][0]
    # Both probe and decode use coded orientation. Record display rotation separately.
    return frames, stream


def source_id(source, include_digest=False):
    s = source.stat()
    identity = dict(path=str(source.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)
    if include_digest:
        with source.open("rb") as video:
            identity["sha256"] = hashlib.file_digest(video, "sha256").hexdigest()
    return identity


def same_source(first, second):
    if first.get("sha256") and second.get("sha256"):
        return first["bytes"] == second["bytes"] and first["sha256"] == second["sha256"]
    return all(first[key] == second[key] for key in ("path", "bytes", "mtime_ns"))


def verify_source(source, expected):
    actual = source_id(source, include_digest=bool(expected.get("sha256")))
    if not same_source(actual, expected):
        raise ValueError("Source does not match the recorded video. Use the same file or regenerate the scan.")


def raw_frames(source, size, start=0, end=None):
    w, h = size
    selection = [] if start == 0 and end is None else [f"select=between(n\\,{start}\\,{end-1})"]
    filters = selection + [f"scale={w}:{h}:flags=area"]
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-noautorotate", "-i", str(source),
           "-map", "0:v:0", "-an", "-sn", "-vf", ",".join(filters), "-fps_mode", "passthrough",
           "-pix_fmt", "bgr24", "-f", "rawvideo", "-threads", "1"]
    if end is not None:
        cmd.extend(["-frames:v", str(end-start)])
    proc = subprocess.Popen(cmd+["pipe:1"], stdout=subprocess.PIPE)
    frame_size = w*h*3
    try:
        while data := proc.stdout.read(frame_size):
            if len(data) != frame_size:
                raise ValueError("FFmpeg returned an incomplete frame.")
            yield np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        if proc.wait():
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def size_for(size, maximum):
    w, h = size
    scale = min(1., maximum/max(w, h))
    return max(1, round(w*scale)), max(1, round(h*scale))


def sheet(out, cells, times, stem, size):
    w, h = size
    for offset in range(0, len(cells), 36):
        pack = cells[offset:offset+36]
        columns = min(6, len(pack))
        rows = math.ceil(len(pack)/columns)
        page = Image.new("RGB", (columns*w, rows*(h+22)), "#15191f")
        pen = ImageDraw.Draw(page)
        for pos, (idx, frame) in enumerate(pack):
            x = (pos%columns)*w
            y = (pos//columns)*(h+22)
            page.paste(Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)), (x, y))
            t = times[idx].get("time_s")
            label = f"f{idx}  {t:.3f}s" if t is not None else f"f{idx}  PTS unavailable"
            pen.text((x+4, y+h+4), label, fill="white")
        page.save(out/f"{stem}-{offset//36+1:03}.jpg", quality=91)


def _frame_optical_signature(image):
    """Lightweight optical-regime signature computed on the already-downsampled (<=180px) scan frame."""
    rgb_f = image[:, :, ::-1].astype(np.float32) / 255.0
    lin = np.power(np.clip(rgb_f, 0.0, 1.0), 2.2)
    lum = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
    sat = cv.cvtColor(image, cv.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    p50 = float(np.percentile(lum, 50))
    p97 = float(np.percentile(lum, 97))
    emissive = float(np.mean((lum >= 0.28) & (sat >= 0.30))) if p50 < 0.38 else 0.0
    contrast = float(p97 / max(0.008, p50)) if p50 < 0.45 else 0.0
    non_void = float(np.mean(lum >= 0.005))
    low_key = float(max(0.0, 0.12 - p50) * non_void)
    return round(emissive, 5), round(contrast, 3), round(low_key, 4), round(p50, 4)


def _select_lookdev_keyframes(opt_records, fps):
    """Pick diverse, non-overlapping optical-regime keyframes across the scanned range."""
    if not opt_records:
        return []
    min_gap = max(6, int(round(max(1.0, fps) * 0.6)))
    n_total = len(opt_records)
    n_windows = max(1, min(6, int(math.ceil(n_total / max(30.0, fps * 2.5)))))
    win_len = max(1, math.ceil(n_total / n_windows))

    selected = []

    def try_add(rec, role):
        if any(abs(rec["frame"] - s["frame"]) < min_gap for s in selected):
            return False
        entry = dict(rec)
        entry["role"] = role
        selected.append(entry)
        return True

    by_em = sorted(opt_records, key=lambda r: r["emissive_score"], reverse=True)
    if by_em and by_em[0]["emissive_score"] >= 0.004:
        try_add(by_em[0], "PEAK_EMISSIVE_OR_HIGHLIGHT")
    by_con = sorted(opt_records, key=lambda r: r["contrast_score"], reverse=True)
    if by_con and by_con[0]["contrast_score"] >= 2.5:
        try_add(by_con[0], "PEAK_TONAL_CONTRAST")
    by_lk = sorted(opt_records, key=lambda r: r["low_key_score"], reverse=True)
    if by_lk and by_lk[0]["low_key_score"] >= 0.02:
        try_add(by_lk[0], "LOW_KEY_MOOD")

    for w in range(n_windows):
        chunk = opt_records[w * win_len : (w + 1) * win_len]
        if not chunk:
            continue
        em_best = max(chunk, key=lambda r: r["emissive_score"])
        if em_best["emissive_score"] >= 0.008:
            try_add(em_best, "WINDOW_EMISSIVE_PEAK")
        mid = chunk[len(chunk) // 2]
        try_add(mid, "WINDOW_MIDPOINT_BASELINE")

    selected.sort(key=lambda r: r["frame"])
    return selected


def scan(source, out, interval, max_candidates, start, end):
    frames, stream = load_source(source)
    end = len(frames) if end is None else end
    if not 0 <= start < end <= len(frames):
        raise ValueError("Scan frame range is outside the source.")
    size = size_for((stream["width"], stream["height"]), 180)
    previous = None
    previous_frame = None
    next_mark = None
    story = []
    candidates = []
    scores = []
    opt_records = []
    for n, image in enumerate(raw_frames(source, size, start, end), start):
        frame = frames[n]
        time = frame["time_s"]
        if n == start or (time is not None and (next_mark is None or time>=next_mark)) or (
            time is None and (n-start)%30 == 0):
            story.append((n, image.copy()))
            next_mark = time+interval if time is not None else None
        em_sc, con_sc, lk_sc, p50_sc = _frame_optical_signature(image)
        opt_records.append(dict(
            frame=n, time_s=time,
            emissive_score=em_sc, contrast_score=con_sc, low_key_score=lk_sc, p50_midtone=p50_sc,
        ))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            mae = float(np.mean(np.abs(gray-previous)))
            scores.append(dict(frame=n, time_s=time, adjacent_luma_mae=mae))
            if len(candidates)<max_candidates or mae>candidates[0][0]:
                candidates.append((mae, n, previous_frame, image.copy()))
                candidates.sort(key=lambda item: item[0])
                if len(candidates)>max_candidates:
                    candidates.pop(0)
        previous, previous_frame = gray, image.copy()
    if len(scores)+1 != end-start:
        raise ValueError("Probe and decoder disagree on frame count for the selected range.")
    metadata = dict(source=source_id(source, include_digest=True), stream=stream, rotation_applied=False,
                    scanned_start=start, scanned_end_exclusive=end,
                    frame_count=len(frames), frames=frames)
    write_json(out/"source.json", metadata)
    write_csv(out/"frames.csv", frames)
    if scores:
        write_csv(out/"adjacent-difference.csv", scores)
    candidates.sort(key=lambda item:item[1])
    write_json(out/"candidate-cuts.json", [dict(frame=n, time_s=frames[n]["time_s"], mae=mae)
                                         for mae,n,_,_ in candidates])
    fps_est = float(stream.get("fps") or 30.0)
    lk_frames = _select_lookdev_keyframes(opt_records, fps_est)
    write_json(out/"lookdev-keyframes.json", dict(
        selection_policy="OPTICAL_REGIME_COVERAGE_V1",
        scanned_start=start,
        scanned_end_exclusive=end,
        keyframes=lk_frames,
    ))
    sheet(out, story, frames, "story", size)
    neighbors = [(idx,img) for _,n,a,b in candidates for idx,img in [(n-1,a),(n,b)]]
    sheet(out, neighbors, frames, "adjacent-pairs", size)


def make_mask(size, rectangles, frame):
    w,h = size
    mask = np.zeros((h,w), np.uint8)
    for item in rectangles:
        if isinstance(item, dict):
            keys = np.asarray(item["keys"], dtype=float)
            if keys.ndim!=2 or keys.shape[1]!=5 or not len(keys) or not np.all(np.diff(keys[:,0])>0):
                raise ValueError("ROI keys must have increasing frame indices and four coordinates.")
            item = [np.interp(frame, keys[:,0], keys[:,col]) for col in range(1,5)]
        l,t,r,b = item
        if not (0<=l<r<=1 and 0<=t<b<=1):
            raise ValueError(f"Invalid normalized rectangle at frame {frame}: {item}")
        x0,y0,x1,y1 = np.rint(np.asarray(item)*[w,h,w,h]).astype(int)
        mask[y0:y1,x0:x1] = 255
    return mask


def fit_ref(a, b, guess, gray, points, templates, mask_b):
    aa,bb = ncc_matches(templates[a], gray[b], points[a], guess)
    if len(bb)<9:
        return None
    xy = np.rint(bb).astype(int)
    h,w = gray[b].shape
    good = (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    good[good] = mask_b[xy[good,1],xy[good,0]] != 0
    if int(sum(good))<9:
        return None
    span = np.minimum(np.ptp(aa[good],axis=0), np.ptp(bb[good],axis=0))
    if span[0]<25 or span[1]<18:
        return None
    fit = solve_pair(aa[good], bb[good])
    if fit is None or fit["holdout_count"]<3 or fit["holdout_p95_px"]>5.5:
        return None
    fit["support_span_px"] = span.tolist()
    return fit


def register(gray, size, shot, anchors):
    lo,hi,ref = (shot[k] for k in ("start","end_exclusive","ref"))
    if hi-lo == 1 or shot.get("geometry") == "none":
        return {}
    masks = {f: make_mask(size, shot["stable"], f) for f in gray}
    for f in masks:
        masks[f][make_mask(size, shot.get("exclude", []),f)>0] = 0
    nodes = sorted(set([ref]+shot.get("bridges", [])+list(range(lo,hi,24))))
    pts = {n:pick(gray[n],masks[n],12) for n in nodes}
    templates = {n:normalize(gray[n]) for n in nodes}
    adjacent = {}
    for f in range(lo,hi-1):
        a,b = flow(gray[f],gray[f+1],pick(gray[f],masks[f],13))
        if len(b):
            xy = np.rint(b).astype(int)
            ok = masks[f+1][xy[:,1],xy[:,0]] != 0
            fit = solve_pair(a[ok],b[ok],False)
            if fit and fit["inlier_observations"]>=7:
                adjacent[f]=np.asarray(fit["H"])
    def seed(a,b):
        h=np.eye(3)
        sign=1 if b>a else -1
        for f in range(a,b,sign):
            step=adjacent.get(min(f,f+sign))
            if step is None:
                return np.eye(3) if abs(a-b)<=24 else None
            try:
                h = (step if sign>0 else np.linalg.inv(step)) @ h
            except np.linalg.LinAlgError:
                return None
        return h
    def register_local(a,b):
        guess=seed(a,b)
        return fit_ref(a,b,guess,gray,pts,templates,masks[b]) if guess is not None else None
    def assemble_fit(a, fit):
        parent=calibrated[a]
        h=np.asarray(fit["H"])@np.asarray(parent["H"])
        # Limit extrapolation across a projective pole at the measured anchors.
        den=np.column_stack([anchors,np.ones(len(anchors))])@h[2]
        xy=project(h,anchors)
        if abs(h[2,2])<1e-8 or np.min(den)<=0 or np.max(den)/np.min(den)>4 or (
                not np.isfinite(xy).all() or abs(xy).max()>5*max(size)):
            return None
        return dict(fit,H=(h/h[2,2]).tolist(),
                    local_reference_frame=a, bridge_depth=parent["bridge_depth"]+(a!=ref),
                    status="validated" if a==ref and fit["holdout_p95_px"]<=2.75 and hi-lo>=3 else "limited")
    calibrated={ref:dict(H=np.eye(3).tolist(),model="identity",bridge_depth=0,status="reference_only")}
    for sign in (1,-1):
        sequence=sorted((n for n in nodes if (n-ref)*sign>0),reverse=sign<0)
        for b in sequence:
            options=[]
            for a in sorted(calibrated,key=lambda n:abs(n-b))[:3]:
                local=register_local(a,b)
                if local and (fit:=assemble_fit(a,local)):
                    options.append(fit)
            if options:
                calibrated[b]=min(options,key=lambda f:f["holdout_p95_px"]+.55*f["bridge_depth"])
    results={}
    for b in gray:
        if b==ref:
            results[b]=calibrated[b]
            continue
        options=[calibrated[b]] if b in calibrated else []
        near=sorted((a for a in calibrated if a!=b),key=lambda n:abs(n-b))[:2]
        for a in dict.fromkeys([ref]+near):
            if abs(b-a)>110:
                continue
            local=register_local(a,b)
            if local and (fit:=assemble_fit(a,local)):
                options.append(fit)
        if options:
            results[b]=min(options,key=lambda f:f["holdout_p95_px"]+.55*f["bridge_depth"])
    return results


def region_optics(bgr, size, f, zones):
    if not zones:
        return {},dict(zone=None,evidence="not_sampled")
    gray=bgr.astype(np.float32) @ np.float32([.0722,.7152,.2126])
    sm=cv.GaussianBlur(gray,(5,5),.75)
    dx=cv.Sobel(sm,cv.CV_32F,1,0,ksize=3,scale=1/8)
    dy=cv.Sobel(sm,cv.CV_32F,0,1,ksize=3,scale=1/8)
    values={}
    for label,rectangles in zones.items():
        prof=sample(gray,dx,dy,make_mask(size,rectangles,f),10)
        values[label]=stats(prof)
    ranked=sorted((v["q30"],name) for name,v in values.items() if v and v["n"]>=5)
    if not ranked:
        return values,dict(zone=None,evidence="unresolved")
    q30,name=ranked[0]
    gap=ranked[1][0]-q30 if len(ranked)>1 else None
    return values,dict(zone=name,xy_px=values[name]["sharp_xy"],runner_up_gap_px=gap,
                       evidence="single_zone" if gap is None else "separated" if gap>=.45 else "ambiguous")


def geometry_metrics(h, anchors):
    p=project(h,anchors)
    u=p[2]-p[1]
    return dict(anchor_x_px=float(p[0,0]),anchor_y_px=float(p[0,1]),
                horizontal_span_px=float(np.linalg.norm(u)),vertical_span_px=float(np.linalg.norm(p[4]-p[3])),
                horizontal_angle_deg=float(np.degrees(np.arctan2(u[1],u[0]))))


def check_anchors(shot, size, f, h):
    checks=[]
    for item in shot.get("independent_points", []):
        xy=item["observations"].get(str(f))
        if xy is None:
            continue
        reference=np.asarray([item["reference"]])*size
        observed=np.asarray(xy)*size
        expected=project(h,reference)[0] if h is not None else None
        checks.append(dict(name=item["name"], observed_px=observed.tolist(),
                           predicted_px=expected.tolist() if expected is not None else None,
                           error_px=float(np.linalg.norm(observed-expected)) if expected is not None else None))
    return checks


def review_frame(rgb, row, shot, anchors):
    w,h=rgb.shape[1],rgb.shape[0]
    output=rgb.copy()
    g=row["geometry"]
    if g.get("H") is not None:
        x,y=anchors[0]
        rx=np.linalg.norm(anchors[2]-anchors[1])/2
        ry=np.linalg.norm(anchors[4]-anchors[3])/2
        color=(40,170,246) if g["status"]=="limited" else (238,177,60)
        lines=[[[a,y-ry],[a,y+ry]] for a in np.linspace(x-rx,x+rx,5)]
        lines += [[[x-rx,b],[x+rx,b]] for b in np.linspace(y-ry,y+ry,5)]
        for ends in lines:
            p=project(g["H"],np.asarray(ends))
            if np.isfinite(p).all() and abs(p).max()<8*max(w,h):
                cv.line(output,*[tuple(z) for z in np.rint(p).astype(int)],color,2,cv.LINE_AA)
    if row["focus"].get("xy_px"):
        cv.circle(output,tuple(np.rint(row["focus"]["xy_px"]).astype(int)),9,(40,170,246),2)
    for point in row["independent_checks"]:
        cv.circle(output,tuple(np.rint(point["observed_px"]).astype(int)),4,(82,224,135),-1)
    cv.putText(output,f'{shot["id"]} | {g["status"]}',(8,24),cv.FONT_HERSHEY_SIMPLEX,.56,(250,245,244),1,cv.LINE_AA)
    return output


def measure(source, scan_path, plan_path, out, cap, review_step, annotated_frames):
    meta=json.loads(scan_path.read_text(encoding="utf-8-sig"))
    verify_source(source, meta["source"])
    shots=json.loads(plan_path.read_text(encoding="utf-8-sig"))["shots"]
    times=meta["frames"]
    coded=(meta["stream"]["width"],meta["stream"]["height"])
    size=size_for(coded,cap)
    if not shots:
        raise ValueError("The plan has no reviewed intervals.")
    ids=[s["id"] for s in shots]
    if len(set(ids))!=len(ids):
        raise ValueError("Interval IDs must be unique.")
    overlay_dir=out/"annotated"
    if annotated_frames:
        overlay_dir.mkdir(exist_ok=True)
    rows=[]
    summary=[]
    review=[]
    for idx,shot in enumerate(shots):
        lo,hi,ref=(shot[k] for k in ("start","end_exclusive","ref"))
        if not (0<=lo<=ref<hi<=len(times)) or (idx and shots[idx-1]["end_exclusive"]>lo):
            raise ValueError(f"Unsorted or invalid frame bounds for {shot['id']}.")
        if any(not lo<=n<hi for n in shot.get("bridges", [])):
            raise ValueError(f"Out-of-shot bridge for {shot['id']}.")
        if shot.get("geometry")!="none" and not shot.get("stable"):
            raise ValueError(f"Select stable features or disable geometry for {shot['id']}.")
        if (hi-lo)*math.prod(size)>82_000_000:
            raise ValueError(f"{shot['id']} is too large for one run. Use overlapping analysis windows or --max-side.")
        anchor=np.asarray(shot.get("anchors",[[.5,.5],[.38,.5],[.62,.5],[.5,.42],[.5,.58]]),float)*size
        if anchor.shape!=(5,2) or not np.isfinite(anchor).all():
            raise ValueError("Expected five finite normalized anchors per shot.")
        rgb={f:a.copy() for f,a in enumerate(raw_frames(source,size,lo,hi),lo)}
        if len(rgb)!=hi-lo:
            raise ValueError("Decoded fewer frames than expected.")
        gray={f:cv.cvtColor(im,cv.COLOR_BGR2GRAY) for f,im in rgb.items()}
        fits=register(gray,size,shot,anchor)
        part=[]
        for f in gray:
            g=fits.get(f,dict(H=None,status="single_frame" if hi-lo==1 else (
                "not_requested" if shot.get("geometry")=="none" else "unresolved")))
            optics,focus=region_optics(rgb[f],size,f,shot.get("zones",{}))
            h=g["H"]
            row=dict(**times[f],shot_id=shot["id"],geometry=g,metrics=geometry_metrics(h,anchor) if h else {},
                     zones=optics,focus=focus,independent_checks=check_anchors(shot,size,f,h))
            part.append(row)
            review_selected=f in (lo,ref,hi-1) or (f-lo)%review_step==0
            if annotated_frames or review_selected:
                preview=review_frame(rgb[f],row,shot,anchor)
            if annotated_frames:
                write_image(overlay_dir / f"{f:09}.png", preview)
            if review_selected:
                review.append((f,cv.resize(preview,size_for(size,240),interpolation=cv.INTER_AREA)))
        rows.extend(part)
        summary.append(dict(id=shot["id"],name=shot.get("name",shot["id"]),
            boundary=shot.get("boundary"),start=lo,end_exclusive=hi,reference_frame=ref,
            anchors_px=anchor.tolist(),start_time_s=times[lo]["time_s"],
            end_time_s=times[hi]["time_s"] if hi<len(times) else (
                times[hi-1]["time_s"]+times[hi-1]["duration_s"] if times[hi-1]["time_s"] is not None and times[hi-1]["duration_s"] else None),
            geometry_states=dict(Counter(r["geometry"]["status"] for r in part))))
        print(f'{shot["id"]}: {hi-lo} frames {summary[-1]["geometry_states"]}',flush=True)
    write_json(out/"measurements.json",dict(source=meta["source"],coded_size=coded,analysis_size=size,
                                            orientation="coded",shots=summary,frames=rows))
    zone_names=list(dict.fromkeys(name for r in rows for name in r["zones"]))
    csv_rows=[]
    for r in rows:
        g=r["geometry"]
        metrics={k:r["metrics"].get(k) for k in ("anchor_x_px","anchor_y_px","horizontal_span_px",
                                                  "vertical_span_px","horizontal_angle_deg")}
        record=dict(frame=r["frame"],time_s=r["time_s"],pts_s=r["pts_s"],shot_id=r["shot_id"],
                    status=g["status"],model=g.get("model"),local_reference_frame=g.get("local_reference_frame"),
                    bridge_depth=g.get("bridge_depth"),holdout_p95_px=g.get("holdout_p95_px"),
                    holdout_count=g.get("holdout_count"),**metrics,
                    focus_zone=r["focus"]["zone"],focus_evidence=r["focus"]["evidence"])
        for y in range(3):
            for x in range(3):
                record[f"H{y+1}{x+1}"]=g["H"][y][x] if g["H"] is not None else None
        for name in zone_names:
            z=r["zones"].get(name) or {}
            for k in ("n","q30","median","xnormal","ynormal"):
                record[f"{name}_{k}"]=z.get(k)
        csv_rows.append(record)
    write_csv(out/"measurements.csv",csv_rows)
    write_csv(out/"shots.csv",summary)
    sheet(out,review,times,"review",size_for(size,240))
    chart(out,rows,summary,zone_names)


def chart(out, rows, shots, zones):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use_time=all(r["time_s"] is not None for r in rows)
    fig,ax=plt.subplots(4,1,figsize=(12,8),sharex=True,constrained_layout=True)
    colors=plt.get_cmap("tab10")
    for i,shot in enumerate(shots):
        group=[r for r in rows if shot["start"]<=r["frame"]<shot["end_exclusive"]]
        t=[r["time_s"] if use_time else r["frame"] for r in group]
        for k,color in [("horizontal_span_px","tab:orange"),("vertical_span_px","tab:blue")]:
            values=[r["metrics"].get(k,np.nan) for r in group]
            ax[0].plot(t,values,color=color,lw=1,label=k if i==0 else None)
            limited=[j for j,r in enumerate(group) if r["geometry"]["status"]=="limited"]
            ax[0].scatter([t[j] for j in limited],[values[j] for j in limited],s=12,
                          facecolors="white",edgecolors=color,linewidths=.6)
        ax[1].plot(t,[r["metrics"].get("horizontal_angle_deg",np.nan) for r in group],color="#276988")
        for z,zone in enumerate(zones):
            values=[(r["zones"].get(zone) or {}).get("q30",np.nan) if (r["zones"].get(zone) or {}).get("n",0)>=5
                    else np.nan for r in group]
            label=f"Zone {z+1}" if f"Zone {z+1}" not in ax[2].get_legend_handles_labels()[1] else None
            if np.isfinite(values).any():
                ax[2].plot(t,values,color=colors(z%10),label=label)
        for status,color in (("validated","tab:blue"),("limited","tab:orange")):
            pairs=[(t[j],r["geometry"]["holdout_p95_px"]) for j,r in enumerate(group)
                   if r["geometry"]["status"]==status and r["geometry"].get("holdout_p95_px") is not None]
            if pairs:
                ax[3].scatter(*zip(*pairs),color=color,s=7)
    for panel,label in zip(ax,("Projected span, px","Line angle, degrees","Edge width P30, px","Local held-out error P95, px")):
        panel.set_ylabel(label)
        panel.grid(alpha=.2)
    ax[0].legend(loc="best")
    if ax[2].get_legend_handles_labels()[0]:
        ax[2].legend(loc="best",ncols=min(4,len(zones)))
        (out/"zone-legend.txt").write_text("\n".join(f"Zone {n+1} = {z}" for n,z in enumerate(zones))+"\n", encoding="utf-8")
    ax[3].set_xlabel("Source time, seconds" if use_time else "Displayed frame index")
    fig.savefig(out/"curves.png",dpi=145)
    plt.close(fig)


def parse_corners(corners_str):
    if not corners_str:
        return None
    pts = []
    for item in corners_str.strip().split(";"):
        if not item.strip():
            continue
        xy = [float(v.strip()) for v in item.split(",")]
        if len(xy) != 2:
            raise ValueError(f"Expected x,y pair in --corners, got: {item}")
        pts.append(xy)
    if len(pts) != 4:
        raise ValueError("Expected exactly 4 corner points 'x1,y1;x2,y2;x3,y3;x4,y4' (TL;TR;BR;BL).")
    return pts


def parse_size_pair(size_str):
    parts = [int(v.strip()) for v in size_str.split(",")]
    if len(parts) != 2 or parts[0] < 16 or parts[1] < 16:
        raise ValueError("Expected 'W,H' with W,H >= 16 for --rect-size.")
    return (parts[0], parts[1])


def scene3d(source, out, frame_idx, window, corners_str, rect_size_str, symmetric_complete, focal_mm, depth_m, objects_plan_path, cap):
    from scene3d_core import (
        analyze_camera_and_projection,
        analyze_lighting_and_volumetrics,
        analyze_material_and_shading,
        analyze_optics_and_motion_3d,
        assess_reconstructability_signals,
        build_lookdev_feature_gates,
        classify_scene_paradigm,
        decompose_3d_objects_and_bom,
        decompose_surface_layers_and_clean_plate,
        rectify_and_analyze_surface,
        save_diagnostics_3d,
    )
    frames_meta, stream = load_source(source)
    if not (0 <= frame_idx < len(frames_meta)):
        raise ValueError(f"Frame {frame_idx} is outside source frame range [0, {len(frames_meta)}).")
    coded = (stream["width"], stream["height"])
    size = size_for(coded, cap)

    half_w = max(1, window // 2)
    lo = max(0, frame_idx - half_w)
    hi = min(len(frames_meta), frame_idx + half_w + 1)
    center_rel = frame_idx - lo

    frames_bgr = list(raw_frames(source, size, lo, hi))
    target_bgr = frames_bgr[center_rel]
    cv.imwrite(str(out / f"frame_{frame_idx:06d}.png"), target_bgr)

    corners_xy = parse_corners(corners_str)
    rect_size = parse_size_pair(rect_size_str)

    cam_info = analyze_camera_and_projection(target_bgr, corners_xy=corners_xy, focal_mm_override=focal_mm, ref_depth_m=depth_m)
    mat_info = analyze_material_and_shading(target_bgr)
    light_info = analyze_lighting_and_volumetrics(target_bgr)
    optics_info = analyze_optics_and_motion_3d(frames_bgr, center_idx=center_rel, cam_info=cam_info, light_info=light_info)
    lookdev_gates = build_lookdev_feature_gates(cam_info, mat_info, light_info, optics_info)
    paradigm_info = classify_scene_paradigm(frames_bgr, cam_info, mat_info, light_info, optics_info)
    reconstructability = assess_reconstructability_signals(cam_info, mat_info, light_info, optics_info, paradigm_info)

    objects_plan = None
    if objects_plan_path is not None:
        raw_plan = json.loads(Path(objects_plan_path).read_text(encoding="utf-8"))
        objects_plan = raw_plan.get("objects", raw_plan) if isinstance(raw_plan, dict) else raw_plan

    bom_info = decompose_3d_objects_and_bom(
        out, frames_bgr, center_rel, cam_info, mat_info, light_info, objects_plan=objects_plan
    )
    write_json(out / "assets_bom.json", bom_info)

    payload = {
        "source": source_id(source, include_digest=True),
        "coded_size": list(coded),
        "analysis_size": list(size),
        "target_frame": frame_idx,
        "target_time_s": frames_meta[frame_idx].get("time_s"),
        "target_pts_s": frames_meta[frame_idx].get("pts_s"),
        "window_range": [lo, hi],
        "scene_paradigm_classification": paradigm_info,
        "reconstructability_signals": reconstructability,
        "lookdev_feature_gates": lookdev_gates,
        "pillar_1_camera_and_projection": cam_info,
        "pillar_3_material_and_bsdf": mat_info,
        "pillar_4_lighting_and_volumetrics": light_info,
        "pillar_5_and_6_motion_and_optics": optics_info,
        "objects_3d_bom": bom_info,
    }

    if corners_xy is not None:
        rect_bgr, topo_info = rectify_and_analyze_surface(
            target_bgr, corners_xy, out_size=rect_size, symmetric_complete=symmetric_complete
        )
        pad_px = light_info["radial_halo_profile"]["recommended_clean_plate_pad_px"]
        fg_mask, mg_mask, height_mask, bg_clean = decompose_surface_layers_and_clean_plate(
            rect_bgr, target_bgr, corners_xy, halo_pad_px=pad_px
        )
        cv.imwrite(str(out / "asset_rectified.png"), rect_bgr)
        cv.imwrite(str(out / "mask_fg_tubes.png"), fg_mask)
        cv.imwrite(str(out / "mask_mg_tubes.png"), mg_mask)
        cv.imwrite(str(out / "mask_height_combined.png"), height_mask)
        cv.imwrite(str(out / "bg_clean.png"), bg_clean)

        # Also save a quantized pixel-art version if voxel step > 1
        step = max(2, topo_info.get("voxel_pixel_step_px") or 4)
        small = cv.resize(rect_bgr, (max(16, rect_size[0] // step), max(16, rect_size[1] // step)), interpolation=cv.INTER_LINEAR)
        pixelated = cv.resize(small, rect_size, interpolation=cv.INTER_NEAREST)
        cv.imwrite(str(out / "asset_pixelated.png"), pixelated)

        payload["pillar_2_geometry_and_topology"] = topo_info
        payload["generated_3d_files"] = {
            "rectified_texture": "asset_rectified.png",
            "pixelated_texture": "asset_pixelated.png",
            "foreground_relief_mask": "mask_fg_tubes.png",
            "midground_relief_mask": "mask_mg_tubes.png",
            "combined_height_mask": "mask_height_combined.png",
            "halo_expanded_clean_background": "bg_clean.png",
            "diagnostics_chart": "diagnostics_3d.png",
            "objects_3d_map": "objects_3d_map.png",
            "assets_bom_json": "assets_bom.json",
            "assets_bom_markdown": "assets_bom.md",
        }

    save_diagnostics_3d(out, target_bgr, cam_info, mat_info, light_info, optics_info)
    write_json(out / "scene3d.json", payload)
    active_gates = [k.replace("enable_", "") for k, v in lookdev_gates.items() if k.startswith("enable_") and v]
    total_gates = sum(1 for k in lookdev_gates if k.startswith("enable_"))
    print(
        f"scene3d frame {frame_idx} ({frames_meta[frame_idx].get('time_s')}s): "
        f"{paradigm_info['detected_paradigm']} | {cam_info['projection_type']} f={cam_info['estimated_focal_mm']}mm | "
        f"geom_conf={reconstructability['geometric_structure_confidence']} engine={reconstructability['suggested_engine']} "
        f"warnings={len(reconstructability['automated_warnings'])} | "
        f"objects={bom_info['object_count']} ({len(bom_info['work_packages_sharding'])} WPs) | "
        f"active_gates({len(active_gates)}/{total_gates})=[{', '.join(active_gates)}]",
        flush=True,
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("scan")
    b=sub.add_parser("measure")
    c=sub.add_parser("scene3d", help="scientifically decompose 3D/Blender camera, objects BOM, geometry, BSDF, lighting, volume, and optics")
    d=sub.add_parser("compare", help="automated bidirectional PASS/FAIL gate comparing original vs replica scene3d.json")
    ab=sub.add_parser("audit-blend", help="static headless audit of a .blend file against source scene3d.json render/lighting requirements")
    ab.add_argument("blend_file", type=Path, help="path to replica .blend project")
    ab.add_argument("--scene3d", type=Path, action="append", default=[], help="one or more source scene3d.json files")
    ab.add_argument("--scene", type=str, action="append", default=None, help="optional Blender scene name(s) to audit")
    ab.add_argument("--out", type=Path, default=None, help="optional directory to save blend_audit.json and blend_audit.md")
    ab.add_argument("--blender", type=Path, default=None, help="optional path to Blender executable")
    ab.add_argument("--strict", action="store_true", help="exit with code 2 when verdict is FAIL")
    fx=sub.add_parser("effects", help="按源时间逐项验收实际导出的效果")
    for name in ("reference", "replica", "plan", "out"):
        fx.add_argument(name, type=Path)
    fx.add_argument("--previous", type=Path, help="同一原片和清单的修前 effects.json")
    fx.add_argument("--decisions", type=Path, help="本次导出和效果清单的人工复核")
    gate=sub.add_parser("continuity", help="gate an exported movie against reference motion and temporal events")
    for name in ("reference", "replica", "out"):
        gate.add_argument(name, type=Path)
    gate.add_argument("--start", type=float, help="first time in seconds on the reference timeline")
    gate.add_argument("--end", type=float, help="exclusive end in seconds, with adjacent seam checks")
    gate.add_argument("--decisions", type=Path, help="reviewed candidate decisions from this exact export and range")
    ld=sub.add_parser("lookdiff", help="compare two videos shot by shot at native pixel scale")
    ld.add_argument("reference",type=Path); ld.add_argument("replica",type=Path); ld.add_argument("out",type=Path)
    for cmd in (a,b,c):
        cmd.add_argument("source", type=Path)
        cmd.add_argument("out", type=Path)
    d.add_argument("orig_json", type=Path, help="original source scene3d.json")
    d.add_argument("render_json", type=Path, help="replica render scene3d.json")
    d.add_argument("--reference-video", type=Path, help="path to the original video if it moved")
    d.add_argument("--render-video", type=Path, help="path to the render video if it moved")
    d.add_argument("--out", type=Path, default=None, help="optional directory to save compare_report.json/.md/.png")
    d.add_argument("--strict", action="store_true", help="exit with code 2 when verdict is FAIL")
    a.add_argument("--interval",type=float,default=2)
    a.add_argument("--max-candidates",type=int,default=40)
    a.add_argument("--start",type=int,default=0)
    a.add_argument("--end",type=int)
    b.add_argument("--scan",type=Path,required=True,help="source.json emitted by scan")
    b.add_argument("--plan",type=Path,required=True,help="human-reviewed interval plan")
    b.add_argument("--max-side",type=int,default=1920)
    b.add_argument("--review-step",type=int,default=12)
    b.add_argument("--annotated-frames",action="store_true",help="also save an overlay PNG for every measured frame")
    c.add_argument("--frame",type=int,required=True,help="target source frame index to decompose")
    c.add_argument("--window",type=int,default=8,help="neighbor frame window around --frame for parallax & shutter analysis")
    c.add_argument("--corners",type=str,default=None,help="optional 4 planar corners 'x1,y1;x2,y2;x3,y3;x4,y4' (TL;TR;BR;BL)")
    c.add_argument("--rect-size",type=str,default="860,280",help="output rectified W,H for --corners")
    c.add_argument("--symmetric-complete",action="store_true",help="complete out-of-frame border/pattern via horizontal symmetry")
    c.add_argument("--focal-mm",type=float,default=None,help="optional focal length override in 36mm equivalent mm")
    c.add_argument("--depth-m",type=float,default=9.0,help="reference center depth in meters for 3D coordinate solve")
    c.add_argument("--objects-plan",type=Path,default=None,help="optional JSON specifying named 3D objects & bboxes to measure and shard into work packages")
    c.add_argument("--max-side",type=int,default=1920,help="maximum analysis image side length")
    args=parser.parse_args()
    if args.command=="effects":
        from effects import run
        raise SystemExit(run(args))
    if args.command=="continuity":
        from continuity import run
        raise SystemExit(run(args, load_source, raw_frames, source_id))
    if args.command=="lookdiff":
        from lookdiff import compare_movies
        result=compare_movies(args.reference,args.replica,args.out)
        print(f"lookdiff: {len(result['windows'])} windows; {len(result['ranked_findings'])} findings")
        return
    if args.command=="audit-blend":
        from blend_audit import audit_blend_file
        rep = audit_blend_file(
            args.blend_file,
            scene3d_paths=args.scene3d,
            out_dir=args.out,
            scene_names=args.scene,
            blender_bin=args.blender,
        )
        s = rep["summary"]
        print(
            f"audit-blend VERDICT={rep['verdict']} | SCENES={s['audited_scene_count']} PASS={s['pass_count']} WARN={s['warn_count']} FAIL={s['fail_count']}",
            flush=True,
        )
        for c in rep["failed_checks"]:
            print(f"  [FAIL] [{c['scene']}] {c['category']}.{c['metric']}: expected={c['expected']} actual={c['actual']} ({c['detail']}) | {c['fix_hint']}", flush=True)
        for c in rep["warning_checks"]:
            print(f"  [WARN] [{c['scene']}] {c['category']}.{c['metric']}: actual={c['actual']} ({c['detail']}) | {c['fix_hint']}", flush=True)
        if args.strict and rep["verdict"] != "PASS":
            raise SystemExit(2)
        return
    if args.command=="compare":
        from scene3d_core import compare_scene3d_pair
        rep = compare_scene3d_pair(
            args.orig_json, args.render_json, out_dir=args.out,
            reference_video=args.reference_video, render_video=args.render_video,
        )
        s = rep["summary"]
        print(
            f"compare VERDICT={rep['verdict']} | PASS={s['pass_count']} WARN={s['warn_count']} FAIL={s['fail_count']}",
            flush=True,
        )
        local = rep["local_appearance"]
        print(f"  局部质感 {local['status']}，{len(local['findings'])} 条证据。", flush=True)
        for c in rep["failed_checks"]:
            print(f"  [FAIL] {c['category']}.{c['metric']}: {c['orig']} -> {c['render']} ({c['detail']}) | {c['fix_hint']}", flush=True)
        for c in rep["warning_checks"]:
            print(f"  [WARN] {c['category']}.{c['metric']}: {c['orig']} -> {c['render']} ({c['detail']})", flush=True)
        if args.strict and rep["verdict"] != "PASS":
            raise SystemExit(2)
        return
    args.out.mkdir(parents=True,exist_ok=True)
    if args.command=="scan":
        if args.interval<=0 or args.max_candidates<1:
            parser.error("scan interval and max-candidates must be positive")
        scan(args.source,args.out,args.interval,args.max_candidates,args.start,args.end)
    elif args.command=="measure":
        if args.max_side<64 or args.review_step<1:
            parser.error("max-side must be at least 64 and review-step positive")
        measure(args.source,args.scan,args.plan,args.out,args.max_side,args.review_step,args.annotated_frames)
    else:
        scene3d(
            args.source, args.out, args.frame, args.window, args.corners,
            args.rect_size, args.symmetric_complete, args.focal_mm, args.depth_m,
            args.objects_plan, args.max_side
        )


if __name__=="__main__":
    main()
