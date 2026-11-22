"""Paired short clips. Each underlying project is assigned to exactly one split in split.json."""

import argparse
import hashlib
import json
import pathlib
import subprocess
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
from lookdiff import compare_movies

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=pathlib.Path)
parser.add_argument(
    "assets", type=pathlib.Path, help="directory containing project source videos"
)
parser.add_argument("out", type=pathlib.Path)
parser.add_argument("phase", choices=("development", "holdout"))
args = parser.parse_args()
root = args.out.resolve()
root.mkdir(parents=True, exist_ok=True)
manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
assets = args.assets
phase = args.phase
paths = [
    (assets / title).resolve(strict=True)
    for split in ("development", "holdout")
    for title in manifest[split]
]
if len(set(paths)) != len(paths):
    parser.error("同一素材不能重复或跨组")
cv.setNumThreads(1)


def encode(path, frames, lossy=False):
    h, w = frames[0].shape[:2]
    codec = (
        ["-c:v", "libx264", "-crf", "32", "-pix_fmt", "yuv420p"]
        if lossy
        else ["-c:v", "ffv1", "-pix_fmt", "bgr0"]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            "25",
            "-i",
            "pipe:0",
            *codec,
            "-threads",
            "1",
            str(path),
        ],
        input=b"".join(f.tobytes() for f in frames),
        check=True,
    )


def run(job):
    number, title, case = job
    key = manifest["cases"].index(case)
    folder = root / phase / f"project{number + 1}" / case
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(str(assets / title))
    frames_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.set(cv.CAP_PROP_POS_FRAMES, int(frames_count * (0.22 + 0.028 * key)))
    frames = []
    for _ in range(5):
        ok, im = cap.read()
        if not ok:
            raise ValueError(title)
        frames.append(cv.resize(im, (320, 224), interpolation=cv.INTER_AREA))
    cap.release()
    # Randomized material position. Neither detector nor ranking receives the box.
    seed = int(hashlib.sha256((title + case).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    n = 100
    x = int(rng.integers(3, 218))
    y = int(rng.integers(3, 122))
    yy, xx = np.mgrid[:n, :n]
    bank = [rng.normal(0, 5, (n, n)) for _ in range(2)]
    ref = []
    test = []
    for j, im in enumerate(frames):

        def make(changed, im=im, j=j):
            result = im.copy()
            roi = result[y : y + n, x : x + n].astype(float)
            period, angle = 7, 22
            if changed and case == "texture_spacing":
                period = 11
            if changed and case == "texture_orientation":
                angle += 34
            if case in ("texture_spacing", "texture_orientation"):
                v = 128 + 29 * np.cos(
                    2
                    * np.pi
                    * (xx * np.cos(np.deg2rad(angle)) + yy * np.sin(np.deg2rad(angle)))
                    / period
                )
                roi = roi * 0.12 + v[:, :, None] * 0.88
            if case == "highlight_width":
                width = 12.5 if changed else 6.5
                v = 70 + 158 * np.exp(
                    -((xx - n / 2 - j * 0.5) ** 2 + (yy - n / 2) ** 2) / (2 * width**2)
                )
                roi = roi * 0.08 + v[:, :, None] * 0.92
            if case == "background_softening" and changed:
                roi = cv.GaussianBlur(roi, (0, 0), 2.3)
            if case == "random_seed":
                roi = 128 + np.repeat(bank[int(changed)][:, :, None], 3, axis=2)
            if case == "smooth":
                result[:] = 128 + int(changed)
                roi[:] = 128 + int(changed)
            result[y : y + n, x : x + n] = np.uint8(np.clip(roi, 0, 255))
            return result

        ref.append(make(False))
        test.append(make(True))
    rp, tp = folder / "reference.mkv", folder / "replica.mkv"
    encode(rp, ref)
    encode(tp, test, case == "codec_only")
    report = compare_movies(rp, tp, folder / "diagnosis")
    expected = {
        "texture_spacing": ("period", "增加"),
        "texture_orientation": ("angle", "增加"),
        "highlight_width": ("highlight_width", "增加"),
        "background_softening": ("edge_width", "增加"),
    }.get(case)

    def localized(f):
        X, Y, W, H = f["roi"]
        inter = max(0, min(x + n, X + W) - max(x, X)) * max(
            0, min(y + n, Y + H) - max(y, Y)
        )
        return (
            x <= X + W / 2 < x + n and y <= Y + H / 2 < y + n and inter >= 0.5 * W * H
        )

    top = report["ranked_findings"][:3]
    result = {
        "title": title,
        "case": case,
        "box": [x, y, n, n],
        "expected": expected,
        "located": any(localized(f) for f in top) if expected else None,
        "directed": any(
            localized(f) and (f["metric"], f["direction"]) == expected for f in top
        )
        if expected
        else None,
        "false_alarm": bool(report["ranked_findings"]) if not expected else None,
        "diagnosis": "diagnosis/lookdiff.json",
    }
    (folder / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        title[:5],
        case,
        result["located"],
        result["directed"],
        result["false_alarm"],
        [(f["metric"], f["direction"]) for f in top],
        flush=True,
    )
    return result


jobs = [
    (i, title, case)
    for i, title in enumerate(manifest[phase])
    for case in manifest["cases"]
]
if len(set(manifest["cases"])) != len(manifest["cases"]) or set(manifest["cases"]) - {
    "texture_spacing",
    "texture_orientation",
    "highlight_width",
    "background_softening",
    "identity",
    "smooth",
    "random_seed",
    "codec_only",
}:
    parser.error("重复或未知的试验名")
with ThreadPoolExecutor(max_workers=3) as pool:
    results = list(pool.map(run, jobs))
summary = {}
for k in ("located", "directed", "false_alarm"):
    vals = [r[k] for r in results if r[k] is not None]
    summary[k] = [sum(vals), len(vals)]
(root / (phase + ".json")).write_text(
    json.dumps(
        {
            "summary": summary,
            "phase": phase,
            "detector_sha256": hashlib.sha256(
                pathlib.Path(__file__).with_name("lookdiff.py").read_bytes()
            ).hexdigest(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "cases": [{k: v for k, v in r.items() if k != "report"} for r in results],
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(summary)
