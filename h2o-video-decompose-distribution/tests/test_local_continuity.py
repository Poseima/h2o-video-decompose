"""Score localized events against known boxes in generated scenes or a held-out video."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from test_continuity import SCRIPT, movie


def score(output, fps, size, source=None):
    w, h = size
    n = round(fps * 3)
    rng = np.random.default_rng(71)
    if source:
        data = subprocess.check_output(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(source),
                "-vf",
                f"fps={fps},scale={w}:{h}",
                "-frames:v",
                str(n),
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "-",
            ]
        )
        backgrounds = np.frombuffer(data, np.uint8).reshape(n, h, w, 3)
    else:
        base = cv.GaussianBlur(
            rng.integers(10, 175, (h, w, 3), dtype=np.uint8), (5, 5), 0.9
        )
        backgrounds = [np.roll(base, round(i * 0.7), axis=1) for i in range(n)]
    bw, bh = w // 5, h // 3
    piece = rng.integers(20, 240, (bh, bw, 3), dtype=np.uint8)
    piece[: bh // 2, :, 1] //= 3

    def draw(im, x, y):
        left, right = max(0, x), min(w, x + bw)
        if left < right:
            im[y : y + bh, left:right] = piece[:, left - x : right - x]

    ref = []
    rep = []
    x, y = w // 3, h // 3
    other = w * 2 // 3
    f1, f2, f3, f4 = [round(t * fps) for t in (0.75, 1.4, 2, 2.5)]
    for i, background in enumerate(backgrounds):
        a = background.copy()
        # The same scripted wipe, border reveal, title, cut and flash in both versions.
        if i > f4 + 2:
            a = np.fliplr(a).copy()
        draw(a, int(i / fps * w * 0.55) - bw, 0)
        if i >= f2:
            cv.putText(
                a, "Title", (5, h - 7), cv.FONT_HERSHEY_SIMPLEX, 0.6, (252, 248, 10), 2
            )
        if i == f4 + 3:
            a = cv.add(a, 36)
        ref.append(a)
        b = a.copy()
        if f1 <= i < f4 and i != f3:
            draw(b, x if i < f2 else other, y)
        rep.append(b)
    a = output / "reference.mp4"
    b = output / "test.mp4"
    movie(a, ref, fps)
    movie(b, rep, fps)

    def review(label, candidate):
        folder = output / label
        code = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "continuity",
                str(a),
                str(candidate),
                str(folder),
            ],
            check=False,
        ).returncode
        report = json.loads((folder / "continuity.json").read_text())
        assert report["complete"], report.get("error")
        return code, [f for f in report["findings"] if "bbox" in f]

    _, findings = review("injected", b)
    clean_code, clear = review("clean", a)
    targets = [(f1, x), (f2, x), (f2, other), (f3, other), (f3 + 1, other), (f4, other)]

    def intersects(box, left):
        l, t, bw2, bh2 = box
        return min(l + bw2, (left + bw) / w) > max(l, left / w) and min(
            t + bh2, (y + bh) / h
        ) > max(t, y / h)

    matches = [
        any(f["frame"] == i and intersects(f["bbox"], left) for f in findings)
        for i, left in targets
    ]
    stray = [
        f
        for f in findings
        if not any(
            f["frame"] == i and intersects(f["bbox"], left) for i, left in targets
        )
    ]
    result = {
        "fps": fps,
        "dimensions": size,
        "matched": matches,
        "negative_boxes": len(clear),
        "clean_exit": clean_code,
        "other_boxes": len(stray),
    }
    (output / "score.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)
    assert not clear
    return result


def check_border_timing(output):
    fps = 30
    rng = np.random.default_rng(17)
    bg = rng.integers(40, 130, (200, 260), dtype=np.uint8)
    bg = cv.cvtColor(bg, cv.COLOR_GRAY2BGR)

    def sequence(offset):
        frames = []
        for i in range(80):
            frame = bg.copy()
            x = min(260, max(185, 260 - (i - offset) * 4))
            cv.rectangle(frame, (x, 20), (x + 95, 86), (26, 75, 214), -1)
            frames.append(frame)
        return frames

    ref = output / "onset-reference.mp4"
    rep = output / "onset-early.mp4"
    movie(ref, sequence(32), fps)
    movie(rep, sequence(23), fps)
    folder = output / "early"
    subprocess.run(
        [sys.executable, str(SCRIPT), "continuity", str(ref), str(rep), str(folder)],
        check=False,
    )
    j = json.loads((folder / "continuity.json").read_text())
    assert j["complete"]
    timing = [f for f in j["findings"] if f["kind"] == "连续入画时机偏差"]
    assert any(
        x["bbox"][0] >= 0.7 and abs(x["offset_seconds"] + 0.3) < 0.06 for x in timing
    ), timing
    assert not any(f["kind"] == "局部轮廓突变" for f in j["findings"])


def check_occlusion(output):
    rng = np.random.default_rng(82)
    background = rng.integers(52, 111, (216, 312, 3), dtype=np.uint8)
    frames = []
    for i in range(66):
        im = background.copy()
        x = max(0, 105 + i * 2)
        cv.rectangle(im, (x, 70), (x + 31, 155), (23, 192, 130), -1)
        im[:, 128:182] = background[:, 128:182] // 2
        frames.append(im)
    reference = output / "occlusion.mp4"
    replica = output / "occlusion-transcoded.mp4"
    movie(reference, frames, 30)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(reference),
            "-vf",
            "scale=280:194",
            "-c:v",
            "libx264",
            "-crf",
            "24",
            str(replica),
        ],
        check=True,
    )
    folder = output / "normal-occlusion"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "continuity",
            str(reference),
            str(replica),
            str(folder),
        ],
        check=False,
    )
    report = json.loads((folder / "continuity.json").read_text())
    assert report["complete"]
    assert not any("bbox" in f for f in report["findings"]), report["findings"]


if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("out", type=Path)
    cmd.add_argument("--holdout", type=Path)
    args = cmd.parse_args()
    for fps, size in [(24, (320, 220)), (30, (220, 320))]:
        path = args.out / str(fps)
        path.mkdir(parents=True, exist_ok=True)
        stats = score(path, fps, size, args.holdout)
        assert all(stats["matched"]), stats

    if not args.holdout:
        check_border_timing(args.out)
        check_occlusion(args.out)
