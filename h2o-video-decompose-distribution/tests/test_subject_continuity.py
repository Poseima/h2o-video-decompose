"""Encode a smooth turn, occluded walker and same-colour distractor with paired defects."""

import json
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from test_continuity import SCRIPT, movie


def sequence(fps, defects):
    rng = np.random.default_rng(893)
    bg = rng.integers(55, 120, (160, 240, 3), dtype=np.uint8)
    frames = []
    for i in range(3 * fps):
        t = i / fps
        im = bg.copy()
        # Continuous turn changes the projected width; swap only the interior head texture.
        width = int(20 + 5 * np.cos(t * 3))
        cv.rectangle(im, (30, 40), (64, 90), (73, 104, 88), -1)
        col = (25, 46, 75) if not defects or t < 0.5 else (56, 68, 96)
        cv.rectangle(im, (47 - width, 20), (47 + width, 45), col, -1)
        cv.line(im, (47 - width, 29), (47 + width, 32), tuple(v + 9 for v in col), 2)
        # Gradual occlusion by a fence, then a full-width pillar and reappearance.
        x = round(62 + t * 42)
        if not defects or t < 1.0:
            cv.rectangle(im, (x, 70), (x + 24, 128), (18, 63, 129), -1)
        for fence in (103, 112, 121, 130):
            im[65:132, fence : fence + 3] = bg[65:132, fence : fence + 3]
        im[65:132, 145:162] = bg[65:132, 145:162]
        # A second object of the same colour remains at a different height.
        cv.rectangle(im, (180, 8), (197, 20), (125, 121, 24), -1)
        x = round(240 - max(0, min(0.52, t - (1.75 if not defects else 1.45))) * 110)
        cv.rectangle(im, (x, 98), (x + 40, 149), (125, 121, 24), -1)
        frames.append(im)
    return frames


def run(out, fps):
    out.mkdir(parents=True, exist_ok=True)
    ref = out / "ref.mp4"
    movie(ref, sequence(fps, False), fps)
    for defects in (False, True):
        candidate = out / ("defects.mp4" if defects else "control.mp4")
        if defects:
            movie(candidate, sequence(fps, True), fps)
        else:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(ref),
                    "-vf",
                    "scale=218:146",
                    "-crf",
                    "22",
                    str(candidate),
                ],
                check=True,
            )
        report = out / candidate.stem
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "continuity",
                str(ref),
                str(candidate),
                str(report),
            ],
            check=False,
        )
        j = json.loads((report / "continuity.json").read_text())
        assert j["complete"]
        items = [
            r
            for r in j["findings"]
            if r["kind"]
            in ("局部外观跨帧跳变", "主体入画时机偏差", "遮挡处可见轨迹中断")
        ]
        if not defects:
            assert not items, items
        else:

            def hit(kind, t, x, y, rows=items):
                return any(
                    r["kind"] == kind
                    and abs(r["time"] - t) <= 1.1 / fps
                    and r["bbox"][0] <= x <= sum(r["bbox"][::2])
                    and r["bbox"][1] <= y <= sum(r["bbox"][1::2])
                    for r in rows
                )

            results = {
                "head": hit("局部外观跨帧跳变", 0.5, 0.2, 0.2),
                "walker": any(
                    r["kind"] == "遮挡处可见轨迹中断"
                    and abs(r["time"] - 1) < 1.1 / fps
                    and 0.4 < r["bbox"][0] < 0.56
                    for r in items
                ),
                "entry": any(
                    r["kind"] == "主体入画时机偏差"
                    and abs(r["offset_seconds"] + 0.3) <= 2 / fps
                    and r["bbox"][1] > 0.5
                    for r in items
                ),
            }
            (out / "scores.json").write_text(json.dumps(results))
            print(fps, results, flush=True)
            assert all(results.values()), results


if __name__ == "__main__":
    for fps in (24, 30):
        run(Path(sys.argv[1]) / str(fps), fps)
