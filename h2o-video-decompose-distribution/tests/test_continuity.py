"""Generate reference/replica fixtures and exercise the public continuity command."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "decompose.py"
WIDTH, HEIGHT = 208, 144


def make_frames():
    rng = np.random.default_rng(17)
    textures = [
        cv.GaussianBlur(
            rng.integers(5, 235, (HEIGHT, WIDTH, 3), dtype=np.uint8), (3, 3), 0.6
        )
        for _ in range(2)
    ]
    for im in textures:
        for _ in range(50):
            x, y = rng.integers(0, [WIDTH, HEIGHT])
            cv.circle(
                im,
                (int(x), int(y)),
                int(rng.integers(2, 9)),
                tuple(map(int, rng.integers(0, 250, 3))),
                -1,
            )
    frames = []
    for i in range(46):
        # Smooth pull and intentionally handheld motion, a common cut, and a common flash.
        mat = cv.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), 0.06 * i, 1 + 0.002 * i)
        mat[0, 2] += i * 1.1 + (2 * np.sin(i * 1.2) if i > 33 else 0)
        im = cv.warpAffine(
            textures[int(i >= 27)],
            mat,
            (WIDTH, HEIGHT),
            borderMode=cv.BORDER_REFLECT_101,
        )
        if i == 30:
            im = cv.add(im, np.full_like(im, 48))
        frames.append(im)
    return frames


def movie(path, frames, fps=24):
    h, w = frames[0].shape[:2]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        input=b"".join(f.tobytes() for f in frames),
        check=True,
    )


def main(folder):
    folder.mkdir(parents=True, exist_ok=True)
    frames = make_frames()
    ref = folder / "ref.mp4"
    movie(ref, frames)

    def run(name, rep, codes, kind=None, extra=(), reference=ref):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "continuity",
                str(reference),
                str(rep),
                str(folder / name),
                *extra,
            ],
            check=False,
        )
        report = json.loads((folder / name / "continuity.json").read_text())
        assert result.returncode in codes, (name, result.returncode, report)
        if kind:
            assert any(kind in issue["kind"] for issue in report["findings"]), (
                name,
                kind,
                report["findings"],
            )
        return report

    good = run("clean", ref, {0})
    assert good["coverage"]["native_pairs"] == [len(frames) - 1] * 2
    cases = {
        "shake": "抖动",
        "jump": "跳位",
        "light": "颜色",
        "cut": "突变",
        "hold": "停顿",
        "local-swap": "局部结构",
    }
    for name, kind in cases.items():
        seq = [im.copy() for im in frames]
        for i in range(12, 20):
            if name == "shake":
                seq[i] = np.roll(seq[i], (1 if i % 2 else -1) * 4, axis=1)
            if name == "jump" and i == 15:
                seq[i] = np.roll(seq[i], 12, axis=0)
            if name == "light":
                seq[i][:75, :110] = cv.add(
                    seq[i][:75, :110], np.full_like(seq[i][:75, :110], 70)
                )
            if name == "cut":
                seq[i] = frames[i + 20]
            if name == "hold":
                seq[i] = seq[11]
            if name == "local-swap":
                seq[i][:55, :77] = frames[i + 20][:55, :77]
        path = folder / (name + ".mp4")
        movie(path, seq)
        run(name, path, {2, 3}, kind)
    flat = folder / "flat.mp4"
    movie(flat, [np.full_like(frames[0], 105) for _ in frames])
    report = run("flat", flat, {3}, "跟踪依据不足", reference=flat)
    decision = folder / "decisions.json"
    decision.write_text(
        json.dumps(
            {
                "review_id": report["review_id"],
                "decisions": {
                    x["id"]: {
                        "status": "通过",
                        "reason": "两片此区间为同一静止纯色画面，原速和慢放核对一致。",
                    }
                    for x in report["findings"]
                },
            }
        )
    )
    run(
        "flat-reviewed", flat, {0}, extra=["--decisions", str(decision)], reference=flat
    )
    short = folder / "short.mp4"
    movie(short, frames[:9])
    assert not run("missing", short, {3})["complete"]
    run("stale", short, {3}, extra=["--decisions", str(decision)])
    # A frozen crop of unrelated texture has no reliable correspondence.
    unrelated = folder / "unrelated.mp4"
    movie(unrelated, [np.flip(im, axis=0).copy() for im in frames])
    run("unrelated", unrelated, {3}, "对应关系待确认")
    # A change exactly at the outgoing range seam must still block approval.
    run(
        "seam-out",
        folder / "jump.mp4",
        {2},
        "跳位",
        extra=["--start", "0.4", "--end", str(14 / 24)],
    )
    # A one-frame misplaced reference flash is detected in both surroundings.
    shifted = folder / "misplaced-flash.mp4"
    changed = [im.copy() for im in frames]
    changed[30] = cv.subtract(frames[30], np.full_like(frames[30], 48))
    changed[31] = cv.add(frames[31], np.full_like(frames[31], 48))
    movie(shifted, changed)
    flash = run("misplaced-flash", shifted, {2}, "颜色")
    assert {30, 31} <= {item["frame"] for item in flash["findings"]}
    # Genuine VFR timestamps and a different encoded resolution.
    vfr = folder / "variable.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(ref),
            "-vf",
            "select='not(eq(mod(n,9),4))',scale=312:216",
            "-fps_mode",
            "vfr",
            str(vfr),
        ],
        check=True,
    )
    run("variable-self", vfr, {0}, reference=vfr)
    # The same continuous timed motion sampled natively at two rates.
    seq = [np.roll(frames[0], int(i * 1.2), axis=1) for i in range(48)]
    fast = folder / "48fps.mp4"
    slow = folder / "24fps.mp4"
    movie(fast, seq, 48)
    movie(slow, seq[::2], 24)
    run("unequal-cadence", fast, {0}, reference=slow)
    print("continuity regression fixtures passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
