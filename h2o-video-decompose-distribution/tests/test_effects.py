"""Deterministic end-to-end positive, negative and abstention cases. Pass an output dir."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

FPS = 24
W, H = 384, 256
N = 18
CLI = Path(__file__).resolve().parents[1] / "scripts/decompose.py"


def tiles(seed):
    rng = np.random.default_rng(seed)
    im = np.full((H, W), 44, np.uint8)
    for y in range(6, H - 10, 18):
        for x in range(6, W - 10, 18):
            v = int(rng.integers(70, 250))
            p = (x + int(rng.integers(-3, 4)), y + int(rng.integers(-3, 4)))
            if rng.integers(2):
                cv.circle(im, p, int(rng.integers(4, 10)), v, -1, cv.LINE_AA)
            else:
                cv.rectangle(im, p, (p[0] + int(rng.integers(5, 15)), p[1] + 12), v, -1)
    return im


def frames(case, bad):
    grid = tiles(5163)
    out = []
    for i in range(N):
        active = 5 <= i <= 12 and bad
        if case == "motion":
            dx = 2 * math.sin(i * 0.51) + (3.8 * (-1) ** i if active else 0)
            im = cv.warpAffine(
                grid,
                np.float32([[1, 0, dx], [0, 1, 0.65 * i]]),
                (W, H),
                borderMode=cv.BORDER_REFLECT,
            )
        elif case == "perspective":
            k = 2 * math.sin(i * 0.24) + (21 if active else 0)
            corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            q = corners + np.float32([[k, 4], [-k, 4], [0, 0], [0, 0]])
            im = cv.warpPerspective(
                grid,
                cv.getPerspectiveTransform(corners, q),
                (W, H),
                borderMode=cv.BORDER_REFLECT,
            )
        elif case == "trail":
            kernel = np.zeros((9, 9), np.float32)
            if active:
                kernel[:, 4] = 1 / 9
            else:
                kernel[4, :] = 1 / 9
            im = cv.filter2D(grid, -1, kernel)
        elif case == "focus":
            near = i >= (11 if bad else 6)
            soft = cv.GaussianBlur(grid, (0, 0), 2.4)
            im = np.hstack(
                [
                    (soft if near else grid)[:, : W // 2],
                    (grid if near else soft)[:, W // 2 :],
                ]
            )
        elif case in ("visibility", "impact"):
            im = grid.copy()
            if (i >= 7 if case == "visibility" else i == 7) or (
                bad and case == "visibility" and i == 4
            ):
                im[50:180, 110:270] = 230
        elif case == "low":
            im = np.full_like(grid, 80)
        else:
            im = grid
        out.append(im)
    return out


def encode(path, imgs, quality=15, audio=None):
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "gray",
        "-video_size",
        f"{W}x{H}",
        "-framerate",
        str(FPS),
        "-i",
        "pipe:0",
    ]
    if audio is not None:
        import wave

        snd = np.zeros(round(N / FPS * 48000), np.int16)
        pos = round(audio / FPS * 48000)
        snd[pos : pos + 600] = (16000 * np.sin(np.arange(600) * 0.17)).astype(np.int16)
        wav = path.with_suffix(".wav")
        with wave.open(str(wav), "wb") as f:
            f.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
            f.writeframes(snd.tobytes())
        cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "192k"]
    subprocess.run(
        cmd
        + [
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-crf",
            str(quality),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        input=b"".join(x.tobytes() for x in imgs),
        check=True,
    )


def inspect(name, case, bad, expected):
    target = ROOT / name
    target.mkdir(exist_ok=True)
    ref = target / "原片.mp4"
    rep = target / "导出.mp4"
    encode(ref, frames(case, False), audio=7 if case == "impact" else None)
    encode(
        rep,
        frames(case, bad),
        quality=21 if name == "轻度重编码" else 15,
        audio=(11 if bad else 7) if case == "impact" else None,
    )
    mode = {"low": "focus", "codec": "motion"}.get(case, case)
    boxes = (
        [[0.04, 0.03, 0.45, 0.97], [0.55, 0.03, 0.96, 0.97]]
        if mode == "focus"
        else [[0.02, 0.02, 0.98, 0.98]]
    )
    plan = target / "效果清单.json"
    plan.write_text(
        json.dumps(
            {
                "effects": [
                    {
                        "id": "一",
                        "subject": name,
                        "criterion": "与原片相同的可辨时序和层次",
                        "evidence": "源片帧 0–17",
                        "mode": mode,
                        "range": [0, N / FPS],
                        "regions": [
                            {"name": f"区域{j + 1}", "ref": b}
                            for j, b in enumerate(boxes)
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        )
    )
    r = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "effects",
            str(ref),
            str(rep),
            str(plan),
            str(target),
        ],
        check=False,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"},
    )
    data = json.loads((target / "effects.json").read_text())
    result = data["effects"][0]
    spans = [
        (round(e["start"] * FPS), round(e["end"] * FPS))
        for e in result["events"]
        if e["status"] == "需返修"
    ]
    match = (
        result["status"] == expected
        and r.returncode == {"通过": 0, "需返修": 2, "待复核": 3}[expected]
    )
    if bad and expected == "需返修" and mode != "impact":
        truth = {
            "motion": (5, 14),
            "perspective": (5, 13),
            "trail": (5, 13),
            "focus": (6, 11),
            "visibility": (4, 5),
        }[mode]
        match = match and any(max(a, truth[0]) < min(b, truth[1]) for a, b in spans)
    if mode == "impact":
        rows = json.loads((target / result["data"]).read_text())
        observation = rows[0]["metrics"][0]
        match = match and all(abs(t - 7 / FPS) < 0.001 for t in observation["visual_s"])
    return {
        "case": name,
        "expected": expected,
        "actual": result["status"],
        "intervals": spans,
        "success": match,
    }


if __name__ == "__main__":
    ROOT = Path(sys.argv[1])
    ROOT.mkdir(parents=True, exist_ok=True)
    cases = [
        ("有意手持", "motion"),
        ("正常转面", "perspective"),
        ("横向拖尾", "trail"),
        ("焦点转移", "focus"),
        ("主体出现", "visibility"),
        ("单一击点", "impact"),
    ]
    results = []
    for name, mode in cases:
        for bad in (False, True):
            results.append(
                inspect(
                    name + ("故障" if bad else "对照"),
                    mode,
                    bad,
                    "需返修" if bad else "通过",
                )
            )
    results.append(inspect("轻度重编码", "codec", False, "通过"))
    results.append(inspect("低纹理", "low", False, "待复核"))
    (ROOT / "验证结果.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    for r in results:
        print(r, flush=True)
    raise SystemExit(int(not all(r["success"] for r in results)))
