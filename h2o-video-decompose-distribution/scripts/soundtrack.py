#!/usr/bin/env python3
"""Prepare synchronized audio evidence and export reviewed sound–picture links."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, stft

from decompose import same_source, source_id, verify_source


STEMS = ("speech", "music", "effects")
AUDIO_LAYERS = ("mix", *STEMS, "music_reference")
BANDIT_SHA256 = "abcfccf65446752a057f4a302c941479a54b7560ebf8d7bca039d2ea98e64cfc"
FREQUENCY_BANDS = ((40, 125), (125, 320), (320, 1300), (1300, 5000), (5000, 18000))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"{args[0]} exited with code {result.returncode}")
    return result.stdout


def audio_stream(path):
    return json.loads(command(
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=index,start_time,sample_rate,channels", "-of", "json", str(path),
    ))["streams"]


def wav_filter(source, target, filters, *options):
    command("ffmpeg", "-nostdin", "-v", "error", "-n", "-copyts", "-i", str(source),
            "-map", "0:a:0", "-af", filters, *options, "-c:a", "pcm_f32le", str(target))


def cut_filter(start, count):
    return (f"atrim=start_sample={start}:end_sample={start + count},"
            f"apad=whole_len={count},atrim=end_sample={count},asetpts=N/SR/TB")


def extract_aligned(source, target, timeline, *, source_pts=None, offset_s=None):
    rate = timeline["sample_rate"]
    if source_pts is not None:
        clock = f"PTS-({source_pts:.12f})/TB"
    else:
        clock = f"PTS-STARTPTS+({offset_s:.12f})/TB"
    wav_filter(source, target,
               f"asetpts={clock},aresample={rate}:async=1:first_pts=0,"
               + cut_filter(timeline["start_sample"], timeline["samples"]))
    check_wav(target, timeline)


def check_wav(path, timeline):
    info = sf.info(path)
    if info.samplerate != timeline["sample_rate"] or info.frames != timeline["samples"]:
        raise ValueError(f"Unexpected sample rate or length in {path.name}")


def frame_range(scan, start, end):
    frames = scan["frames"]
    if not 0 <= start < end <= len(frames):
        raise ValueError("Requested audio range is outside the source frames.")
    origin = frames[0]["pts_s"]
    lower = frames[start]["pts_s"]
    if end == len(frames):
        last = frames[-1]
        upper = last["pts_s"] + last["duration_s"] if None not in (
            last["pts_s"], last["duration_s"],
        ) else None
    else:
        upper = frames[end]["pts_s"]
    if None in (origin, lower, upper) or not origin <= lower < upper:
        raise ValueError("Source frame timestamps do not resolve the audio range.")
    return origin, lower - origin, upper - origin


def prepare_timeline(scan, start, end, stream, rate):
    origin, lower, upper = frame_range(scan, start, end)
    timeline = dict(video_origin_pts_s=origin, start_s=lower, end_s=upper)
    if rate:
        first_sample = round(lower * rate)
        timeline.update(sample_rate=rate, start_sample=first_sample,
                        samples=round(upper * rate) - first_sample,
                        sample_origin_s=first_sample / rate)
        if timeline["samples"] < 1:
            raise ValueError("The requested video range is shorter than one audio sample.")
    if stream:
        audio_start = stream.get("start_time")
        timeline.update(audio_stream=stream["index"], channels=stream["channels"],
                        audio_stream_start_pts_s=float(audio_start) if audio_start not in (None, "N/A") else None)
    return timeline


def track_inputs(path):
    if path is None:
        return {}
    tracks = read_json(path)
    unknown = set(tracks) - (set(STEMS) | {"music_reference"})
    if unknown:
        raise ValueError(f"Unknown track names: {sorted(unknown)}")
    for track in tracks.values():
        track["path"] = str((path.parent / track["path"]).resolve())
        if not math.isfinite(track["start_s"]) or not track["evidence"].strip():
            raise ValueError("Each external track needs a finite source offset and alignment evidence.")
    return tracks


def reuse_stems(path, scan, timeline, out):
    previous = read_json(path)
    old = previous["timeline"]
    start = timeline["start_sample"] - old["start_sample"]
    if (not same_source(previous["source"], scan["source"]) or old["sample_rate"] != timeline["sample_rate"]
            or old["video_origin_pts_s"] != timeline["video_origin_pts_s"]
            or start < 0 or start + timeline["samples"] > old["samples"]):
        raise ValueError("Previous separation has no verified coverage on the requested source clock.")
    result = {}
    with TemporaryDirectory(dir=out, prefix=".reuse-") as directory:
        for stem in STEMS:
            item = previous["tracks"].get(stem)
            if item is None or item["status"] != "available":
                continue
            origin = path.parent / item["file"]
            check_wav(origin, old)
            target = Path(directory) / f"{stem}.wav"
            wav_filter(origin, target, cut_filter(start, timeline["samples"]))
            check_wav(target, timeline)
            result[stem] = dict(status="available", file=target.name, method="reused",
                                evidence=str(path.resolve()), prior_method=item["method"])
        for item in result.values():
            file = item["file"]
            (Path(directory) / file).replace(out / file)
    return result


def write_model_stems(mix, checkpoint, temporary, device, torch):
    from eluate.core.separator import BanditSeparator
    from eluate.utils.paths import get_bundled_config_path

    config = get_bundled_config_path("bandit_v2.yaml")
    if config is None:
        raise ValueError("ELUATE does not contain its Bandit v2 configuration.")
    model = BanditSeparator(config_path=config, checkpoint_path=checkpoint,
                            device=torch.device(device)).model
    rate = 48000
    window, hop = 8 * rate, 6 * rate
    overlap = window - hop
    fade = np.linspace(0, 1, overlap, endpoint=False, dtype=np.float32)
    # Keep only the two overlapping windows in memory for long videos.
    with ExitStack() as stack, torch.inference_mode():
        source = stack.enter_context(sf.SoundFile(mix))
        outputs = [
            stack.enter_context(sf.SoundFile(temporary / f"{name}-48k.wav", mode="w",
                                            samplerate=rate, channels=2, subtype="FLOAT"))
            for name in STEMS
        ]
        prior_tail = None
        for start in range(0, source.frames, hop):
            source.seek(start)
            samples = source.read(window, dtype="float32", always_2d=True).T
            count = samples.shape[1]
            padded = np.pad(samples, ((0, 0), (0, window - count)), mode="reflect")
            prediction = model(torch.from_numpy(np.ascontiguousarray(padded))[None].to(device))
            values = prediction.cpu().numpy()[0, :, :, :count]
            if values.shape != (len(STEMS), 2, count) or not np.isfinite(values).all():
                raise ValueError("Bandit v2 returned invalid audio.")
            if prior_tail is not None:
                values[:, :, :overlap] = prior_tail * (1 - fade) + values[:, :, :overlap] * fade
            last = start + count == source.frames
            for output, stem in zip(outputs, values):
                output.write(stem[:, :count if last else hop].T)
            if last:
                break
            prior_tail = values[:, :, hop:].copy()
            if device == "mps" and (start // hop) % 4 == 3:
                torch.mps.empty_cache()


def separate(mix, out, timeline, names, checkpoint, device):
    if checkpoint is None:
        raise ValueError("Set H2O_BANDIT_CHECKPOINT or pass --checkpoint with the official multi weights.")
    with checkpoint.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != BANDIT_SHA256:
            raise ValueError("Bandit v2 multilingual checkpoint failed SHA-256 verification.")

    import torch

    torch.set_num_threads(min(4, torch.get_num_threads()))
    selected = ("mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
    notes = []
    with TemporaryDirectory(dir=out, prefix=".bandit-") as directory:
        temporary = Path(directory)
        model_mix = temporary / "model-input.wav"
        filters = "aresample=48000"
        if sf.info(mix).channels == 1:
            filters = "pan=stereo|c0=c0|c1=c0," + filters
        wav_filter(mix, model_mix, filters, "-ac", "2")
        try:
            write_model_stems(model_mix, checkpoint, temporary, selected, torch)
        except (RuntimeError, NotImplementedError) as error:
            if selected != "mps" or device != "auto":
                raise
            notes.append(f"MPS fallback: {error}")
        if notes:
            torch.mps.empty_cache()
            selected = "cpu"
            write_model_stems(model_mix, checkpoint, temporary, selected, torch)
        for name in names:
            path = temporary / f"{name}.wav"
            wav_filter(temporary / f"{name}-48k.wav", path,
                       f"aresample={timeline['sample_rate']}," + cut_filter(0, timeline["samples"]))
            check_wav(path, timeline)
        for name in names:
            (temporary / f"{name}.wav").replace(out / f"{name}.wav")
    return selected, notes


def transients(path, layer, offset):
    with sf.SoundFile(path) as source:
        rate = source.samplerate
        window = max(8, round(0.0232 * rate))
        hop = max(1, round(0.005 * rate))
        block = round(24 * rate)
        margin = round(0.15 * rate)
        for start in range(0, source.frames, block):
            lower = max(0, start - margin)
            upper = min(source.frames, start + block + margin)
            source.seek(lower)
            data = source.read(upper - lower, dtype="float32", always_2d=True)
            if len(data) < window:
                continue
            frequency, time, spectrum = stft(data, rate, nperseg=window,
                                             noverlap=window - hop, axis=0, padded=False)
            magnitude = np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=1))
            flux = np.maximum(0, np.diff(np.log1p(1000 * magnitude), axis=-1))
            for low, high in FREQUENCY_BANDS:
                bins = (frequency >= low) & (frequency < high)
                if not bins.any():
                    continue
                values = flux[bins].mean(axis=0)
                middle = np.median(values)
                spread = np.median(np.abs(values - middle))
                peaks, _ = find_peaks(values, height=max(0.005, middle + 4 * spread),
                                      prominence=max(0.005, 3 * spread), distance=7)
                for peak in peaks:
                    position = lower / rate + time[peak + 1]
                    if start / rate <= position < (start + block) / rate and position < source.frames / rate:
                        yield dict(layer=layer, peak_candidate_s=round(offset + position, 9),
                                   band_low_hz=low, band_high_hz=high,
                                   spectral_flux=round(float(values[peak]), 6))


def music_levels(path, timeline):
    offset = timeline["sample_origin_s"]
    rate = timeline["sample_rate"]
    step = max(1, round(rate / 4))
    levels = []
    for index, samples in enumerate(sf.blocks(path, blocksize=step, dtype="float32", always_2d=True)):
        rms = float(np.sqrt(np.mean(samples ** 2)))
        levels.append(dict(start_s=round(offset + index * step / rate, 9),
                           end_s=round(offset + (index * step + len(samples)) / rate, 9),
                           rms_dbfs=round(20 * math.log10(max(rms, 1e-8)), 2)))
    return levels


def beat_candidates(path, timeline, beat_checkpoint, levels):
    if not levels or max(row["rms_dbfs"] for row in levels) <= -100:
        return [], "silent"
    import torch
    from beat_this.inference import File2Beats

    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = File2Beats(checkpoint_path=beat_checkpoint, device="cpu", dbn=False)
    beats, downbeats = model(path)
    offset, rate = timeline["sample_origin_s"], timeline["sample_rate"]
    duration = timeline["samples"] / rate
    # The model returns timestamps on a 50 Hz timeline. Keep labels as candidates.
    labels = {round(float(value), 6): "beat" for value in beats if math.isfinite(value)}
    labels.update({round(float(value), 6): "downbeat" for value in downbeats if math.isfinite(value)})
    events = [dict(time_s=round(offset + time, 9), kind=kind, status="candidate")
              for time, kind in sorted(labels.items()) if 0 <= time < duration]
    return events, "candidate" if events else "no_candidates"


def external_sample_rate(supplied, reuse, scan, notes):
    if reuse:
        try:
            previous = read_json(reuse)
            if same_source(previous["source"], scan["source"]):
                return previous["timeline"]["sample_rate"]
        except (OSError, ValueError, KeyError) as error:
            notes.append(f"Previous separation: {error}")
    for name, item in supplied.items():
        try:
            streams = audio_stream(Path(item["path"]))
            if streams:
                return int(streams[0]["sample_rate"])
            notes.append(f"External {name}: no audio stream")
        except (OSError, ValueError) as error:
            notes.append(f"External {name}: {error}")
    return None


def prepare(args):
    scan = read_json(args.scan)
    verify_source(args.source, scan["source"])
    start = args.start if args.start is not None else scan["scanned_start"]
    end = args.end if args.end is not None else scan["scanned_end_exclusive"]
    supplied = track_inputs(args.tracks)
    args.out.mkdir(parents=True, exist_ok=False)
    mix = args.out / "mix.wav"
    result = dict(source=scan["source"], frame_start=start, frame_end_exclusive=end,
                  tracks={}, notes=[], beat_status="unavailable")
    tracks = result["tracks"]
    try:
        streams = audio_stream(args.source)
        stream = streams[0] if streams else None
        rate = int(stream["sample_rate"]) if stream else external_sample_rate(
            supplied, args.reuse, scan, result["notes"],
        )
        result["timeline"] = timeline = prepare_timeline(scan, start, end, stream, rate)
        if rate is None:
            result["status"] = "no_audio"
            write_json(args.out / "soundtrack.json", result)
            return
    except (OSError, RuntimeError, ValueError) as error:
        result.update(status="audio_unavailable")
        result["notes"].append(str(error))
        write_json(args.out / "soundtrack.json", result)
        return

    tracks["mix"] = dict(status="unavailable", method="source_mix")
    if stream:
        with TemporaryDirectory(dir=args.out, prefix=".mix-") as directory:
            target = Path(directory) / mix.name
            try:
                extract_aligned(args.source, target, timeline, source_pts=timeline["video_origin_pts_s"])
                target.replace(mix)
                tracks["mix"] = dict(status="available", file=mix.name, method="source_mix")
            except (OSError, RuntimeError, ValueError) as error:
                result["notes"].append(f"Source mix: {error}")
    if args.reuse:
        try:
            tracks.update(reuse_stems(args.reuse, scan, timeline, args.out))
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            result["notes"].append(f"Previous separation: {error}")
    for name, item in supplied.items():
        # Stage external tracks so a failed decode leaves an existing verified track intact.
        with TemporaryDirectory(dir=args.out, prefix=".track-") as directory:
            target = Path(directory) / f"{name}.wav"
            try:
                extract_aligned(Path(item["path"]), target, timeline, offset_s=item["start_s"])
                target.replace(args.out / target.name)
            except (OSError, RuntimeError, ValueError) as error:
                result["notes"].append(f"External {name}: {error}")
                continue
        tracks[name] = dict(status="available", file=target.name, method="external_aligned",
                            path=item["path"], start_s=item["start_s"], evidence=item["evidence"])
    missing = [name for name in STEMS if name not in tracks]
    if missing:
        for name in missing:
            tracks[name] = dict(status="unavailable", method="bandit_v2_multi")
        try:
            if tracks["mix"]["status"] != "available":
                raise ValueError("No readable source mix for the missing stems.")
            selected, notes = separate(mix, args.out, timeline, missing, args.checkpoint, args.device)
            result["notes"].extend(notes)
            for name in missing:
                tracks[name] = dict(status="available", file=f"{name}.wav",
                                    method="bandit_v2_multi", device=selected,
                                    checkpoint_sha256=BANDIT_SHA256)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            result["notes"].append(f"Bandit v2: {error}")
    if all(tracks[name]["status"] == "available" for name in STEMS):
        result["status"] = "ready"
    elif any(item["status"] == "available" for item in tracks.values()):
        result["status"] = "mix_or_partial"
    else:
        result["status"] = "audio_unavailable"
    write_json(args.out / "soundtrack.json", result)

    offset = timeline["sample_origin_s"]
    candidates = []
    for layer in ("mix", "effects"):
        if tracks[layer]["status"] == "available":
            candidates.extend(transients(args.out / tracks[layer]["file"], layer, offset))
    write_csv(args.out / "transients.csv",
              ("layer", "peak_candidate_s", "band_low_hz", "band_high_hz", "spectral_flux"),
              sorted(candidates, key=lambda row: row["peak_candidate_s"]))

    music = "music_reference" if "music_reference" in tracks else "music"
    if tracks[music]["status"] == "available":
        result["beat_input"] = music
        level_layer = "music" if tracks["music"]["status"] == "available" else music
        result["music_level_input"] = level_layer
        levels = music_levels(args.out / tracks[level_layer]["file"], timeline)
        write_csv(args.out / "music-level.csv", ("start_s", "end_s", "rms_dbfs"), levels)
        beat_audio = args.out / tracks[music]["file"]
        try:
            beat_levels = levels if level_layer == music else music_levels(beat_audio, timeline)
            beats, result["beat_status"] = beat_candidates(
                beat_audio, timeline, args.beat_checkpoint, beat_levels,
            )
            write_csv(args.out / "beats.csv", ("time_s", "kind", "status"), beats)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            result["notes"].append(f"Beat This!: {error}")
    write_json(args.out / "soundtrack.json", result)


def reviewed_rows(review, scan, manifest):
    frames = scan["frames"]
    if not same_source(scan["source"], manifest["source"]):
        raise ValueError("The frame index and soundtrack manifest belong to different sources.")
    lower, upper = manifest["frame_start"], manifest["frame_end_exclusive"]
    time_range = manifest.get("timeline", {})
    for number, event in enumerate(review["events"], 1):
        frame = event.get("frame")
        if frame is not None and (type(frame) is not int or not lower <= frame < upper):
            raise ValueError(f"Event {number} is outside the prepared frame range.")
        frame_time = frames[frame]["time_s"] if frame is not None else None
        frame_gap = None
        if frame_time is not None:
            next_time = frames[frame + 1]["time_s"] if frame + 1 < len(frames) else None
            frame_gap = next_time - frame_time if next_time is not None else frames[frame]["duration_s"]
            if frame_gap is not None and frame_gap <= 0:
                frame_gap = None
        audio = {key: event.get(f"{key}_s") for key in ("onset", "peak", "end", "beat", "boundary")}
        for time in audio.values():
            if time is not None and (not math.isfinite(time) or
                                     not time_range["start_s"] <= time <= time_range["end_s"]):
                raise ValueError(f"Event {number} audio time is outside the prepared range.")
        anchor = event.get("anchor")
        if anchor is not None and (anchor not in audio or audio[anchor] is None):
            raise ValueError(f"Event {number} anchor needs a matching audio time.")
        if anchor is None and any(time is not None for time in audio.values()) and frame_time is not None:
            raise ValueError(f"Event {number} must name the audio time used for its comparison.")
        confidence, layer = event["confidence"], event["layer"]
        if (confidence not in ("high", "medium", "low", "unmatched")
                or layer not in AUDIO_LAYERS or not event["evidence"].strip()):
            raise ValueError(f"Event {number} needs a valid layer, confidence and review evidence.")
        audio_time = audio.get(anchor)
        delta = (audio_time - frame_time) * 1000 if None not in (audio_time, frame_time) else None
        yield dict(id=number, frame=frame, frame_pts_s=frames[frame]["pts_s"] if frame is not None else None,
                   picture_time_s=frame_time,
                   frame_interval_ms=round(frame_gap * 1000, 3) if frame_gap is not None else None,
                   action=event.get("action"), phase=event.get("phase"), layer=layer,
                   audio_event=event.get("audio_event"), **{f"{key}_s": value for key, value in audio.items()},
                   anchor=anchor, audio_minus_picture_ms=round(delta, 3) if delta is not None else None,
                   relation=event.get("relation"), confidence=confidence, evidence=event["evidence"])


def write_clips(clips, manifest, source, out):
    timeline = manifest.get("timeline", {})
    if not clips or "sample_rate" not in timeline or not any(
        manifest["tracks"].get(layer, {}).get("status") == "available" for layer in AUDIO_LAYERS
    ):
        return
    rate = timeline["sample_rate"]
    for number, (lower, upper) in enumerate(clips, 1):
        if not timeline["start_s"] <= lower < upper <= timeline["end_s"]:
            raise ValueError(f"Listening window {number} is outside the prepared range.")
        first, last = round(lower * rate), round(upper * rate)
        if first == last:
            raise ValueError(f"Listening window {number} is shorter than one sample.")
        folder = out / f"listen-{number:02d}"
        folder.mkdir()
        write_json(folder / "window.json", dict(start_s=lower, end_s=upper,
                                               sample_origin_s=first / rate, samples=last - first))
        for layer in AUDIO_LAYERS:
            item = manifest["tracks"].get(layer)
            if item and item["status"] == "available":
                wav_filter(source / item["file"], folder / f"{layer}.wav",
                           cut_filter(first - timeline["start_sample"], last - first))


def match(args):
    review = read_json(args.review)
    scan = read_json(args.scan)
    manifest = read_json(args.soundtrack)
    rows = list(reviewed_rows(review, scan, manifest))
    args.out.mkdir(parents=True, exist_ok=False)
    columns = ("id", "frame", "frame_pts_s", "picture_time_s", "frame_interval_ms", "action",
               "phase", "layer", "audio_event", "onset_s", "peak_s", "end_s", "beat_s",
               "boundary_s", "anchor", "audio_minus_picture_ms", "relation", "confidence", "evidence")
    write_csv(args.out / "av-events.csv", columns, rows)
    write_clips(review.get("listen", []), manifest, args.soundtrack.parent, args.out)


def _merge_audio_peaks(rows, merge_gap_s=0.07):
    """Merge multi-band spectral flux detections occurring within merge_gap_s into single onset peaks."""
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: float(r["time_s"]))
    merged = []
    cur = dict(ordered[0])
    cur["bands"] = [cur.get("band", "beat")]
    for item in ordered[1:]:
        if float(item["time_s"]) - float(cur["time_s"]) <= merge_gap_s:
            if float(item["strength"]) > float(cur["strength"]):
                cur["time_s"] = item["time_s"]
                cur["strength"] = item["strength"]
                cur["kind"] = item["kind"]
            if item.get("band") and item["band"] not in cur["bands"]:
                cur["bands"].append(item["band"])
        else:
            merged.append(cur)
            cur = dict(item)
            cur["bands"] = [cur.get("band", "beat")]
    merged.append(cur)
    return merged


def beat_lock(args):
    """Automatically align audio transient/beat peaks with video cuts, flash jumps, and motion impact spikes."""
    import cv2 as cv
    from decompose import load_source, raw_frames, size_for

    scan_data = read_json(args.scan) if args.scan is not None else None
    if scan_data is not None:
        verify_source(args.source, scan_data["source"])
        frames_meta = scan_data["frames"]
        coded = (int(scan_data["stream"]["width"]), int(scan_data["stream"]["height"]))
        src_meta = scan_data["source"]
    else:
        frames_meta, stream_info = load_source(args.source)
        coded = (int(stream_info["width"]), int(stream_info["height"]))
        src_meta = source_id(args.source)

    start = 0 if args.start is None else args.start
    end = len(frames_meta) if args.end is None else min(len(frames_meta), args.end)
    if not (0 <= start < end <= len(frames_meta)):
        raise ValueError("Requested beat-lock frame range is outside source frames.")

    # 1. Collect audio onset / beat peaks (either from existing soundtrack.json dir or directly from source video)
    audio_events = []
    if args.soundtrack is not None:
        sdir = args.soundtrack.parent
        t_csv = sdir / "transients.csv"
        b_csv = sdir / "beats.csv"
        if t_csv.exists():
            with t_csv.open("r", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    audio_events.append({
                        "time_s": float(r["peak_candidate_s"]),
                        "strength": float(r["spectral_flux"]),
                        "kind": f"transient_{r['layer']}",
                        "band": f"{r['band_low_hz']}-{r['band_high_hz']}Hz",
                    })
        if b_csv.exists():
            with b_csv.open("r", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    audio_events.append({
                        "time_s": float(r["time_s"]),
                        "strength": 1.2 if r.get("kind") == "downbeat" else 0.85,
                        "kind": r.get("kind", "beat"),
                        "band": "beat_grid",
                    })
    else:
        streams = audio_stream(args.source)
        if not streams:
            raise ValueError("No audio stream found in source video for beat-lock alignment.")
        a_stream = streams[0]
        fake_scan = {"frames": frames_meta}
        timeline = prepare_timeline(fake_scan, start, end, a_stream, int(a_stream["sample_rate"]))
        with TemporaryDirectory(prefix=".beatlock-") as tmp_dir:
            mix_wav = Path(tmp_dir) / "mix.wav"
            extract_aligned(args.source, mix_wav, timeline, source_pts=timeline["video_origin_pts_s"])
            for r in transients(mix_wav, "mix", timeline["sample_origin_s"]):
                audio_events.append({
                    "time_s": float(r["peak_candidate_s"]),
                    "strength": float(r["spectral_flux"]),
                    "kind": "transient_mix",
                    "band": f"{r['band_low_hz']}-{r['band_high_hz']}Hz",
                })

    origin_pts = frames_meta[0].get("pts_s") or 0.0
    t_lo = (frames_meta[start].get("pts_s") or frames_meta[start].get("time_s") or 0.0) - origin_pts
    t_hi = (frames_meta[end - 1].get("pts_s") or frames_meta[end - 1].get("time_s") or 0.0) - origin_pts + (
        frames_meta[end - 1].get("duration_s") or 0.04
    )
    audio_events = [e for e in audio_events if t_lo - 0.05 <= e["time_s"] <= t_hi + 0.05]
    merged_audio = _merge_audio_peaks(audio_events, merge_gap_s=0.07)

    # 2. Extract lightweight per-frame visual impulse curves (cut, luminance flash, motion impact)
    small_size = size_for(coded, 320)
    prev_gray = None
    prev_hist = None
    prev_lum = None
    frame_indices = []
    frame_times = []
    cut_scores = []
    flash_jumps = []
    motion_scores = []

    for idx, bgr in enumerate(raw_frames(args.source, small_size, start, end), start):
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).astype(np.float32)
        hist /= max(1.0, float(np.sum(hist)))
        mean_l = float(np.mean(gray))
        t_s = float((frames_meta[idx].get("pts_s") or frames_meta[idx].get("time_s") or 0.0) - origin_pts)
        frame_indices.append(idx)
        frame_times.append(t_s)
        if prev_gray is None:
            cut_scores.append(0.0)
            flash_jumps.append(0.0)
            motion_scores.append(0.0)
        else:
            mad = float(np.mean(np.abs(gray - prev_gray)))
            hist_diff = float(np.sum(np.abs(hist - prev_hist)))
            lum_jump = abs(mean_l - prev_lum)
            cut_scores.append(0.5 * hist_diff + 1.2 * mad)
            flash_jumps.append(lum_jump)
            motion_scores.append(mad)
        prev_gray, prev_hist, prev_lum = gray, hist, mean_l

    cut_arr = np.asarray(cut_scores, dtype=np.float64)
    flash_arr = np.asarray(flash_jumps, dtype=np.float64)
    mot_arr = np.asarray(motion_scores, dtype=np.float64)

    visual_impulses = []
    if len(frame_indices) >= 3:
        cut_med, cut_mad = float(np.median(cut_arr)), float(np.median(np.abs(cut_arr - np.median(cut_arr))))
        fl_med, fl_mad = float(np.median(flash_arr)), float(np.median(np.abs(flash_arr - np.median(flash_arr))))
        mt_med, mt_mad = float(np.median(mot_arr)), float(np.median(np.abs(mot_arr - np.median(mot_arr))))

        cut_peaks, _ = find_peaks(cut_arr, height=max(0.42, cut_med + 5.0 * max(0.02, cut_mad)), distance=3)
        flash_peaks, _ = find_peaks(flash_arr, height=max(0.09, fl_med + 4.5 * max(0.008, fl_mad)), distance=2)
        mot_peaks, _ = find_peaks(mot_arr, height=max(0.045, mt_med + 3.2 * max(0.005, mt_mad)), distance=2)

        by_frame = {}
        for p in mot_peaks:
            by_frame[int(p)] = ("MOTION_IMPULSE", float(mot_arr[p]))
        for p in flash_peaks:
            by_frame[int(p)] = ("FLASH_JUMP", float(flash_arr[p]))
        for p in cut_peaks:
            by_frame[int(p)] = ("HARD_CUT", float(cut_arr[p]))
        for p, (vtype, vval) in sorted(by_frame.items()):
            visual_impulses.append({
                "frame": frame_indices[p],
                "time_s": round(frame_times[p], 4),
                "visual_type": vtype,
                "visual_score": round(vval, 4),
            })

    # 3. Match each audio peak to nearest visual impulse within tolerance_ms
    tol_s = float(args.tolerance_ms) / 1000.0
    lock_rows = []
    locked_count = 0
    for idx_a, ap in enumerate(merged_audio, 1):
        at = float(ap["time_s"])
        best_v = None
        best_dt = tol_s + 1.0
        for vi in visual_impulses:
            dt = abs(vi["time_s"] - at)
            if dt <= tol_s and dt < best_dt:
                best_dt = dt
                best_v = vi
        if best_v is not None:
            locked_count += 1
            rel = f"LOCKED_{best_v['visual_type']}"
            delta_ms = round((at - best_v["time_s"]) * 1000.0, 1)
            v_frame = best_v["frame"]
            v_time = best_v["time_s"]
            v_score = best_v["visual_score"]
        else:
            rel = "AUDIO_ONLY_BEAT"
            delta_ms = None
            v_frame = None
            v_time = None
            v_score = None
        lock_rows.append({
            "id": idx_a,
            "audio_time_s": round(at, 4),
            "audio_kind": ap["kind"],
            "audio_strength": round(float(ap["strength"]), 4),
            "audio_bands": ";".join(ap["bands"]),
            "visual_frame": v_frame,
            "visual_time_s": v_time,
            "visual_score": v_score,
            "audio_minus_visual_ms": delta_ms,
            "sync_relation": rel,
        })

    args.out.mkdir(parents=True, exist_ok=True)
    sync_ratio = round(locked_count / max(1, len(merged_audio)), 3)
    payload = {
        "source": src_meta,
        "frame_range": [start, end],
        "tolerance_ms": args.tolerance_ms,
        "audio_peak_count": len(merged_audio),
        "visual_impulse_count": len(visual_impulses),
        "locked_av_pair_count": locked_count,
        "audio_to_visual_lock_ratio": sync_ratio,
        "visual_impulses": visual_impulses,
        "beat_lock_pairs": lock_rows,
    }
    write_json(args.out / "av_beat_lock.json", payload)
    write_csv(
        args.out / "av_beat_lock.csv",
        ("id", "audio_time_s", "audio_kind", "audio_strength", "audio_bands",
         "visual_frame", "visual_time_s", "visual_score", "audio_minus_visual_ms", "sync_relation"),
        lock_rows,
    )
    print(
        f"beat-lock frames [{start}, {end}): audio_peaks={len(merged_audio)}, "
        f"visual_impulses={len(visual_impulses)}, locked={locked_count} ({sync_ratio*100:.1f}%)",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    setup = modes.add_parser("prepare", help="Extract aligned stems, audio and beat candidates.")
    setup.add_argument("source", type=Path)
    setup.add_argument("out", type=Path)
    setup.add_argument("--scan", type=Path, required=True, help="source.json from decompose.py scan")
    setup.add_argument("--start", type=int, help="Inclusive source frame; defaults to scan range")
    setup.add_argument("--end", type=int, help="Exclusive source frame; defaults to scan range")
    setup.add_argument("--tracks", type=Path, help="Reviewed external track paths and offsets")
    setup.add_argument("--reuse", type=Path, help="soundtrack.json from a previous run on this source")
    setup.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    setup.add_argument("--checkpoint", type=Path, default=os.environ.get("H2O_BANDIT_CHECKPOINT"))
    setup.add_argument("--beat-checkpoint", default=os.environ.get("H2O_BEAT_CHECKPOINT", "final0"))
    report = modes.add_parser("match", help="Calculate offsets and export reviewed listening windows.")
    report.add_argument("out", type=Path)
    report.add_argument("--scan", type=Path, required=True)
    report.add_argument("--soundtrack", type=Path, required=True)
    report.add_argument("--review", type=Path, required=True)
    lock = modes.add_parser("beat-lock", help="Auto-align audio beat/transient peaks with video cuts, flashes, and motion impacts.")
    lock.add_argument("source", type=Path)
    lock.add_argument("out", type=Path)
    lock.add_argument("--scan", type=Path, default=None, help="optional source.json from decompose.py scan")
    lock.add_argument("--soundtrack", type=Path, default=None, help="optional soundtrack.json from prepare")
    lock.add_argument("--start", type=int, default=None, help="Inclusive source frame")
    lock.add_argument("--end", type=int, default=None, help="Exclusive source frame")
    lock.add_argument("--tolerance-ms", type=float, default=85.0, help="Max audio-visual alignment window in ms")
    args = parser.parse_args()
    try:
        dispatch = {"prepare": prepare, "match": match, "beat-lock": beat_lock}
        dispatch[args.mode](args)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
