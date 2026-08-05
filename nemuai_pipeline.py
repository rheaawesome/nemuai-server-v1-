# nemuai_pipeline.py
# Ported 1:1 from NemuAI.ipynb (the original Colab inference notebook) so a server
# can import and call it directly instead of running it as a notebook script.
#
# Behavior is intentionally unchanged from the notebook: same windowing, same
# YAMNet gating logic, same feature pooling, same centering/scaling/logit-adjustment
# math, same report schema ("nemuai-final-colab-1"). If you need to verify this
# ported version matches the original, run both against the same audio file and
# diff the resulting JSON reports.

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import tensorflow_hub as hub
from scipy.signal import stft

DISCLAIMER = (
    "NemuAI is an experimental audio-based screening prototype. It may flag possible "
    "obstructive or mixed apnea-like sound patterns, but it does not diagnose sleep apnea, "
    "calculate clinical AHI, or replace polysomnography or evaluation by a qualified medical professional."
)

REQUIRED_BUNDLE_FILES = [
    "nemuai_final_model.joblib",
    "nemuai_final_scaler.joblib",
    "nemuai_final_threshold.json",
    "nemuai_final_config.json",
    "nemuai_final_selection.json",
    "README.txt",
]

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


class NemuAIBundle:
    """Holds the loaded model, scaler, config, and YAMNet handle. Load once at
    server startup and reuse across requests -- loading YAMNet per-request would
    be very slow."""

    def __init__(self, model, scaler, config, event_threshold, selection_metadata, yamnet):
        self.model = model
        self.scaler = scaler
        self.config = config
        self.event_threshold = event_threshold
        self.selection_metadata = selection_metadata
        self.yamnet = yamnet


def load_bundle(model_dir: Path) -> NemuAIBundle:
    model_dir = Path(model_dir)
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Bundle is missing required files: {missing}")

    model = joblib.load(model_dir / "nemuai_final_model.joblib")
    scaler = joblib.load(model_dir / "nemuai_final_scaler.joblib")
    config = json.loads((model_dir / "nemuai_final_config.json").read_text())
    threshold_metadata = json.loads((model_dir / "nemuai_final_threshold.json").read_text())
    selection_metadata = json.loads((model_dir / "nemuai_final_selection.json").read_text())
    event_threshold = float(threshold_metadata["event_threshold"])

    if not hasattr(model, "predict_proba") or not hasattr(scaler, "transform"):
        raise TypeError("The model or scaler artifact is incompatible")

    yamnet = hub.load(config["yamnet_url"])

    return NemuAIBundle(
        model=model,
        scaler=scaler,
        config=config,
        event_threshold=event_threshold,
        selection_metadata=selection_metadata,
        yamnet=yamnet,
    )


# ---------------------------------------------------------------------------
# Helper functions (ported unchanged from the notebook, config passed as a param
# instead of read from a notebook global)
# ---------------------------------------------------------------------------

def format_timestamp(seconds):
    seconds = max(0, float(seconds))
    whole = int(seconds)
    return f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"


def probe_audio(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=sample_rate,channels,duration,codec_name", "-show_entries", "format=format_name,duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    fmt = data.get("format", {})
    return {
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": int(stream.get("channels", 0)),
        "duration": float(stream.get("duration") or fmt.get("duration") or 0),
        "codec": stream.get("codec_name"),
        "format_name": fmt.get("format_name"),
    }


def decode_audio(path, sample_rate=16000):
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise ValueError("FFmpeg could not decode the audio: " + result.stderr.decode(errors="replace"))
    waveform = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Decoded audio is empty or invalid")
    peak = float(np.max(np.abs(waveform)))
    if peak > 1.0:
        waveform /= peak
    return np.clip(waveform, -1, 1), sample_rate


def generate_windows(waveform, sample_rate, length_seconds, hop_seconds, minimum_final_seconds):
    length = int(length_seconds * sample_rate)
    hop = int(hop_seconds * sample_rate)
    rows = []
    for index, start in enumerate(range(0, len(waveform), hop)):
        real = waveform[start:min(start + length, len(waveform))]
        if len(real) < minimum_final_seconds * sample_rate:
            break
        sufficient = len(real) == length
        padded = np.pad(real, (0, length - len(real))).astype(np.float32)
        rows.append({
            "index": index,
            "start_seconds": start / sample_rate,
            "end_seconds": min((start + length) / sample_rate, len(waveform) / sample_rate),
            "waveform": padded,
            "sufficient_duration": sufficient,
        })
        if start + length >= len(waveform) and not sufficient:
            break
    return rows


def assess_quality(waveform, sample_rate, sufficient_duration, config):
    q = config["quality"]
    if not sufficient_duration:
        return {"status": "insufficient_duration", "usable": False, "reason": "Window has less than 30 seconds of real audio"}
    if waveform.size == 0 or not np.isfinite(waveform).all():
        return {"status": "corrupted", "usable": False, "reason": "Empty or non-finite audio"}
    absolute = np.abs(waveform)
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    clipping = float(np.mean(absolute >= q["clipping_amplitude"]))
    near_zero = float(np.mean(absolute <= q["near_zero_amplitude"]))
    zcr = float(np.mean(np.diff(np.signbit(waveform)) != 0))
    _, _, spectrum = stft(waveform, fs=sample_rate, nperseg=1024)
    power = np.square(np.abs(spectrum)) + 1e-12
    flatness = float(np.mean(np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)))
    activity_threshold = max(q["near_zero_amplitude"] * 5, rms * 0.25)
    activity = float(np.mean(absolute > activity_threshold))
    status, reason, usable = "good", None, True
    if clipping > q["max_clipping_ratio"]:
        status, reason, usable = "clipped", "Excessive clipping", False
    elif near_zero > q["max_near_zero_ratio"] and activity < q["minimum_activity_ratio"]:
        status, reason, usable = "mostly_empty", "Almost no measurable signal", False
    elif rms < q["minimum_rms"]:
        status, reason, usable = "too_quiet", "Below minimum measurable RMS", False
    elif flatness > q["excessive_noise_flatness"] and rms > 0.05:
        status, reason, usable = "movement_or_environmental_noise", "Broadband noise dominates", False
    return {
        "status": status, "usable": usable, "reason": reason, "rms": rms,
        "clipping_ratio": clipping, "near_zero_percentage": near_zero * 100,
        "spectral_flatness": flatness, "zero_crossing_rate": zcr, "signal_activity": activity,
    }


def reliability_gate(quality, yamnet_scores, config):
    if not quality["usable"]:
        return quality["status"], quality["reason"]
    groups = config["yamnet_class_groups"]
    speech_score = float(np.max(yamnet_scores[:, groups["speech_tv"]]))
    environment_score = float(np.max(yamnet_scores[:, groups["environment"]]))
    if speech_score >= config["speech_tv_threshold"]:
        return "speech_or_tv_contamination", f"YAMNet speech/TV score {speech_score:.3f}"
    if environment_score >= config["environment_threshold"]:
        return "movement_or_environmental_noise", f"YAMNet noise/movement score {environment_score:.3f}"
    if quality["signal_activity"] < config["quality"]["minimum_activity_ratio"]:
        return "uncertain_audio", "Insufficient measurable activity"
    return "suitable_breathing_audio", None


def pool_yamnet_embeddings(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return np.concatenate([
        embeddings.mean(0), embeddings.std(0), embeddings.max(0),
        np.percentile(embeddings, [25, 50, 75], axis=0).reshape(-1),
    ]).astype(np.float32)


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


def logit(probabilities):
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def smooth_probabilities(probabilities, size=3):
    radius = size // 2
    result = []
    for index, value in enumerate(probabilities):
        if value is None:
            result.append(None)
            continue
        neighbors = [p for p in probabilities[max(0, index - radius):index + radius + 1] if p is not None]
        result.append(float(np.mean(neighbors)))
    return result


def merge_events(rows, gap_seconds=5):
    flagged = [row for row in rows if row["predicted_label"] == "possible_obstructive_event"]
    if not flagged:
        return []
    groups = [[flagged[0]]]
    for row in flagged[1:]:
        if row["start_seconds"] <= max(item["end_seconds"] for item in groups[-1]) + gap_seconds:
            groups[-1].append(row)
        else:
            groups.append([row])
    events = []
    for event_id, group in enumerate(groups, 1):
        start = min(x["start_seconds"] for x in group)
        end = max(x["end_seconds"] for x in group)
        probabilities = [x["smoothed_probability"] for x in group]
        events.append({
            "event_id": event_id, "start_seconds": start, "end_seconds": end,
            "start_timestamp": format_timestamp(start), "end_timestamp": format_timestamp(end),
            "duration_seconds": end - start, "mean_probability": float(np.mean(probabilities)),
            "maximum_probability": float(np.max(probabilities)),
            "contributing_windows": [x["window_number"] for x in group],
        })
    return events


def reliable_coverage(rows, duration):
    intervals = sorted(
        (row["start_seconds"], row["end_seconds"])
        for row in rows if row["audio_suitability"] == "suitable_breathing_audio"
    )
    covered = 0.0
    cursor = 0.0
    for start, end in intervals:
        if end > cursor:
            covered += end - max(start, cursor)
            cursor = end
    return covered, (100 * covered / duration if duration else 0)


# ---------------------------------------------------------------------------
# Main entry point for the server
# ---------------------------------------------------------------------------

def run_inference(audio_path: Path, bundle: NemuAIBundle) -> dict:
    """Runs the full NemuAI pipeline on one audio file and returns the report
    dict, matching the "nemuai-final-colab-1" schema produced by the notebook."""
    audio_path = Path(audio_path)
    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported audio format: {audio_path.suffix}")

    config = bundle.config

    original = probe_audio(audio_path)
    waveform, processed_rate = decode_audio(audio_path, config["sample_rate"])
    duration = len(waveform) / processed_rate
    windows = generate_windows(
        waveform, processed_rate, config["window_seconds"], config["hop_seconds"], config["minimum_final_window_seconds"]
    )

    results, features, suitable_indexes = [], [], []
    for window in windows:
        quality = assess_quality(window["waveform"], processed_rate, window["sufficient_duration"], config)
        state, reason, feature = quality["status"], quality["reason"], None
        if quality["usable"]:
            scores, embeddings, _ = bundle.yamnet(window["waveform"].astype(np.float32))
            scores, embeddings = np.asarray(scores), np.asarray(embeddings)
            state, reason = reliability_gate(quality, scores, config)
            if state == "suitable_breathing_audio":
                feature = pool_yamnet_embeddings(embeddings)
        result = {
            "window_number": window["index"] + 1, "start_seconds": window["start_seconds"],
            "end_seconds": window["end_seconds"], "start_time": format_timestamp(window["start_seconds"]),
            "end_time": format_timestamp(window["end_seconds"]), "audio_suitability": state,
            "exclusion_reason": reason, "raw_probability": None, "calibrated_probability": None,
            "smoothed_probability": None, "predicted_label": "excluded" if feature is None else "normal_breathing",
        }
        results.append(result)
        if feature is not None:
            features.append(feature)
            suitable_indexes.append(window["index"])

    if features:
        feature_matrix = np.stack(features)
        centered = feature_matrix - config["recording_center_alpha"] * np.median(feature_matrix, axis=0)
        scaled = bundle.scaler.transform(centered[:, :config["feature_size"]])
        raw_probabilities = bundle.model.predict_proba(scaled)[:, 1]
        adjusted_logits = logit(raw_probabilities)
        adjusted_logits -= config["score_center_beta"] * (
            np.median(adjusted_logits) - config["training_subject_logit_reference"]
        )
        calibrated_probabilities = sigmoid(adjusted_logits)
        for index, raw_probability, calibrated_probability in zip(suitable_indexes, raw_probabilities, calibrated_probabilities):
            results[index]["raw_probability"] = float(raw_probability)
            results[index]["calibrated_probability"] = float(calibrated_probability)

    smoothed = smooth_probabilities([row["calibrated_probability"] for row in results], config["smoothing_window_count"])
    for row, probability in zip(results, smoothed):
        row["smoothed_probability"] = probability
        if probability is not None:
            row["predicted_label"] = "possible_obstructive_event" if probability >= bundle.event_threshold else "normal_breathing"

    events = merge_events(results, config["event_merge_gap_seconds"])
    reliable_seconds, reliable_percentage = reliable_coverage(results, duration)
    if reliable_seconds < config["minimum_reliable_seconds"] or reliable_percentage < config["minimum_reliable_percentage"]:
        overall_result = "insufficient_reliable_audio"
    elif events:
        overall_result = "possible_repeated_obstructive_pattern"
    else:
        overall_result = "no_repeated_pattern_detected"

    state_counts = dict(Counter(row["audio_suitability"] for row in results))
    excluded_counts = dict(Counter(row["audio_suitability"] for row in results if row["audio_suitability"] != "suitable_breathing_audio"))

    report = {
        "schema_version": "nemuai-final-colab-1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "filename": audio_path.name, "original_format": audio_path.suffix.lower().lstrip("."),
            "original_sample_rate": original["sample_rate"], "processed_sample_rate": processed_rate,
            "original_channels": original["channels"], "processed_channels": 1,
            "duration_seconds": round(duration, 3),
        },
        "processing": {
            "window_seconds": config["window_seconds"], "hop_seconds": config["hop_seconds"],
            "recording_center_alpha": config["recording_center_alpha"],
            "score_center_beta": config["score_center_beta"], "total_windows": len(results),
        },
        "reliability": {
            "suitable_windows": sum(row["audio_suitability"] == "suitable_breathing_audio" for row in results),
            "excluded_windows": sum(row["audio_suitability"] != "suitable_breathing_audio" for row in results),
            "reliable_audio_seconds": round(reliable_seconds, 3), "reliable_percentage": round(reliable_percentage, 2),
            "state_counts": state_counts, "exclusion_counts": excluded_counts,
        },
        "model": {
            "name": config["model_name"], "version": config["model_version"],
            "target_event_types": config["target_event_types"], "target_label": config["target_label"],
            "selected_threshold": bundle.event_threshold, "production_model": False,
        },
        "validation_metrics": bundle.selection_metadata["validation"],
        "windows": results,
        "merged_events": events,
        "overall_experimental_result": overall_result,
        "disclaimer": DISCLAIMER,
    }

    # Same sanity assertions the notebook makes before saving the report.
    assert report["schema_version"] == "nemuai-final-colab-1"
    assert all(row["raw_probability"] is None or isinstance(row["raw_probability"], float) for row in report["windows"])
    assert all(row["raw_probability"] is None for row in report["windows"] if row["audio_suitability"] != "suitable_breathing_audio")

    return report
