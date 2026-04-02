#!/usr/bin/env python3
"""
Extract Vietnamese subtitles from video using OCR.
Output: SRT file alongside the input video.

Available backends (--ocr):
    surya   — Surya OCR (default, offline, runs on MPS/CUDA/CPU)

Usage:
    python extract_subtitles.py video.mp4
    python extract_subtitles.py video.mp4 --ocr surya --fps 2
    python extract_subtitles.py https://youtu.be/xxx
"""

import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


# ── Tunable constants ────────────────────────────────────────────────────────
ROI_TOP_FRACTION = 0.75   # crop bottom 25% of frame as subtitle region
SAMPLE_FPS = 2.0           # frames/sec to sample (2fps good for ~24fps source)
BATCH_SIZE = 16            # images per OCR batch (reduce if OOM)
MAX_MERGE_GAP_SEC = 1.5    # merge subtitle entries within this gap

# Noise filter: min Vietnamese chars ratio to accept a line
MIN_TEXT_LEN = 4
MIN_VIET_RATIO = 0.5       # at least 50% of chars must be Vietnamese/latin

# Vietnamese character set (latin + diacritics used in Vietnamese)
_VIET_CHARS = re.compile(r"[a-zA-ZàáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ\s,.'\"!?;:-]")


@dataclass
class SubtitleEntry:
    start_sec: float
    end_sec: float
    text: str


# ── Frame helpers ────────────────────────────────────────────────────────────

def crop_subtitle_region(frame: np.ndarray) -> np.ndarray:
    """Keep only the bottom portion of the frame where subtitles appear."""
    h = frame.shape[0]
    return frame[int(h * ROI_TOP_FRACTION):, :]


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def normalize_text(text: str) -> str:
    # Surya uses <br> for intra-line breaks; convert to newline for SRT
    text = text.replace("<br>", "\n")
    # Collapse whitespace within each line, preserve newlines
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(l for l in lines if l).strip()


def is_valid_subtitle(text: str) -> bool:
    """Filter out noise: too short, mostly non-Vietnamese chars, or repetitive garbage."""
    flat = text.replace("\n", " ")
    if len(flat) < MIN_TEXT_LEN:
        return False
    # Reject repetitive words (e.g. "second second second second...")
    words = flat.split()
    if len(words) >= 4 and len(set(words)) / len(words) < 0.4:
        return False
    # Vietnamese/latin ratio check
    viet_chars = sum(1 for _ in _VIET_CHARS.finditer(flat))
    return viet_chars / len(flat) >= MIN_VIET_RATIO


def texts_are_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Character-level Jaccard similarity between two strings."""
    if a == b:
        return True
    if not a or not b:
        return False
    set_a = set(a)
    set_b = set(b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return (intersection / union) >= threshold


# ── SRT helpers ──────────────────────────────────────────────────────────────

def seconds_to_srt_time(s: float) -> str:
    ms = int((s % 1) * 1000)
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def merge_raw_results(
    raw: list[tuple[float, str]], sample_fps: float, max_gap_sec: float
) -> list[SubtitleEntry]:
    """Merge consecutive identical/similar frames into SRT entries."""
    if not raw:
        return []

    frame_duration = 1.0 / sample_fps
    entries: list[SubtitleEntry] = []
    start, prev_text = raw[0]
    prev_time = start

    for ts, text in raw[1:]:
        gap = ts - prev_time
        if texts_are_similar(text, prev_text) and gap <= max_gap_sec:
            prev_time = ts  # extend current entry
        else:
            if prev_text:
                entries.append(SubtitleEntry(start, prev_time + frame_duration, prev_text))
            start = ts
            prev_text = text
            prev_time = ts

    if prev_text:
        entries.append(SubtitleEntry(start, prev_time + frame_duration, prev_text))

    return entries


def write_srt(entries: list[SubtitleEntry], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, e in enumerate(entries, 1):
            f.write(f"{i}\n")
            f.write(f"{seconds_to_srt_time(e.start_sec)} --> {seconds_to_srt_time(e.end_sec)}\n")
            f.write(f"{e.text}\n\n")


# ── OCR backend ──────────────────────────────────────────────────────────────

BACKENDS = {
    "surya": "backends.surya",
    # Add more here, e.g.:
    # "qwen": "backends.qwen",
    # "google": "backends.google_vision",
}


def load_backend(name: str):
    import importlib
    if name not in BACKENDS:
        sys.exit(f"ERROR: Unknown OCR backend '{name}'. Available: {', '.join(BACKENDS)}")
    module = importlib.import_module(BACKENDS[name])
    state = module.load()
    return module, state


# ── Main pipeline ────────────────────────────────────────────────────────────

def extract_subtitles(
    video_path: Path,
    output_srt: Path,
    backend,
    state,
    sample_fps: float = SAMPLE_FPS,
    batch_size: int = BATCH_SIZE,
) -> list[SubtitleEntry]:

    print(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    frame_interval = max(1, int(video_fps / sample_fps))

    print(f"Video: {video_fps:.2f}fps, {total_frames} frames, {duration:.1f}s")
    print(f"Sampling every {frame_interval} frames (~{sample_fps}fps) → ~{int(duration * sample_fps)} frames total\n")

    pending_frames: list[np.ndarray] = []
    pending_timestamps: list[float] = []
    raw_results: list[tuple[float, str]] = []
    frames_processed = 0

    def flush_batch():
        nonlocal frames_processed
        if not pending_frames:
            return
        crops = [crop_subtitle_region(f) for f in pending_frames]
        pil_images = [bgr_to_pil(c) for c in crops]
        texts = backend.run(pil_images, state)
        for ts, text in zip(pending_timestamps, texts):
            t = normalize_text(text)
            if t and is_valid_subtitle(t):
                raw_results.append((ts, t))
        frames_processed += len(pending_frames)
        print(
            f"  [{frames_processed} frames] {len(raw_results)} subtitle candidates...",
            end="\r",
        )
        pending_frames.clear()
        pending_timestamps.clear()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            pending_frames.append(frame)
            pending_timestamps.append(frame_idx / video_fps)
            if len(pending_frames) >= batch_size:
                flush_batch()
        frame_idx += 1

    flush_batch()
    cap.release()

    print(f"\n\nOCR complete. {len(raw_results)} subtitle frames → merging...")
    entries = merge_raw_results(raw_results, sample_fps, MAX_MERGE_GAP_SEC)
    write_srt(entries, output_srt)
    print(f"Wrote {len(entries)} SRT entries to: {output_srt}")
    return entries


def download_youtube(url: str, output_dir: Path) -> Path:
    """Download a YouTube video at 720p (or best available ≤720p) using yt-dlp."""
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
        if not path.exists():
            path = Path(ydl.prepare_filename(info))
        print(f"Downloaded: {path}")
        return path


def main():
    parser = argparse.ArgumentParser(description="Extract Vietnamese subtitles from video using OCR")
    parser.add_argument("video", help="Input video file path or YouTube URL")
    parser.add_argument("--ocr", choices=list(BACKENDS), default="surya", help=f"OCR backend (default: surya)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output SRT path (default: <video>.srt)")
    parser.add_argument("--fps", type=float, default=SAMPLE_FPS, help=f"Sample rate in fps (default: {SAMPLE_FPS})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--download-dir", type=Path, default=Path("downloads"), help="Directory for downloaded videos (default: downloads/)")
    args = parser.parse_args()

    # YouTube URL detection
    if args.video.startswith("http://") or args.video.startswith("https://"):
        video_path = download_youtube(args.video, args.download_dir)
    else:
        video_path = Path(args.video).resolve()
        if not video_path.exists():
            sys.exit(f"ERROR: File not found: {video_path}")

    output_srt = args.output or video_path.with_suffix(".srt")

    backend, state = load_backend(args.ocr)
    entries = extract_subtitles(video_path, output_srt, backend, state, sample_fps=args.fps, batch_size=args.batch)

    if entries:
        print("\nFirst 5 entries preview:")
        for e in entries[:5]:
            print(f"  [{seconds_to_srt_time(e.start_sec)} → {seconds_to_srt_time(e.end_sec)}] {e.text!r}")


if __name__ == "__main__":
    main()
