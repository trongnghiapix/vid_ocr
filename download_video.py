#!/usr/bin/env python3
"""
Step 1: Download video from YouTube.

Downloads at 720p (or best available ≤720p) into a target folder.
The output file is named "vid.<ext>" (default: vid.mp4).

Usage:
    python download_video.py "https://youtube.com/watch?v=..."
    python download_video.py "https://youtube.com/watch?v=..." --output-dir downloads
    python download_video.py "https://youtube.com/watch?v=..." --output-dir downloads --name vid
"""

import argparse
import sys
from pathlib import Path

import yt_dlp


def download_youtube(url: str, output_dir: Path, name: str = "vid") -> Path:
    """Download a YouTube video at 720p (or best available ≤720p) using yt-dlp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / f"{name}.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # ensure .mp4 extension after merge
        path = Path(filename).with_suffix(".mp4")
        if not path.exists():
            path = Path(filename)
        print(f"Downloaded: {path}")
        return path


def main():
    parser = argparse.ArgumentParser(description="Step 1: Download video from YouTube")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output-dir", "-d", type=Path, default=Path("downloads"),
        help="Directory to save the video (default: downloads/)",
    )
    parser.add_argument(
        "--name", "-n", default="vid",
        help="Output filename without extension (default: vid)",
    )
    args = parser.parse_args()

    if not (args.url.startswith("http://") or args.url.startswith("https://")):
        sys.exit("ERROR: Please provide a valid YouTube URL.")

    path = download_youtube(args.url, args.output_dir, args.name)
    print(f"\nDone → {path}")


if __name__ == "__main__":
    main()
