#!/usr/bin/env python3
"""
Step 3: Clean an SRT file using Gemini AI.
  - Fix Vietnamese spelling/diacritics
  - Remove nonsensical/noise entries
  - Merge near-duplicate consecutive entries

Output: <name>.cleaned.srt

Usage:
    python clean_srt.py downloads/vid.srt
    python clean_srt.py downloads/vid.srt --model gemini-2.0-flash
    GEMINI_API_KEY=xxx python clean_srt.py downloads/vid.srt
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


@dataclass
class SrtEntry:
    index: int
    start: str
    end: str
    text: str


# ── SRT parse/write ──────────────────────────────────────────────────────────

def parse_srt(path: Path) -> list[SrtEntry]:
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r"\n{2,}", content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", lines[1])
        if not m:
            continue
        text = "\n".join(lines[2:]).strip()
        entries.append(SrtEntry(index=idx, start=m.group(1), end=m.group(2), text=text))
    return entries


def write_srt(entries: list[SrtEntry], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, e in enumerate(entries, 1):
            f.write(f"{i}\n{e.start} --> {e.end}\n{e.text}\n\n")


# ── Gemini ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia hiệu đính phụ đề tiếng Việt được trích xuất bằng OCR.

Nhiệm vụ:
1. Sửa lỗi chính tả, dấu thanh tiếng Việt sai (ví dụ "câu" → "cậu", "Ői" → "Ôi")
2. XÓA các entry vô nghĩa: chuỗi ký tự rác, số liệu kỹ thuật
3. GỘP các entry gần giống nhau, chỉ khác lỗi nhỏ hoặc thiếu dấu, thành 1 entry duy nhất với nội dung chính xác nhất.
4. GIỮ NGUYÊN timing (cột START --> END), chỉ sửa/xóa phần TEXT
5. GIỮ NGUYÊN các entry có nội dung hội thoại hợp lệ dù có lỗi nhỏ

Trả về ĐÚNG định dạng SRT, KHÔNG thêm giải thích hay markdown. Nếu xóa entry, bỏ qua luôn, đánh số lại từ 1.

Ví dụ entry vô nghĩa cần XÓA:
- "69 sta .29 pvcs .88"
- "-3014 Disc. - Enterprises 16"
- "The second second second second..."
- "A ......... I"
- Chuỗi chỉ có dấu chấm hoặc gạch ngang"""


def clean_with_gemini(entries: list[SrtEntry], api_key: str, model: str) -> list[SrtEntry]:
    from google import genai

    client = genai.Client(api_key=api_key)

    # Format entries as SRT block to send
    srt_block = ""
    for e in entries:
        srt_block += f"{e.index}\n{e.start} --> {e.end}\n{e.text}\n\n"

    print(f"Sending {len(entries)} entries to {model}...")

    response = client.models.generate_content(
        model=model,
        contents=SYSTEM_PROMPT + "\n\nSRT cần xử lý:\n\n" + srt_block,
    )

    if not response.text:
        print("Warning: Gemini returned empty response, returning original entries.")
        return entries

    cleaned_text = response.text.strip()

    # Strip markdown code fences if model wraps in ```
    cleaned_text = re.sub(r"^```[^\n]*\n?", "", cleaned_text)
    cleaned_text = re.sub(r"\n?```$", "", cleaned_text)

    return parse_srt_from_string(cleaned_text)


def parse_srt_from_string(content: str) -> list[SrtEntry]:
    blocks = re.split(r"\n{2,}", content.strip())
    entries = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", lines[1])
        if not m:
            continue
        text = "\n".join(lines[2:]).strip()
        entries.append(SrtEntry(index=idx, start=m.group(1), end=m.group(2), text=text))
    return entries


# ── Chunked processing (free tier: 15 rpm) ───────────────────────────────────

def _texts_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Character-level Jaccard similarity, ignoring case and whitespace."""
    a, b = a.lower().replace("\n", " ").strip(), b.lower().replace("\n", " ").strip()
    if a == b:
        return True
    if not a or not b:
        return False
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) >= threshold


def dedup_boundaries(entries: list[SrtEntry]) -> list[SrtEntry]:
    """Remove consecutive near-duplicate entries that may span chunk boundaries.

    When two adjacent entries have similar text, keep the first and extend its
    end time to cover the second, then drop the second.
    """
    if not entries:
        return entries

    result = [entries[0]]
    for curr in entries[1:]:
        prev = result[-1]
        if _texts_similar(prev.text, curr.text):
            # Extend previous entry to cover current's end time
            prev.end = curr.end
        else:
            result.append(curr)
    return result


def clean_in_chunks(
    entries: list[SrtEntry], api_key: str, model: str, chunk_size: int = 30
) -> list[SrtEntry]:
    """Split into chunks to stay within free-tier context/rate limits."""
    chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]
    result = []

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}/{len(chunks)} ({len(chunk)} entries)...")
        cleaned = clean_with_gemini(chunk, api_key, model)
        result.extend(cleaned)
        if i < len(chunks) - 1:
            time.sleep(4)  # stay under 15 rpm free tier

    result = dedup_boundaries(result)

    # Re-index
    for i, e in enumerate(result, 1):
        e.index = i

    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean SRT with Gemini AI")
    parser.add_argument("srt", type=Path, help="Input SRT file")
    parser.add_argument("--api-key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API key")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Gemini model (default: gemini-3.1-flash-lite-preview)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output SRT (default: <name>.cleaned.srt)")
    parser.add_argument("--chunk-size", type=int, default=30, help="Entries per API call (default: 30)")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("ERROR: Gemini API key required. Use --api-key or set GEMINI_API_KEY env var.")

    srt_path = args.srt.resolve()
    if not srt_path.exists():
        sys.exit(f"ERROR: File not found: {srt_path}")

    output_path = args.output or srt_path.with_suffix("").with_suffix(".cleaned.srt")

    entries = parse_srt(srt_path)
    print(f"Loaded {len(entries)} entries from {srt_path.name}")

    cleaned = clean_in_chunks(entries, args.api_key, args.model, args.chunk_size)
    write_srt(cleaned, output_path)

    removed = len(entries) - len(cleaned)
    print(f"\nDone: {len(cleaned)} entries kept, {removed} removed → {output_path}")

    print("\nPreview (first 5):")
    for e in cleaned[:5]:
        print(f"  [{e.start} → {e.end}] {e.text!r}")


if __name__ == "__main__":
    main()
