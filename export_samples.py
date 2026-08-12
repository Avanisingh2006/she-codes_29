"""Write the scripted clips out as .mp4 files.

Run this if you want shareable video files — for a slide deck, or to test the
video-upload path. The app does not need them: Sample clip mode plays the same
clips directly from code.

    python export_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.synthetic import CLIPS, export_video

OUT = Path(__file__).resolve().parent / "samples"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for clip in CLIPS:
        path = OUT / f"{clip.key}.mp4"
        export_video(clip, str(path))
        size = path.stat().st_size / 1024
        print(f"  {path.name:<24} {size:7.0f} KB   {clip.caption}")
    print(f"\nWrote {len(CLIPS)} clips to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
