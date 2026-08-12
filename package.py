"""Build a zip that runs on someone else's machine.

Includes the pose model so the recipient needs no network on first run, and
excludes the virtualenv, caches and any previously exported videos.

    python package.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "movewise.zip"

INCLUDE_SUFFIX = {".py", ".txt", ".md", ".task", ".bat", ".sh", ".toml"}
# data/ holds the local user's recorded sessions — personal, never shipped.
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", ".idea", ".vscode", "samples", "data"}


def main() -> int:
    if OUT.exists():
        OUT.unlink()

    written, total = 0, 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
                continue
            if path.suffix.lower() not in INCLUDE_SUFFIX:
                continue
            if path.name == OUT.name:
                continue
            zf.write(path, Path("movewise") / path.relative_to(ROOT))
            written += 1
            total += path.stat().st_size

    print(f"  {written} files, {total / 1024 / 1024:.1f} MB uncompressed")
    print(f"  -> {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB zipped)")

    model = ROOT / "models" / "pose_landmarker_lite.task"
    if model.exists():
        print("  pose model bundled — recipient needs no network on first run")
    else:
        print("  NOTE: no pose model found. Run the app once first, or the "
              "recipient will need network access to download it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
