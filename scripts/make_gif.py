#!/usr/bin/env python3
"""Assemble README frames (PNG) into an optimised GIF. Needs Pillow (dev only; not a plugin dependency).

    python scripts/make_gif.py <frames-dir> docs/branchpoint-demo.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


def main(frames_dir: str, out: str, width: int = 1100) -> int:
    files = sorted(Path(frames_dir).glob("f*.png"))
    if not files:
        print("no frames", file=sys.stderr)
        return 2
    frames = []
    for f in files:
        im = Image.open(f).convert("RGB")
        frames.append(im.resize((width, int(im.height * width / im.width)), Image.LANCZOS))
    uniq = [frames[0]]
    for im in frames[1:]:
        if im.tobytes()[::997] != uniq[-1].tobytes()[::997]:
            uniq.append(im)
    pal = [im.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in uniq]
    durs = [500] * len(pal)
    durs[-1] = 4000
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print(f"{len(frames)} frames -> {len(uniq)} unique, {Path(out).stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
