#!/usr/bin/env python3
"""Compose the demo video from rendered UI frames + title/terminal cards (dev tooling: needs Pillow and ffmpeg).

    python scripts/make_video.py <frames-dir> <adopt-output.txt> <out.mp4>
Frames: g03..g36.png (live view, one per event), forge36.png, ink36.png, record.png.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1500, 860, 15
FD = "/usr/share/fonts/truetype/dejavu/"
MONO = ImageFont.truetype(FD + "DejaVuSansMono.ttf", 23)
MONOB = ImageFont.truetype(FD + "DejaVuSansMono-Bold.ttf", 23)
BIG = ImageFont.truetype(FD + "DejaVuSans-Bold.ttf", 78)
MID = ImageFont.truetype(FD + "DejaVuSans.ttf", 30)
SMALL = ImageFont.truetype(FD + "DejaVuSans.ttf", 19)
BG, FG, DIM, ACC, GOOD = (10, 13, 19), (230, 236, 247), (138, 152, 180), (122, 162, 255), (61, 220, 151)


class Out:
    def __init__(self, d: Path):
        self.d, self.n = d, 0
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    def add(self, im: Image.Image, secs: float = 1 / FPS):
        im = im.convert("RGB")
        for _ in range(max(1, round(secs * FPS))):
            im.save(self.d / f"f{self.n:05d}.png")
            self.n += 1


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def title_card() -> Image.Image:
    im, d = canvas()
    d.text((110, 270), "Branch", font=BIG, fill=FG)
    d.text((110 + d.textlength("Branch", font=BIG), 270), "point", font=BIG, fill=ACC)
    d.text((110, 380), "/rewind is undo-and-retry.", font=MID, fill=FG)
    d.text((110, 424), "Branchpoint is undo-and-explore.", font=MID, fill=GOOD)
    d.text((110, 700), "Replay of a real run on the bundled demo service. The model replies are scripted;", font=SMALL, fill=DIM)
    d.text((110, 728), "the git worktrees, pytest runs, blast tracing, blind council and adopt are the real engine.", font=SMALL, fill=DIM)
    return im


def terminal(lines: list[tuple[str, tuple[int, int, int]]], upto_chars: int | None = None, caption: str = "") -> Image.Image:
    im, d = canvas()
    d.rounded_rectangle((60, 60, W - 60, H - 60), 18, fill=(16, 21, 31), outline=(36, 48, 71), width=2)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((90 + i * 32, 88, 108 + i * 32, 106), fill=c)
    y, left = 140, upto_chars
    for text, col in lines:
        if left is not None:
            if left <= 0:
                break
            text = text[:left]
            left -= len(text)
        # wrap long lines
        while text:
            seg, text = text[:92], text[92:]
            d.text((96, y), seg, font=MONO, fill=col)
            y += 33
        y += 4
    if caption:
        d.text((96, H - 120), caption, font=SMALL, fill=DIM)
    return im


def main(frames: str, adopt_txt: str, out: str) -> int:
    fr = Path(frames)
    o = Out(Path("/tmp/bpt/vid"))
    o.add(title_card(), 3.5)

    cmd = '/branchpoint rewind 1 --goal "add caching to the slow GET /report/{customer_id} endpoint" --n 4 --execute all'
    typed = [("# you already built an in-memory TTL cache with your agent. Was it the right call?", DIM), ("$ " + cmd, FG)]
    total = sum(len(t) for t, _ in typed)
    for n in range(0, total + 1, 4):
        o.add(terminal(typed, n, "Branchpoint rewinds to prompt 1; what you built becomes route A (the baseline)"))
    out_lines = typed + [
        ("", DIM),
        ("4 distinct routes:", GOOD),
        ("  A  [library-choice]  Hand-rolled in-memory TTL cache   <- what you built", FG),
        ("  B  [infrastructure]  Shared Redis cache", FG),
        ("  C  [api-contract]    HTTP ETag caching", FG),
        ("  D  [do-less]         Precompute on a schedule", FG),
        ("", DIM),
        ("Your working tree is not touched until you run /branchpoint adopt.", DIM),
        ("Live view: http://127.0.0.1:<port>/?t=<token>", ACC),
    ]
    o.add(terminal(out_lines, None, "Near-duplicate strategies are rejected and regenerated before anything is built"), 4.5)

    paths = sorted(fr.glob("g*.png"))
    prev = Image.open(paths[0]).convert("RGB")
    o.add(prev, 0.8)
    for p in paths[1:]:
        cur = Image.open(p).convert("RGB")
        for k in range(1, 5):
            o.add(Image.blend(prev, cur, k / 4), 1 / FPS)
        hold = 0.45
        o.add(cur, hold)
        prev = cur
    o.add(prev, 2.6)

    prev_img = prev
    for name, secs, cap in (("record.png", 4.0, "The verdict: scores, why it won by how much, and the worst flaw of every route"),
                            ("record2.png", 3.4, "Why the winner won and the others lost: every sentence links to evidence"),
                            ("changes.png", 5.6, "Changes ±: the real diff, with the justification attached to each change")):
        im = Image.open(fr / name).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((W // 2 - 520, H - 82, W // 2 + 520, H - 30), 12, fill=(0, 0, 0))
        d.text((W // 2 - d.textlength(cap, font=SMALL) / 2, H - 68), cap, font=SMALL, fill=(255, 255, 255))
        for k in range(1, 6):
            o.add(Image.blend(prev_img, im, k / 5), 1 / FPS)
        o.add(im, secs)
        prev_img = im

    def captioned(path: Path, text: str) -> Image.Image:
        im = Image.open(path).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((W // 2 - 330, H - 120, W // 2 + 330, H - 66), 12, fill=(0, 0, 0))
        d.text((W // 2 - d.textlength(text, font=MID) / 2, H - 114), text, font=MID, fill=(255, 255, 255))
        return im

    for name, label in (("record.png", "Glass"), ("record-forge.png", "Forge"), ("record-ink.png", "Ink"),
                        ("changes-forge.png", "Forge"), ("changes-ink.png", "Ink")):
        o.add(captioned(fr / name, f"Decision Record · {label} look"), 2.0)

    adopt = Path(adopt_txt).read_text().strip().splitlines()
    typed2 = [("$ /branchpoint adopt C merge", FG)] + [("", DIM)] + [(ln, GOOD if "PASS" in ln else FG) for ln in adopt]
    total2 = sum(len(t) for t, _ in typed2)
    for n in range(0, total2 + 1, 6):
        o.add(terminal(typed2, n, "Adopt makes your tree equal the winning route, keeps a safety snapshot, and runs your tests. /branchpoint undo reverses it."))
    o.add(terminal(typed2, None, "Adopt makes your tree equal the winning route, keeps a safety snapshot, and runs your tests. /branchpoint undo reverses it."), 4.5)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(o.d / "f%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", out], check=True)
    print(f"{o.n} frames, {o.n / FPS:.0f}s -> {out} ({Path(out).stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
