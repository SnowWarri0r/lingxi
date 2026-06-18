"""Turn static pet sprites into gentle idle-animation APNGs (full alpha).

Motion = a slow sway (small rotation about the feet) + a vertical bob, eased
on a sine. NOT a uniform scale pulse (that reads as the image "swelling").
This is a cheap "it's alive" loop from a single still — real per-part motion
(blink, mouth) still needs multi-frame art.

Usage:
    python tools/make_idle_apng.py <sprite_dir> [--states a,b,c]
Writes <name>.apng next to each <name>.png. The pet window plays .apng with
its own full-alpha frame player.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image


def make_idle_apng(
    src_path: Path,
    out_path: Path,
    *,
    frames: int = 18,
    fps: int = 9,
    sway_deg: float = 2.0,
    bob_px: int = 3,
    pivot_y: float = 0.85,
    pad: int = 12,
    max_size: int = 460,
) -> None:
    src = Image.open(src_path).convert("RGBA")
    # The pet shows ~240px (×2 on Retina); a 1024px source makes a 20MB+ APNG
    # that's slow to decode. Downscale so frames stay small + sharp enough.
    if max(src.size) > max_size:
        src.thumbnail((max_size, max_size), Image.LANCZOS)
    w, h = src.size
    canvas = (w + 2 * pad, h + 2 * pad)
    dur = int(round(1000 / fps))
    out = []
    for i in range(frames):
        ph = 2 * math.pi * i / frames
        angle = sway_deg * math.sin(ph)
        dy = bob_px * math.sin(ph + math.pi / 2)
        rot = src.rotate(
            angle, resample=Image.BICUBIC, center=(w / 2, h * pivot_y), expand=False
        )
        frame = Image.new("RGBA", canvas, (0, 0, 0, 0))
        frame.paste(rot, (pad, pad + int(round(dy))), rot)
        out.append(frame)
    out[0].save(
        out_path, save_all=True, append_images=out[1:],
        duration=dur, loop=0, disposal=2,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sprite_dir")
    ap.add_argument("--states", default="", help="comma list; default = all *.png")
    args = ap.parse_args()

    d = Path(args.sprite_dir)
    if args.states:
        names = args.states.split(",")
        pngs = [d / f"{n}.png" for n in names]
    else:
        pngs = sorted(p for p in d.glob("*.png"))

    for png in pngs:
        if not png.exists():
            print(f"skip (missing): {png}")
            continue
        out = png.with_suffix(".apng")
        make_idle_apng(png, out)
        print(f"✓ {out.name}")


if __name__ == "__main__":
    main()
