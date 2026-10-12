"""background.py: put GitHub's dark grey behind the banner, so one picture serves both themes.

    python3 tools/logo/background.py docs/logo/defuqumat-banner.png [out.png] [--colour 0d1117]

``logo.py render`` writes the banner with a transparent film and then runs this on it
(``--background`` there, ``none`` to skip), so a bare ``render`` still regenerates the
committed file. The composite is a straight-alpha blend in sRGB space, pixel by pixel,
``rgb * a + colour * (1 - a)``, which is what a browser does with a transparent PNG on a
page of that colour; a blend in linear light would be the physically right one for
light adding up, but it moves the partly transparent pixels (the depth-of-field halos,
a third of the banner) by up to 64 of 255 from the picture that was approved, so the
display-space blend is the one kept. With no ``out``, the file is replaced in place; the
output is 8-bit RGB.
"""
import argparse

import numpy as np
from PIL import Image


def composite(path, out, colour="0d1117"):
    bg = np.array([int(colour[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255
    im = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255
    rgb, a = im[..., :3], im[..., 3:4]
    flat = np.clip(np.round((rgb * a + bg * (1 - a)) * 255), 0, 255).astype(np.uint8)
    Image.fromarray(flat, "RGB").save(out, optimize=True)
    return int((a[..., 0] < 0.998).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("png")
    p.add_argument("out", nargs="?")
    p.add_argument("--colour", default="0d1117", help="hex, GitHub's dark theme by default")
    args = p.parse_args()
    n = composite(args.png, args.out or args.png, args.colour.lstrip("#"))
    print(f"{args.out or args.png}: #{args.colour.lstrip('#')} behind {n} transparent pixels")


if __name__ == "__main__":
    main()
