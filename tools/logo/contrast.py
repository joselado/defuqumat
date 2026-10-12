"""contrast.py: the legibility of the banner's wordmark on GitHub's two themes, as numbers.

    python3 tools/logo/contrast.py docs/logo/defuqumat-banner.png mask.png

``mask.png`` is the same scene rendered with ``logo.py ... --mask-only``, so its
alpha is the letters. The banner is composited onto white (GitHub's light theme)
and onto ``#0d1117`` (its dark one); the letter pixels are where the mask's alpha
is above a half, the surround is a ring around them ``--ring`` of the height wide
(the honeycomb or the theme, whichever shows there), and the report is, per theme,
the median relative luminance of the letters and of the ring, their WCAG contrast
ratio (L1 + 0.05) / (L2 + 0.05) with L1 the brighter, and the share of letter pixels
whose own luminance is at least a ratio of 3 (the WCAG level for large text) from
the ring's median. ``--json`` prints the figures as JSON instead.
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage

THEMES = {"light": (1.0, 1.0, 1.0), "dark": (13 / 255, 17 / 255, 23 / 255)}


def _linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(rgb):
    lin = _linear(rgb)
    return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]


def composite(rgba, bg):
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + np.asarray(bg) * (1 - a)


def ratio(l1, l2):
    hi, lo = np.maximum(l1, l2), np.minimum(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def measure(banner, mask, ring_frac=0.015):
    im = np.asarray(Image.open(banner).convert("RGBA"), dtype=float) / 255
    m = np.asarray(Image.open(mask).convert("RGBA"), dtype=float)[..., 3] / 255 > 0.5
    if m.shape != im.shape[:2]:
        raise SystemExit(f"mask {m.shape} and banner {im.shape[:2]} differ in size")
    r = max(1, int(round(ring_frac * im.shape[0])))
    ring = ndimage.binary_dilation(m, iterations=r) & ~m
    out = {"letter_pixels": int(m.sum()), "ring_pixels": int(ring.sum()), "ring_px": r}
    for theme, bg in THEMES.items():
        lum = luminance(composite(im, bg))
        letters, around = lum[m], lum[ring]
        med_l, med_r = float(np.median(letters)), float(np.median(around))
        per_pixel = ratio(letters, med_r)
        out[theme] = {
            "letters_median": round(med_l, 4), "ring_median": round(med_r, 4),
            "ratio_of_medians": round(float(ratio(med_l, med_r)), 2),
            "share_at_least_3": round(float((per_pixel >= 3).mean()), 3),
            "ratio_10th_percentile": round(float(np.percentile(per_pixel, 10)), 2),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("banner")
    p.add_argument("mask")
    p.add_argument("--ring", type=float, default=0.015, help="ring width, a fraction of the height")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    out = measure(args.banner, args.mask, args.ring)
    if args.json:
        print(json.dumps(out, indent=1))
        return
    print(f"{out['letter_pixels']} letter pixels, a ring {out['ring_px']} px wide around them")
    print(f"{'theme':6s} {'letters':>8s} {'ring':>8s} {'ratio':>6s} {'>=3':>6s} {'10th pct':>9s}")
    for theme in THEMES:
        t = out[theme]
        print(f"{theme:6s} {t['letters_median']:8.3f} {t['ring_median']:8.3f} "
              f"{t['ratio_of_medians']:6.2f} {t['share_at_least_3']:6.3f} {t['ratio_10th_percentile']:9.2f}")


if __name__ == "__main__":
    main()
