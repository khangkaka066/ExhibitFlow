"""Render a schematic top-down floor_plan.png from data/zones.json.

Scene: MuseumVisitors (Bargello hall) - see the "note" field in data/zones.json
(reference points are still rough estimates, refine with tools/pick_reference_points.py
once available). Re-run whenever
zones.json changes:  python tools/make_floor_plan.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONES = os.path.join(HERE, "data", "zones.json")

ZONE_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860",
               "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def main():
    with open(ZONES, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    fp = cfg["floor_plan"]
    W, H = float(fp["width_m"]), float(fp["height_m"])
    ppm = float(fp["px_per_m"])
    out = os.path.join(HERE, "data", fp["image"])

    fig, ax = plt.subplots(figsize=(W * ppm / 100.0, max(H * ppm / 100.0, 2.4)), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # +y down, origin top-left (matches image / data contract)
    ax.set_aspect("equal")
    ax.set_facecolor("#f4efe9")

    # --- static scene furniture (visual reference only, not analytical) ---
    ax.axhspan(0.0, 0.25, color="#cbb89a", zorder=1)          # left-side shop wall strip
    ax.axhspan(H - 0.25, H, color="#cbb89a", zorder=1)        # right-side shop wall strip

    # --- zones ---
    for i, z in enumerate(cfg.get("zones", [])):
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        poly = MplPolygon(z["polygon_m"], closed=True, facecolor=color, alpha=0.30,
                          edgecolor=color, linewidth=1.8, zorder=3)
        ax.add_patch(poly)
        xs = [p[0] for p in z["polygon_m"]]
        ys = [p[1] for p in z["polygon_m"]]
        ax.text(sum(xs) / len(xs), sum(ys) / len(ys), f"{z['id']}\n{z['name']}",
                ha="center", va="center", fontsize=7, color=color, weight="bold", zorder=4)

    # --- reference points ---
    for r in cfg.get("reference_points", []):
        fx, fy = r["floor_m"]
        ax.plot(fx, fy, "x", color="#b00020", ms=8, mew=2, zorder=5)
        ax.text(fx + 0.15, fy + 0.15, r["id"], fontsize=6, color="#b00020", zorder=5)

    # 1 m grid
    for gx in range(0, int(W) + 1):
        ax.axvline(gx, color="#00000010", lw=0.5, zorder=0)
    for gy in range(0, int(H) + 1):
        ax.axhline(gy, color="#00000010", lw=0.5, zorder=0)

    ax.set_xticks(range(0, int(W) + 1, 1))
    ax.set_yticks(range(0, int(H) + 1, 1))
    ax.set_xlabel("x (mét)")
    ax.set_ylabel("y (mét)")
    ax.set_title("ExhibitFlow - Sơ đồ mặt bằng (MuseumVisitors - Sảnh Donatello, Bargello)",
                 fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print(f"wrote {out}  ({W}x{H} m @ {ppm} px/m)")


if __name__ == "__main__":
    main()
