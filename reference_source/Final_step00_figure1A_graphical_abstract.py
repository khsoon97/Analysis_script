# %%
"""Build Figure 1A from one Python canvas.

Only the text-free histology/CODEX illustration is raster. All labels,
connectors, cells, motifs, score bars, and validation icons are Matplotlib
vector artists exported to a native-shape PowerPoint slide.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("FINAL_LIGHT_IMPORTS", "1")
os.environ["FINAL_FIGURE_ONLY_WRITE_GUARD"] = "1"
from Final_utils import _final_figure_stem_registered, save_final_figure_pair
from editable_pptx import _validate_native_shape_slide


FINAL_ROOT = Path("__ROIDER_ROOT__/Analysis/python/Final")
FIG_ROOT = FINAL_ROOT / "figures" / "v2"
ASSET_ROOT = FINAL_ROOT / "references" / "figure1A_assets"
DEFAULT_OUT = FIG_ROOT / "Final_Figure1A_GraphicalAbstract"
DEFAULT_ASSET = ASSET_ROOT / "Figure1A_cohort_dlbcl_codex_v3.png"
STEP00_FIGURE_ONLY = True

CANVAS_W = 7.5
# Match the 7.5 x 3.65 in Figure-1A slot on slide 1 of the layout reference PPTX.
# The wider landscape ratio is part of the source canvas, never a post-save crop.
CANVAS_H = 3.65
FIGURE_DPI = 300
# Keep the direct-save PNG at 2250×1095 px while the native PowerPoint slide
# remains exactly 7.5×3.65 in. No crop or post-save resize is applied.
PNG_DPI = 300.0

ARIAL_REGULAR = Path("__FONT_ROOT__/arial.ttf")
ARIAL_BOLD = Path("__FONT_ROOT__/arialbd.ttf")

COLORS = {
    "ink": "#111827",
    "muted": "#4B5563",
    "soft": "#F8FAFC",
    "line": "#475569",
    "blue": "#1739C8",
    "blue_fill": "#F2F6FF",
    "orange": "#F05A28",
    "orange_fill": "#FFF9F5",
    "teal": "#0B6473",
    "teal_fill": "#F1FAFB",
    "red": "#C9252D",
    "green": "#4C7A27",
    "purple": "#7A2D91",
    "dr_pos": "#0D3B8E",
    "dr_neg": "#5A8CFF",
    "t_cell": "#548235",
}


def _register_arial() -> dict[str, str]:
    """Register and resolve the exact Arial Regular/Bold files."""
    for path in (ARIAL_REGULAR, ARIAL_BOLD):
        if not path.is_file():
            raise FileNotFoundError(f"Required Arial font is missing: {path}")
        fm.fontManager.addfont(str(path))
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": 7.0,
            "svg.fonttype": "path",
        }
    )
    resolved = {}
    for name, weight, expected in (
        ("regular", "normal", ARIAL_REGULAR),
        ("bold", "bold", ARIAL_BOLD),
    ):
        actual = Path(
            fm.findfont(
                fm.FontProperties(family=["Arial"], weight=weight),
                fallback_to_default=False,
            )
        ).resolve()
        if actual != expected.resolve():
            raise RuntimeError(
                f"Arial {name} resolution mismatch: {actual} != {expected.resolve()}"
            )
        resolved[name] = str(actual)
    return resolved


def _text(
    ax,
    x,
    y,
    value,
    *,
    size=7.0,
    weight="normal",
    color=None,
    ha="center",
    va="center",
    linespacing=1.08,
    zorder=20,
):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontfamily="Arial",
        fontweight=weight,
        fontstyle="normal",
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        linespacing=linespacing,
        zorder=zorder,
    )


def _round_box(
    ax,
    x,
    y,
    w,
    h,
    *,
    edge,
    face="white",
    lw=1.0,
    radius=0.055,
    linestyle="-",
    zorder=2,
):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _tag(ax, x, y, w, label, *, edge, face="white", color=None):
    """Compact scope tag used inside a module, never as a workflow node."""
    _round_box(
        ax,
        x,
        y,
        w,
        0.17,
        edge=edge,
        face=face,
        lw=0.55,
        radius=0.026,
        zorder=9,
    )
    _text(
        ax,
        x + w / 2,
        y + 0.085,
        label,
        size=4.8,
        weight="bold",
        color=color or edge,
    )


def _arrow(ax, start, end, *, color=None, lw=1.0, mutation=8, zorder=10):
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color or COLORS["ink"],
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _elbow_arrow(ax, start, end, x_mid, *, color=None, lw=0.9):
    color = color or COLORS["ink"]
    x0, y0 = start
    x1, y1 = end
    ax.plot([x0, x_mid, x_mid], [y0, y0, y1], color=color, lw=lw, zorder=8)
    return _arrow(ax, (x_mid, y1), (x1, y1), color=color, lw=lw, mutation=7)


def _cell(ax, x, y, r, *, color, edge=None, nucleus=True, zorder=12):
    edge = edge or color
    ax.add_patch(
        patches.Circle(
            (x, y), r, facecolor=color, edgecolor=edge, linewidth=0.8, zorder=zorder
        )
    )
    if nucleus:
        ax.add_patch(
            patches.Circle(
                (x, y),
                r * 0.46,
                facecolor="#23405F",
                edgecolor="none",
                alpha=0.72,
                zorder=zorder + 1,
            )
        )


def _person(ax, x, y, *, color, scale=1.0):
    head_r = 0.032 * scale
    ax.add_patch(patches.Circle((x, y), head_r, facecolor=color, edgecolor="none", zorder=12))
    ax.add_patch(
        patches.FancyBboxPatch(
            (x - 0.034 * scale, y + 0.040 * scale),
            0.068 * scale,
            0.128 * scale,
            boxstyle=f"round,pad=0.004,rounding_size={0.018 * scale}",
            facecolor=color,
            edgecolor="none",
            zorder=12,
        )
    )
    for sign in (-1, 1):
        ax.plot(
            [x + sign * 0.020 * scale, x + sign * 0.032 * scale],
            [y + 0.164 * scale, y + 0.238 * scale],
            color=color,
            lw=3.0 * scale,
            solid_capstyle="round",
            zorder=12,
        )


def _model_pill(ax, x, y, label):
    _round_box(
        ax,
        x,
        y,
        0.245,
        0.205,
        edge=COLORS["blue"],
        face="white",
        lw=0.8,
        radius=0.035,
        zorder=5,
    )
    _text(ax, x + 0.1225, y + 0.103, label, size=6.5, weight="bold")


def _cover_crop(image: np.ndarray, target_ratio: float) -> np.ndarray:
    """Center-crop without stretching to cover a requested width/height ratio."""
    height, width = image.shape[:2]
    ratio = width / height
    if ratio > target_ratio:
        new_width = max(1, int(round(height * target_ratio)))
        x0 = (width - new_width) // 2
        return image[:, x0 : x0 + new_width]
    new_height = max(1, int(round(width / target_ratio)))
    y0 = (height - new_height) // 2
    return image[y0 : y0 + new_height, :]


def _draw_cohort_panel(ax, asset_path: Path):
    x, y, w, h = 0.07, 0.07, 1.76, 2.61
    # Keep the cohort panel face transparent so the PowerPoint raster overlay
    # remains visible behind the native vector border and labels.
    _round_box(ax, x, y, w, h, edge=COLORS["blue"], face="none", lw=1.15, radius=0.06)
    _text(ax, x + w / 2, y + 0.15, "Published CODEX cohort", size=8.7, weight="bold", color=COLORS["blue"])
    _text(
        ax,
        x + w / 2,
        y + 0.34,
        "5.45M cells · 50 proteins · 19 patients · 35 tissues",
        size=5.2,
    )

    image = plt.imread(asset_path)
    ix0, iy0, iw, ih = x + 0.26, y + 0.45, 1.24, 1.17
    image = _cover_crop(image, iw / ih)
    ax.imshow(image, extent=(ix0, ix0 + iw, iy0 + ih, iy0), aspect="auto", zorder=3)
    _round_box(
        ax,
        ix0 - 0.012,
        iy0 - 0.012,
        iw + 0.024,
        ih + 0.024,
        edge="#CBD5E1",
        face="none",
        lw=0.7,
        radius=0.025,
        zorder=4,
    )

    divider_y = y + 1.70
    ax.plot(
        [x + 0.06, x + w - 0.06],
        [divider_y, divider_y],
        color=COLORS["blue"],
        lw=0.9,
        ls=(0, (4, 2)),
        zorder=6,
    )
    _text(ax, x + w / 2, divider_y + 0.14, "Reference / target", size=7.4, weight="bold", color=COLORS["blue"])
    _text(ax, x + w / 2, divider_y + 0.29, "non-DLBCL · DLBCL", size=5.3)

    left_center, right_center = x + 0.50, x + 1.40
    _text(ax, left_center, divider_y + 0.47, "Reference", size=5.4, weight="bold", color=COLORS["blue"])
    _text(ax, right_center, divider_y + 0.47, "DLBCL target", size=5.4, weight="bold", color=COLORS["red"])
    # Reference comprises 15 patients versus four DLBCL target patients.
    # The glyph ratio is intentionally reference-heavy; exact sizes follow.
    for idx, dx in enumerate((-0.34, -0.17, 0.0, 0.17, 0.34)):
        _person(
            ax,
            left_center + dx,
            divider_y + 0.57 + (0.01 if idx != 1 else 0),
            color=COLORS["blue"],
            scale=0.42,
        )
    for idx, dx in enumerate((-0.12, 0.12)):
        _person(
            ax,
            right_center + dx,
            divider_y + 0.57 + (0.01 if idx % 2 else 0),
            color="#D40000",
            scale=0.42,
        )
    _round_box(
        ax,
        x + 0.08,
        y + 2.34,
        0.96,
        0.20,
        edge=COLORS["blue"],
        face=COLORS["blue"],
        lw=0,
        radius=0.055,
        zorder=8,
    )
    _text(
        ax,
        x + 0.56,
        y + 2.44,
        "4.45M cells\n15 patients · 27 tissues",
        size=3.85,
        weight="bold",
        color="white",
        linespacing=0.88,
    )
    _round_box(
        ax,
        x + 1.08,
        y + 2.34,
        0.60,
        0.20,
        edge="#D40000",
        face="#D40000",
        lw=0,
        radius=0.055,
        zorder=8,
    )
    _text(
        ax,
        x + 1.38,
        y + 2.44,
        "995,424 cells\n4 patients · 8 tissues",
        size=3.55,
        weight="bold",
        color="white",
        linespacing=0.88,
    )


def _draw_annotation_panel(ax):
    x, y, w, h = 1.91, 0.07, 1.49, 2.61
    _round_box(ax, x, y, w, h, edge=COLORS["blue"], face="white", lw=1.15, radius=0.06)
    _text(
        ax,
        x + w / 2,
        y + 0.15,
        "Reference-trained\nannotation",
        size=7.0,
        weight="bold",
        color=COLORS["blue"],
        linespacing=1.0,
    )

    _round_box(ax, x + 0.11, y + 0.37, w - 0.22, 0.42, edge=COLORS["blue"], face=COLORS["blue_fill"], lw=0.9, radius=0.040)
    _text(
        ax,
        x + w / 2,
        y + 0.47,
        "Patient-grouped models",
        size=5.5,
        weight="bold",
    )
    for xpos, label in zip(
        (x + 0.15, x + 0.45, x + 0.75, x + 1.05),
        ("RF", "XGB", "ENet", "MLP"),
    ):
        _model_pill(ax, xpos, y + 0.56, label)

    _arrow(ax, (x + w / 2, y + 0.81), (x + w / 2, y + 0.88), lw=0.85, mutation=6)
    _round_box(ax, x + 0.14, y + 0.90, w - 0.28, 0.22, edge=COLORS["blue"], face=COLORS["blue_fill"], lw=0.85, radius=0.038)
    _text(ax, x + w / 2, y + 1.01, "Broad lineage · XGB + MLP", size=5.35, weight="bold")

    _arrow(ax, (x + w / 2, y + 1.14), (x + w / 2, y + 1.21), lw=0.85, mutation=6)
    _round_box(ax, x + 0.14, y + 1.23, w - 0.28, 0.27, edge=COLORS["blue"], face="white", lw=0.85, radius=0.038)
    _text(
        ax,
        x + w / 2,
        y + 1.365,
        "Low-confidence rescue\nSpatial KNN · marker guard",
        size=5.0,
        weight="bold",
        linespacing=0.95,
    )

    _arrow(ax, (x + w / 2, y + 1.52), (x + w / 2, y + 1.59), lw=0.85, mutation=6)
    _round_box(ax, x + 0.24, y + 1.61, w - 0.48, 0.18, edge=COLORS["blue"], face=COLORS["blue_fill"], lw=0.70, radius=0.035)
    _text(ax, x + w / 2, y + 1.70, "5 broad lineages", size=5.35, weight="bold")

    _arrow(ax, (x + w / 2, y + 1.81), (x + w / 2, y + 1.87), lw=0.85, mutation=6)
    _round_box(
        ax,
        x + 0.11,
        y + 1.89,
        w - 0.22,
        0.27,
        edge=COLORS["blue"],
        face="white",
        lw=0.9,
        radius=0.045,
        linestyle=(0, (4, 2)),
    )
    ax.add_patch(
        patches.Polygon(
            [
                (x + 0.14, y + 1.95),
                (x + 0.32, y + 1.95),
                (x + 0.27, y + 2.03),
                (x + 0.27, y + 2.10),
                (x + 0.22, y + 2.13),
                (x + 0.22, y + 2.03),
            ],
            closed=True,
            facecolor="white",
            edgecolor=COLORS["ink"],
            lw=0.8,
            zorder=10,
        )
    )
    _text(ax, x + 0.40, y + 1.97, "Lineage routing", size=5.35, weight="bold", ha="left")
    _text(
        ax,
        x + 0.40,
        y + 2.09,
        "B · T · myeloid · stromal",
        size=4.65,
        ha="left",
    )

    _arrow(ax, (x + w / 2, y + 2.18), (x + w / 2, y + 2.23), lw=0.85, mutation=6)
    _round_box(ax, x + 0.13, y + 2.25, w - 0.26, 0.29, edge=COLORS["blue"], face=COLORS["blue_fill"], lw=0.85, radius=0.040)
    _text(ax, x + w / 2, y + 2.32, "Annotations in DLBCL tissues", size=5.4, weight="bold")
    _text(ax, x + w / 2, y + 2.43, "Lineage-routed subtypes", size=4.65, weight="bold")
    _text(ax, x + w / 2, y + 2.51, "GRZB-derived CD4 CTL proxy", size=4.15)


def _draw_spatial_sector(ax):
    """One sector containing four parallel post-annotation analyses."""
    _round_box(
        ax,
        3.50,
        0.07,
        3.93,
        3.51,
        edge=COLORS["orange"],
        face="#FFFDFC",
        lw=1.15,
        radius=0.06,
        zorder=1,
    )
    _text(ax, 5.47, 0.16, "Post-annotation spatial analyses", size=8.7, weight="bold")


def _draw_nearest_distance(ax, x, y, w, h):
    _round_box(ax, x, y, w, h, edge=COLORS["orange"], face=COLORS["orange_fill"], lw=1.05, radius=0.055)
    _text(ax, x + 0.14, y + 0.22, "Nearest T→B distance", size=7.3, weight="bold", ha="left")
    _tag(ax, x + 0.14, y + 0.43, 0.78, "T-cell anchored", edge=COLORS["blue"], face="white")
    _text(ax, x + 1.02, y + 0.515, "HLA-DR+ / HLA-DR− B cells", size=5.2, ha="left")
    tx, ty = x + 2.95, y + 0.38
    ax.add_patch(
        patches.Circle(
            (tx, ty),
            0.082,
            facecolor="#EDF5E8",
            edgecolor=COLORS["t_cell"],
            lw=0.75,
            alpha=0.90,
            zorder=7,
        )
    )
    _cell(ax, tx, ty, 0.052, color=COLORS["t_cell"], edge="#34531B")

    nearest_b = (
        (x + 2.55, y + 0.19, "pos"),
        (x + 3.34, y + 0.57, "neg"),
    )
    for bx, by, kind in nearest_b:
        ax.plot([tx, bx], [ty, by], color=COLORS["line"], lw=0.75, ls=(0, (3, 2)), zorder=6)
        if kind == "pos":
            _cell(ax, bx, by, 0.050, color=COLORS["dr_pos"], edge="#05275F")
        else:
            _cell(ax, bx, by, 0.050, color="white", edge=COLORS["dr_neg"], nucleus=False)


def _draw_gradient(ax, x, y, w, h, start, end, label):
    start_rgb = np.array(mpl.colors.to_rgb(start))
    end_rgb = np.array(mpl.colors.to_rgb(end))
    for idx in range(36):
        t = idx / 35
        color = start_rgb * (1 - t) + end_rgb * t
        ax.add_patch(
            patches.Rectangle(
                (x + w * idx / 36, y),
                w / 36 + 0.002,
                h,
                facecolor=color,
                edgecolor="none",
                zorder=10,
            )
        )
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="none", edgecolor=COLORS["ink"], lw=0.7, zorder=12))
    _text(ax, x, y - 0.055, label, size=4.7, ha="left", va="bottom")
    _text(ax, x, y + h + 0.08, "Low", size=4.4, ha="left")
    _text(ax, x + w, y + h + 0.08, "High", size=4.4, ha="right")


def _draw_function(ax, x, y, w, h):
    _round_box(ax, x, y, w, h, edge=COLORS["orange"], face=COLORS["orange_fill"], lw=1.05, radius=0.055)
    _text(
        ax,
        x + 0.15,
        y + 0.22,
        "T-cell activation / exhaustion",
        size=7.0,
        weight="bold",
        ha="left",
    )
    _tag(ax, x + 0.14, y + 0.43, 0.78, "T-cell anchored", edge=COLORS["blue"], face="white")
    _draw_gradient(ax, x + 2.20, y + 0.23, 1.22, 0.065, "#FFFDF7", "#E40000", "Activation")
    _draw_gradient(ax, x + 2.20, y + 0.52, 1.22, 0.065, "#FFFDF7", "#5B007F", "Exhaustion")


def _motif_card(ax, x, y, label, b_kinds):
    w, h = 0.48, 0.52
    _round_box(ax, x, y, w, h, edge=COLORS["line"], face="white", lw=0.65, radius=0.035)
    b_positions = (
        (x + 0.11, y + 0.11),
        (x + 0.37, y + 0.11),
        (x + 0.11, y + 0.31),
        (x + 0.37, y + 0.31),
    )
    # Every motif is defined around the same anchor T cell; only the number
    # and HLA-DR state of neighboring B cells change between motif classes.
    anchor_x, anchor_y = x + w / 2, y + 0.21
    ax.add_patch(
        patches.Circle(
            (anchor_x, anchor_y),
            0.064,
            facecolor="#EDF5E8",
            edgecolor=COLORS["t_cell"],
            linewidth=0.65,
            alpha=0.85,
            zorder=10,
        )
    )
    _cell(ax, anchor_x, anchor_y, 0.042, color=COLORS["t_cell"], edge="#34531B")
    for (cx, cy), kind in zip(b_positions, b_kinds):
        if kind is None:
            continue
        if kind == "pos":
            _cell(ax, cx, cy, 0.032, color=COLORS["dr_pos"], edge="#05275F")
        elif kind == "neg":
            _cell(ax, cx, cy, 0.032, color="white", edge=COLORS["dr_neg"], nucleus=False)
    _text(ax, x + w / 2, y + 0.43, label, size=4.65, weight="bold", linespacing=0.88)


def _draw_motifs(ax, x, y, w, h):
    _round_box(ax, x, y, w, h, edge=COLORS["orange"], face=COLORS["orange_fill"], lw=1.05, radius=0.055)
    _text(
        ax,
        x + 0.14,
        y + 0.20,
        "20 µm B-cell motifs",
        size=7.2,
        weight="bold",
        ha="left",
        linespacing=1.0,
    )
    _tag(ax, x + 0.14, y + 0.43, 0.78, "T-cell anchored", edge=COLORS["blue"], face="white")
    cards = (
        ("DR+\nB-dominant", ("pos", "pos", "pos", "neg")),
        ("Mixed B", ("pos", "neg", "neg", "pos")),
        ("DR−\nB-dominant", ("neg", "neg", "neg", "pos")),
        ("B-poor", ("pos", None, None, None)),
    )
    for offset, (label, b_kinds) in zip((1.30, 1.84, 2.38, 2.92), cards):
        _motif_card(ax, x + offset, y + 0.11, label, b_kinds)


def _darker(color):
    return mpl.colors.to_hex(np.clip(np.array(mpl.colors.to_rgb(color)) * 0.70, 0, 1))


def _lr_pair(ax, x, y, ligand, receptor, color):
    _cell(ax, x, y, 0.032, color=color, edge=_darker(color))
    _text(ax, x + 0.07, y, ligand, size=4.05, ha="left")
    _arrow(ax, (x + 0.36, y), (x + 0.46, y), lw=0.70, mutation=5)
    _cell(ax, x + 0.52, y, 0.032, color=color, edge=_darker(color))
    _text(ax, x + 0.59, y, receptor, size=4.05, ha="left")


def _draw_lr(ax, x, y, w, h):
    _round_box(ax, x, y, w, h, edge=COLORS["orange"], face=COLORS["orange_fill"], lw=1.05, radius=0.055)
    _text(ax, x + 0.14, y + 0.22, "L→R proximity enrichment", size=7.1, weight="bold", ha="left")
    _tag(ax, x + 0.14, y + 0.46, 0.91, "All annotated cells", edge=COLORS["orange"], face="white")
    _text(ax, x + 0.14, y + 0.72, "≤50 µm · permutation · BH-FDR", size=4.7, ha="left")
    _lr_pair(ax, x + 1.82, y + 0.27, "CXCL13", "CXCR5", COLORS["purple"])
    _lr_pair(ax, x + 2.75, y + 0.27, "HLA-DR", "LAG3", "#FF6200")
    _lr_pair(ax, x + 1.82, y + 0.59, "CD34", "CD62L", "#1756A9")
    _lr_pair(ax, x + 2.75, y + 0.59, "PDPN", "CD44", COLORS["green"])


def _draw_right_modules(ax):
    modules = (
        (_draw_nearest_distance, 3.67, 0.29, 3.66, 0.73),
        (_draw_function, 3.67, 1.08, 3.66, 0.73),
        (_draw_motifs, 3.67, 1.87, 3.66, 0.73),
        (_draw_lr, 3.67, 2.66, 3.66, 0.82),
    )
    for drawer, x, y, w, h in modules:
        drawer(ax, x, y, w, h)

    # A shared branch from the final annotations makes the four modules
    # explicitly parallel. The scope tags describe anchor/all-cell subsets.
    branch_x = 3.57
    centers = (0.655, 1.445, 2.235, 3.07)
    ax.plot([branch_x, branch_x], [centers[0], centers[-1]], color=COLORS["ink"], lw=0.8, zorder=8)
    for center_y in centers:
        _arrow(ax, (branch_x, center_y), (3.65, center_y), color=COLORS["ink"], lw=0.8, mutation=6)


def _validation_icon(ax, cx, cy, kind):
    color = COLORS["teal"]
    if kind == "matrix":
        for row in range(3):
            for col in range(3):
                fill = color if row == col else "white"
                ax.add_patch(
                    patches.Rectangle(
                        (cx - 0.13 + col * 0.085, cy - 0.13 + row * 0.085),
                        0.066,
                        0.066,
                        facecolor=fill,
                        edgecolor=color,
                        lw=0.55,
                        zorder=11,
                    )
                )
    elif kind == "branch":
        ax.plot([cx - 0.12, cx - 0.02, cx - 0.02, cx + 0.12], [cy, cy, cy - 0.10, cy - 0.10], color=color, lw=1.0, zorder=11)
        ax.plot([cx - 0.02, cx - 0.02, cx + 0.12], [cy, cy + 0.10, cy + 0.10], color=color, lw=1.0, zorder=11)
        for px, py in ((cx - 0.14, cy), (cx + 0.14, cy - 0.10), (cx + 0.14, cy + 0.10)):
            ax.add_patch(patches.Rectangle((px - 0.028, py - 0.028), 0.056, 0.056, facecolor="white", edgecolor=color, lw=0.9, zorder=12))
    elif kind == "locked_heatmap":
        heat = ("#DFF2F0", "#7FC8C1", "#2B7A78", "#164E63")
        for row in range(4):
            for col in range(4):
                ax.add_patch(
                    patches.Rectangle(
                        (cx - 0.14 + col * 0.065, cy - 0.13 + row * 0.065),
                        0.052,
                        0.052,
                        facecolor=heat[(row + 2 * col) % len(heat)],
                        edgecolor="white",
                        lw=0.25,
                        zorder=11,
                    )
                )
        ax.add_patch(patches.FancyBboxPatch((cx + 0.06, cy - 0.18), 0.12, 0.11, boxstyle="round,pad=0.01,rounding_size=0.015", facecolor="white", edgecolor=color, lw=0.8, zorder=13))
        ax.add_patch(patches.Arc((cx + 0.12, cy - 0.17), 0.08, 0.10, theta1=180, theta2=360, edgecolor=color, lw=0.8, zorder=13))
    elif kind == "datasets":
        for pane_x, fill in ((cx - 0.10, "#2471A3"), (cx + 0.10, "#C0392B")):
            ax.add_patch(patches.Rectangle((pane_x - 0.08, cy - 0.14), 0.16, 0.28, facecolor="white", edgecolor=color, lw=0.75, zorder=10))
            for dx, dy in ((-0.035, -0.06), (0.03, -0.02), (-0.01, 0.06), (0.04, 0.09)):
                ax.add_patch(patches.Circle((pane_x + dx, cy + dy), 0.016, facecolor=fill, edgecolor="none", zorder=12))
    elif kind == "score_metrics":
        fills = ("#7FC8C1", "#2B7A78", "#164E63")
        heights = ((0.17, 0.24, 0.29), (0.15, 0.22, 0.27))
        for group_idx, group_heights in enumerate(heights):
            group_x = cx - 0.22 + group_idx * 0.27
            for arm_idx, (height, fill) in enumerate(zip(group_heights, fills)):
                bx = group_x + arm_idx * 0.055
                ax.add_patch(
                    patches.Rectangle(
                        (bx, cy + 0.15 - height),
                        0.043,
                        height,
                        facecolor=fill,
                        edgecolor=color,
                        lw=0.45,
                        zorder=11,
                    )
                )
        ax.plot([cx - 0.25, cx + 0.24], [cy + 0.16, cy + 0.16], color=color, lw=0.85, zorder=12)


def _draw_validation(ax):
    x, y, w, h = 0.07, 2.75, 3.33, 0.83
    _round_box(ax, x, y, w, h, edge=COLORS["teal"], face=COLORS["teal_fill"], lw=1.05, radius=0.06)
    _text(ax, x + w / 2, y + 0.11, "Validation", size=7.1, weight="bold", color=COLORS["teal"])
    _text(ax, x + 0.60, y + 0.25, "DLBCL benchmarks", size=4.9, weight="bold", color=COLORS["teal"])
    _text(ax, x + 2.27, y + 0.25, "External scRNA", size=4.9, weight="bold", color=COLORS["teal"])
    ax.plot([x + 1.18, x + 1.18], [y + 0.18, y + 0.76], color=COLORS["teal"], lw=0.60, alpha=0.42, zorder=8)

    # Broad and subtype benchmarks are parallel evidence. The subtype result is
    # broad-lineage conditional rather than an end-to-end score.
    _validation_icon(ax, x + 0.32, y + 0.48, "matrix")
    _text(ax, x + 0.32, y + 0.70, "Broad", size=4.35, weight="bold")
    _validation_icon(ax, x + 0.85, y + 0.48, "branch")
    _text(ax, x + 0.85, y + 0.70, "Subtype*", size=4.35, weight="bold")
    _text(ax, x + 0.60, y + 0.79, "*broad-lineage conditional", size=3.65, color=COLORS["teal"])

    scrna_items = (
        (x + 1.50, "locked_heatmap", "Frozen"),
        (x + 2.08, "datasets", "VRJUNV · GSE182434"),
        (x + 2.82, "score_metrics", "3 scoring arms\nBAcc · Macro F1"),
    )
    for idx, (cx, kind, label) in enumerate(scrna_items):
        _validation_icon(ax, cx, y + 0.48, kind)
        label_y = y + (0.76 if kind == "score_metrics" else 0.72)
        _text(ax, cx, label_y, label, size=3.75, linespacing=0.88)
        if idx < len(scrna_items) - 1:
            _arrow(
                ax,
                (cx + 0.18, y + 0.48),
                (scrna_items[idx + 1][0] - 0.18, y + 0.48),
                color=COLORS["teal"],
                lw=0.65,
                mutation=5,
            )


def build_figure(asset_path: Path):
    if not asset_path.is_file():
        raise FileNotFoundError(f"Figure 1A raster asset is missing: {asset_path}")
    resolved_fonts = _register_arial()
    fig = plt.figure(figsize=(CANVAS_W, CANVAS_H), dpi=FIGURE_DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    _draw_cohort_panel(ax, asset_path)
    _draw_annotation_panel(ax)
    _draw_spatial_sector(ax)
    _draw_right_modules(ax)
    _draw_validation(ax)
    _arrow(ax, (1.84, 1.34), (1.90, 1.34), color=COLORS["ink"], lw=0.90, mutation=6)
    # Final annotations feed four parallel analyses. T-cell anchoring is a
    # within-module scope tag, not an intermediate workflow node.
    _arrow(ax, (3.41, 2.40), (3.57, 2.40), color=COLORS["ink"], lw=0.90, mutation=6)
    fig.canvas.draw()
    return fig, resolved_fonts


def _validate_canvas_artists(fig):
    """Fail if visible text or patch geometry escapes the designed canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    offenders = []
    tick_text_ids = {
        id(label)
        for axis in fig.axes
        for label in (*axis.get_xticklabels(minor=False), *axis.get_xticklabels(minor=True),
                      *axis.get_yticklabels(minor=False), *axis.get_yticklabels(minor=True))
    }
    artists = list(
        fig.findobj(match=lambda obj: isinstance(obj, (mpl.text.Text, patches.Patch)))
    )
    for artist in artists:
        if not artist.get_visible():
            continue
        if isinstance(artist, mpl.spines.Spine) or id(artist) in tick_text_ids:
            continue
        if isinstance(artist, mpl.text.Text) and not artist.get_text().strip():
            continue
        try:
            bbox = artist.get_window_extent(renderer)
        except Exception:
            continue
        if not np.all(np.isfinite(bbox.extents)):
            continue
        if (
            bbox.x0 < canvas.x0 - 1
            or bbox.y0 < canvas.y0 - 1
            or bbox.x1 > canvas.x1 + 1
            or bbox.y1 > canvas.y1 + 1
        ):
            label = (
                artist.get_text()[:50]
                if isinstance(artist, mpl.text.Text)
                else type(artist).__name__
            )
            offenders.append(f"{label}: {tuple(round(v, 1) for v in bbox.extents)}")
    if offenders:
        raise RuntimeError(
            "Figure 1A artist is outside the canvas:\n  - "
            + "\n  - ".join(offenders[:20])
        )


def save_figure(fig, out_stem: Path):
    """Save the canonical stem either directly or in one isolated preview directory."""
    out_stem = Path(out_stem).resolve()
    if not STEP00_FIGURE_ONLY:
        raise RuntimeError("Final Step00 is a Figure-only entry point")
    if out_stem.name != DEFAULT_OUT.name or not _final_figure_stem_registered(out_stem):
        raise PermissionError(f"Unregistered Step00 Figure stem: {out_stem}")
    relative_parent = out_stem.parent.relative_to(FIG_ROOT.resolve())
    if len(relative_parent.parts) > 1 or (
        relative_parent.parts and not relative_parent.parts[0].startswith("_tmp_")
    ):
        raise PermissionError(
            "Step00 output must be canonical or one direct _tmp_ preview directory: "
            f"{out_stem.parent}"
        )
    _validate_canvas_artists(fig)
    png_path = out_stem.with_suffix(".png")
    pptx_path = out_stem.with_suffix(".pptx")
    saved_png, saved_pptx = save_final_figure_pair(
        fig,
        png_path,
        pptx_path,
        dpi=int(PNG_DPI),
        facecolor="white",
        auxiliary_tight=False,
        validate=True,
        preserve_target_directory=bool(relative_parent.parts),
        pptx_rasterize_dense_layers=True,
        pptx_dense_layer_point_threshold=100,
        show=False,
        close=False,
    )
    counts = _validate_native_shape_slide(Path(saved_pptx))
    return Path(saved_png), Path(saved_pptx), int(counts["native_shape"])


def main():
    preview_id = os.environ.get("FINAL_FIGURE1A_PREVIEW_ID", "").strip()
    asset_override = os.environ.get("FINAL_FIGURE1A_ASSET")
    if preview_id:
        if not preview_id.startswith("_tmp_") or not preview_id.replace("_", "a").isalnum():
            raise ValueError(
                "FINAL_FIGURE1A_PREVIEW_ID must use only letters, digits, underscores "
                "and start with _tmp_"
            )
        preview_root = FIG_ROOT / preview_id
        if preview_root.exists():
            raise FileExistsError(f"Refusing to overwrite Step00 preview: {preview_root}")
        out_stem = preview_root / DEFAULT_OUT.name
    else:
        out_stem = DEFAULT_OUT
    if asset_override and not preview_id:
        raise ValueError(
            "FINAL_FIGURE1A_ASSET is allowed only for an isolated preview"
        )
    asset_path = Path(asset_override).resolve() if asset_override else DEFAULT_ASSET
    fig, resolved_fonts = build_figure(asset_path)
    try:
        png_path, pptx_path, native_shape_count = save_figure(fig, out_stem)
    finally:
        plt.close(fig)
    print(f"Arial Regular: {resolved_fonts['regular']}")
    print(f"Arial Bold: {resolved_fonts['bold']}")
    print(f"Native PowerPoint shapes: {native_shape_count}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pptx_path}")


if __name__ == "__main__":
    main()
