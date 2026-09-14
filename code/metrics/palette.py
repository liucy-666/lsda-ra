"""Shared publication-style palette and axes styling.

Vivid scheme: baseline = light gray, ours = bright blue, competitor = vivid orange.
"""

OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#9E9E9E",
}

# Arm colors (vivid).
ARM_COLORS = {
    "SS": "#BDBDBD",        # baseline: light gray
    "LSDA": "#1E88E5",      # ours: bright blue
    "Binding": "#FB8C00",   # competitor: vivid orange
}

# Five-method palette (native / binding / uniform / knowledge / routed).
METHOD_COLORS = ["#BDBDBD", "#FB8C00", "#42A5F5", "#AB47BC", "#1E88E5"]

# Objective-metric series palette.
SERIES_COLORS = ["#1E88E5", "#FB8C00", "#43A047", "#8E24AA", "#00ACC1"]

EDGE = "#333333"


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.7, alpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#444444")
    ax.tick_params(colors="#333333")
