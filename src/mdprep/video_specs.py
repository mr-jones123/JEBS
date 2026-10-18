"""Shared clip geometry, timing, and task-family constants."""

SOURCE = "generated_temporal_videos"
FPS = 2
FRAME_COUNT = 4
FRAME_SIZE = (256, 192)
SAMPLE_TIMES_S = tuple(index / FPS for index in range(FRAME_COUNT))
FAMILY_CYCLE = ("bounce", "color_sequence", "gate_order", "collision")
OOD_FAMILY = "zigzag_guides"
PALETTE = {
    "amber": "#E49A31",
    "blue": "#3978B8",
    "cyan": "#2B9EA1",
    "green": "#4C956C",
    "magenta": "#B04A82",
    "red": "#C94F46",
    "violet": "#7656A5",
    "yellow": "#E0B83D",
}
