import cv2
import numpy as np
import time
from src.config import *
from src.geometry import *

def field_outline(map_x, map_y, start, stop, position_inner, position_outer):
    """Returns the outline of one grid rectangle as a polygon in the real image."""
    rows = rows_between(start, stop)
    inner, outer = to_column(position_inner), to_column(position_outer)

    # Walk out along the outer edge, then back along the inner edge
    x = np.r_[map_x[rows, outer], map_x[rows[::-1], inner]]
    y = np.r_[map_y[rows, outer], map_y[rows[::-1], inner]]
    return np.round(np.column_stack([x, y])).astype(np.int32)


def build_fields(sectors, rings, map_x, map_y):
    """Returns one entry with name, score and outline for each of the 82 fields."""
    fields = []

    for number, start, stop in sectors:
        # (name, score, position of the inner edge, position of the outer edge)
        parts = [(f"S{number} inner", number, rings["outer_bull"], rings["treble_inner"]),
                 (f"T{number}", 3 * number, rings["treble_inner"], rings["treble_outer"]),
                 (f"S{number} outer", number, rings["treble_outer"], rings["double_inner"]),
                 (f"D{number}", 2 * number, rings["double_inner"], rings["double_outer"])]

        for name, score, inner, outer in parts:
            fields.append(dict(name=name, score=score,
                               outline=field_outline(map_x, map_y, start, stop, inner, outer)))

    # The bulls are full circles, so they have only one edge
    for name, score, ring in [("Outer bull", 25, "outer_bull"), ("Bull", 50, "bull")]:
        circle = np.column_stack([map_x[:, to_column(rings[ring])], map_y[:, to_column(rings[ring])]])
        fields.append(dict(name=name, score=score, outline=np.round(circle).astype(np.int32)))

    return fields


def build_field_map(fields, image_shape):
    """Returns an image where each pixel holds the number of its field (-1 = no field)."""
    field_map = np.full(image_shape[:2], -1, np.int16)

    # The bull comes after the outer bull in the list, so it is painted on top of it
    for number, field in enumerate(fields):
        cv2.fillPoly(field_map, [field["outline"]], number)

    return field_map


def field_at(x, y, fields, field_map):
    """Returns the name and score at a pixel in the image."""
    number = field_map[int(y), int(x)]
    if number < 0:
        return "Miss", 0
    return fields[number]["name"], fields[number]["score"]

