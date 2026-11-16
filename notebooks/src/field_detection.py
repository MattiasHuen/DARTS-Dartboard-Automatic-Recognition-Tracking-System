from src.config import *
from src.geometry import to_column, rows_between, unroll_board
from src.scoring import build_fields, build_field_map
import cv2
import numpy as np
import time

def boundary_profile(gray):
    """Returns how strongly each row looks like a sector boundary."""
    # Blur along the columns and slightly along the rows, then take the slope along the rows
    slope = np.abs(cv2.Sobel(cv2.blur(gray, (5, 3)), cv2.CV_32F, 0, 1, ksize=5))

    # Average over the two single areas, the parts of the board with a clean pattern
    inner_single = slope[:, to_column(0.15):to_column(0.55)].mean(axis=1)
    outer_single = slope[:, to_column(0.65):to_column(0.90)].mean(axis=1)
    return inner_single + outer_single


def find_sector_rows(profile, search=8):
    """Returns the 20 rows of the sector boundaries, sorted from low to high."""
    # Comb: for each starting row, add up the profile at 20 evenly spaced rows
    teeth = ROWS_PER_SECTOR * np.arange(N_SECTORS)
    comb_score = [profile[(start + teeth) % N_ROWS].sum() for start in range(ROWS_PER_SECTOR)]
    best_start = int(np.argmax(comb_score))

    # Move each tooth to the highest peak close to it.
    # % N_ROWS wraps around, because row 720 is the same direction as row 0.
    rows = []
    for tooth in best_start + teeth:
        window = np.arange(tooth - search, tooth + search + 1) % N_ROWS
        rows.append(int(window[np.argmax(profile[window])]))

    return np.sort(rows)


def equalise_sectors(rows):
    """Moves the boundaries so that every second sector has the same average width as the others."""
    # Width of each sector. rows[0] + N_ROWS closes the circle.
    widths = np.diff(np.r_[rows, rows[0] + N_ROWS])

    # Sectors alternate between dark and bright, so compare every second sector. Moving all
    # boundaries by `shift` changes the difference by 4 * shift (2 * shift for each type of sector).
    shift = (widths[0::2].mean() - widths[1::2].mean()) / 4

    # Even boundaries start the even sectors, so moving them forward and the
    # odd boundaries backward makes the even sectors narrower
    moved = rows.astype(np.float64)
    moved[0::2] += shift
    moved[1::2] -= shift

    return np.sort(np.round(moved).astype(int) % N_ROWS)

def find_rings(gray, search=0.03):
    """Returns the measured position of each ring and the profile they were found in."""
    # Top-hat along the columns: only thin bright lines are left
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((1, 15), np.uint8))

    # The median over all directions keeps what goes all the way round the board
    wire_profile = np.median(gray - opened, axis=0)

    rings = {}
    for name, guess in RING_GUESSES.items():
        window = np.arange(to_column(guess - search), to_column(guess + search) + 1)
        rings[name] = float(window[np.argmax(wire_profile[window])] / (N_COLUMNS - 1))

    # rings["double_outer"] = 1.0
    return rings, wire_profile

def name_sectors(gray, sector_rows):
    """
    Finds which sector is 20.

    Returns a list with one (number, first row, last row) for each sector, and the
    brightness of the sectors, in the order they are found in the unrolled image.
    """
    # Sector i goes from boundary i to boundary i + 1. % N_ROWS wraps around for the last one.
    starts = sector_rows
    stops = np.roll(sector_rows, -1)

    # Brightness of each sector, measured in the outer single area
    # (a few rows are skipped at each end to stay away from the boundaries)
    outer_single = gray[:, to_column(0.65):to_column(0.90)]
    brightness = []
    for start, stop in zip(starts, stops):
        rows = np.arange(start + 6, start + (stop - start) % N_ROWS - 6) % N_ROWS
        brightness.append(outer_single[rows].mean())
    brightness = np.array(brightness)

    # The middle row of each sector, as the distance in rows from straight up (row 18)
    middles = starts + ((stops - starts) % N_ROWS) / 2
    distance_to_top = np.abs((middles - 18 + N_ROWS / 2) % N_ROWS - N_ROWS / 2)

    # Sector 20 is the dark sector that is closest to the top
    dark = brightness < np.median(brightness)
    first = int(np.argmin(np.where(dark, distance_to_top, np.inf)))

    # Sectors go clockwise, which is the direction of increasing rows
    sectors = []
    for k, number in enumerate(SECTOR_NUMBERS):
        i = (first + k) % N_SECTORS
        sectors.append((number, int(starts[i]), int(stops[i])))

    return sectors, brightness



def detect_fields(image, board):
    """Runs steps 2-6 on one image and returns everything that was found."""
    unrolled, map_x, map_y = unroll_board(image, board["inner"], board["bullseye"], np.linspace(0, 1, N_COLUMNS), radians = True)                     # step 2
    gray = cv2.cvtColor(unrolled, cv2.COLOR_BGR2GRAY).astype(np.float32)

    sector_rows = equalise_sectors(find_sector_rows(boundary_profile(gray)))   # step 3
    rings, _ = find_rings(gray)                                             # step 4
    sectors, brightness = name_sectors(gray, sector_rows)                   # step 5
    fields = build_fields(sectors, rings, map_x, map_y)                     # step 6
    field_map = build_field_map(fields, image.shape)

    return dict(image=image, sector_rows=sector_rows, rings=rings, sectors=sectors,
                brightness=brightness, fields=fields, field_map=field_map)
