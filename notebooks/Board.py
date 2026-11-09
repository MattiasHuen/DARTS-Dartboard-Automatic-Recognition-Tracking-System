import cv2
import numpy as np

from board_functions import calibrate_from_video, crop_around
N_ROWS = 720           # one row per 0.5 degrees around the board
N_COLUMNS = 800        # columns from the bullseye (0.0) to the inner ring (1.0)
N_SECTORS = 20
ROWS_PER_SECTOR = N_ROWS // N_SECTORS

# The sector numbers clockwise, starting from the top of a dartboard
SECTOR_NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
RING_GUESSES = dict(bull=0.04, outer_bull=0.10, treble_inner=0.56, treble_outer=0.62, double_inner=0.93)

def to_column(fraction):
    """Converts a position from 0.0 (bullseye) to 1.0 (inner ring) to a column number."""
    return int(round(fraction * (N_COLUMNS - 1)))


def unroll_board(image, board):
    """
    Samples the image on a grid of rows (directions) and columns (positions out to the inner ring).

    Returns the unrolled image, and map_x and map_y, which say where each grid position
    is in the original image. We need those to go back to the image in step 6.
    """
    (ex, ey), (width, height), angle = board["inner"]
    bx, by = board["bullseye"]
    a = np.radians(angle)

    # Table of all combinations of position s and direction t. Direction 0 is straight up,
    # and directions grow clockwise, because y points down in an image.
    s, t = np.meshgrid(np.linspace(0, 1, N_COLUMNS), np.radians(np.arange(N_ROWS) * 360 / N_ROWS - 9))

    # Convert the direction to the angle around the ellipse, measured from its own axis
    p = t - np.pi / 2 - a

    # The centre moves from the bullseye (s = 0) to the centre of the inner ellipse (s = 1),
    # because perspective shifts the centre of a circle more the larger the circle is
    cx = bx + s**2 * (ex - bx)
    cy = by + s**2 * (ey - by)

    # Point on an ellipse: half-axes * (cos p, sin p), rotated by the ellipse angle,
    # scaled by s and moved to the centre (cx, cy)
    map_x = (cx + s * (width / 2 * np.cos(p) * np.cos(a) - height / 2 * np.sin(p) * np.sin(a))).astype(np.float32)
    map_y = (cy + s * (width / 2 * np.cos(p) * np.sin(a) + height / 2 * np.sin(p) * np.cos(a))).astype(np.float32)

    # remap reads the image at all these positions
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR), map_x, map_y

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


def detect_fields(image, board):
    """Runs steps 2-6 on one image and returns everything that was found."""
    unrolled, map_x, map_y = unroll_board(image, board)                     # step 2
    gray = cv2.cvtColor(unrolled, cv2.COLOR_BGR2GRAY).astype(np.float32)

    sector_rows = equalise_sectors(find_sector_rows(boundary_profile(gray)))   # step 3
    rings, _ = find_rings(gray)                                             # step 4
    sectors, brightness = name_sectors(gray, sector_rows)                   # step 5
    fields = build_fields(sectors, rings, map_x, map_y)                     # step 6
    field_map = build_field_map(fields, image.shape)

    return dict(image=image, sector_rows=sector_rows, rings=rings, sectors=sectors,
                brightness=brightness, fields=fields, field_map=field_map)

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

    rings["double_outer"] = 1.0
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

def rows_between(start, stop):
    """Returns the rows from start to stop, also when they wrap around past row 720."""
    return np.arange(start, start + (stop - start) % N_ROWS + 1) % N_ROWS


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

def draw_fields(image, fields, field_map):
    """Returns the image with every field coloured and outlined."""
    # Each field gets a colour from its number, so that neighbouring fields differ
    rng = np.random.default_rng(0)
    colours = rng.integers(60, 255, size=(len(fields), 3), dtype=np.uint8)

    # Blend the colours with the image, only for pixels that belong to a field
    coloured = image.copy()
    inside = field_map >= 0
    coloured[inside] = colours[field_map[inside]]
    output = cv2.addWeighted(image, 0.55, coloured, 0.45, 0)

    for field in fields:
        cv2.polylines(output, [field["outline"]], True, (255, 255, 255), 1, cv2.LINE_AA)

    return output

def field_at(x, y, fields, field_map):
    """Returns the name and score at a pixel in the image."""
    number = field_map[int(y), int(x)]
    if number < 0:
        return "Miss", 0
    return fields[number]["name"], fields[number]["score"]

def detect_fields(image, board):
    """Runs steps 2-6 on one image and returns everything that was found."""
    unrolled, map_x, map_y = unroll_board(image, board)                     # step 2
    gray = cv2.cvtColor(unrolled, cv2.COLOR_BGR2GRAY).astype(np.float32)

    sector_rows = equalise_sectors(find_sector_rows(boundary_profile(gray)))   # step 3
    rings, _ = find_rings(gray)                                             # step 4
    sectors, brightness = name_sectors(gray, sector_rows)                   # step 5
    fields = build_fields(sectors, rings, map_x, map_y)                     # step 6
    field_map = build_field_map(fields, image.shape)

    return dict(image=image, sector_rows=sector_rows, rings=rings, sectors=sectors,
                brightness=brightness, fields=fields, field_map=field_map)




class DartBoard:
    """Detect and manage a dartboard."""

    N_ROWS = 720
    N_COLUMNS = 800
    N_SECTORS = 20
    ROWS_PER_SECTOR = N_ROWS // N_SECTORS

    SECTOR_NUMBERS = [
        20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
        3, 19, 7, 16, 8, 11, 14, 9, 12, 5
    ]

    def __init__(self):
        self.frame = None
        self.frame_gray = None
        self.board_homography = None

        # Unrolled board
        self.unrolled = None
        self.map_x = None
        self.map_y = None

        # Detected board structure
        self.sector_rows = None
        self.rings = None
        self.sectors = None
        self.brightness = None

        # Scoring fields
        self.fields = None
        self.field_map = None

    def update_frame(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Invalid Frame Provided")

        self.frame = frame
        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def calibrate(self, frames):
        self.board_homography = calibrate_from_video(frames)

    def unroll(self):
        if self.board_homography is None:
            raise ValueError("Board not calibrated.")

        if self.frame is None:
            raise ValueError("No frame available.")

        self.unrolled, self.map_x, self.map_y = unroll_board(
            self.frame,
            self.board_homography
        )

        return self.unrolled, self.map_x, self.map_y

    def detect(self):
        if self.frame is None:
            raise ValueError("No frame available.")

        if self.board_homography is None:
            raise ValueError("Board not calibrated.")

        results = detect_fields(
            self.frame,
            self.board_homography
        )

        self.sector_rows = results["sector_rows"]
        self.rings = results["rings"]
        self.sectors = results["sectors"]
        self.brightness = results["brightness"]
        self.fields = results["fields"]
        self.field_map = results["field_map"]

        return results

    def get_field(self, x, y):
        if self.field_map is None:
            raise ValueError("Fields have not been detected.")

        return field_at(x, y, self.fields, self.field_map)

    def draw_field_values(self, image):
        """Draw the score of each field at its center."""
        output = image.copy()

        for field in self.fields:
            outline = field["outline"]
            score = field["score"]

            # Find a point inside the field
            moments = cv2.moments(outline)

            if moments["m00"] == 0:
                continue

            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            text = str(score)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            # Center the text around the point
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            x = cx - text_width // 2
            y = cy + text_height // 2

            # Black outline for readability
            cv2.putText(
                output, text, (x, y),
                font, font_scale, (0, 0, 0), 3, cv2.LINE_AA
            )

            # White text
            cv2.putText(
                output, text, (x, y),
                font, font_scale, (255, 255, 255),
                thickness, cv2.LINE_AA
            )

        return output

    def get_inner_board(self, return_mask=False):
        if self.board_homography is None:
            raise ValueError("Board has not been calibrated.")

        if self.frame is None:
            raise ValueError("No frame available.")
        
        ellipse = self.board_homography["inner"]
        # Binary mask of the inner scoring area
        mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, -1)

        # Extract pixels inside the ellipse
        inner_board = cv2.bitwise_and(
            self.frame, self.frame, mask=mask
        )

        # Crop to the bounding rectangle
        x, y, w, h = cv2.boundingRect(
            cv2.findNonZero(mask)
        )

        inner_board = inner_board[y:y+h, x:x+w]
        cropped_mask = mask[y:y+h, x:x+w]

        if return_mask:
            return inner_board, cropped_mask

        return inner_board
        


    def draw(self):
        if self.field_map is None:
            raise ValueError("Fields have not been detected.")
        output = draw_fields(
            self.frame,
            self.fields,
            self.field_map
        )
        output = self.draw_field_values(output)
        return output

    

        


