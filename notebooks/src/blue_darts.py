"""
Functions for finding a blue dart and its tip.

The idea:
    1. Find all blue pixels on the board (HSV colour threshold).
    2. Keep only the blue pixels that were not there before the throw (the new dart).
    3. Clean the mask and keep the largest blob (the dart).
    4. Thin the blob down to a one pixel wide line (skeleton).
    5. The two ends of the skeleton are the tip and the back of the dart.
       The back is the end next to the (white, non-blue) flights, so the tip is the other end.
"""
import cv2
import numpy as np

# Default colour limits for "blue" in OpenCV HSV (hue goes from 0 to 179 in OpenCV)
BLUE_HUE = (95, 130)    # blue is around 120 in OpenCV, red ~0, green ~60
MIN_SATURATION = 80     # the board's black and white have low saturation
MIN_VALUE = 60          # very dark pixels have an unreliable hue, so they are ignored


def ellipse_mask(shape, ellipse):
    """Returns a mask that is 255 inside the ellipse and 0 outside."""
    mask = np.zeros(shape[:2], np.uint8)
    (cx, cy), (width, height), angle = ellipse
    cv2.ellipse(mask, (int(cx), int(cy)), (int(width / 2), int(height / 2)), angle, 0, 360, 255, -1)
    return mask


def blue_mask(image, hue=BLUE_HUE, min_saturation=MIN_SATURATION, min_value=MIN_VALUE):
    """Returns a mask (0/255) of the blue pixels in a BGR image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # A pixel is blue if its hue is in the blue range AND it is coloured (saturation)
    # AND it is bright enough for the hue to be trusted (value)
    blue = (h >= hue[0]) & (h <= hue[1]) & (s >= min_saturation) & (v >= min_value)
    return blue.astype(np.uint8) * 255


def new_blue(blue_now, blue_before, grow=7):
    """
    Returns the blue pixels that are new since the reference frame.

    The old blue mask is grown a little first (dilation), so that darts that were
    already on the board are not counted again if the board moves by a pixel or two.
    """
    grown_before = cv2.dilate(blue_before, np.ones((grow, grow), np.uint8))
    return cv2.bitwise_and(blue_now, cv2.bitwise_not(grown_before))


def clean_mask(mask, open_size=2, close_size=5):
    """Removes single noisy pixels (opening) and fills small gaps in the dart (closing)."""
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))


def largest_blob(mask):
    """Returns a mask with only the largest connected blob, and its area in pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:                      # label 0 is the background, so there is no blob
        return np.zeros_like(mask), 0

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip the background
    biggest = 1 + int(np.argmax(areas))
    return (labels == biggest).astype(np.uint8) * 255, int(areas.max())


def skeletonize(mask):
    """
    Thins a blob down to a one pixel wide line (Zhang-Suen thinning).

    Each pass removes boundary pixels from the blob, but only if removing them
    does not break the blob into pieces and is not the end of a line.
    This is repeated until nothing changes. What is left is the centre line.
    """
    skeleton = (mask > 0).astype(np.uint8)

    while True:
        removed_any = False

        for sub_step in (0, 1):
            # The 8 neighbours of every pixel, named clockwise from the top (P2 ... P9)
            p = np.pad(skeleton, 1)
            P2, P3, P4, P5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            P6, P7, P8, P9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            neighbours = [P2, P3, P4, P5, P6, P7, P8, P9]

            # B: number of neighbours that are part of the blob
            B = sum(n.astype(int) for n in neighbours)
            # A: number of 0 -> 1 changes when walking around the pixel.
            #    A == 1 means the pixel is on a simple edge (removing it does not split the blob)
            A = sum(((neighbours[i] == 0) & (neighbours[(i + 1) % 8] == 1)).astype(int) for i in range(8))

            # The two sub-steps remove pixels from opposite sides (south-east, then north-west)
            if sub_step == 0:
                side = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                side = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)

            remove = (skeleton == 1) & (B >= 2) & (B <= 6) & (A == 1) & side
            if remove.any():
                skeleton[remove] = 0
                removed_any = True

        if not removed_any:
            return skeleton * 255


def skeleton_ends(skeleton):
    """Returns the (x, y) end points of a skeleton: pixels with exactly one neighbour."""
    on = (skeleton > 0).astype(np.uint8)
    # The sum of the 3x3 neighbourhood includes the pixel itself, so an end point gives 2
    neighbourhood = cv2.filter2D(on, -1, np.ones((3, 3)), borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.nonzero((on == 1) & (neighbourhood == 2))
    return np.column_stack([xs, ys])


def two_furthest_ends(ends):
    """Returns the two end points that are furthest apart (the two ends of the dart)."""
    if len(ends) < 2:
        return None
    distances = np.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=2)
    a, b = np.unravel_index(np.argmax(distances), distances.shape)
    return ends[a], ends[b]


def change_mask(before, after, threshold=30):
    """Returns a mask of the pixels that changed between two frames (grayscale difference)."""
    difference = cv2.absdiff(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(after, cv2.COLOR_BGR2GRAY))
    changed = (difference > threshold).astype(np.uint8) * 255
    # Thin lines from the board moving slightly are removed; the flights are a larger blob
    return cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def pixels_near(mask, point, radius):
    """Counts the mask pixels inside a circle around a point."""
    circle = np.zeros_like(mask)
    cv2.circle(circle, (int(point[0]), int(point[1])), radius, 255, -1)
    return int(np.count_nonzero((mask > 0) & (circle > 0)))


def choose_tip(end_a, end_b, flights, radius=25):
    """
    Decides which end of the skeleton is the tip.
    The flights are not blue but they do change the image, so the end with the most
    changed (non-blue) pixels around it is the back of the dart. The tip is the other end.
    """
    near_a = pixels_near(flights, end_a, radius)
    near_b = pixels_near(flights, end_b, radius)
    tip, back = (end_a, end_b) if near_a < near_b else (end_b, end_a)
    return tip, back, (near_a, near_b)


def extend_tip(tip, back, pixels):
    """
    Moves the tip a few pixels further along the dart (from the back towards the tip).
    The steel point is not painted and the end of the barrel is often in shadow,
    so the blue part stops a little before the point where the dart enters the board.
    """
    tip, back = np.asarray(tip, float), np.asarray(back, float)
    direction = (tip - back) / np.linalg.norm(tip - back)
    return tip + pixels * direction


def find_dart(before, after, board_mask, min_pixels=40, tip_extension=5):
    """
    Finds the new dart between two frames and returns every step of the way.
    'before' is the frame before the throw, 'after' the frame once the dart has settled.
    """
    steps = {}
    steps["blue_before"] = cv2.bitwise_and(blue_mask(before), board_mask)
    steps["blue_after"] = cv2.bitwise_and(blue_mask(after), board_mask)
    steps["new_blue"] = new_blue(steps["blue_after"], steps["blue_before"])
    steps["cleaned"] = clean_mask(steps["new_blue"])
    steps["dart"], steps["area"] = largest_blob(steps["cleaned"])
    steps["found"] = steps["area"] >= min_pixels
    if not steps["found"]:
        return steps

    steps["skeleton"] = skeletonize(steps["dart"])
    steps["ends"] = skeleton_ends(steps["skeleton"])
    pair = two_furthest_ends(steps["ends"])
    if pair is None:
        steps["found"] = False
        return steps

    # Changed pixels that are not part of the blue dart: mostly the flights
    flights = change_mask(before, after)
    flights[cv2.dilate(steps["dart"], np.ones((5, 5), np.uint8)) > 0] = 0
    steps["flights"] = cv2.bitwise_and(flights, board_mask)

    steps["blue_end"], steps["back"], steps["flight_counts"] = choose_tip(pair[0], pair[1], steps["flights"])
    steps["tip"] = extend_tip(steps["blue_end"], steps["back"], tip_extension)
    return steps


def draw_dart(image, steps, colour=(0, 255, 0)):
    """Draws the found dart on a copy of the image: dart in blue, skeleton in red, tip as a circle."""
    output = image.copy()
    if not steps.get("found"):
        return output
    output[steps["dart"] > 0] = (255, 128, 0)
    output[steps["skeleton"] > 0] = (0, 0, 255)
    cv2.circle(output, tuple(int(v) for v in steps["back"]), 6, (0, 255, 255), 2)
    cv2.circle(output, tuple(int(v) for v in steps["tip"]), 6, colour, 2)
    return output
