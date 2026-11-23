import cv2
import numpy as np


def calculate_difference(
    reference,
    new,
    timestamp,
    mean_change_threshold: float = 4,
    pixel_threshold: int = 30
):
    """Calculate image differences and detect significant changes."""
    hsv_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    hsv_new = cv2.cvtColor(new, cv2.COLOR_BGR2HSV)

    # Blue color range (adjust for your dart)
    lower_blue = np.array([95, 80, 50])
    upper_blue = np.array([135, 255, 255])

    mask_ref = cv2.inRange(hsv_ref, lower_blue, upper_blue)
    mask_new = cv2.inRange(hsv_new, lower_blue, upper_blue)

    difference_blue = cv2.absdiff(mask_ref, mask_new)
    # gray_difference = cv2.cvtColor(
    #         difference, cv2.COLOR_BGR2GRAY
    #     )
    
    # # difference = cv2.absdiff(reference, new)
    # gray_difference = cv2.cvtColor(
    #     difference, cv2.COLOR_BGR2GRAY
    # )

    mean_difference = float(np.mean(difference_blue))
    max_pixel_difference = int(np.max(difference_blue))

    changed_percentage = float(
        np.mean(difference_blue > pixel_threshold) * 100
    )

    change_detected = (
        mean_difference > mean_change_threshold
    )

    return {
        "time": round(timestamp, 3),
        "change_detected": change_detected,
        "mean_difference": round(mean_difference, 3),
        "max_pixel_difference": max_pixel_difference,
        "changed_percentage": round(changed_percentage, 3),
        "persistent_change": np.nan,
        "dart_landed": 0,
        "dart_missed": 0
    }



# --- Blue-dart tip detection: HSV, colour change, morphology, skeleton ---
# The whole dart (barrel, shaft and flights) is painted blue. The idea:
#   1. Blue pixels (HSV) that have also CHANGED colour since the frame before the throw
#      are the new dart. Requiring a change lets the blue limits be loose, so flights
#      in shadow (dark blue) and in light (pale blue) are both found.
#   2. Clean the mask and keep the largest blob (the flights) plus the smaller pieces
#      close to it (the thin shaft often breaks where it crosses the wires).
#   3. Thin the dart to a one pixel wide line (skeleton).
#   4. The back is the thickest point of the dart (the middle of the flights).
#      The tip is the skeleton end furthest away from it.
# calculate_difference() above (used for throw detection) is not changed.

BLUE_HUE = (90, 135)     # blue is around 120 in OpenCV (hue 0-179)
MIN_SATURATION = 40      # loose: the colour-change test removes the board's own pixels
MIN_VALUE = 25           # dark blue flights in shadow are still kept
MIN_COLOUR_CHANGE = 30   # a pixel must change this much (0-255, any colour channel)


def ellipse_mask(shape, ellipse):
    """Returns a mask that is 255 inside the ellipse and 0 outside."""
    mask = np.zeros(shape[:2], np.uint8)
    (cx, cy), (width, height), angle = ellipse
    cv2.ellipse(mask, (int(cx), int(cy)),
                (int(width / 2), int(height / 2)), angle, 0, 360, 255, -1)
    return mask


def blue_mask(image, hue=BLUE_HUE,
              min_saturation=MIN_SATURATION, min_value=MIN_VALUE):
    """Returns a mask (0/255) of the blue pixels in a BGR image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    blue = ((h >= hue[0]) & (h <= hue[1]) &
            (s >= min_saturation) & (v >= min_value))
    return blue.astype(np.uint8) * 255


def colour_change(before, after, threshold=MIN_COLOUR_CHANGE):
    """Returns a mask (0/255) of the pixels where any colour channel changed more than 'threshold'."""
    difference = cv2.absdiff(before, after).max(axis=2)
    return (difference > threshold).astype(np.uint8) * 255


def clean_mask(mask, open_size=2, close_size=5):
    """Remove noisy pixels (opening) and connect small gaps (closing)."""
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              np.ones((open_size, open_size), np.uint8))
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                            np.ones((close_size, close_size), np.uint8))


def dart_blobs(mask, near=40, min_piece=5):
    """
    Returns a mask with the dart, and the area of its largest blob.

    The largest blob is the flights and the barrel. The shaft is thin and often breaks
    into pieces, so every other blob of at least 'min_piece' pixels that lies within
    'near' pixels of the largest blob is kept as part of the dart.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:                      # label 0 is the background, so there is no blob
        return np.zeros_like(mask), 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    dart = (labels == biggest).astype(np.uint8) * 255

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * near + 1, 2 * near + 1))
    close_by = cv2.dilate(dart, disk) > 0
    for label in range(1, count):
        piece = labels == label
        if label != biggest and stats[label, cv2.CC_STAT_AREA] >= min_piece and close_by[piece].any():
            dart[piece] = 255
    return dart, int(areas.max())


def skeletonize(mask):
    """Zhang-Suen thinning: reduce the dart blob to its one-pixel centreline."""
    skeleton = (mask > 0).astype(np.uint8)
    while True:
        removed_any = False
        for sub_step in (0, 1):
            p = np.pad(skeleton, 1)
            P2, P3, P4, P5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            P6, P7, P8, P9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            neighbours = [P2, P3, P4, P5, P6, P7, P8, P9]
            B = sum(n.astype(int) for n in neighbours)
            A = sum(((neighbours[i] == 0) &
                     (neighbours[(i + 1) % 8] == 1)).astype(int)
                    for i in range(8))
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
    """Find skeleton pixels with exactly one active 8-connected neighbour."""
    on = (skeleton > 0).astype(np.uint8)
    neighbourhood = cv2.filter2D(on, -1, np.ones((3, 3)),
                                  borderType=cv2.BORDER_CONSTANT)
    ys, xs = np.nonzero((on == 1) & (neighbourhood == 2))
    return np.column_stack([xs, ys])


def thickest_point(mask):
    """
    Returns the (x, y) point deepest inside the mask (largest distance to the background).
    For a dart this is the middle of the flights, the widest part.
    """
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    return np.array([x, y])


def extend_tip(tip, back, pixels):
    """Extend the blue end of the dart towards the unpainted steel point."""
    tip, back = np.asarray(tip, float), np.asarray(back, float)
    norm = np.linalg.norm(tip - back)
    if norm == 0:
        return tip
    direction = (tip - back) / norm
    return tip + pixels * direction


def find_dart(before, after, board_mask, min_pixels=40, tip_extension=4):
    """Detect a NEW blue dart in two full-size frames; return diagnostic masks and tip."""
    if before.shape != after.shape or board_mask.shape != after.shape[:2]:
        raise ValueError("before, after and board_mask must have the same image dimensions")

    steps = {}
    steps["blue_after"] = cv2.bitwise_and(blue_mask(after), board_mask)
    steps["changed"] = colour_change(before, after)
    steps["new_blue"] = cv2.bitwise_and(steps["blue_after"], steps["changed"])
    steps["cleaned"] = clean_mask(steps["new_blue"])
    steps["dart"], steps["area"] = dart_blobs(steps["cleaned"])
    steps["found"] = steps["area"] >= min_pixels
    if not steps["found"]:
        return steps

    # Thinning only needs the box around the dart (plus a 1 pixel border of background),
    # which is much faster than thinning the full frame and gives the same skeleton
    ys, xs = np.nonzero(steps["dart"])
    y1, y2 = max(ys.min() - 1, 0), ys.max() + 2
    x1, x2 = max(xs.min() - 1, 0), xs.max() + 2
    steps["skeleton"] = np.zeros_like(steps["dart"])
    steps["skeleton"][y1:y2, x1:x2] = skeletonize(steps["dart"][y1:y2, x1:x2])
    steps["ends"] = skeleton_ends(steps["skeleton"])
    if len(steps["ends"]) == 0:
        steps["found"] = False
        return steps

    # Back = middle of the flights, tip = the skeleton end furthest away from it
    steps["back"] = thickest_point(steps["dart"])
    distances = np.linalg.norm(steps["ends"] - steps["back"], axis=1)
    steps["blue_end"] = steps["ends"][np.argmax(distances)]
    steps["tip"] = extend_tip(steps["blue_end"], steps["back"], tip_extension)
    return steps


def draw_dart(image, steps, colour=(0, 255, 0)):
    """Draw the detected dart, skeleton and estimated tip."""
    output = image.copy()
    if not steps.get("found"):
        return output
    output[steps["dart"] > 0] = (255, 128, 0)
    output[steps["skeleton"] > 0] = (0, 0, 255)
    cv2.circle(output, tuple(int(v) for v in steps["back"]), 6, (0, 255, 255), 2)
    cv2.circle(output, tuple(int(v) for v in steps["tip"]), 6, colour, 2)
    return output
