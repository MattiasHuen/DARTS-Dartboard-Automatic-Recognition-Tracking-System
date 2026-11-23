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



# --- Blue-dart tip detection: HSV, morphology, skeleton and endpoints ---
# This section is the user's existing algorithm, kept separate from
# calculate_difference() above (which is used for throw detection).

BLUE_HUE = (95, 135)
MIN_SATURATION = 80
MIN_VALUE = 50


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


def new_blue(blue_now, blue_before, grow=7):
    """Keep blue pixels not already present before the throw."""
    grown_before = cv2.dilate(blue_before, np.ones((grow, grow), np.uint8))
    return cv2.bitwise_and(blue_now, cv2.bitwise_not(grown_before))


def clean_mask(mask, open_size=2, close_size=5):
    """Remove noisy pixels and connect small gaps."""
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                              np.ones((open_size, open_size), np.uint8))
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE,
                            np.ones((close_size, close_size), np.uint8))


def largest_blob(mask):
    """Return a mask for the largest connected blue component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    return (labels == biggest).astype(np.uint8) * 255, int(areas.max())


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


def two_furthest_ends(ends):
    """Take the two skeleton endpoints furthest apart."""
    if len(ends) < 2:
        return None
    distances = np.linalg.norm(ends[:, None, :] - ends[None, :, :], axis=2)
    a, b = np.unravel_index(np.argmax(distances), distances.shape)
    return ends[a], ends[b]


def change_mask(before, after, threshold=30):
    """Find non-blue regions (such as flights) that changed between frames."""
    difference = cv2.absdiff(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY),
                             cv2.cvtColor(after, cv2.COLOR_BGR2GRAY))
    changed = (difference > threshold).astype(np.uint8) * 255
    return cv2.morphologyEx(changed, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))


def pixels_near(mask, point, radius):
    """Count mask pixels inside a circle near a candidate endpoint."""
    circle = np.zeros_like(mask)
    cv2.circle(circle, (int(point[0]), int(point[1])), radius, 255, -1)
    return int(np.count_nonzero((mask > 0) & (circle > 0)))


def choose_tip(end_a, end_b, flights, radius=25):
    """Treat the end nearest the white flights as the back of the dart."""
    near_a = pixels_near(flights, end_a, radius)
    near_b = pixels_near(flights, end_b, radius)
    tip, back = (end_a, end_b) if near_a < near_b else (end_b, end_a)
    return tip, back, (near_a, near_b)


def extend_tip(tip, back, pixels):
    """Extend the chosen blue endpoint towards the unpainted steel tip."""
    tip, back = np.asarray(tip, float), np.asarray(back, float)
    norm = np.linalg.norm(tip - back)
    if norm == 0:
        return tip
    direction = (tip - back) / norm
    return tip + pixels * direction


def find_dart(before, after, board_mask, min_pixels=5, tip_extension=1):
    """Detect a NEW blue dart in two full-size frames; return diagnostic masks and tip."""
    if before.shape != after.shape or board_mask.shape != after.shape[:2]:
        raise ValueError("before, after and board_mask must have the same image dimensions")

    steps = {}
    steps["blue_before"] = cv2.bitwise_and(blue_mask(before), board_mask)
    steps["blue_after"] = cv2.bitwise_and(blue_mask(after), board_mask)
    steps["new_blue"] = new_blue(steps["blue_after"], steps["blue_before"])
    steps["cleaned"] = clean_mask(steps["new_blue"])
    steps["dart"], steps["area"] = largest_blob(steps["cleaned"])
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
    pair = two_furthest_ends(steps["ends"])
    if pair is None:
        steps["found"] = False
        return steps

    flights = change_mask(before, after)
    flights[cv2.dilate(steps["dart"], np.ones((5, 5), np.uint8)) > 0] = 0
    steps["flights"] = cv2.bitwise_and(flights, board_mask)

    steps["blue_end"], steps["back"], steps["flight_counts"] = choose_tip(
        pair[0], pair[1], steps["flights"]
    )
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
