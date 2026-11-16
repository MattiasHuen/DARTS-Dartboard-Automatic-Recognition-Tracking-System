from src.config import *
from src.geometry import *
import cv2
import numpy as np
import time


def median_frame(frame_list, amount: int = 1, video: str = True):
    """
    Calculate a median image from a video or list of frames,
    returns the median image and first frame
    """
    if amount < 1:
        raise ValueError("Amount must be at least 1")

    frames = []

    if video:
        cap = cv2.VideoCapture(str(frame_list))

        if not cap.isOpened():
            raise ValueError(f"Could not open video {frame_list}")
        try:
            for _ in range(amount):
                ok, frame = cap.read()

                if not ok:
                    break

                frames.append(frame)
        finally:
            cap.release()
    else:
        frames = list(frame_list[:amount])

    first_frame = frames[0].copy()
    median_image = np.median(np.stack(frames), axis=0).astype(np.uint8)
    return median_image, first_frame

def find_centre(
        image,
        clahe_lim = 2.5,
        clahe_grid = (9,9)
        ):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 1)
    clahe = cv2.createCLAHE(clipLimit=clahe_lim, tileGridSize=clahe_grid)
    gray_clahe = clahe.apply(gray)

    edges = cv2.Canny(gray_clahe, 100, 150)
    gx = cv2.Sobel(gray_clahe, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray_clahe, cv2.CV_32F, 0, 1)

    ys, xs = np.nonzero(edges)
    magnitude = np.hypot(gx[ys, xs], gy[ys, xs]) + 1e-6

    dx = -gy[ys, xs] / magnitude
    dy = gx[ys, xs] / magnitude

    votes = np.zeros(gray_clahe.shape, np.float32)
    line_length = gray_clahe.shape[0] // 8

    for t in range(-line_length, line_length + 1):
        px = np.round(xs + t * dx).astype(int)
        py = np.round(ys + t * dy).astype(int)
        inside = (px >= 0) & (px < gray_clahe.shape[1]) & (py >= 0) & (py < gray_clahe.shape[0])
        np.add.at(votes, (py[inside], px[inside]), 1)

    votes = cv2.GaussianBlur(votes, (0, 0), 3)

    y, x = np.unravel_index(np.argmax(votes), votes.shape)
    peak = (x, y)   

    return (peak[0],peak[1]), edges, votes



def thin_bright_lines(polar):
    """
    Keeps only bright lines that are thin along the radius (a top-hat filter).

    The result is bright on the white wire and close to 0 almost everywhere else.
    """
    opened = cv2.morphologyEx(polar, cv2.MORPH_OPEN, np.ones((1, 11), np.uint8))
    return polar - opened


def darkest_ring(polar):
    """
    Finds the rough distance of the outer black number ring from the centre.

    Returns that distance and the brightness profile it was found in.
    """
    without_digits = cv2.morphologyEx(polar, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))

    profile = np.convolve(np.median(without_digits, axis=0), np.ones(15) / 15, mode="same")

    return 30 + int(np.argmin(profile[30:-30])), profile



def find_wire_points(top_hat, centre, r_dark):
    """Finds one rough wire position in each direction and returns them as (x, y) points."""
    r_max = min(int(1.5 * r_dark), top_hat.shape[1] - 1)
    band = top_hat[:, r_dark:r_max]

    strong = band >= 0.5 * band.max(axis=1, keepdims=True)

    outermost = band.shape[1] - 1 - np.argmax(strong[:, ::-1], axis=1)

    return polar_to_points(centre, r_dark + outermost)




def snap_to_wire(ellipse, top_hat, centre, search=8):
    """Moves each point onto the wire by searching close to the current ellipse."""
    expected = ellipse_radius_per_angle(ellipse, centre)
    radii = np.zeros(N_ANGLES)

    for i in range(N_ANGLES):
        # Only look `search` pixels to each side of the ellipse in this direction
        start = max(int(expected[i]) - search, 0)
        segment = top_hat[i, start:int(expected[i]) + search + 1]

        # Light smoothing, so one noisy pixel cannot win
        segment = np.convolve(segment, [0.25, 0.5, 0.25], mode="same")

        # The strongest response is the wire. Add `start` to get the distance from the centre.
        radii[i] = start + np.argmax(segment)

    return polar_to_points(centre, radii)



def find_outer_ellipse(gray, centre):
    """
    Finds the ellipse on the white wire.

    Returns the ellipse, all final wire points and the points that agree with the ellipse.
    """
    polar = polar_image(gray, centre, gray.shape[0] // 2)
    top_hat = thin_bright_lines(polar)
    r_dark, _ = darkest_ring(polar)

    ellipse, _ = fit_ellipse_ransac(find_wire_points(top_hat, centre, r_dark))

    for _ in range(3):
        points = snap_to_wire(ellipse, top_hat, centre)
        ellipse, inliers = fit_ellipse_ransac(points)

    return ellipse, points, inliers




def rough_ring_edge(brightness, scales):
    """
    Finds the rough position of the double ring's outer edge.

    Returns the column of the edge and the brightness threshold that was used.
    """
    def column(scale):
        # Converts a scale (e.g. 0.85) to the matching column number
        return int(np.searchsorted(scales, scale))

    # Typical brightness of the black number ring and of the bright parts of the sectors
    black = np.median(brightness[column(0.85):column(0.95)])
    cream = np.median(brightness[column(0.45):column(0.65)])

    # Threshold just above black. Basing it on these two levels means that it
    # adapts to the exposure of the video.
    threshold = black + 0.15 * (cream - black)

    # Walk inward from the number ring until the brightness passes the threshold
    j = column(0.93)
    while brightness[j] < threshold:
        j -= 1

    return j, threshold


def find_ring_points(value, map_x, map_y, rough_edge, scales):
    """Finds the exact ring edge in each direction and returns them as (x, y) points."""
    # Search width: 0.03 in scale to each side of the rough edge, converted to columns
    half_width = int(0.03 / (scales[1] - scales[0]))
    start, stop = rough_edge - half_width, rough_edge + half_width

    # Blur lightly over 3 columns, then take the Sobel slope along the scale axis.
    # A large negative value means that the image goes from bright to dark going outwards.
    step = cv2.Sobel(cv2.blur(value, (3, 1)), cv2.CV_32F, 1, 0, ksize=5)

    # For each row, the column with the most negative slope is the edge
    columns = start + np.argmin(step[:, start:stop], axis=1)

    # map_x and map_y tell us where each of these positions is in the original image
    rows = np.arange(N_ANGLES)
    return np.column_stack([map_x[rows, columns], map_y[rows, columns]])

def find_inner_edge(image, outer_ellipse, bullseye):
    """
    Finds points on the outer edge of the double ring.

    Returns the points and a dictionary with the intermediate results for the plots.
    """
    # 700 scales from 0.3 to 1.0 of the outer ellipse
    scales = np.linspace(0.3, 1.0, 700)
    unrolled, map_x, map_y = unroll_board(image, outer_ellipse, bullseye, scales)

    # V channel of HSV: V = max(R, G, B), so red and green are both bright
    value = cv2.cvtColor(unrolled, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)

    # Median of each column ignores digits and logos, because they only appear in
    # some directions. The moving average of 5 columns smooths the curve.
    brightness = np.convolve(np.median(value, axis=0), np.ones(5) / 5, mode="same")

    # First the rough position of the edge, then the exact position in each direction
    rough_edge, threshold = rough_ring_edge(brightness, scales)
    points = find_ring_points(value, map_x, map_y, rough_edge, scales)

    details = dict(unrolled=unrolled, brightness=brightness, scales=scales,
                   rough_edge=scales[rough_edge], threshold=threshold)
    return points, details





def detect_board(image):
    """
    Runs steps 2-6 on one image.

    Returns a dictionary with the bullseye, both ellipses, the inner/outer size ratio
    and how many of the directions agree with each ellipse.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    bullseye, _, _ = find_centre(image)                                    # step 2
    outer_ellipse, _, outer_inliers = find_outer_ellipse(gray, bullseye)   # steps 3 and 4
    inner_points, _ = find_inner_edge(image, outer_ellipse, bullseye)      # step 5
    inner_ellipse, ratio, inner_inliers = fit_same_shape_ellipse(inner_points, outer_ellipse)   # step 6

    return dict(
        bullseye=bullseye, outer=outer_ellipse, inner=inner_ellipse, ratio=ratio,
        outer_agreement=len(outer_inliers) / N_ANGLES,
        inner_agreement=inner_inliers.sum() / N_ANGLES,
    )


def calibrate_from_video(frame):
    """Runs the full detector on one video: step 1 first, then steps 2-6 on the clean image."""
    clean = frame

    # Only the detection is timed, not reading the video
    start = time.perf_counter()
    board = detect_board(clean)
    board["seconds"] = time.perf_counter() - start

    board["image"] = clean   # kept so the result can be drawn on it later
    return board

