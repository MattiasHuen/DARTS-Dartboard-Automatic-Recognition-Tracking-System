"""
Functions for finding the dartboard in a video.

These are the functions from board_detector.ipynb, where each step is explained
with example images. Import them from here to use the board in other notebooks:

    from board_functions import calibrate_from_video
"""

import time

import cv2
import numpy as np

N_ANGLES = 720     # number of directions around the board (0.5 degree steps)


def crop_around(image, centre, half_size):
    """Cuts out a square of 2 * half_size pixels around a point."""
    x, y = int(centre[0]), int(centre[1])

    # Add a black border first, so the crop still works close to the image edge.
    # The border moves every pixel by half_size, so the point (x, y) is now at
    # (x + half_size, y + half_size), which is the middle of the slice below.
    padded = cv2.copyMakeBorder(image, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT)
    return padded[y:y + 2 * half_size, x:x + 2 * half_size]


def direction_angles():
    """Returns the angle in radians of each of the N_ANGLES directions around the board."""
    return 2 * np.pi * np.arange(N_ANGLES) / N_ANGLES


def median_frame(frames):
    """
    Combines a video into one clean image.

    Returns the median image and the list of frames that were used.
    """
    return np.median(np.stack(frames), axis=0).astype(np.uint8), frames


def find_centre(image):
    """
    Finds the bullseye: the point where the most edge directions cross.

    Returns the centre (x, y) in full resolution, the edge image and the vote image.
    """
    # Half resolution is enough for a starting point and makes the voting faster.
    # INTER_AREA averages the removed pixels, which avoids aliasing.
    small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Blur before differentiating, because derivatives amplify noise
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)

    # Canny gives thin edge lines: gradients above 150 are always edges, gradients
    # between 50 and 150 are only kept if they connect to a strong edge
    edges = cv2.Canny(gray, 50, 150)

    # Sobel gives the brightness slope in x and in y for every pixel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

    # Positions of all edge pixels, and the length of the gradient at each of them.
    # The tiny 1e-6 avoids dividing by zero below.
    ys, xs = np.nonzero(edges)
    magnitude = np.hypot(gx[ys, xs], gy[ys, xs]) + 1e-6

    # The gradient points across the edge. Rotating it by 90 degrees gives the
    # direction along the edge, and dividing by the length gives a vector of length 1.
    dx = -gy[ys, xs] / magnitude
    dy = gx[ys, xs] / magnitude

    # Start with an empty vote image. Each edge pixel will vote along a line that
    # reaches line_length pixels to each side of it.
    votes = np.zeros(gray.shape, np.float32)
    line_length = gray.shape[0] // 8

    # For each distance t, move all edge pixels t steps along their own edge direction
    for t in range(-line_length, line_length + 1):
        px = np.round(xs + t * dx).astype(int)
        py = np.round(ys + t * dy).astype(int)

        # Ignore positions outside the image
        inside = (px >= 0) & (px < gray.shape[1]) & (py >= 0) & (py < gray.shape[0])

        # Add one vote at each position (np.add.at also counts repeated positions)
        np.add.at(votes, (py[inside], px[inside]), 1)

    # Blur the votes so that lines that almost cross still add up, then take the highest peak
    votes = cv2.GaussianBlur(votes, (0, 0), 3)
    _, _, _, peak = cv2.minMaxLoc(votes)

    # We worked at half resolution, so multiply by 2 to get full resolution coordinates
    return (2 * peak[0], 2 * peak[1]), edges, votes


def polar_image(gray, centre, max_radius):
    """
    Unwraps the image around the centre.

    In the result, row i is direction i * 360 / N_ANGLES degrees and column j is
    the distance j from the centre. Circles around the centre become vertical lines.
    """
    angles = direction_angles()
    radii = np.arange(max_radius, dtype=np.float32)[None, :]

    # For each output pixel, calculate where to read in the input image:
    # x = cx + r * cos(angle), y = cy + r * sin(angle).
    # [:, None] makes this a table with one row per direction and one column per distance.
    map_x = centre[0] + radii * np.cos(angles)[:, None]
    map_y = centre[1] + radii * np.sin(angles)[:, None]

    # remap reads the image at those positions. The positions fall between pixels,
    # so INTER_LINEAR interpolates between the nearest pixels.
    return cv2.remap(gray, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)


def thin_bright_lines(polar):
    """
    Keeps only bright lines that are thin along the radius (a top-hat filter).

    The result is bright on the white wire and close to 0 almost everywhere else.
    """
    # Structuring element: 1 row high and 11 columns wide, so 11 px along the radius.
    # Opening removes bright details that are thinner than this.
    opened = cv2.morphologyEx(polar, cv2.MORPH_OPEN, np.ones((1, 11), np.uint8))

    # What the opening removed is exactly the thin bright lines
    return polar - opened


def darkest_ring(polar):
    """
    Finds the rough distance of the outer black number ring from the centre.

    Returns that distance and the brightness profile it was found in.
    """
    # An opening with a square element removes the thin digits and wires
    without_digits = cv2.morphologyEx(polar, cv2.MORPH_OPEN, np.ones((13, 13), np.uint8))

    # Median brightness for each distance (axis=0 combines all directions).
    # The moving average of 15 px smooths out small bumps.
    profile = np.convolve(np.median(without_digits, axis=0), np.ones(15) / 15, mode="same")

    # Darkest distance. The first and last 30 px are skipped, because the moving
    # average is unreliable at the ends and the bullseye is not the ring we want.
    return 30 + int(np.argmin(profile[30:-30])), profile


def polar_to_points(centre, radius_per_angle):
    """Converts one radius per direction back to (x, y) points in the image."""
    angles = direction_angles()
    x = centre[0] + radius_per_angle * np.cos(angles)
    y = centre[1] + radius_per_angle * np.sin(angles)
    return np.column_stack([x, y]).astype(np.float32)


def find_wire_points(top_hat, centre, r_dark):
    """Finds one rough wire position in each direction and returns them as (x, y) points."""
    # The wire lies outside the middle of the black ring, so only search this band
    r_max = min(int(1.5 * r_dark), top_hat.shape[1] - 1)
    band = top_hat[:, r_dark:r_max]

    # Mark positions that are at least half as strong as the strongest one in the same row
    strong = band >= 0.5 * band.max(axis=1, keepdims=True)

    # We want the last marked position, because the digits sit further in than the wire.
    # Flipping each row with [:, ::-1] turns the last one into the first one, and argmax
    # finds the first True. The subtraction converts it back to the original position.
    outermost = band.shape[1] - 1 - np.argmax(strong[:, ::-1], axis=1)

    # Positions in the band start at r_dark, so add it to get the distance from the centre
    return polar_to_points(centre, r_dark + outermost)


def ellipse_error_px(points, ellipse):
    """Returns the rough distance in pixels from each point to the ellipse."""
    (cx, cy), (width, height), angle = ellipse
    t = np.radians(angle)

    # Position of each point relative to the ellipse centre
    dx, dy = points[:, 0] - cx, points[:, 1] - cy

    # Rotate into the ellipse's own axes and divide by the half-axes.
    # A point exactly on the ellipse now has sqrt(u^2 + v^2) = 1.
    u = (dx * np.cos(t) + dy * np.sin(t)) / (width / 2)
    v = (-dx * np.sin(t) + dy * np.cos(t)) / (height / 2)

    # The difference from 1 is the relative error. Multiplying by the
    # average radius of the ellipse converts it to pixels.
    return np.abs(np.sqrt(u**2 + v**2) - 1) * (width + height) / 4


def fit_ellipse_ransac(points, tolerance=2.0, tries=400):
    """
    Fits an ellipse that ignores wrong points (RANSAC).

    Returns the ellipse and the points that agree with it (the inliers).
    """
    # A fixed seed gives the same "random" samples, so every run gives the same result
    rng = np.random.default_rng(0)
    best_ellipse, best_count = None, 0

    for _ in range(tries):
        # 1. Fit an ellipse through 5 random points, the minimum that defines an ellipse
        sample = points[rng.choice(len(points), 5, replace=False)]
        ellipse = cv2.fitEllipse(sample)
        axes = ellipse[1]

        # Skip impossible results (infinite, zero sized or far too large ellipses)
        if not np.all(np.isfinite(axes)) or min(axes) <= 0 or max(axes) > 2000:
            continue

        # 2. Count how many of all the points lie close to this ellipse
        count = np.sum(ellipse_error_px(points, ellipse) < tolerance)

        # 3. Remember the ellipse that most points agree with so far
        if count > best_count:
            best_ellipse, best_count = ellipse, count

    # 4. Final fit through all points that agree with the best ellipse
    inliers = points[ellipse_error_px(points, best_ellipse) < tolerance]
    return cv2.fitEllipse(inliers), inliers


def ellipse_radius_per_angle(ellipse, centre):
    """Returns the distance from `centre` to the ellipse in each of the N_ANGLES directions."""
    (cx, cy), (width, height), angle = ellipse

    # Walk around the ellipse and calculate 4000 points on it
    t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    a = np.radians(angle)
    x = cx + width / 2 * np.cos(t) * np.cos(a) - height / 2 * np.sin(t) * np.sin(a)
    y = cy + width / 2 * np.cos(t) * np.sin(a) + height / 2 * np.sin(t) * np.cos(a)

    # Direction and distance of each of those points as seen from `centre`.
    # % (2 * pi) puts the angles between 0 and 2 * pi, like direction_angles().
    point_angles = np.arctan2(y - centre[1], x - centre[0]) % (2 * np.pi)
    point_radii = np.hypot(x - centre[0], y - centre[1])
    order = np.argsort(point_angles)

    # Look up the radius in exactly our N_ANGLES directions. period=2 * pi tells
    # np.interp that the angles wrap around, so 360 degrees is the same as 0 degrees.
    return np.interp(direction_angles(), point_angles[order], point_radii[order], period=2 * np.pi)


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
    # Prepare the polar image, the thin bright lines and the rough board size
    polar = polar_image(gray, centre, gray.shape[0] // 2)
    top_hat = thin_bright_lines(polar)
    r_dark, _ = darkest_ring(polar)

    # First estimate from the rough wire points
    ellipse, _ = fit_ellipse_ransac(find_wire_points(top_hat, centre, r_dark))

    # Each round finds better points close to the ellipse and fits again
    for _ in range(3):
        points = snap_to_wire(ellipse, top_hat, centre)
        ellipse, inliers = fit_ellipse_ransac(points)

    return ellipse, points, inliers


def unroll_board(image, outer_ellipse, bullseye, scales):
    """
    Samples the image along scaled copies of the outer ellipse.

    In the result, each row is one direction around the ellipse and each column is
    one scale, so the outer ellipse becomes a straight line at scale 1.0.
    Also returns map_x and map_y, which say where each result pixel came from in the image.
    """
    (ex, ey), (width, height), angle = outer_ellipse
    a = np.radians(angle)

    # Table of all combinations of scale s and position t around the ellipse
    s, t = np.meshgrid(scales, direction_angles())

    # The ellipse centre moves from the bullseye (s = 0) to the outer ellipse centre (s = 1).
    # Under perspective the shift grows quickly with the size of the circle, hence s**2.
    cx = bullseye[0] + s**2 * (ex - bullseye[0])
    cy = bullseye[1] + s**2 * (ey - bullseye[1])

    # Point on an ellipse: half-axes * (cos t, sin t), rotated by the ellipse angle,
    # scaled by s and moved to the centre (cx, cy)
    map_x = (cx + s * (width / 2 * np.cos(t) * np.cos(a) - height / 2 * np.sin(t) * np.sin(a))).astype(np.float32)
    map_y = (cy + s * (width / 2 * np.cos(t) * np.sin(a) + height / 2 * np.sin(t) * np.cos(a))).astype(np.float32)

    # remap reads the image at all these positions
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR), map_x, map_y


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


def to_unit_circle(points, ellipse):
    """Moves, rotates and stretches the points so that `ellipse` becomes the unit circle."""
    (ex, ey), (width, height), angle = ellipse
    a = np.radians(angle)

    # Move: the ellipse centre becomes (0, 0)
    dx, dy = points[:, 0] - ex, points[:, 1] - ey

    # Rotate into the ellipse's own axes, and stretch by dividing by the half-axes
    u = (dx * np.cos(a) + dy * np.sin(a)) / (width / 2)
    v = (-dx * np.sin(a) + dy * np.cos(a)) / (height / 2)
    return u, v


def fit_circle(u, v):
    """
    Fits a circle to the points and returns its centre (a, b) and radius.

    The circle equation (u - a)^2 + (v - b)^2 = R^2 can be rewritten as
    u^2 + v^2 = 2*a*u + 2*b*v + c, where c = R^2 - a^2 - b^2.
    This is linear in a, b and c, so least squares solves it directly.
    """
    A = np.column_stack([2 * u, 2 * v, np.ones(len(u))])
    a, b, c = np.linalg.lstsq(A, u**2 + v**2, rcond=None)[0]

    # Get the radius back from c
    return a, b, np.sqrt(c + a**2 + b**2)


def fit_same_shape_ellipse(points, outer_ellipse, tolerance_px=1.5, tries=500):
    """
    Fits the inner ellipse with the same shape and tilt as the outer ellipse.

    Returns the inner ellipse, its size relative to the outer ellipse (the radius in
    unit circle coordinates) and a True/False array of which points agree with it.
    """
    (ex, ey), (width, height), angle = outer_ellipse

    # After this transform the outer ellipse is the unit circle, and the double ring
    # is a circle as well, so we only have to find a centre and a radius
    u, v = to_unit_circle(points, outer_ellipse)

    # The radius of the outer ellipse is now 1, so convert the tolerance from pixels
    tolerance = tolerance_px / ((width + height) / 4)

    rng = np.random.default_rng(0)
    best_circle, best_count = None, 0

    # RANSAC as in step 4, but with 3 points, since 3 points define a circle
    for _ in range(tries):
        i = rng.choice(len(u), 3, replace=False)
        a, b, radius = fit_circle(u[i], v[i])

        # The double ring must be inside the outer ellipse (and the fit must be valid)
        if not np.isfinite(radius) or radius >= 1:
            continue

        # Count the points whose distance to the circle is within the tolerance
        count = np.sum(np.abs(np.hypot(u - a, v - b) - radius) < tolerance)
        if count > best_count:
            best_circle, best_count = (a, b, radius), count

    # Final fit through all points that agree with the best circle
    a, b, radius = best_circle
    inliers = np.abs(np.hypot(u - a, v - b) - radius) < tolerance
    a, b, radius = fit_circle(u[inliers], v[inliers])

    # Back to image coordinates: undo the stretch and rotation for the circle centre
    t = np.radians(angle)
    cx = ex + a * width / 2 * np.cos(t) - b * height / 2 * np.sin(t)
    cy = ey + a * width / 2 * np.sin(t) + b * height / 2 * np.cos(t)

    # Same angle and same shape as the outer ellipse, scaled by `radius`
    return ((cx, cy), (radius * width, radius * height), angle), radius, inliers


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


def draw_board(image, board, thickness=1):
    """Returns a copy of the image with the outer ellipse in green and the inner ellipse in red."""
    output = image.copy()   # copy, so the original image is not drawn on
    cv2.ellipse(output, board["outer"], (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.ellipse(output, board["inner"], (0, 0, 255), thickness, cv2.LINE_AA)
    return output


def calibrate_from_video(frames):
    """Runs the full detector on one video: step 1 first, then steps 2-6 on the clean image."""
    clean, frames = median_frame(frames)

    # Only the detection is timed, not reading the video
    start = time.perf_counter()
    board = detect_board(clean)
    board["seconds"] = time.perf_counter() - start

    board["image"] = clean   # kept so the result can be drawn on it later
    return board
