from src.config import *
import cv2
import numpy as np
import time

def to_column(fraction):
    """Converts a position from 0.0 (bullseye) to 1.0 (inner ring) to a column number."""
    return int(round(fraction * (N_COLUMNS - 1)))


def rows_between(start, stop):
    """Returns the rows from start to stop, also when they wrap around past row 720."""
    return np.arange(start, start + (stop - start) % N_ROWS + 1) % N_ROWS


def direction_angles():
    """Returns the angle in radians of each of the N_ANGLES directions around the board."""
    return 2 * np.pi * np.arange(N_ANGLES) / N_ANGLES

def crop_around(image, centre, half_size):
    """Cuts out a square of 2 * half_size pixels around a point."""
    x, y = int(centre[0]), int(centre[1])
    padded = cv2.copyMakeBorder(image, half_size, half_size, half_size, half_size, cv2.BORDER_CONSTANT)
    return padded[y:y + 2 * half_size, x:x + 2 * half_size]

def polar_image(gray, centre, max_radius):
    """
    Unwraps the image around the centre.

    In the result, row i is direction i * 360 / N_ANGLES degrees and column j is
    the distance j from the centre. Circles around the centre become vertical lines.
    """
    angles = direction_angles()
    radii = np.arange(max_radius, dtype=np.float32)[None, :]
    # [:, None] makes this a table with one row per direction and one column per distance.
    map_x = centre[0] + radii * np.cos(angles)[:, None]
    map_y = centre[1] + radii * np.sin(angles)[:, None]

    return cv2.remap(gray, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)

def polar_to_points(centre, radius_per_angle):
    """Converts one radius per direction back to (x, y) points in the image."""
    angles = direction_angles()
    x = centre[0] + radius_per_angle * np.cos(angles)
    y = centre[1] + radius_per_angle * np.sin(angles)
    return np.column_stack([x, y]).astype(np.float32)


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

def unroll_board(image, ellipse, bullseye, scales, radians: bool = False):
    """
    Samples the image along scaled copies of the outer ellipse.

    In the result, each row is one direction around the ellipse and each column is
    one scale, so the outer ellipse becomes a straight line at scale 1.0.
    Also returns map_x and map_y, which say where each result pixel came from in the image.
    """
    (ex, ey), (width, height), angle = ellipse

    a = np.radians(angle)

    if radians:
        s, t = np.meshgrid(scales, np.radians(np.arange(N_ROWS) * 360 / N_ROWS - 9))
        t = t - np.pi / 2 - a
    else:
        s, t = np.meshgrid(scales, direction_angles())

    cx = bullseye[0] + s**2 * (ex - bullseye[0])
    cy = bullseye[1] + s**2 * (ey - bullseye[1])

    map_x = (cx + s * (width / 2 * np.cos(t) * np.cos(a) - height / 2 * np.sin(t) * np.sin(a))).astype(np.float32)
    map_y = (cy + s * (width / 2 * np.cos(t) * np.sin(a) + height / 2 * np.sin(t) * np.cos(a))).astype(np.float32)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR), map_x, map_y

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

