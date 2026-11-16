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

    # Difference between reference and current frame
    difference = cv2.absdiff(reference, new)

    gray_difference = cv2.cvtColor(
        difference, cv2.COLOR_BGR2GRAY
    )

    # Difference statistics
    mean_difference = float(np.mean(gray_difference))
    max_pixel_difference = int(np.max(gray_difference))

    changed_percentage = float(
        np.mean(gray_difference > pixel_threshold) * 100
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