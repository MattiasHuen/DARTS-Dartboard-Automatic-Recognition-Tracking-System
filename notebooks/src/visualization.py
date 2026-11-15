import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(images, titles, size=6):
    """Shows a list of images next to each other, each with a title."""
    fig, axs = plt.subplots(1, len(images), figsize=(size * len(images), size * 0.6))
    axs = np.atleast_1d(axs)

    for ax, image, title in zip(axs, images, titles):
        if image.ndim == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def draw_board(image, board, thickness=1):
    """Returns a copy of the image with the outer ellipse in green and the inner ellipse in red."""
    output = image.copy()   # copy, so the original image is not drawn on
    cv2.ellipse(output, board["outer"], (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.ellipse(output, board["inner"], (0, 0, 255), thickness, cv2.LINE_AA)
    return output


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

