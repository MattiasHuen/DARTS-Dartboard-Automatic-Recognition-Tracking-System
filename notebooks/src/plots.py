import numpy as np
import matplotlib.pyplot as plt


def _get_ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
    return ax


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 16)
    ax.set_xlabel("Time (s)", fontsize = 16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=12)


def _plot_threshold(ax, data, column, threshold, name, above=True):
    if threshold is None:
        return

    x = data["time"].to_numpy()
    y = data[column].to_numpy(dtype=float)
    valid = np.isfinite(y)

    condition = y > threshold if above else y < threshold
    operator = ">" if above else "<"

    ax.axhline(
        threshold,
        color="darkorange",
        linestyle="--",
        linewidth=2,
        label=f"{name}: {operator} {threshold:g}"
    )

    ax.fill_between(
        x, y, threshold,
        where=valid & condition,
        color="green",
        alpha=0.15,
        interpolate=True
    )

    ax.fill_between(
        x, y, threshold,
        where=valid & ~condition,
        color="red",
        alpha=0.08,
        interpolate=True
    )


def plot_events(data, column, ax=None):
    ax = _get_ax(ax)

    if "dart_landed" in data.columns:
        landed = data["dart_landed"].eq(1)

        ax.scatter(
            data.loc[landed, "time"],
            data.loc[landed, column],
            color="green",
            edgecolors="black",
            s=80,
            zorder=5,
            label="Landed"
        )

    if "dart_missed" in data.columns:
        missed = data["dart_missed"].eq(1)

        ax.scatter(
            data.loc[missed, "time"],
            data.loc[missed, column],
            color="red",
            marker="x",
            s=90,
            zorder=5,
            label="Missed / bounced"
        )

    if "dart_thrown" in data.columns:
        detected = data["dart_thrown"].eq(1)

        ax.scatter(
            data.loc[detected, "time"],
            data.loc[detected, column],
            color="purple",
            marker="^",
            s=90,
            zorder=5,
            label="Throw detected"
        )
    return ax


def plot_mean_difference(data, ax=None):
    ax = _get_ax(ax)

    ax.plot(
        data["time"],
        data["mean_difference"],
        color="steelblue",
        label="Mean difference"
    )

    if "baseline" in data.columns:
        ax.plot(
            data["time"],
            data["baseline"],
            color="gray",
            linestyle="--",
            label="Recent baseline"
        )

    plot_events(data, "mean_difference", ax)
    _style(ax, "Mean Difference and Baseline", "Mean pixel difference")

    return ax


def plot_changed_percentage(data, pixel_threshold=30, ax=None):
    ax = _get_ax(ax)

    ax.plot(
        data["time"],
        data["changed_percentage"],
        color="steelblue",
        label="Changed pixels (%)"
    )

    plot_events(data, "changed_percentage", ax)

    _style(
        ax,
        f"Changed Pixel Percentage (PIXEL_CHANGE_THRESHOLD = {pixel_threshold})",
        "Changed pixels (%)"
    )

    return ax


def plot_stability(data, threshold=None, ax=None):
    ax = _get_ax(ax)

    ax.plot(
        data["time"],
        data["stability"],
        color="steelblue",
        label="Frame-to-frame difference"
    )

    _plot_threshold(
        ax, data, "stability",
        threshold,
        name="STABILITY_THRESHOLD",
        above=False
    )

    plot_events(data, "stability", ax)
    _style(ax, "Frame-to-Frame Stability", "Mean pixel difference")

    return ax


def plot_max_pixel_difference(data, pixel_threshold=None, ax=None):
    ax = _get_ax(ax)

    ax.plot(
        data["time"],
        data["max_pixel_difference"],
        color="steelblue",
        label="Maximum pixel difference"
    )

    if pixel_threshold is not None:
        ax.axhline(
            pixel_threshold,
            color="darkorange",
            linestyle="--",
            linewidth=2,
            label=f"PIXEL_CHANGE_THRESHOLD = {pixel_threshold:g}"
        )

    plot_events(data, "max_pixel_difference", ax)

    _style(
        ax,
        "Maximum Pixel Difference",
        "Pixel intensity difference (0–255)"
    )

    return ax


def plot_persistent_change(data, threshold=None, ax=None):
    ax = _get_ax(ax)

    if "persistent_change" not in data.columns:
        raise ValueError("Missing 'persistent_change' column")

    valid = data["persistent_change"].notna()

    ax.scatter(
        data.loc[valid, "time"],
        data.loc[valid, "persistent_change"],
        color="steelblue",
        s=45,
        label="Persistent change",
        zorder=3
    )

    _plot_threshold(
        ax, data, "persistent_change",
        threshold,
        name="MIN_PERSISTENT_CHANGE",
        above=True
    )

    plot_events(data, "persistent_change", ax)

    _style(
        ax,
        "Persistent Change: Landing Classification",
        "Mean pixel difference"
    )

    return ax


def plot_impact_change(data, threshold=None, ax=None):
    ax = _get_ax(ax)

    ax.plot(
        data["time"],
        data["impact_change"],
        color="steelblue",
        label="Impact change"
    )

    _plot_threshold(
        ax, data, "impact_change",
        threshold,
        name="MEAN_CHANGE_THRESHOLD",
        above=True
    )

    if "dart_thrown" in data.columns:
        thrown = data["dart_thrown"].eq(1)

        ax.scatter(
            data.loc[thrown, "time"],
            data.loc[thrown, "impact_change"],
            color="purple",
            marker="^",
            s=90,
            zorder=6,
            label="Throw detected"
        )

    _style(
        ax,
        "Impact Detection",
        "Mean difference above baseline"
    )

    return ax


def plot_detection_state(data, ax=None):
    ax = _get_ax(ax)

    state_map = {
        "WAITING": 0,
        "SETTLING": 1,
        "SETTLE": 1,
        "COOLDOWN": 2
    }

    states = data["state"].map(state_map)

    ax.step(
        data["time"],
        states,
        where="post",
        color="purple",
        linewidth=2,
        label="Detection state"
    )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["WAITING", "SETTLING", "COOLDOWN"])

    _style(ax, "Detection State Transitions", "State")

    return ax


def plot_stable_frames(data, required=10, ax=None):
    ax = _get_ax(ax)

    if "stable_frames" not in data.columns:
        raise ValueError("Missing 'stable_frames' column")

    ax.step(
        data["time"],
        data["stable_frames"],
        where="post",
        color="steelblue",
        label="Consecutive stable frames"
    )

    ax.axhline(
        required,
        color="darkorange",
        linestyle="--",
        linewidth=2,
        label=f"STABLE_FRAMES_REQUIRED: >= {required}"
    )

    ax.fill_between(
        data["time"],
        data["stable_frames"],
        required,
        where=data["stable_frames"] >= required,
        color="green",
        alpha=0.15
    )

    _style(ax, "Consecutive Stable Frames", "Frame count")

    return ax