import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_cameras(frame_path):
    """Plot camera positions and look-at lines"""
    # Load frame data
    with open(frame_path) as f:
        data = json.load(f)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot true cameras (blue)
    for pos, look_at in zip(
        data["true"]["camera"]["positions"], data["true"]["camera"]["look_at"]
    ):
        # Plot camera position
        ax.scatter(pos[0], pos[2], pos[1], color="blue", s=100, label="True Camera")
        # Plot line from camera to look-at point
        ax.plot(
            [pos[0], look_at[0]],
            [pos[2], look_at[2]],
            [pos[1], look_at[1]],
            "b--",
            alpha=0.5,
        )

    # Plot predicted cameras (red)
    for pos, look_at in zip(
        data["pred"]["camera"]["positions"], data["pred"]["camera"]["look_at"]
    ):
        # Plot camera position
        ax.scatter(pos[0], pos[2], pos[1], color="red", s=100, label="Pred Camera")
        # Plot line from camera to look-at point
        ax.plot(
            [pos[0], look_at[0]],
            [pos[2], look_at[2]],
            [pos[1], look_at[1]],
            "r--",
            alpha=0.5,
        )

    # Plot true objects (green)
    for pos in data["true"]["objects"]["positions"]:
        ax.scatter(pos[0], pos[2], pos[1], color="green", s=50, label="True Object")

    # Plot predicted objects (orange)
    for pos in data["pred"]["objects"]["positions"]:
        ax.scatter(pos[0], pos[2], pos[1], color="orange", s=50, label="Pred Object")

    # Plot origin
    ax.scatter(0, 0, 0, color="black", s=100, label="Origin")

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title("Camera and Object Positions")

    # Handle legend duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.show()


if __name__ == "__main__":
    # Plot first frame of latest iteration
    plot_cameras("viz_server/data/run_1744505997/iter_0000/frame_0000.json")
