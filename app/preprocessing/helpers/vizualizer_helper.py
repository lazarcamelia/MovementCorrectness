import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

matplotlib.use('Qt5Agg')

slider = None

# Define the connections between adjacent points (lines) using indexes
lines = [
    (0, 1), (1, 2), (2, 3),  # SpineBase -> SpineMid -> Neck -> Head
    (2, 20),  # Neck -> SpineShoulder
    (20, 4), (20, 8),  # SpineShoulder -> ShoulderLeft -> ShoulderRight
    (4, 5), (5, 6), (6, 7),  # ShoulderLeft -> ElbowLeft -> WristLeft -> HandLeft
    (7, 21), (7, 22),  # HandLeft -> HandTipLeft -> ThumbLeft
    (8, 9), (9, 10), (10, 11),  # ShoulderRight -> ElbowRight -> WristRight -> HandRight
    (11, 23), (11, 24),  # HandRight -> HandTipRight -> ThumbRight
    (0, 12), (12, 13), (13, 14), (14, 15),  # SpineBase -> HipLeft -> KneeLeft -> AnkleLeft -> FootLeft
    (0, 16), (16, 17), (17, 18), (18, 19)  # SpineBase -> HipRight -> KneeRight -> AnkleRight -> FootRight
]

def plot_frame_wrapper_single(frames):
    global slider

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax2 = fig.add_axes([0.1, 0.85, 0.8, 0.1])

    def plot_frame(frame_index):
        ax.clear()
        points = np.array(frames[frame_index])
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])

        for line in lines:
            p1_idx, p2_idx = line
            x_values = [points[p1_idx][0], points[p2_idx][0]]
            y_values = [points[p1_idx][1], points[p2_idx][1]]
            z_values = [points[p1_idx][2], points[p2_idx][2]]
            ax.plot(x_values, y_values, z_values, c='r')

        ax.set_title(f'Frame {frame_index}')
        fig.canvas.draw_idle()

    slider = Slider(ax2, 'Frame Index', 0, len(frames) - 1, valinit=0, valstep=1)
    slider.on_changed(plot_frame)
    plot_frame(0)
    plt.show()