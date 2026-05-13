from matplotlib import pyplot as plt
import torch
import numpy as np
import matplotlib.patches as patches

def draw_checkerboard(ax, n=4, color='steelblue', alpha=0.25):
    L = 2 * np.pi
    size = L / n
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                rect = patches.Rectangle(
                    (-np.pi + i * size, -np.pi + j * size), size, size,
                    linewidth=0, facecolor=color, alpha=alpha
                )
                ax.add_patch(rect)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_aspect('equal')