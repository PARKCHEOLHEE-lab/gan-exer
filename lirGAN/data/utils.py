import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from typing import Callable, List, Union


def runtime_calculator(func: Callable) -> Callable:
    """A decorator function for measuring the runtime of another function.

    Args:
        func (Callable): Function to measure

    Returns:
        Callable: Decorator
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"The function {func.__name__} took {runtime} seconds to run.")
        return result

    return wrapper


def vectorize_polygon_from_array(binary_grid: np.ndarray) -> np.ndarray:
    """Convert a binary grid-shaped polygon represented as a 2D array of 1s and 0s into a vectorized Numpy array.

    Args:
        binary_grid (np.ndarray): 2D Numpy array representing the binary image.

    Raises:
        ValueError: occurs if there are no contours

    Returns:
        np.ndarray: Numpy array representing the vertices of the largest polygon.
    """

    image = np.uint8(binary_grid * 255)

    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the array.")

    polygon_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.01 * cv2.arcLength(polygon_contour, True)
    approx_polygon = cv2.approxPolyDP(polygon_contour, epsilon, True)

    return np.array(approx_polygon).squeeze()


def get_binary_grid_shaped_polygon(coordinates: np.ndarray, canvas_size: np.ndarray) -> np.ndarray:
    """Convert a given polygon coordinates to the binary grid-shaped polygon

    Args:
        coordinates (np.ndarray): polygon coordinates

    Returns:
        np.ndarray: binary grid
    """

    binary_grid_shaped_polygon = np.zeros(canvas_size, np.uint8)
    cv2.fillPoly(binary_grid_shaped_polygon, [coordinates], 255)

    binary_grid_shaped_polygon = (binary_grid_shaped_polygon == 255).astype(np.uint8)

    return binary_grid_shaped_polygon


def visualize_binary_grids(binary_grids: List[np.ndarray], colormap: Union[List[str], str] = None) -> None:
    """visualize multiple binary grids in a row

    Args:
        binary_grids (List[np.ndarray]): list of binary grids
        colormap (Union[List[str], str], optional): colormap to use for displaying the binary grids. Defaults to None.
    """

    n = len(binary_grids)
    _, axs = plt.subplots(1, n, figsize=(n * 5, 5))

    # Handle colormap
    matplotlib_colormap = "Greys"
    if colormap is not None:
        if isinstance(colormap, list):
            matplotlib_colormap = mcolors.ListedColormap(colormap)
        elif isinstance(colormap, str):
            matplotlib_colormap = colormap

    for i, grid in enumerate(binary_grids):
        if n == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.imshow(grid, cmap=matplotlib_colormap)
        ax.axis("off")

    plt.show()
