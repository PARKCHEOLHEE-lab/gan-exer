import time
from typing import Callable

import cv2
import numpy as np


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


def visualize_binary_grid(binary_grid: np.ndarray) -> None:
    """Shows binary grid by convert it to the binary image using OpencV

    Args:
        binary_grid (np.ndarray): binary grid
    """
    
    color_grid = cv2.cvtColor(binary_grid.astype(np.float32), cv2.COLOR_GRAY2BGR)

    color_grid[binary_grid == 1] = [255, 255, 255]  
    color_grid[binary_grid == 0] = [0, 0, 0]        

    cv2.imshow('Binary Grid', color_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()