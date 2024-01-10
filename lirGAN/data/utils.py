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
        binary_grid: 2D Numpy array representing the binary image.

    Returns:
        Numpy array representing the vertices of the largest polygon.
    """

    image = np.uint8(binary_grid * 255)

    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the array.")

    polygon_contour = max(contours, key=cv2.contourArea)

    epsilon = 0.01 * cv2.arcLength(polygon_contour, True)
    approx_polygon = cv2.approxPolyDP(polygon_contour, epsilon, True)

    return np.array(approx_polygon).squeeze()