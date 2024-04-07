import os
import time
import imageio

from typing import Callable


def create_animation_gif(
    images_directory: str,
    save_path: str,
    format: str = "png",
    loop: int = 0,
    duration: int = 1,
    sorting: bool = True,
    add_reverse_frames: bool = False,
) -> None:
    """Create an animated GIF from a directory of images.

    Args:
        images_directory (str): Directory path containing images.
        save_path (str): File path to save the generated GIF.
        format (str, optional): Format of the image files (default is "png").
        loop (int, optional): Number of loops for the GIF (0 for infinite looping, default is 0).
        duration (int, optional): Duration (in milliseconds) of each frame (default is 1).
        sorting (bool, optional): Whether to sort the images by their names (default is False).
        add_reverse_frames (bool, optional): Whether to add reverse frames (default is False).
    """

    files = os.listdir(images_directory)
    if sorting:
        files = sorted(files, key=lambda x: int(x.split("-")[-1].split(".")[0]))

    files = [os.path.abspath(os.path.join(images_directory, file)) for file in files if file.endswith(format)]

    if add_reverse_frames:
        files += files[::-1]

    images_data = []
    for file in files:
        data = imageio.imread(file)
        images_data.append(data)

    imageio.mimwrite(save_path, images_data, format=".gif", duration=duration, loop=loop)


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


def add_debugvisualizer(globals_dict: dict) -> None:
    """Add libs for debugging to the global namespace.

    Args:
        globals_dict (dict): The global namespace.
    """

    from debugvisualizer.debugvisualizer import Plotter
    from shapely import geometry
    import trimesh

    globals_dict["Plotter"] = Plotter
    globals_dict["geometry"] = geometry
    globals_dict["trimesh"] = trimesh
