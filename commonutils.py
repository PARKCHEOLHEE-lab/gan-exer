from PIL import Image
import os


def create_animation_gif(
    images_directory: str, save_path: str, format: str = "png", duration: int = 100, loop: int = 0
) -> None:
    """Create an animated GIF from a directory of images.

    Args:
        images_directory (str): Directory path containing images.
        save_path (str): File path to save the generated GIF.
        format (str, optional): Format of the image files (default is "png").
        duration (int, optional): Duration (in milliseconds) of each frame (default is 100).
        loop (int, optional): Number of loops for the GIF (0 for infinite looping, default is 0).
    """

    files = sorted(os.listdir(images_directory), key=lambda x: int(x.split("-")[-1].split(".")[0]))
    files = [file for file in files if file.endswith(format)]

    frames = []
    for filename in files:
        file_path = os.path.join(images_directory, filename)
        img = Image.open(file_path)
        frames.append(img)

    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=duration, loop=loop)
