import os
from pathlib import Path
from PIL import Image
import imageio

import cs3arl

DEMO_PATH = os.path.join(cs3arl.__path__[0], "outputs/demo/")


def save_gif(frames: list[Image.Image], gif_name: str, path: str=None):
    """ Save a list of PIL.Image frames as a gif. """

    format_ok = True
    for frame in frames:
        if not isinstance(frame, Image.Image):
            format_ok = False
            break
    if not format_ok:
        raise ValueError("All frames should be PIL.Image.Image instances.")

    path = path if path is not None else DEMO_PATH
    Path(path).mkdir(parents=True, exist_ok=True)

    gif_filepath = os.path.join(path, f"{gif_name}.gif")

    imageio.mimsave(
        gif_filepath,
        frames,
        duration=500, # ms/frame
    )

    print(f"Saved gif with {len(frames)} frames at: {gif_filepath}")