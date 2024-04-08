import os
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import imageio
import matplotlib.pyplot as plt

import cs3arl

DEMO_PATH = os.path.join(cs3arl.__path__[0], "outputs/demo/")


@dataclass
class EpisodeData:
    """ Dataclass to store the frames of an episode. """
    map_id: str = None # init map in the episode collection of the episode
    frames: list = None # list of frames (PIL.Image.Image)
    length: int = None # length of the episode

    def add_frame(self, frame: Image.Image):
        if self.frames is None:
            self.frames = [frame]
        else:
            self.frames.append(frame)
        self.length = len(self.frames)


def save_gif(episode_data: EpisodeData, tag: str, path: str=None):
    """ Save a list of PIL.Image frames as a gif. """

    format_ok = True
    for frame in episode_data.frames:
        if not isinstance(frame, Image.Image):
            format_ok = False
            break
    if not format_ok:
        raise ValueError("All frames should be PIL.Image.Image instances.")

    path = path if path is not None else DEMO_PATH
    Path(path).mkdir(parents=True, exist_ok=True)

    gif_filepath = os.path.join(path, f"{tag}.gif")

    imageio.mimsave(
        gif_filepath,
        episode_data.frames,
        duration=500, # ms/frame
    )

    print(f"Saved gif with {episode_data.length} frames at: {gif_filepath}")


def save_frames_mosaic(episode_data: EpisodeData, tag: str=None, n_cols: int=6, path: str=None):
    """ Plot a mosaic of frames. """

    title = f"Episode of length {episode_data.length} from map {episode_data.map_id}"
    tag = title if tag is None else tag
    n_rows = episode_data.length // n_cols + 1 if episode_data.length % n_cols != 0 else episode_data.length // n_cols

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2, n_rows * 2))

    # plot frames
    for idx, frame in enumerate(episode_data.frames):
        ax = axs[idx // n_cols, idx % n_cols] if n_rows > 1 else axs[idx]
        ax.imshow(frame)
        ax.axis("off")
    
    # disable empty subplots
    for idx in range(episode_data.length, n_rows * n_cols):
        ax = axs[idx // n_cols, idx % n_cols] if n_rows > 1 else axs[idx]
        ax.axis("off")

    fig.suptitle(title, fontweight="bold", fontsize=16)
    plt.tight_layout()

    path = path if path is not None else DEMO_PATH
    Path(path).mkdir(parents=True, exist_ok=True)

    fig.savefig(os.path.join(path, f"{tag}.png"))