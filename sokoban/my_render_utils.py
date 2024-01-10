import os
from pathlib import Path
import numpy as np
from PIL import Image


def create_alpha_image(filepath: str) -> None:
    """ Create a player image with alpha channel for black background transparency. """

    image = Image.open(filepath)
    image = image.convert("RGBA")
    np_image = np.array(image)

    new_np_image = np_image.copy()
    for row in range(np_image.shape[0]):
        for col in range(np_image.shape[1]):
            if np.sum(np_image[row, col, :3]) == 0: # background
                new_np_image[row, col, 3] = 0 # transparent

    Image.fromarray(new_np_image).save(os.path.join(Path(filepath).parent, "player_alpha.png"))


class SokobanRenderingEngine():

    PLAYER_ID = 5

    DEFAULT_DATAPATH = "surface/"
    
    SYMBOLS_MATCHING = {
        0: "wall.png",
        1: "floor.png",
        2: "box_target.png",
        3: "box_on_target.png",
        4: "box.png",
        PLAYER_ID: "player_alpha.png",
    }

    ELEMENT_SIZE = 16

    if not os.path.exists(DEFAULT_DATAPATH): # check if data path exists
        raise ValueError(f"Data path {DEFAULT_DATAPATH} does not exist.")
    
    if not os.path.isfile(os.path.join(DEFAULT_DATAPATH, SYMBOLS_MATCHING[PLAYER_ID])): # create alpha image for the player
        create_alpha_image(os.path.join(DEFAULT_DATAPATH, "player.png"))

    def __init__(
            self,
            data_path: str = None,
            symbols_matching: dict = None,
            player_id: int = None,
            element_size: int = None,
        ) -> None:

        self.data_path = SokobanRenderingEngine.DEFAULT_DATAPATH if data_path is None else data_path
        self.symbols_matching = SokobanRenderingEngine.SYMBOLS_MATCHING if symbols_matching is None else symbols_matching
        self.player_id = SokobanRenderingEngine.PLAYER_ID if player_id is None else player_id
        self.element_size = SokobanRenderingEngine.ELEMENT_SIZE if element_size is None else element_size

        if not os.path.exists(self.data_path):
            raise ValueError(f"Data path {self.data_path} does not exist.")

        self.elements = {}
        for id, filename in SokobanRenderingEngine.SYMBOLS_MATCHING.items():
            filepath = os.path.join(self.data_path, filename)
            if not os.path.exists(filepath):
                raise ValueError(f"File {filepath} does not exist.")
            self.elements[id] = Image.open(filepath).convert("RGBA")
        
    def create_scene(
            self, 
            map: np.ndarray, 
            player_position: tuple[int,int] = None, 
            output_format: str = None
        ):
        """ Create the matrix representing the scene of map. """
            
        map_dim = map.shape

        scene = np.zeros(shape=(map_dim[0] * self.element_size, map_dim[1] * self.element_size, 4), dtype=np.uint8)
        for i in range(map_dim[0]):
            for j in range(map_dim[1]):
                if not map[i, j] in self.elements:
                    raise ValueError(f"Element with ID {map[i, j]} not found in the elements dictionary.")
                if player_position is not None and (i,j) == player_position: # add player on cell
                    cell_image = self.elements[map[i, j]].copy()
                    cell_image.paste(self.elements[self.player_id], (0,0), self.elements[self.player_id])
                else:
                    cell_image = self.elements[map[i, j]]
                scene[i * self.element_size: (i + 1) * self.element_size, j * self.element_size: (j + 1) * self.element_size, :] = cell_image
        
        if output_format.lower() == "np":
            return scene
        elif output_format.lower() == "image":
            return Image.fromarray(scene)
        else:
            return scene
    
    def save_scene(self, map: np.ndarray, player_position: tuple[int,int] = None, filepath: str = None) -> None:
        """ Save the scene of map to filepath. """
        scene_image = self.create_scene(map, player_position, output_format="image") # create scene
        filepath = filepath if filepath is not None else "outputs/scene.png" # set default filepath
        Path(filepath).parent.mkdir(parents=True, exist_ok=True) # create directory
        scene_image.save(filepath) # save scene


if __name__ == "__main__":

    from dataloaders import MySokobanLoader

    # load a map
    easy_custom_levels = MySokobanLoader(level="easy", file_id=0)
    player_position, sokoban_map = easy_custom_levels[0]

    # render the map
    rendering_engine = SokobanRenderingEngine()
    rendering_engine.save_scene(sokoban_map, player_position, filepath="outputs/test_scene.png")

