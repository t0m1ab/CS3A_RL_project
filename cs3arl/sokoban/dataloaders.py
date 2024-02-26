import os
import numpy as np

import cs3arl


def create_datafilename(id: int, n_digits: int = 3):
    """ id=42 --> "042.txt" if n_digits=3 """
    name = f"{id}.txt"
    return "0" * max(4 + n_digits - len(name),0) + name


def sokoban_datafile_parser(raw_content: str, symbols_matching: dict):
    """
    Parse Sokoban levels file and returns a dict of levels after matching symbols to integers.
    The file should respect the Boxoban format defined by DeepMind here: https://github.com/google-deepmind/boxoban-levels/tree/master
    """
    lines = raw_content.split("\n")
    levels = {}
    rows = []
    for line in lines:
        line = line.strip(" ")
        if line.startswith(";"):
            if len(rows) > 0: # add stored level data to the levels dict
                levels[level_id] = np.array(rows, dtype=np.uint8)
                rows = []
            level_id = int(line.split(";")[-1])
        elif len(line) > 0: # data line
            rows.append([symbols_matching[symbol] for symbol in line])
    if len(rows) > 0: # store last level of the file
        levels[level_id] = np.array(rows, dtype=np.int8)
    return levels


class SokobanDataLoader():

    def __init__(
            self, 
            data_path: str = None, 
            level: str = None, 
            split: str = None, 
            file_id: int = None,
            symbols_matching: dict = None,
            player_id: int = None,
        ) -> None:

        self.data_path = data_path
        self.level = level
        self.split = split
        self.file_id = file_id

        self.levels_filepath = os.path.join(self.data_path, self.level, self.split, create_datafilename(self.file_id))
        if not os.path.exists(self.levels_filepath):
            raise ValueError(f"File does not exist: {self.levels_filepath}")
        
        with open(self.levels_filepath, "r") as f:
            raw_content = f.read()

        self.symbols_matching = symbols_matching
        self.player_id = player_id
    
        self.levels = sokoban_datafile_parser(raw_content, self.symbols_matching)

        self.map_dim = self.levels[0].shape
        self.n_levels = len(self.levels)
        self.auto_extract_player = True
    
    def set_auto_extract_player(self, extract: bool) -> None:
        """ Set whether the player should be automatically extracted from the map when fetching the map with __getitem__(). """
        self.auto_extract_player = extract
    
    def __extract_player(self, map: np.ndarray) -> tuple[tuple[int,int], np.ndarray]:

        if not self.player_id in map:
            raise ValueError("Player not found in the map.")
        
        coords = np.argwhere(map == self.player_id)

        if len(coords) > 1:
            raise ValueError("Multiple players found in the map.")
        
        coords = (coords[0][0], coords[0][1])
        map[coords] = 1 # remove player from the map
        
        return coords, map

    def __getitem__(self, game_id: int) -> tuple[tuple[int,int], np.ndarray] | np.ndarray:

        if not game_id in self.levels:
            raise ValueError(f"Game ID {game_id} does not exist in levels file {self.levels_filepath}")

        if self.auto_extract_player:
            return self.__extract_player(self.levels[game_id].copy())
        else:
            return self.levels[game_id].copy()
    
    def __len__(self) -> int:
        return self.n_levels


class DeepMindBoxobanLoader(SokobanDataLoader):

    PLAYER_ID = 5

    DEFAULT_DATAPATH = os.path.join(cs3arl.sokoban.__path__[0], "boxoban-levels/")

    DEFAULT_LEVELS = ["medium", "hard", "unfiltered"]
    
    SYMBOLS_MATCHING = {
        "#": 0, # wall
        " ": 1, # empty
        ".": 2, # target
        "$": 4, # box
        "@": PLAYER_ID, # player
    }

    def __init__(
            self, 
            data_path: str = None, 
            level: str = None, 
            split: str = None, 
            file_id: int = None,
            symbols_matching: dict = None,
            player_id: int = None,
        ) -> None:

        kwargs = {
            "data_path": DeepMindBoxobanLoader.DEFAULT_DATAPATH if data_path is None else data_path,
            "level": DeepMindBoxobanLoader.DEFAULT_LEVELS[0] if level is None else level,
            "split": "train" if split is None else split,
            "file_id": 0 if file_id is None else file_id,
            "symbols_matching": DeepMindBoxobanLoader.SYMBOLS_MATCHING if symbols_matching is None else symbols_matching,
            "player_id": DeepMindBoxobanLoader.PLAYER_ID if player_id is None else player_id,
        }

        super().__init__(**kwargs)


class MySokobanLoader(SokobanDataLoader):

    PLAYER_ID = 5

    DEFAULT_DATAPATH = os.path.join(cs3arl.sokoban.__path__[0], "my-sokoban-levels/")

    DEFAULT_LEVELS = ["easy", "medium", "hard"]
    
    SYMBOLS_MATCHING = {
        "0": 0, # wall
        "1": 1, # empty
        "2": 2, # target
        "4": 4, # box
        str(PLAYER_ID): PLAYER_ID, # player
    }

    def __init__(
            self, 
            data_path: str = None, 
            level: str = None, 
            file_id: int = None,
            symbols_matching: dict = None,
            player_id: int = None,
        ) -> None:

        kwargs = {
            "data_path": MySokobanLoader.DEFAULT_DATAPATH if data_path is None else data_path,
            "level": MySokobanLoader.DEFAULT_LEVELS[0] if level is None else level,
            "split": "", # no split for custom levels yet
            "file_id": 0 if file_id is None else file_id,
            "symbols_matching": MySokobanLoader.SYMBOLS_MATCHING if symbols_matching is None else symbols_matching,
            "player_id": MySokobanLoader.PLAYER_ID if player_id is None else player_id,
        }
        
        super().__init__(**kwargs)


def main():

    ## DEEP MIND DATA loader

    medium_DeepMinds_levels = DeepMindBoxobanLoader(level="medium", split="train", file_id=0)
    print(f"Number of DEEPMIND levels: {medium_DeepMinds_levels.n_levels}")
    rand_level_id = np.random.randint(medium_DeepMinds_levels.n_levels)
    player_position, rand_level = medium_DeepMinds_levels[rand_level_id]
    # print(f"Level ID: {rand_level_id}")
    # print(f"Player position: {player_position}")
    # print("Level:")
    # print(rand_level)
    assert rand_level.shape == medium_DeepMinds_levels.map_dim
    assert rand_level[player_position] == 1

    ## MY CUSTOM DATA loader

    easy_custom_levels = MySokobanLoader(level="easy", file_id=0)
    print(f"Number of CUSTOM levels: {easy_custom_levels.n_levels}")
    rand_level_id = np.random.randint(easy_custom_levels.n_levels)
    player_position, rand_level = easy_custom_levels[rand_level_id]
    # print(f"Level ID: {rand_level_id}")
    # print(f"Player position: {player_position}")
    # print("Level:")
    # print(rand_level)
    assert rand_level.shape == easy_custom_levels.map_dim
    assert rand_level[player_position] == 1


if __name__ == "__main__":
    main()