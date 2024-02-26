import os
import numpy as np
from aenum import Enum, NoAlias
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Tuple
from gymnasium.utils import seeding

from cs3arl.sokoban.dataloaders import SokobanDataLoader, sokoban_datafile_parser
from cs3arl.sokoban.render_utils import SokobanRenderingEngine


def generate_map(map_dim: int, num_boxes: int, gen_steps: int):
    """ Generate a map with the given dimensions and number of boxes. """
    print("WARNING: generating a random map...")
    fake_map = np.zeros(map_dim, dtype=np.uint8)
    fake_map[1,4] = 5
    return fake_map


class SokobanEnv(Env):
    """
    Implementation of the Sokoban environment on a discrete torus gridworld.
    """

    PLAYER_ID = 5

    CELL_LOOKUP = {
        0: "wall",
        1: "empty space",
        2: "box target",
        3: "box on target",
        4: "box not on target",
        PLAYER_ID: "player",
    }

    ACTION_LOOKUP = {
        0: "push up",
        1: "push down",
        2: "push left",
        3: "push right",
        4: "move up",
        5: "move down",
        6: "move left",
        7: "move right",
    }

    class ActionResult(Enum):
        """ Enum containing all the possible undirected result of actions 
            in the environement and their associated reward value. """
        _settings_ = NoAlias # avoid grouping items with same values (for NULL and STEP)
        NULL = -0.1
        STEP = -0.1
        BOX_ON_TARGET = 1.0
        BOX_OFF_TARGET = -1.0
        DONE = 10.0

        @classmethod
        def to_dict(cls):
            return {x.name: x.value for x in cls}

    RENDERING_MODES = [
        "rgb_array", 
        "human", 
    ]

    RESET_MODES = [
        "random",
        "next",
    ]

    CELL_DIM = 16 # length of the edge of a square cell for visualization

    def __init__(
            self,
            map_txt: str = None,
            map_collection: SokobanDataLoader = None,
            map_dim: tuple = (10, 10),
            num_boxes: int = 1,
            gen_steps: int = 120,
            max_steps: int = None,
            reset_mode: str = None,
            merge_move_push: bool = False,
        ):
        """
        INPUTS:
            - map_txt: path to a txt file containing a map with the same symbols as in SokobanEnv.CELL_LOOKUP
            - map_collection: SokobanDataLoader containing a collection of maps
            - map_dim: tuple containing the dimensions of the map to generate
            - num_boxes: number of boxes to put in the generated map
            - gen_steps: maximum number of steps for map generation
            - max_steps: maximum number of steps for an episode
            - reset_mode: mode for the reset method, see options in SokobanEnv.RESET_MODES
            - merge_move_push: if True, reduce the action space to 4 actions keeping only push actions and allowing player to move with it
        """

        ## set up a map
        self.map_collection = None
        self.map_id = 0
        self.map = None
        if map_collection is not None: # load a collection of maps
            if not isinstance(map_collection, SokobanDataLoader):
                raise ValueError(f"map_collection must be a SokobanDataLoader but is a {type(map_collection)}.")
            self.map_collection = map_collection
            self.map_collection.set_auto_extract_player(False)
            self.map = self.map_collection[self.map_id]
        elif map_txt is not None: # load a single map
            self.map = self.__load_map_from_txt(filepath=map_txt)
        else: # generate a map
            gen_steps = int(1.7 * (map_dim[0] + map_dim[1])) if gen_steps is None else gen_steps
            if map_dim is None or num_boxes is None or gen_steps is None:
                raise ValueError("map_dim, num_boxes and gen_steps must be specified to generate a map.")
            self.map = generate_map(map_dim=map_dim, num_boxes=num_boxes, gen_steps=gen_steps)
            
        ## save the initial map and don't modify it during the episode
        self.init_map  = self.map.copy()
        self.player_position = self.__get_player_position(extract_player=True) # virtually free the cell where the player is
        self.map_dim = self.map.shape

        ## set map and episode attributes
        self.num_boxes = self.__count_boxes() # number of boxes in the map
        self.boxes_on_target = 0 # number of boxes correctly placed on a target
        self.last_reward = 0 # last reward obtained
        self.num_env_steps = 0 # counter of the number of steps taken from the initial state
        self.max_steps = max_steps # maximum number of steps allowed before truncation (no limit if None)
        self.merge_move_push = merge_move_push # if True, move actions allow agent to push boxes just like push actions
        # mode for the reset method, either "random" or "next"
        self.reset_mode = reset_mode if reset_mode in SokobanEnv.RESET_MODES else SokobanEnv.RESET_MODES[0]
        self.action_lookup = {i: SokobanEnv.ACTION_LOOKUP[i] for i in range(4 if self.merge_move_push else 8)}
        
        ## action_space: int for each action
        self.action_space = Discrete(len(self.action_lookup))
        
        ## observation_space: Box = map without player | Tuple = player position in the map
        self.observation_space = Tuple((
            Box(low=0, high=len(SokobanEnv.CELL_LOOKUP)-1, shape=(self.map_dim[0], self.map_dim[1]), dtype=np.uint8),
            Tuple((Discrete(self.map_dim[0]), Discrete(self.map_dim[1]))),
        ))

        ## initialize the rendering engine
        self.rendering_engine = SokobanRenderingEngine()

    def __load_map_from_txt(self, filepath: str) -> np.ndarray:
        """ Load a map from a txt file with direct str->int symbol matching. """

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File does not exist: {filepath}")
        
        with open(filepath, "r") as f:
            raw_content = f.read()

        symbols_matching = {str(x): x for x in SokobanEnv.CELL_LOOKUP.keys()}

        levels = sokoban_datafile_parser(raw_content, symbols_matching=symbols_matching)

        if len(levels) == 0:
            raise ValueError(f"No map found in file: {filepath}")
        
        return list(levels.values())[0]

    def __get_player_position(self, extract_player: bool = False):

        if not SokobanEnv.PLAYER_ID in self.map:
            raise ValueError("Player not found in the map.")
        
        coords = np.argwhere(self.map == SokobanEnv.PLAYER_ID)

        if len(coords) > 1:
            raise ValueError("Multiple players found in the map.")
        
        if extract_player:
            self.map[coords[0][0], coords[0][1]] = 1
        
        return (coords[0][0], coords[0][1])

    def __count_boxes(self, xmap: np.ndarray = None) -> int:
        """ Count the number of boxes in the map. """
        xmap = self.map if xmap is None else xmap
        return np.sum(xmap == 3) + np.sum(xmap == 4)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset_episode(self) -> None:
        """ Reset the environment to the initial state of the map. """
        self.map = self.init_map.copy().astype(np.uint8)
        self.num_boxes = self.__count_boxes()
        self.boxes_on_target = 0
        self.last_reward = 0
        self.num_env_steps = 0
        self.player_position = self.__get_player_position(extract_player=True) # virtually free the cell where the player is
    
    def reset(self, seed: int = None, options: dict = None) -> tuple[tuple | np.ndarray, dict]:
        """
        Reset the environment using another map is a map_collection is given.
        Otherwise just perform a reset_episode().
        """
        
        if seed is not None:
            self.seed(seed)
        
        if self.map_collection is not None:
            if self.reset_mode == "random":
                self.map_id = np.random.randint(0, len(self.map_collection))
            elif self.reset_mode == "next":
                self.map_id = (self.map_id + 1) % len(self.map_collection)
            self.init_map = self.map_collection[self.map_id].copy().astype(np.uint8)
        
        self.reset_episode()

        return (np.copy(self.map), self.player_position), {}

    def __is_done(self) -> bool:
        """ Returns True if all boxes are on target or if the max_steps is reached. """
        if self.max_steps is None:
            return self.boxes_on_target == self.num_boxes
        else:
            return (self.boxes_on_target == self.num_boxes) or (self.num_env_steps >= self.max_steps)

    def __get_ahead_cells(self, action: int):
        """
        Returns the coordinates of the cell subject to the action and the cell after that.
        """

        if action % 4 == 0: # up
            cell_ahead = ((self.player_position[0]-1)%self.map_dim[0], self.player_position[1])
            cell_after = ((self.player_position[0]-2)%self.map_dim[0], self.player_position[1])
        elif action % 4 == 1: # down
            cell_ahead = ((self.player_position[0]+1)%self.map_dim[0], self.player_position[1])
            cell_after = ((self.player_position[0]+2)%self.map_dim[0], self.player_position[1])
        elif action % 4 == 2: # left
            cell_ahead = (self.player_position[0], (self.player_position[1]-1)%self.map_dim[1])
            cell_after = (self.player_position[0], (self.player_position[1]-2)%self.map_dim[1])
        elif action % 4 == 3: # right
            cell_ahead = (self.player_position[0], (self.player_position[1]+1)%self.map_dim[1])
            cell_after = (self.player_position[0], (self.player_position[1]+2)%self.map_dim[1])
        
        return cell_ahead, cell_after
    
    def __is_free_cell(self, cell: tuple) -> bool:
        """ Returns True if the cell is free, False otherwise. """
        return self.map[cell] in [1,2]
    
    def __contains_box(self, cell: tuple) -> bool:
        """ Returns True if the cell contains a box, False otherwise. """
        return self.map[cell] in [3,4]

    def __push(self, cell_push: tuple, cell_after: tuple) -> ActionResult:
        """
        Pushes the box in the direction of the action, if possible.
        If there is no box in the direction of the action, try to move.
        Returns the result of the action which contains itself the reward.
        """

        if self.__contains_box(cell_push) and self.__is_free_cell(cell_after): # push box
            
            if self.map[cell_after] == 1: # empty space
                self.map[cell_after] = 4
                if self.map[cell_push] == 3: # box on target
                    self.map[cell_push] = 2
                    self.boxes_on_target -= 1
                    result = SokobanEnv.ActionResult.BOX_OFF_TARGET
                else: # box not on target
                    self.map[cell_push] = 1
                    result = SokobanEnv.ActionResult.STEP
            
            elif self.map[cell_after] == 2: # box target
                self.map[cell_after] = 3
                if self.map[cell_push] == 3: # box on target
                    self.map[cell_push] = 2
                    result = SokobanEnv.ActionResult.STEP # no positive gain from pushing box from target to target
                else: # box not on target
                    self.map[cell_push] = 1
                    result = SokobanEnv.ActionResult.BOX_ON_TARGET
                self.boxes_on_target += 1
                if self.boxes_on_target == self.num_boxes:
                    result = SokobanEnv.ActionResult.DONE
            
            else:
                raise ValueError(f"Cell {cell_after} was supposed to be free but is {self.map[cell_after]}...")
            
            self.player_position = cell_push
            return result

        elif self.__contains_box(cell_push): # impossible to push box
            return SokobanEnv.ActionResult.STEP

        else: # no box to push
            if self.merge_move_push:
                return self.__move(cell_move=cell_push) # move instead
            else:
                return SokobanEnv.ActionResult.NULL

    def __move(self, cell_move: tuple) -> ActionResult:
        """
        Moves the player in the direction of the action, if possible.
        Returns the reward obtained from the action.
        """

        if self.__is_free_cell(cell_move):
            self.player_position = cell_move
            return SokobanEnv.ActionResult.STEP
        else:
            return SokobanEnv.ActionResult.NULL

    def step(self, action: int) -> tuple[tuple | np.ndarray, float, bool, bool, dict]:
        
        if not action in self.action_lookup:
            raise ValueError(f"Invalid action ID: {action}")
        
        cell_ahead, cell_after = self.__get_ahead_cells(action)

        if action < 4: # push action
            action_result = self.__push(cell_push=cell_ahead, cell_after=cell_after)
        else: # move action
            action_result = self.__move(cell_move=cell_ahead)

        self.last_reward = action_result.value
        self.num_env_steps += 1
        
        terminated = self.__is_done()
        truncated = False if self.max_steps is None else (self.num_env_steps >= self.max_steps)

        info = {
            "action_name": self.action_lookup[action],
            "action_result": action_result.name,
            "player_moved": action_result.name != "NULL",
            "box_moved": (action_result.name != "STEP") and (action_result.name != "NULL"),
        }
        if terminated or truncated:
            info["steps_used"] = self.num_env_steps
            info["all_boxes_on_target"] = (self.boxes_on_target == self.num_boxes)

        return (np.copy(self.map), self.player_position), self.last_reward, terminated, truncated, info
    
    @staticmethod
    def to_bloc_state(map: np.ndarray, player_position: tuple) -> np.ndarray:
        """ 
            Build a tensor (3D np.ndarray) observation from the map and the player position. 
            The shape of the tensor is (4, map_dim[0], map_dim[1]) with WTBP channels:
            - channel 1 = channel W : 1 if the cell is a wall, 0 otherwise
            - channel 2 = channel T : 1 if the cell is a target, 0 otherwise
            - channel 3 = channel B : 1 if the cell is a box, 0 otherwise
            - channel 4 = channel P : 1 if the cell is the player, 0 otherwise
            The elements of map must match the keys of SokobanEnv.CELL_LOOKUP.
        """
        tensor = np.zeros((4, map.shape[0], map.shape[1]), dtype=np.int8)
        tensor[0] = (map == 0).astype(np.int8) # W channel
        tensor[1] = (map == 2).astype(np.int8) # T channel
        tensor[2] = ((map == 3) | (map == 4)).astype(np.int8) # B channel
        tensor[3, player_position[0], player_position[1]] = 1 # P channel
        return tensor

    def get_image(self, mode: str = None):
        """ Only mode "human" is supported for now and is the default rendering given by the engine. """

        if mode is None:
            mode = "human"
        
        if mode not in SokobanEnv.RENDERING_MODES:
            raise ValueError(f"Invalid rendering mode: {mode}")

        image = self.rendering_engine.create_scene(
            map=self.map,
            player_position=self.player_position,
            output_format="image",
        )

        return image

    def render(self, mode: str = None):
        """ Only mode "human" is supported for now and is the default rendering given by the engine. """
        
        image = self.get_image(mode=mode)
        image.show()

    def get_action_lookup(self):
        return self.action_lookup
    
    def get_CELL_LOOKUP(self):
        return SokobanEnv.CELL_LOOKUP
    
    def get_rewards(self):
        return SokobanEnv.ActionResult.to_dict()


def main():

    import gymnasium as gym

    namespace = "sokoban"
    env_id = "sokoban-v0"

    with gym.envs.registration.namespace(ns=namespace):
        gym.register(
            id=env_id,
            entry_point=SokobanEnv,
        )

    env = gym.make(
        id=f"{namespace}/{env_id}",
        merge_move_push=False,
        reset_mode="random"
    )

    print("Sokoban environment is ready!")

    # gym.pprint_registry()


if __name__ == "__main__":
    main()