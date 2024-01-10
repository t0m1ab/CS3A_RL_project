import numpy as np
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Tuple
from gymnasium.utils import seeding

# from .room_utils import generate_room
# from .render_utils import room_to_rgb, room_to_tiny_world_rgb


def generate_map(map_dim: int, num_boxes: int, gen_steps: int):
    fake_map = np.zeros(map_dim)
    fake_map[1,4] = 5
    return fake_map


class SokobanEnv(Env):
    """
    Implementation of the Sokoban environment on a discrete torus gridworld.
    """

    PLAYER_ID = 5

    TYPE_LOOKUP = {
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

    REWARDS = {
        "step": -0.1,
        "box_on_target": 1,
        "box_off_target": -1,
        "done": 10,
    }

    RENDERING_MODES = [
        "rgb_array", 
        "human", 
        "tiny_rgb_array", 
        "tiny_human", 
        "raw",
    ]

    CELL_DIM = 16 # length of the edge of a square cell for visualization

    def __init__(
            self,
            map_json: str = None,
            map_txt: np.ndarray = None,
            map_dim: tuple = (10, 10),
            num_boxes: int = 1,
            gen_steps: int = 120,
            max_steps: int = None,
            # reset: bool = True,
        ):
        """
        INPUTS:
            - map_json: path to a json file containing a map
            - map_data: numpy array containing a map
            - map_dim: tuple containing the dimensions of the map
            - num_boxes: number of boxes to put in the map
            - max_steps: maximum number of steps for map generation
        """

        # number of generation steps
        self.gen_steps = int(1.7 * (map_dim[0] + map_dim[1])) if gen_steps is None else gen_steps
        # number of boxes in the map
        self.num_boxes = num_boxes
        # number of boxes correctly placed on a target
        self.boxes_on_target = 0
        # last reward obtained
        self.last_reward = 0
        # counter of the number of steps taken from the initial state
        self.num_env_steps = 0
        # maximum number of steps allowed before truncation (no limit if None)
        self.max_steps = max_steps

        # build the map
        self.map = None
        if map_json is not None and map_txt is not None:
            raise ValueError("Only one of map_json and map_txt can be specified.")
        if map_json is not None:
            self.map = self.__load_map_from_json(filepath=map_json)
            self.map_dim = self.map.shape
        if map_txt is not None:
            self.map = self.__load_map_from_txt(filepath=map_txt)
            self.map_dim = self.map.shape
        if self.map is None:
            self.map_dim = map_dim
            if self.map_dim is None or self.num_boxes is None:
                raise ValueError("Either a map file or map_dim must be specified.")
            self.map = generate_map(map_dim=self.map_dim, num_boxes=self.num_boxes, gen_steps=self.gen_steps)
        
        # save the initial map and don't modify the original
        self.init_map  = self.map.copy()
        self.player_position = self.__get_player_position()
        self.map[self.player_position] = 1 # virtually free the cell where the player is

        # initialize the environment spaces
        self.action_space = Discrete(len(SokobanEnv.ACTION_LOOKUP))
        self.observation_space = Tuple((
            Box(low=0, high=len(SokobanEnv.TYPE_LOOKUP)-1, shape=(map_dim[0], map_dim[1]), dtype=np.uint8),
            Tuple((Discrete(map_dim[0]), Discrete(map_dim[1]))),
        )) # Box = map without player | Tuple = player position in the map
        
        # if reset:
        #     self.reset()

    def __load_map_from_json(self, filepath: str):
        return np.zeros((42,42))

    def __load_map_from_txt(self, filepath: str):
        return np.zeros((42,42))

    def __get_player_position(self):

        if not SokobanEnv.PLAYER_ID in self.map:
            raise ValueError("Player not found in the map.")
        
        coords = np.argwhere(self.map == SokobanEnv.PLAYER_ID)

        if len(coords) > 1:
            raise ValueError("Multiple players found in the map.")
        
        return (coords[0][0], coords[0][1])

    def reset_episode(self) -> None:
        """
        Reset the environment to the initial state of the map.
        """
        self.map = self.init_map.copy()
        self.boxes_on_target = 0
        self.last_reward = 0
        self.num_env_steps = 0
        self.player_position = self.__get_player_position()
        self.map[self.player_position] = 1 # virtually free the cell where the player is
    
    def reset(self) -> None:
        """
        Reset the environment to the initial state of the map.
        """
        self.reset_episode()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
        """
        Returns True if the cell is free, False otherwise.
        """
        return self.map[cell] in [1,2]
    
    def __contains_box(self, cell: tuple) -> bool:
        """
        Returns True if the cell contains a box, False otherwise.
        """
        return self.map[cell] in [3,4]

    def __push(self, cell_push: tuple, cell_after: tuple) -> float:
        """
        Pushes the box in the direction of the action, if possible.
        If there is no box in the direction of the action, try to move.
        Returns the reward obtained from the action.
        """

        if self.__contains_box(cell_push) and self.__is_free_cell(cell_after): # push box
            
            if self.map[cell_after] == 1: # empty space
                self.map[cell_after] = 4
                if self.map[cell_push] == 3: # box on target
                    self.map[cell_push] = 2
                    self.boxes_on_target -= 1
                    reward = SokobanEnv.REWARDS["box_off_target"]
                else: # box not on target
                    self.map[cell_push] = 1
                    reward = SokobanEnv.REWARDS["step"]
            
            elif self.map[cell_after] == 2: # box target
                self.map[cell_after] = 3
                if self.map[cell_push] == 3: # box on target
                    self.map[cell_push] = 2
                    reward = SokobanEnv.REWARDS["step"] # no positive gain from pushing box from target to target
                else: # box not on target
                    self.map[cell_push] = 1
                    reward = SokobanEnv.REWARDS["box_on_target"]
                self.boxes_on_target += 1
                if self.boxes_on_target == self.num_boxes:
                    reward = SokobanEnv.REWARDS["done"]
            
            else:
                raise ValueError(f"Cell {cell_after} was supposed to be free but is {self.map[cell_after]}...")
            
            self.player_position = cell_push
            return reward

        elif self.__contains_box(cell_push): # impossible to push box
            return SokobanEnv.REWARDS["step"]

        else: # no box to push
            return self.__move(cell_move=cell_push)

    def __move(self, cell_move: tuple) -> float:
        """
        Moves the player in the direction of the action, if possible.
        Returns the reward obtained from the action.
        """

        if self.__is_free_cell(cell_move):
            self.player_position = cell_move
        
        return SokobanEnv.REWARDS["step"]

    def step(self, action: int):

        if not action in SokobanEnv.ACTION_LOOKUP.keys():
            raise ValueError("Invalid action: {}".format(action))
        
        cell_ahead, cell_after = self.__get_ahead_cells(action)

        if action < 4: # push action
            self.last_reward = self.__push(cell_push=cell_ahead, cell_after=cell_after)
        else: # move action
            self.last_reward = self.__move(cell_move=cell_ahead, cell_after=cell_after)
  
        self.num_env_steps += 1
        
        terminated = (self.boxes_on_target == self.num_boxes) or (self.num_env_steps == self.max_steps)
        truncated = False if self.max_steps is None else (self.num_env_steps >= self.max_steps)

        info = {
            "action_name": SokobanEnv.ACTION_LOOKUP[action],
            # "action_moved_player": player_moved,
            # "action_moved_box": box_moved,
        }
        if terminated or truncated:
            info["steps_used"] = self.num_env_steps
            info["all_boxes_on_target"] = (self.boxes_on_target == self.num_boxes)

        return (self.map, self.player_position), self.last_reward, terminated, truncated, info

    def render(self, mode='raw', close=None, scale=1):

        raise NotImplementedError("Need to check things before...")
    
        # assert mode in RENDERING_MODES

        # img = self.get_image(mode, scale)

        # if 'rgb_array' in mode:
        #     return img

        # #elif 'human' in mode:
        # #    from gym.envs.classic_control import rendering
        # #    if self.viewer is None:
        # #        self.viewer = rendering.SimpleImageViewer()
        # #    self.viewer.imshow(img)
        # #    return self.viewer.isopen

        # if mode=='raw':
        #     arr_walls = (self.room_fixed == 0).view(np.int8)
        #     arr_goals = (self.room_fixed == 2).view(np.int8)
        #     arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
        #     arr_player = (self.room_state == 5).view(np.int8)

        #     return arr_walls, arr_goals, arr_boxes, arr_player

        # else:
        #     super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):

        raise NotImplementedError("Need to check things before...")
        
        # if mode.startswith('tiny_'):
        #     img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        # else:
        #     img = room_to_rgb(self.room_state, self.room_fixed)

        # return img

    def get_action_lookup(self):
        return SokobanEnv.ACTION_LOOKUP
    
    def get_type_lookup(self):
        return SokobanEnv.TYPE_LOOKUP
    
    def get_rewards(self):
        return SokobanEnv.REWARDS


if __name__ == "__main__":

    import gymnasium as gym

    namespace = "sokoban"
    env_id = "sokoban-v0"

    with gym.envs.registration.namespace(ns=namespace):
        gym.register(
            id=env_id,
            entry_point=SokobanEnv,
        )

    env = gym.make(id=f"{namespace}/{env_id}")

    assert env.unwrapped.num_env_steps == 0

    print("Sokoban environment is ready!")

    # gym.pprint_registry()