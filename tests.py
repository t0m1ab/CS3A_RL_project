import cs3arl
MODULE_PATH = cs3arl.__path__[0] # absolute path to cs3arl module

from cs3arl.blackjack.agents import main as agents_main
from cs3arl.blackjack.trainers import main as trainers_main
from cs3arl.blackjack.utils import main as utils_main

from cs3arl.sokoban.sokoban_env import main as sokoban_env_main
from cs3arl.sokoban.dataloaders import main as dataloaders_main
from cs3arl.sokoban.render_utils import main as render_utils_main

print(">>> Testing blackjack code...")
agents_main()
trainers_main()
utils_main()

print("\n>>> Testing sokoban code...")
sokoban_env_main()
dataloaders_main()
render_utils_main()

print("\n>>> ALL TESTS PASSED! <<<")