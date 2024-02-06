import sys

def tests():
    """ Run tests for the cs3arl package """

    import cs3arl
    PACKAGE_PATH = cs3arl.__path__[0] # absolute path to cs3arl module

    from cs3arl.blackjack.agents import main as agents_main
    from cs3arl.blackjack.trainers import main as trainers_main

    from cs3arl.sokoban.sokoban_env import main as sokoban_env_main
    from cs3arl.sokoban.dataloaders import main as dataloaders_main
    from cs3arl.sokoban.render_utils import main as render_utils_main

    from cs3arl.deeprl.buffers import main as buffers_main
    from cs3arl.deeprl.networks import main as networks_main
    from cs3arl.deeprl.agents import main as agents_main
    from cs3arl.deeprl.trainers import main as trainers_main

    print(f"Running tests for package located at: {PACKAGE_PATH}")

    print("\n(1/3) Testing blackjack code...")
    agents_main()
    trainers_main()

    print("\n(2/3) Testing sokoban code...")
    sokoban_env_main()
    dataloaders_main()
    render_utils_main()

    print("\n(3/3) Testing deeprl code...")
    buffers_main()
    networks_main()
    agents_main()
    trainers_main()

    print("\n>>> ALL TESTS PASSED! <<<")


def main():
    """ 
    Entry point for the application script.
        - sys.argv: list of arguments given to the program (as strings separated by spaces)
    """

    if ("--help" in sys.argv) or ("-h" in sys.argv):
        print("Welcome to the cs3arl package!")
        print("\nThe submodules available are: \n")
        print("\t- blackjack: implements classic RL methods for the game of blackjack")
        print("\t- sokoban: implements a sokoban environment respecting the API specifications of gym")
        print("\t- deeprl: implements deep reinforcement learning methods like DQN")

    elif ("--test" in sys.argv) or ("-t" in sys.argv):
        tests()

    else:
        print("command 'cs3arl' is working: try --help or --test")

if __name__ == "__main__":
    main()