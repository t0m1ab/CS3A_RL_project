import sys
import os
import cs3arl


def print_help(help_msg_relative_path: str = "docs/help.txt"):
    """ Print the help message for the cs3arl package """
    filepath = os.path.join(cs3arl.__path__[0], help_msg_relative_path)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as file:
        print(file.read())


def tests():
    """ Run tests for the cs3arl package """

    import cs3arl
    PACKAGE_PATH = cs3arl.__path__[0] # absolute path to cs3arl module

    from cs3arl.classicrl.agents import main as classic_agents_main
    from cs3arl.classicrl.trainers import main as classic_trainers_main

    from cs3arl.sokoban.sokoban_env import main as sokoban_env_main
    from cs3arl.sokoban.dataloaders import main as dataloaders_main
    from cs3arl.sokoban.render_utils import main as render_utils_main

    from cs3arl.deeprl.buffers import main as buffers_main
    from cs3arl.deeprl.networks import main as networks_main
    from cs3arl.deeprl.agents import main as deep_agents_main
    from cs3arl.deeprl.trainers import main as deep_trainers_main

    print(f"Running tests for package located at: {PACKAGE_PATH}")

    print("\n(1/3) Testing classis agents/trainers code...")
    classic_agents_main()
    classic_trainers_main()

    print("\n(2/3) Testing sokoban code...")
    sokoban_env_main()
    dataloaders_main()
    render_utils_main()

    print("\n(3/3) Testing deep agents/trainers code...")
    buffers_main()
    networks_main()
    deep_agents_main()
    deep_trainers_main()

    print("\n>>> ALL TESTS PASSED! <<<")


def main():
    """ 
    Entry point for the application script.
    (sys.argv = list of arguments given to the program as strings separated by spaces)
    """

    if ("--help" in sys.argv) or ("-h" in sys.argv):
        print_help()

    elif ("--test" in sys.argv) or ("-t" in sys.argv):
        tests()

    else:
        print("command 'cs3arl' is working: try --help or --test")


if __name__ == "__main__":
    main()