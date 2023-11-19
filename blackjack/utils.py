import numpy as np


def thorp_basic_strategy():
    """ Returns Thorp basic strategy according to Sutton & Barto (figure 5.2) 
        - Dealer showing range = {A, 1, 2, ..., 10} on dimension 0
        - Player sum range = {12, 13 , 14, ..., 21} on dimension 1
        - Stick = 0
        - Hit = 1
        RETURNS : {"usable_ace": np.ndarray, "no_usable_ace": np.ndarray} """
    
    usable_ace = np.ones((10, 10))
    no_usable_ace = np.ones((10, 10))
    # build usable ace strategy
    usable_ace[:4, :] = 0
    usable_ace[3, -2:] = 1
    # build no usable ace strategy
    no_usable_ace[:5, :] = 0
    no_usable_ace[:, 1:6] = 0
    no_usable_ace[-1, [1,2]] = 1

    strategy = {"usable_ace": np.rot90(usable_ace, 3), "no_usable_ace": np.rot90(no_usable_ace, 3)}

    return strategy


if __name__ == "__main__":

    # create a plot showing Thorp basic strategy
    from visualization import create_policy_plots
    thorp_strat = thorp_basic_strategy()
    create_policy_plots(thorp_strat["usable_ace"], thorp_strat["no_usable_ace"], "Thorp basic strategy", show=False, save_dir="outputs/", save=True)