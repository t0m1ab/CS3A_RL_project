import numpy as np
from abc import abstractmethod
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetType(Enum):
    """ Enumerate types of neural networks. """
    FC = "fully_connected"
    CONV = "convolutional"

    @staticmethod
    def get_type(string_type: str):
        if string_type is None:
            return NetType.CONV
        for net_type in NetType:
            if net_type.value == string_type:
                return net_type
        raise ValueError(f"Unknown network type: {string_type}")


class DQN(nn.Module):

    def __init__(self, verbose: bool=False):
        super(DQN, self).__init__()
        self.__name__ = "DQN"
        self.net_type = None
        self.verbose = verbose
    
    def print(self, *args):
        if self.verbose:
            print(*args)
    
    def enumerate_parameters(self):
        for pname, param in self.named_parameters():
            print(f"{pname} : {tuple(param.size())}")
 
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @abstractmethod
    def float(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConvDQNCartPole(DQN):

    def __init__(self, n_observations: int, n_actions: int, verbose: bool=False):
        super().__init__(verbose=verbose)
        self.__name__ = "DQNCartPole"
        self.net_type = NetType.CONV
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class ConvDQNSokoban(DQN):

    def __init__(self, n_observations: int, n_actions: int, verbose: bool=False):
        """ 
        ARGUMENTS:
            - n_observations: should be equal to the number of cells in the map which is considered squared (e.g. 10x10 => 100)
            - n_actions: should be equal to the number of possible actions (4 or 8 for Sokoban)
        """
        super().__init__(verbose=verbose)
        self.__name__ = "ConvDQNSokoban"
        self.net_type = NetType.CONV

        map_edge_size = int(n_observations ** 0.5)
        if map_edge_size < 4:
            raise ValueError("The map should be at least 4x4")

        self.conv1 = nn.Conv2d(4, 16, kernel_size=2, padding="same", padding_mode="reflect")
        self.conv2 = nn.Conv2d(16, 16, kernel_size=2, stride=2, padding="valid")
        self.conv3 = nn.Conv2d(16, 16, kernel_size=2, stride=2, padding="valid")
        self.flatten = nn.Flatten()
        self.fcc_dim = 16 * (map_edge_size // 4) ** 2 # should be 64 for a 10x10 map
        self.fcc1 = nn.Linear(self.fcc_dim, self.fcc_dim // 2)
        self.fcc2 = nn.Linear(self.fcc_dim // 2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.print("input ->", x.shape)
        x = F.relu(self.conv1(x))
        self.print("conv1 ->", x.shape)
        x = F.relu(self.conv2(x))
        self.print("conv2 ->", x.shape)
        x = F.relu(self.conv3(x))
        self.print("conv3 ->", x.shape)
        x = self.flatten(x)
        self.print("flatten ->", x.shape)
        x = F.relu(self.fcc1(x))
        self.print("fcc1 ->", x.shape)
        x = self.fcc2(x)
        self.print("fcc2 ->", x.shape)
        return x


class FCDQNSokoban(DQN):

    def __init__(self, n_observations: int, n_actions: int, verbose: bool=False):
        """ 
        ARGUMENTS:
            - n_observations: should be equal to the number of cells in the map which is considered squared (e.g. 10x10 => 100)
            - n_actions: should be equal to the number of possible actions (4 or 8 for Sokoban)
        """
        super().__init__(verbose=verbose)
        self.__name__ = "FCDQNSokoban"
        self.net_type = NetType.FC

        map_edge_size = int(n_observations ** 0.5)
        if map_edge_size < 4:
            raise ValueError("The map should be at least 4x4")

        self.dim1 = n_observations
        self.dim2 = n_observations // 2

        self.flatten = nn.Flatten()
        self.fc_in = nn.Linear(4 * n_observations, self.dim1)
        self.fc1 = nn.Linear(self.dim1, self.dim2)
        self.fc_out = nn.Linear(self.dim2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.print("input ->", x.shape)
        x = self.flatten(x)
        self.print("flatten ->", x.shape)
        x = F.relu(self.fc_in(x))
        self.print("fc_in ->", x.shape)
        x = F.relu(self.fc1(x))
        self.print("fc_1 ->", x.shape)
        x = self.fc_out(x)
        self.print("fc_out ->", x.shape)
        return x


def main():

    VERBOSE = False

    # test ConvDQNCartPole
    net = ConvDQNCartPole(100, 10, verbose=VERBOSE)
    print(f"{net.__name__} of type {net.net_type.value} is ready!")

    map_edge_size = 8
    input = torch.randn(32, 4, map_edge_size, map_edge_size) # 32 = batch size, 4 = number of channels

    # test ConvDQNSokoban
    net = ConvDQNSokoban(map_edge_size ** 2, 8, verbose=VERBOSE)
    _ = net(input) # don't forget to unsqueeze(0) if the batch contains a single element
    print(f"{net.__name__} of type {net.net_type.value} is ready!")
    # net.enumerate_parameters()
    # print(f"Number of parameters = {net.count_parameters()}")

    net = FCDQNSokoban(map_edge_size ** 2, 8, verbose=VERBOSE)
    _ = net(input) # don't forget to unsqueeze(0) if the batch contains a single element
    print(f"{net.__name__} of type {net.net_type.value} is ready!")
    # net.enumerate_parameters()
    # print(f"Number of parameters = {net.count_parameters()}")


if __name__ == "__main__":
    main()