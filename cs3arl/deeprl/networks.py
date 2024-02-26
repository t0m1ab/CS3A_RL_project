import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.__name__ = "DQN"
    
    def enumerate_parameters(self):
        for pname, param in self.named_parameters():
            print(f"{pname} : {tuple(param.size())}")
 
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DQNCartPole(DQN):

    def __init__(self, n_observations: int, n_actions: int):
        super(DQNCartPole, self).__init__()
        self.__name__ = "DQNCartPole"
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

    def __init__(self, n_observations: int, n_actions: int):
        """ 
        ARGUMENTS:
            - n_observations: should be equal to the number of cells in the map which is considered squared (e.g. 10x10 => 100)
            - n_actions: should be equal to the number of possible actions (4 or 8 for Sokoban)
        """
        super().__init__()
        self.__name__ = "ConvDQNSokoban"

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
        print_shapes = False
        print(x.shape) if print_shapes else None
        x = F.relu(self.conv1(x))
        print(x.shape) if print_shapes else None
        x = F.relu(self.conv2(x))
        print(x.shape) if print_shapes else None
        x = F.relu(self.conv3(x))
        print(x.shape) if print_shapes else None
        x = self.flatten(x)
        print(x.shape) if print_shapes else None
        x = F.relu(self.fcc1(x))
        print(x.shape) if print_shapes else None
        x = self.fcc2(x)
        print(x.shape) if print_shapes else None
        return x


class FCDQNSokoban(DQN):

    def __init__(self, n_observations: int, n_actions: int):
        """ 
        ARGUMENTS:
            - n_observations: should be equal to the number of cells in the map which is considered squared (e.g. 10x10 => 100)
            - n_actions: should be equal to the number of possible actions (4 or 8 for Sokoban)
        """
        super().__init__()
        self.__name__ = "FCDQNSokoban"

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
        print_shapes = False
        x = self.flatten(x)
        print(x.shape) if print_shapes else None
        x = F.relu(self.fc_in(x))
        print(x.shape) if print_shapes else None
        x = F.relu(self.fc1(x))
        print(x.shape) if print_shapes else None
        x = self.fc_out(x)
        print(x.shape) if print_shapes else None
        return x


def main():

    # test DQNCartPole
    net1 = DQNCartPole(100, 10)
    print(f"{net1.__name__} is ready!")

    map_edge_size = 8
    input = torch.randn(32, 4, map_edge_size, map_edge_size) # 32 = batch size, 4 = number of channels

    # test ConvDQNSokoban
    conv_net = ConvDQNSokoban(map_edge_size ** 2, 8)
    _ = conv_net(input) # don't forget to unsqueeze(0) if the batch contains a single element
    print(f"{conv_net.__name__} is ready!")
    # conv_net.enumerate_parameters()
    # print(f"Number of parameters = {conv_net.count_parameters()}")

    fc_net = FCDQNSokoban(map_edge_size ** 2, 8)
    _ = fc_net(input) # don't forget to unsqueeze(0) if the batch contains a single element
    print(f"{fc_net.__name__} is ready!")
    # fc_net.enumerate_parameters()
    # print(f"Number of parameters = {fc_net.count_parameters()}")

if __name__ == "__main__":
    main()