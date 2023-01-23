import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

import math
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Tuple, Union
from typing_extensions import Final

class GroupedLinearEinsum(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        # self.weight: Tensor
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        assert input_size % groups == 0
        self.ws = input_size // groups
        self.register_parameter(
            "weight",
            Parameter(
                torch.zeros(groups, input_size // groups, hidden_size // groups), requires_grad=True
            ),
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., I]
        x = x.unflatten(-1, (self.groups, self.ws))  # [..., G, I/G]
        x = torch.einsum("...gi,...gih->...gh", x, self.weight)  # [..., G, H/G]
        x = x.flatten(2, 3)  # [B, T, H]
        return x


class GroupedLinear(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]
    shuffle: Final[bool]

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1, shuffle: bool = True):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.layers = nn.ModuleList(
            nn.Linear(self.input_size, self.hidden_size) for _ in range(groups)
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            outputs.append(layer(x[..., i * self.input_size : (i + 1) * self.input_size]))
        output = torch.cat(outputs, dim=-1)
        if self.shuffle:
            orig_shape = output.shape
            output = (
                output.view(-1, self.hidden_size, self.groups).transpose(-1, -2).reshape(orig_shape)
            )
        return output