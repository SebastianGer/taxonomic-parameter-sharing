from typing import List

import torch


class TPS(torch.nn.Module):
    def __init__(self, d_in: int, classes_per_level: List[int]):
        super(TPS, self).__init__()
        self.fc_list = torch.nn.ModuleList()
        for classes_at_level in classes_per_level:
            self.fc_list.append(torch.nn.Linear(d_in, classes_at_level, bias=True))

    def forward(self, x):
        outputs = []
        for output_layer in self.fc_list:
            output = output_layer(x)
            outputs.append(output)
        return outputs
