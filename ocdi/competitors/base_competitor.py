import torch
from torch import nn
import numpy as np


class CompetitorWrapperER(nn.Module):
    def __init__(self, net, buffer_size, input_size):
        super(CompetitorWrapperER, self).__init__()
        self.net = net
        self.buffer_size = buffer_size

    def forward(self, x):
        return self.net(x)

    def update_buffer(self, x, y):
        pass

    def retrieve_buffer(self,  num_retrieve=None, return_indices=False, optimizer=None, current_loss=None):
        # return only the portion of the buffer filled so far! (use the n_seen_so_far)

        possible_retrieve = min(num_retrieve, self.current_index)
        if possible_retrieve > 0:
            buffer_idx = torch.multinomial(torch.ones(self.current_index).float(), possible_retrieve, replacement=False)
        else:
            return None, None

        x = self.sample_buffer[buffer_idx]
        y = self.label_buffer[buffer_idx]

        if return_indices:
            return x, y, buffer_idx
        else:
            return x, y
