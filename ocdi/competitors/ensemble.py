from torch import nn
import torch
import copy


# function to reset parameters
def _reset_parameters(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


class Ensemble(nn.Module):
    def __init__(self, net, options, num_classes):
        num_nets = options.ensembled_models
        super(Ensemble, self).__init__()
        self.nets = nn.ModuleList()
        for i in range(num_nets):
            new_net = copy.deepcopy(net)
            new_net.apply(_reset_parameters)
            self.nets.append(new_net)
        pass

    def forward(self, x):
        outputs = [net(x) for net in self.nets]
        return torch.mean(torch.stack(outputs), dim=0)
