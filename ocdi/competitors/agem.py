from torch import nn
import torch.nn.functional as F
from competitors import ERReservoir
import torch


class AGEM(ERReservoir):
    def __init__(self, net, buffer_size, input_size, options):
        super(AGEM, self).__init__(net, buffer_size, input_size)

    def agem_step(self, replay_x, replay_y, optimizer, criterion=None):
        if replay_x.size(0) > 0:
            params = [p for p in self.net.parameters() if p.requires_grad]

            # gradient was already computed using current batch (as in MIR)  #  TODO maybe size issue, current batch is 1 sample only
            grad = [p.grad.clone() for p in params]
            mem_logits = self.net(replay_x)
            if criterion is not None:
                loss_mem = criterion(mem_logits, replay_y)
            else:
                loss_mem = F.cross_entropy(mem_logits, replay_y, reduction='mean')

            optimizer.zero_grad()
            # compute gradient using memory samples
            loss_mem.backward()
            grad_ref = [p.grad.clone() for p in params]

            # inner product of grad and grad_ref
            prod = sum([torch.sum(g * g_r) for g, g_r in zip(grad, grad_ref)])
            if prod < 0:
                prod_ref = sum([torch.sum(g_r ** 2) for g_r in grad_ref])
                # do projection
                grad = [g - prod / prod_ref * g_r for g, g_r in zip(grad, grad_ref)]
            # replace params' grad
            for g, p in zip(grad, params):
                p.grad.data.copy_(g)
