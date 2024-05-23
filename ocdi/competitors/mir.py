import torch
from competitors import ERReservoir
import numpy as np
import copy
import torch.nn.functional as F


class MIR(ERReservoir):
    def __init__(self, net, buffer_size, input_size, options):
        super(MIR, self).__init__(net, buffer_size, input_size)
        self.beta_m = options.beta_m
        # label_buffer
        # sample_buffer
        # buffer_size

    def retrieve_buffer(self, num_retrieve=None, return_indices=False, optimizer=None, current_loss=None):

        possible_retrieve = min(num_retrieve, self.current_index)
        if possible_retrieve == 0:
            return None, None

        sub_x, sub_y = super().retrieve_buffer(num_retrieve,
                                               return_indices)
        grad_dims = []
        for param in self.net.parameters():
            grad_dims.append(param.data.numel())
        # gather all gradients in a single vector. Notice!!! The gradients are the new ones computed on the new batch!
        grad_vector = self.get_grad_vector(self.net.parameters, grad_dims)
        # create a virtual model updating its parameters with the gradients computed before
        model_temp = self.get_future_step_parameters(self.net, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            with torch.no_grad():
                logits_pre = self.net(sub_x)
                logits_post = model_temp.forward(sub_x)
                pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')  # TODO same  reduction as main?
                post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
                scores = post_loss - pre_loss
                big_ind = scores.sort(descending=True)[1][:num_retrieve]
            return sub_x[big_ind], sub_y[big_ind]
        else:
            return sub_x, sub_y

    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.beta_m * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1

    def get_grad_vector(self, pp, grad_dims):
        """
            gather the gradients in one vector
        """
        grads = torch.Tensor(sum(grad_dims))
        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if cnt == 0:
                grads.to(param.device)
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(param.grad.data.view(-1))
            cnt += 1
        return grads
