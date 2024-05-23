import torch
import math
import torch.nn.functional as F
from collections.abc import Iterable
from mh.mhs import MHs


class Linear(MHs):
    def __init__(self, in_features, out_features, bias=True, device=None,
                 shared_keys=True, key_mem_units=2, psi_fn='identity', key_size=None, layer_norm=False, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.layer_norm = layer_norm
        if "memory_id_propagation" in kwargs:
            self.memory_id_propagation = kwargs["memory_id_propagation"]
            self.propagate = kwargs["propagate"]
            self.memory_id_routing = kwargs["memory_id_routing"]
            self.memory_id_mappings = kwargs["memory_id_mappings"]
            self.memory_id_type = kwargs["memory_id_type"]
        else:
            self.memory_id_propagation = None
            self.propagate = None
            self.memory_id_routing = None
            self.memory_id_mappings = None
            self.memory_id_type = None

        if kwargs is not None:
            assert 'q' not in kwargs, "The number of MHs is automatically determined, do not set argument 'q'"
            assert 'd' not in kwargs, "The size of each key can be specified with argument 'key_size', " \
                                      "do not set argument 'd'"
            assert 'm' not in kwargs, "The number of keys and memory units can be specified with argument " \
                                      "'key_mem_units', do not set argument 'm'"
            assert 'u' not in kwargs, "Size of each memory unit is automatically determined, do not set argument 'u'"

        # number of keys/memory units
        kwargs['m'] = key_mem_units

        # size of each key
        kwargs['d'] = in_features if key_size is None else key_size

        if "query_class_mappings" in kwargs and kwargs["query_class_mappings"] is not None:
            # increase key dimension when we use the class mappings concatenated to keys
            kwargs['d'] = kwargs['d'] + kwargs["query_class_mappings"].shape[1]

        if not self.memory_id_propagation:
            if "memory_id_routing" in kwargs and kwargs["memory_id_routing"] is not None:
                # increase key dimension when we use the memory ids concatenated to keys
                id_size = kwargs["memory_id_size"] if kwargs["memory_id_routing"] in ["top1", "topk_binary"] else \
                    kwargs["memory_id_size"] * \
                    kwargs["delta"]

                kwargs['d'] = kwargs['d'] + id_size

        # function used to compare input against keys
        kwargs['psi_fn'] = psi_fn

        if not shared_keys:
            # each neuron is an independent MH, with its own keys and its own memory units
            kwargs['q'] = self.out_features
            kwargs['u'] = self.in_features + (1 if self.bias else 0)
        else:
            # all the MHs of the layer share the same keys, thus their memory units are concatenated
            kwargs['q'] = 1
            kwargs['u'] = self.out_features * (self.in_features + (1 if self.bias else 0))
        kwargs['u'] += (2 * self.in_features) if self.layer_norm else 0

        # creating neurons
        super(Linear, self).__init__(**kwargs)

        # switching device
        if device is not None:
            self.to(device)

    def reset_parameters(self):
        super().reset_parameters()
        if self.layer_norm:
            torch.nn.init.ones_(self.M[:, :, -(2 * self.in_features):-self.in_features])
            torch.nn.init.zeros_(self.M[:, :, -self.in_features:])

    def forward(self, x, y=None):

        # init layer norm stuff
        G = None
        B = None

        # getting weights
        W = self.compute_weights(x, y)

        # unpacking layer normalizers
        if self.layer_norm:
            G = W[:, :, -(2 * self.in_features):-self.in_features]
            B = W[:, :, -self.in_features:]
            W = W[:, :, 0:-(2 * self.in_features)]

        # ensuring the shape is right (needed when neurons share the same keys)
        W = W.reshape((x.shape[0], self.out_features, -1))  # [b,q,1] => [b, out_features,(in_features + 1-if-bias)]

        # splitting into weights and biases
        if self.bias:
            weights = W[:, :, :-1]  # [b,out_features,in_features]
            bias = W[:, :, -1]  # [b,out_features]
        else:
            weights = W  # [b,out_features,in_features]
            bias = None

        if self.layer_norm:
            weights = G * weights
            if bias is not None:
                bias = bias + torch.sum(weights * B, dim=2)
            else:
                bias = torch.sum(weights * B, dim=2)
            x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.var(x, dim=1, correction=1, keepdim=True) + 1e-5)

        # batched linear projection: matmul([b,out_features,in_features], [b,in_features,1]) = [b,out_features,1]
        # that we squeeze to [b,out_features]
        o = torch.matmul(weights, x.unsqueeze(2)).squeeze(2)  # [b,out_features]
        if bias is not None:
            o += bias

        # concatenate memory id for routing propagation
        if self.propagate:
            o = self.concatenate_memory_ids(o)

        return o

    def concatenate_memory_ids(self, o):

        memory_ids = self.top_indices_bqk_hook
        maps = self.memory_id_mappings
        if self.memory_id_proproute == "top1":
            ids = maps[memory_ids[..., 0].squeeze(1)]
            o = torch.cat((o, ids), dim=1)
        elif self.memory_id_proproute == "topk":
            ids = maps[memory_ids.squeeze(1)].flatten(1)
            o = torch.cat((o, ids), dim=1)
        elif self.memory_id_proproute == "topk_binary" and self.memory_id_type == "onehot":
            ids = maps[memory_ids.squeeze(1)].sum(dim=1)
            o = torch.cat((o, ids), dim=1)
        return o

    def __str__(self):
        s = "- in_features = " + str(self.in_features) + "\n"
        s += "- out_features = " + str(self.out_features) + "\n"
        s += "- bias = " + str(self.bias) + "\n"
        return "[MH-based Linear Layer]\n" + s + super().__str__()
