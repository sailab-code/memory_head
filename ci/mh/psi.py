import torch
import torch.nn.functional as F
import math


def psi(x, mode, key_size, normalize=True, query_class=None, memory_routing=None, routing_type=None,
        query_class_mappings=None, memory_id_mappings=None, input_for_keys=None, training=None,
        memory_id_propagation=None):

    if mode == "identity":
        o = x.flatten(start_dim=1)
    elif mode == "sign":
        o = torch.sign(x.flatten(start_dim=1))
    elif mode == "resize1d":
        o = resize1d(x, key_size)
    elif mode == "resize2d":
        o = resize2d(x, key_size)
    elif mode == "resize2d_sign":
        o = torch.sign(resize2d(x, key_size))
    else:
        raise NotImplementedError

    if query_class_mappings is not None:
        o = torch.cat((o, query_class_mappings[query_class]), dim=1)
        if training:
            o = torch.cat((o, query_class_mappings[query_class]), dim=1)
        else:
            o = torch.cat((o, torch.zeros((o.shape[0], query_class_mappings.shape[1]), device=o.device)), dim=1)

    if not memory_id_propagation:
        ##### Routing mode inside psi only when the id propagation is off

        if memory_routing and memory_routing == "top1":
            # TODO handle the case of shared_keys=False
            ids = memory_id_mappings[input_for_keys[..., 0].squeeze(1)].to(o.device)
            o = torch.cat((o, ids), dim=1)
        elif memory_routing and memory_routing == "topk":
            ids = memory_id_mappings[input_for_keys.squeeze(1)].flatten(1).to(o.device)
            o = torch.cat((o, ids), dim=1)
        elif memory_routing and memory_routing == "topk_binary" and routing_type == "onehot":
            ids = memory_id_mappings[input_for_keys.squeeze(1)].sum(dim=1).to(o.device)
            o = torch.cat((o, ids), dim=1)

    assert o.shape[1] == key_size, \
        "The selected psi function (" \
        + str(mode) + ") cannot map data to the target " \
                      "key_size (data_size: " + str(o.shape[1]) + ", key_size: " + str(key_size) + ")"
    if normalize:
        o = F.normalize(o, p=2.0, dim=1, eps=1e-12, out=None)
    return o


def resize1d(I, key_size):
    if I.shape[1] == key_size:
        pass
    else:
        I = F.interpolate(I.unsqueeze(1), size=key_size, mode="linear").squeeze(1)
    return I


def resize2d(I, key_size):
    b, c, h, w = I.shape
    spatial_key_size = key_size // c
    ratio = float(spatial_key_size) / float(w * h)
    w = int(round(math.sqrt(ratio) * w))
    h = spatial_key_size // w
    remainder = key_size - (c * h * w)
    o = F.interpolate(I, size=(h, w), mode="bilinear").flatten(start_dim=1)
    if h * w < spatial_key_size:
        o = torch.cat([o, torch.zeros((b, remainder), device=o.device, dtype=o.dtype)], dim=1)
    return o
