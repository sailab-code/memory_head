import torch
from models.slca import SLCA, MHHeads

def get_model(model_name, args):
    name = model_name.lower()
    if 'slca' in name:
        return SLCA(args)
    elif 'mh' in name:
        return MHHeads(args)
    else:
        assert 0
