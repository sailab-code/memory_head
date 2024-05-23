'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers.weight_init import trunc_normal_
from timm.models.layers import Mlp
from copy import deepcopy
from mh.layers import Linear as MHLinear


class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=True))
        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))

        _fc = nn.Linear(self.embed_dim, nb_classes, bias=True)
        trunc_normal_(_fc.weight, std=.02)
        nn.init.constant_(_fc.bias, 0)
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False

        self.heads.append(new_head)

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(self.heads[ti](fc_inp))
        out = {'logits': torch.cat(out, dim=1)}
        return out


class SimpleMHLinear(SimpleContinualLinear):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False, args=None):
        # calling super of parent class, it needs to be an nn.Module
        super(SimpleContinualLinear, self).__init__()
        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        self.args = args
        heads = []
        single_head = []

        single_head.append(MHLinear(in_features=embed_dim, out_features=nb_classes, shared_keys=True, bias=True,
                                    key_mem_units=args["key_mem_units"],
                                    psi_fn="identity",
                                    upd_m=args["upd_m"],  # choices=["vanilla", "WTA"]
                                    upd_k=args["upd_k"],  # choices=["ad_hoc_WTA", "grad_WTA", "grad_not_WTA"]
                                    beta_k=args["beta_k"],
                                    gamma_alpha=args["gamma_alpha"],
                                    tau_alpha=args["tau_alpha"],
                                    tau_mu=args["tau_mu"],
                                    tau_eta=args["tau_eta"],
                                    scramble=args["scramble"],
                                    delta=args["delta"],
                                    layer_norm=args["layer_norm"],
                                    distance=args["distance"],  # choices=["cosine", "euclidean", "dot_scaled"]
                                    sigma_eu=args["sigma_eu"]
                                    ))

        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        args = self.args
        _fc = MHLinear(in_features=self.embed_dim, out_features=nb_classes, shared_keys=True, bias=True,
                       key_mem_units=args["key_mem_units"],
                       psi_fn="identity",
                       upd_m=args["upd_m"],  # choices=["vanilla", "WTA"]
                       upd_k=args["upd_k"],  # choices=["ad_hoc_WTA", "grad_WTA", "grad_not_WTA"]
                       beta_k=args["beta_k"],
                       gamma_alpha=args["gamma_alpha"],
                       tau_alpha=args["tau_alpha"],
                       tau_mu=args["tau_mu"],
                       tau_eta=args["tau_eta"],
                       scramble=args["scramble"],
                       delta=args["delta"],
                       layer_norm=args["layer_norm"],
                       distance=args["distance"],  # choices=["cosine", "euclidean", "dot_scaled"]
                       sigma_eu=args["sigma_eu"]
                       )

        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False
            for h in self.heads:
                # set to 0 the keys learning rate
                h[0].beta_k = 0.0
                h[0].eval()

        self.heads.append(new_head)
