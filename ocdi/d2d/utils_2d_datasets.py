import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mh.layers import Linear as CMLinear
import wandb
import torch.nn.functional as F


# load a previously generated d2d dataset from file
def load_toy_dataset(file_name):
    loaded_data = np.load(file_name, allow_pickle=True)
    toy_dataset_dict = {
        'dataset': loaded_data['dataset'][()],
        'order': loaded_data['order'][()],
        'X_train': loaded_data['X_train'],
        'y_train': loaded_data['y_train'],
        'domain_train': loaded_data['domain_train'],
        'distributions_train': loaded_data['distributions_train'],
        'X_val': loaded_data['X_val'],
        'y_val': loaded_data['y_val'],
        'domain_val': loaded_data['domain_val'],
        'distributions_val': loaded_data['distributions_val'],
        'X_test': loaded_data['X_test'],
        'y_test': loaded_data['y_test'],
        'domain_test': loaded_data['domain_test'],
        'distributions_test': loaded_data['distributions_test']
    }
    d = toy_dataset_dict['X_train'].shape[1]  # input dimensionality
    c = len(toy_dataset_dict['dataset']['classes_ratio'])  # number of classes
    return toy_dataset_dict, d, c


# convert the d2d data (loaded from file) to pytorch tensors
def convert_toy_dataset_to_pytorch_tensors(loaded_toy_data, device):
    X_train, y_train, distributions_train = \
        torch.from_numpy(loaded_toy_data['X_train']).to(torch.float).to(device), \
        torch.from_numpy(loaded_toy_data['y_train']).to(torch.long).to(device), \
        torch.from_numpy(loaded_toy_data['distributions_train']).to(torch.long).to(device)
    X_val, y_val, distributions_val = \
        torch.from_numpy(loaded_toy_data['X_val']).to(torch.float).to(device), \
        torch.from_numpy(loaded_toy_data['y_val']).to(torch.long).to(device), \
        torch.from_numpy(loaded_toy_data['distributions_val']).to(torch.long).to(device)
    X_test, y_test, distributions_test = \
        torch.from_numpy(loaded_toy_data['X_test']).to(torch.float).to(device), \
        torch.from_numpy(loaded_toy_data['y_test']).to(torch.long).to(device), \
        torch.from_numpy(loaded_toy_data['distributions_test']).to(torch.long).to(device)
    return X_train, y_train, distributions_train, X_val, y_val, distributions_val, X_test, y_test, distributions_test


# classic accuracy in [0,1]
def accuracy(_o, _y):
    return torch.mean((torch.eq(_o, _y)).to(torch.float)).item()


# average accuracy
def avg_accuracy(_CCM):
    return torch.mean(_CCM[-1, :]).item()


# average forgetting
def avg_forgetting(_CCM):
    if _CCM.shape[0] == 1:
        return 0.
    else:
        _acc_star, _ = torch.max(_CCM[0:-1, 0:-1], dim=0)  # discarding last row and last column
        _acc = _CCM[-1, 0:-1]  # this is the last row, discarding the last column
        return torch.mean(_acc_star - _acc).item()


# positive backward transfer
def backward_transfer(_CCM):
    if _CCM.shape[0] == 1:
        return 0.
    else:
        _CCM_diff = _CCM - torch.diag(_CCM)
        _bwt = (2. * torch.sum(torch.tril(_CCM_diff, diagonal=-1))) / float(_CCM.shape[0] * (_CCM.shape[0] - 1))
        return max(_bwt.item(), 0)


# forward transfer
def forward_transfer(_CCM):
    if _CCM.shape[0] == 1:
        return 0.
    else:
        _fwd = (2. * torch.sum(torch.triu(_CCM, diagonal=1))) / float(_CCM.shape[0] * (_CCM.shape[0] - 1))
        return _fwd.item()


# create an empty continual confusion matrix given an array with distribution indices
def create_continual_confusion_matrix(distributions):
    distributions_count = torch.max(distributions) + 1
    return -torch.ones(distributions_count, distributions_count, dtype=torch.float)


# updating the continual confusion matrix (_CCM) given:
# - the index of the current distribution (_distribution) - i.e., the row index,
# - the current network (_net),
# - the data (_X: samples, _y: targets, _distributions: distribution indices)
def fill_row_in_continual_confusion_matrix(_CCM, _distribution, _net, _X, _y, _distributions, chunk_size=-1):
    if _distribution > _CCM.shape[0] - 1:
        return

    with torch.no_grad():
        _net.eval()
        if chunk_size == -1:
            _o = torch.argmax(_net(_X), dim=1)
        else:
            _o = []
            chunk_idx = 0
            while chunk_idx < _X.shape[0]:
                _o.append(torch.argmax(_net(_X[chunk_idx:min(chunk_idx+chunk_size, _X.shape[0])]), dim=1))
                chunk_idx+=chunk_size
            _o = torch.concat(_o, dim=0)
        _distributions_count = torch.max(_distributions) + 1

        for j in range(0, _distributions_count):
            _idx_distribution_j = _distributions == j
            _CCM[_distribution, j] = accuracy(_o[_idx_distribution_j], _y[_idx_distribution_j])

        _net.train()


# cut the continual confusion matrix up to the given distribution index (_distribution),
# which is automatically set to the last filled-up row index if not provided (if None)
def cut_continual_confusion_matrix(_CCM, _distribution=None):
    if _distribution > _CCM.shape[0] - 1:
        return _CCM

    if _distribution is None:
        _min, _idx = torch.min(_CCM[:, 0], dim=0, keepdim=False)
        _distribution = _CCM.shape[0] - 1 if _min != -1 else _idx
    return _CCM[0:(_distribution + 1), 0:(_distribution + 1)]


# print continual confusion matrix named _name
def print_continual_confusion_matrix(_name, _CCM):
    s = "CCM (" + _name + "):\n"
    s += "    M/D"
    for j in range(0, _CCM.shape[1]):
        s += " | {:3d}".format(j) + " "
    s += '\n'
    s += "    ----"
    for j in range(0, _CCM.shape[1]):
        s += "+------"
    s += '\n'
    for i in range(0, _CCM.shape[0]):
        s += "   {:3d}".format(i) + " "
        for j in range(0, _CCM.shape[1]):
            a = _CCM[i][j].item()
            s += " | {:.2f}".format(a) if a != -1 else " | N.A."
        s += '\n' if i < _CCM.shape[0] - 1 else ''
    print(s)


def create_metrics_dictionary(distributions):
    distributions_count = torch.max(distributions) + 1
    _metrics = {
        'distribution': torch.arange(0, distributions_count),
        'avg_accuracy': -torch.ones(distributions_count),
        'avg_forgetting': -torch.ones(distributions_count),
        'backward_transfer': -torch.ones(distributions_count),
        'forward_transfer': -torch.ones(distributions_count),
    }
    return _metrics


# update the _metrics dictionary with the information on the current distribution (_distribution)
def update_metrics(_metrics, _CCM, _distribution):
    if _distribution > _CCM.shape[0] - 1:
        return

    _CCM_cut = cut_continual_confusion_matrix(_CCM, _distribution)

    _metrics['avg_accuracy'][_distribution] = avg_accuracy(_CCM_cut)
    _metrics['avg_forgetting'][_distribution] = avg_forgetting(_CCM_cut)
    _metrics['backward_transfer'][_distribution] = backward_transfer(_CCM_cut)
    _metrics['forward_transfer'][_distribution] = forward_transfer(_CCM_cut)


# print metrics (named with string _name) computed right after having processed a given distribution (_distribution)
def print_metrics(_name, _metrics, _distribution):
    if _distribution > _metrics['distribution'][-1].item():
        _distribution = _metrics['distribution'][-1].item()

    s = "Metrics (" + _name + ")\tat distribution ID " + str(_distribution) + ": "
    s += "avg_accuracy={:.2f}".format(_metrics['avg_accuracy'][_distribution].item()) + ", "
    s += "avg_forgetting={:.2f}".format(_metrics['avg_forgetting'][_distribution].item()) + ", "
    s += "backward_transfer={:.2f}".format(_metrics['backward_transfer'][_distribution].item()) + ", "
    s += "forward_transfer={:.2f}".format(_metrics['forward_transfer'][_distribution].item())
    print(s)


def get_current_metrics(_name, _metrics, _distribution, _steps=False):
    dict_metr = {}
    if _distribution > _metrics['distribution'][-1].item():
        _distribution = _metrics['distribution'][-1].item()

    acc, forg, back, forw = f"{_name}/avg_accuracy", f"{_name}/avg_forgetting", f"{_name}/backward_transfer", \
                            f"{_name}/forward_transfer"

    # add distribution number to have separate metrics
    acc_s = acc + f"_{_distribution}"
    forg_s = forg + f"_{_distribution}"
    back_s = back + f"_{_distribution}"
    forw_s = forw + f"_{_distribution}"

    dict_metr[acc] = _metrics['avg_accuracy'][_distribution].item()
    dict_metr[acc_s] = _metrics['avg_accuracy'][_distribution].item()
    dict_metr[forg] = _metrics['avg_forgetting'][_distribution].item()
    dict_metr[forg_s] = _metrics['avg_forgetting'][_distribution].item()
    dict_metr[back] = _metrics['backward_transfer'][_distribution].item()
    dict_metr[back_s] = _metrics['backward_transfer'][_distribution].item()
    dict_metr[forw] = _metrics['forward_transfer'][_distribution].item()
    dict_metr[forw_s] = _metrics['forward_transfer'][_distribution].item()

    return dict_metr


# checking validity of the distribution indices (better be sure!):
# we assume that val and test distributions are either equal to the train ones or to an initial portion of it
def check_validity_of_distributions(distributions_train, distributions_val, distributions_test):
    device = distributions_train.device
    distributions_train_count = torch.max(distributions_train).item() + 1  # number of p(x|I_z) in train set
    distributions_val_count = torch.max(distributions_val).item() + 1  # number of p(x|I_z) in val set
    distributions_test_count = torch.max(distributions_test).item() + 1  # number of p(x|I_z) in test set

    assert distributions_train_count >= distributions_val_count, "Unexpected!"
    assert distributions_train_count >= distributions_test_count, "Unexpected!"
    assert torch.sum(torch.abs(distributions_train - torch.sort(distributions_train)[0])).item() == 0, "Unexpected!"
    assert torch.sum(torch.abs(distributions_val - torch.sort(distributions_val)[0])).item() == 0, "Unexpected!"
    assert torch.sum(torch.abs(distributions_test - torch.sort(distributions_test)[0])).item() == 0, "Unexpected!"
    assert torch.sum(torch.abs(torch.unique(distributions_train, sorted=True) -
                               torch.arange(0, distributions_train_count).to(device))).item() == 0, "Unexpected!"
    assert torch.sum(torch.abs(torch.unique(distributions_val, sorted=True) -
                               torch.arange(0, distributions_val_count).to(device))).item() == 0, "Unexpected!"
    assert torch.sum(torch.abs(torch.unique(distributions_test, sorted=True) -
                               torch.arange(0, distributions_test_count).to(device))).item() == 0, "Unexpected!"


# plot 2D data accordingly to their labels, using the label map to prepare a legend, and plot 2D space predictions
def plot_2d_data_and_predictions(title, data, labels, label_map, predictor=None, limits=None):
    if data.ndim > 2 or data.shape[1] != 2:
        return


    markers = ['o', 'x', '+', '^', '.', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', 'd']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#ff0000', '#0000ff',
              '#ffd700', '#ffc0cb', '#9acd32', '#a0522d', '#e6e6fa',
              '#800080', '#030764', '#ffa500', '#40e0d0', '#9acd32']
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)  # squared plots
    # fig.tight_layout()

    # plotting data, class-by-class
    for k, v in label_map.items():
        data_single_class = data[labels == v].cpu()
        if "K" in k:
            mark = '*'
            markersize = 25
        else:
            mark = markers[v % len(markers)]
            markersize = 10
        ax.plot(data_single_class[:, 0], data_single_class[:, 1],
                linestyle='none',
                marker=mark,
                label=k,
                markerfacecolor=colors[v % len(colors)],
                markersize=markersize)

    # ax.set_title(title, fontsi2ze=10)
    #ax.legend(bbox_to_anchor=(1.2, 1.12), ncol=5)
    ax.set_xticks([])
    ax.set_yticks([])

    # plotting background (predictions)
    if predictor is not None:

        predictor.eval()
        step_size = 0.01
        _min = torch.min(data, dim=0)[0] - 0.05
        _max = torch.max(data, dim=0)[0] + 0.05
        # x1_min = _min[0].item()
        # x1_max = _max[0].item()
        # x2_min = _min[1].item()
        # x2_max = _max[1].item()

        x1_min = -1.01
        x1_max = 1.01
        x2_min = -1.01
        x2_max = 1.01

        # limits given by keys too
        if limits is not None:
            x1_min = torch.min(_min[0], limits[0][0]).item()
            x1_max = torch.max(_max[0], limits[0][1]).item()
            x2_min = torch.min(_min[1], limits[1][0]).item()
            x2_max = torch.max(_max[1], limits[1][1]).item()

        xx1, xx2 = torch.meshgrid(torch.arange(x1_min, x1_max + step_size, step_size, dtype=torch.float32),
                                  torch.arange(x2_min, x2_max + step_size, step_size, dtype=torch.float32),
                                  indexing='ij')

        zz = predictor(torch.stack([xx1, xx2], dim=2).view(-1, 2).to(data.device))
        zz = torch.argmax(zz, dim=1).reshape(xx1.shape)  # warning: taking argmax here!

        _colors = []
        for j in range(0, len(label_map)):
            _colors.append(colors[j % len(colors)])

        plt.contourf(xx1.cpu(), xx2.cpu(), zz.cpu(), range(-1, len(label_map)), colors=_colors, alpha=0.25)
        plt.xlim([-1.01, 1.01])
        plt.ylim([-1.01, 1.01])

        predictor.train()
    else:
        plt.xlim([-1.1, 1.1])
        plt.ylim([-1.1, 1.1])
    return plt


def get_keys(net, model_name):
    # go over all the layers in the sequential
    keys_dict = {}
    keys_list = []
    for l_id, layer in enumerate(net):
        # access the modulelist of basicblocks
        # case cm_layer => one single basicblock
        if "mh" in model_name:
            if l_id > 0 and "_1" in model_name:  # TODO remove this ugly condition!
                break  # jump away for the second layer, it does not have d2d
            if "Cm" in type(layer).__name__:
                for bb_id, bbk in enumerate(layer.moduleList):
                    key = bbk.K.data.T.squeeze().cpu()  # put into the right format
                    title = f"keys_l_{l_id}_bb_{bb_id}"
                    keys_dict[title] = key
                    keys_list.append(key)

        else:

            if l_id > 0 and "_1" in model_name:  # TODO remove this ugly condition!
                break  # jump away for the second layer, it does not have d2d
            if "Cm" in type(layer).__name__:
                for bb_id, bbk in enumerate(layer.moduleList):
                    key = bbk.K.data.T.squeeze().cpu()  # put into the right format
                    title = f"keys_l_{l_id}_bb_{bb_id}"
                    keys_dict[title] = key
                    keys_list.append(key)

    all_keys = torch.cat(keys_list, dim=0)
    min_xcoord = torch.min(all_keys[:, 0])
    max_xcoord = torch.max(all_keys[:, 0])
    min_ycoord = torch.min(all_keys[:, 1])
    max_ycoord = torch.max(all_keys[:, 1])
    return all_keys, keys_dict, [[min_xcoord, max_xcoord], [min_ycoord, max_ycoord]]


def plot_keys_on_pred(plt_pred, k_list):
    plt_pred.plot(k_list[:, 0], k_list[:, 1], 'k*', markersize=12, label="Keys")
    plt.legend()
    return plt


def plot_ever_baseblock(k_dict):
    key_plots = {}
    for key, val in k_dict.items():
        fig, ax = plt.subplots()
        ax.plot(val[:, 0], val[:, 1], 'k*', markersize=12, label="Keys")
        ax.set_title(key, fontsize=10)
        plt.legend()
        key_plots[f"plots/{key}"] = wandb.Image(plt)
        # plt.show()
    return key_plots


class D2dModelFactory:
    @staticmethod
    def createModel(options, in_features, num_classes):
        model_name = options.model

        if model_name == "mlp":
            return nn.Sequential(
                nn.Linear(in_features=in_features, out_features=options.hidden, bias=options.bias),
                nn.Tanh(),
                nn.Linear(in_features=options.hidden, out_features=num_classes, bias=options.bias)
            )
        elif model_name == "lp":
            return nn.Sequential(
                nn.Linear(in_features=in_features, out_features=num_classes, bias=options.bias),
            )
        elif model_name == "mh":
            return torch.nn.Sequential(
                CMLinear(in_features, num_classes, shared_keys=options.shared_keys, bias=options.bias,
                         key_mem_units=options.key_mem_units,
                         psi_fn=options.psi, upd_m=options.upd_m, upd_k=options.upd_k, beta_k=options.beta_k,
                         key_size=options.key_size, gamma_alpha=options.gamma_alpha, tau_alpha=options.tau_alpha,
                         tau_mu=options.tau_mu,
                         tau_eta=options.tau_eta, scramble=options.scramble,
                         delta=options.delta)
            )
        else:
            raise NotImplementedError


def add_keys_from_continual_memory_neurons(net, X, y, label_map):
    assert isinstance(net, nn.Sequential), "This function expects the network to be implemented as torch.nn.Sequential!"
    K_list = []
    K_names = []
    c = len(label_map)

    for i, module in enumerate(net):
        if hasattr(module, 'K'):
            K = module.K
            n = K.shape[0]  # assumed to be the dim of the key matrices in this layer (1 per layer, or 1 per neuron)
            d = K.shape[1]
            assert d == 2, "This function assumes the 2nd dimension of the key tensor K is the length of each key, " \
                           "and here we can only handle keys of length 2 (i.e., 2-dimensional, for plotting purposes)"

            if n == 1:
                K_list.append(K.t())
                K_names.append('K Layer ' + str(i))
            else:
                for j in range(0, n):
                    K_list.append(K[j, :, :].t())
                    K_names.append('K Layer ' + str(i) + " Neuron " + str(j))

    if len(K_list) > 0:
        X = torch.cat([X] + K_list, dim=0)
        y = torch.cat([y] + torch.arange(c, c + len(K_list)), dim=0)
        for j, K_name in enumerate(K_names):
            label_map[K_name] = c - 1 + j

    return X, y, label_map


def add_keys_from_mh(net, X, y, label_map):
    if not isinstance(net, nn.Sequential):
        # handling wrapper in case of competitors
        net = net.net
    # assert isinstance(net, nn.Sequential), "This function expects the network to be implemented as torch.nn.Sequential!"
    K_list = []
    K_names = []
    c = len(label_map)
    y_list = []
    y_counter = c

    for i, module in enumerate(net):
        if hasattr(module, 'K'):
            K = module.K.detach()
            q = K.shape[0]  # assumed to be the dim of the key matrices in this layer (1 per layer, or 1 per neuron)
            m = K.shape[1]  # memory size
            d = K.shape[2]
            if d > 2:
                """
                print("This function assumes the 2nd dimension of the key tensor K is the length of each key, " \
                               "and here we can only handle keys of length 2 (i.e., 2-dimensional, for plotting purposes)")
                """
                pass
            else:
                for j in range(0, q):
                    K_list.append(K[j, :, :])
                    for mem in range(m):
                        # K_names.append('K Layer ' + str(i) + " Neuron " + str(j) + " Memory " + str(mem))
                        # K_names.append('K. l-' + str(i) + " n-" + str(j))
                        K_names.append('Keys')
                        y_list.append(y_counter)
                    # a different identifier for each neuron keys
                    label_map[K_names[-1]] = y_counter
                    y_counter += 1

    if len(K_list) > 0:
        X = torch.cat([X] + K_list, dim=0)
        y = torch.cat([y] + [torch.tensor(y_list).to(y.device)], dim=0)

    return X, y, label_map


def compute_loss(o, y, classes, loss, ):
    if loss == "xent":
        loss = nn.functional.cross_entropy(o, y, reduction='mean')
    elif loss == "hinge":
        target_ = (F.one_hot(y, num_classes=classes) * 2.0) - 1.0
        loss = torch.sum(torch.maximum(torch.zeros_like(o), -target_ * o))
    else:
        raise NotImplementedError
    return loss
