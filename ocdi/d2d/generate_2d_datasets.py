import sklearn.datasets as dt
import numpy as np
import os

# --------------------
# CUSTOMIZABLE OPTIONS
# --------------------
import torch

save_to = "generated_datasets"  # sub-folder were data will be saved
n = 1000  # number of training samples (validation and test samples will be approx 1/4 and 1/2 of it, respectively)
scale_to_m1p1 = True  # if the data should be scaled in [-1, 1] (usually yes!)

# properties of the d2d data sets to generate
datasets = [
    {'name': 'bi-modals',
     'classes_ratio': [0.2, 0.4, 0.1, 0.3],
     'centers1': np.array([[0.25, 0.25], [0.6, 0.65], [0.15, 0.5], [0.85, 0.3]]),
     'std1': np.array([0.02, 0.08, 0.015, 0.03]),
     'centers2': np.array([[0.2, 0.75], [1.0, 0.1], [1.15, 0.75], [0.3, 0.5]]),
     'std2': np.array([0.04, 0.02, 0.06, 0.015]),
     'seeds1': [1234, 5678, 9101112, 13141516],
     'seeds2': [1, 5, 9, 13],
     'n': n,
     'scale_to_m1p1': True},
    {'name': 'bi-moons',
     'classes_ratio': [0.4, 0.1, 0.3, 0.2],
     'offsets1': np.array([[0.0, 0.0], [-3.0, 1.0]]),
     'offsets2': np.array([[2.0, 1.0], [3.0, -1.0]]),
     'std1': np.array([0.08, 0.06]),
     'std2': np.array([0.04, 0.03]),
     'seeds1': [1234, 5678],
     'seeds2': [1, 5],
     'n': n,
     'scale_to_m1p1': True}
]
# --------------------

# UNUSED
old_datasets_not_used_anymore = [
    {'name': 'single-modes',
     'classes_ratio': [0.2, 0.4, 0.1, 0.3],
     'centers': np.array([[0.25, 0.25], [0.6, 0.65], [0.15, 0.5], [0.85, 0.3]]),
     'std': np.array([0.02, 0.08, 0.015, 0.03]),
     'seeds': [1234, 5678, 9101112, 13141516],
     'n': n,
     'scale_to_m1p1': True},
    {'name': 'single-moons',
     'classes_ratio': [0.4, 0.1, 0.3, 0.2],
     'offsets': np.array([[0.0, 0.0], [2.0, 2.0]]),
     'std': np.array([0.08, 0.06]),
     'seeds': [1234, 5678],
     'n': n,
     'scale_to_m1p1': True}
]


def class_incremental(_X, _y, _domain, incremental_domain_too=False):
    _distributions = None

    if not incremental_domain_too:
        _idx = np.argsort(_y)
        _X = np.take_along_axis(_X, _idx.reshape(_idx.shape[0], 1), axis=0)
        _y = np.take_along_axis(_y, _idx, axis=0)
        _domain = np.zeros_like(_domain)
        _distributions = compute_distributions_markers_in_domain_and_class_incremental(_y, None)
    else:
        _idx = np.argsort(_domain)
        _X = np.take_along_axis(_X, _idx.reshape(_idx.shape[0], 1), axis=0)
        _y = np.take_along_axis(_y, _idx, axis=0)
        _domain = np.take_along_axis(_domain, _idx, axis=0)

        __y_ = []
        __X_ = []
        __domain_ = []
        for ii in range(0, np.max(_domain) + 1):
            sub = _domain == ii
            __y = _y[sub]
            __X = _X[sub, :]
            __domain = _domain[sub]
            _idx = np.argsort(__y)
            __X = np.take_along_axis(__X, _idx.reshape(_idx.shape[0], 1), axis=0)
            __y = np.take_along_axis(__y, _idx, axis=0)
            __domain = np.take_along_axis(__domain, _idx, axis=0)
            __y_.append(__y)
            __X_.append(__X)
            __domain_.append(__domain)

        _X = np.concatenate(__X_, axis=0)
        _y = np.concatenate(__y_)
        _domain = np.concatenate(__domain_)
        _distributions = compute_distributions_markers_in_domain_and_class_incremental(_y, _domain)
    return _X, _y, _domain, _distributions


def shuffle(_X, _y, _domain):
    np.random.seed(1234)
    shuffler = np.random.permutation(_X.shape[0])
    _X = _X[shuffler, :]
    _y = _y[shuffler]
    _domain = _domain[shuffler]
    return _X, _y, _domain


def iid(_X, _y, _domain):
    _X, _y, _ = shuffle(_X, _y, _domain)
    _domain = np.zeros_like(_y)
    _distributions = np.zeros_like(_y)
    return _X, _y, _domain, _distributions


def compute_distributions_markers_in_domain_and_class_incremental(_y, _domain):
    _distributions = np.zeros_like(_y)
    kk = 0
    for jj in range(0, _distributions.shape[0]):
        if jj > 0:
            if (_y is not None and _y[jj] != _y[jj - 1]) or (_domain is not None and _domain[jj] != _domain[jj - 1]):
                kk += 1
        _distributions[jj] = kk
    return _distributions


def dependent_sampling_within_each_distribution(_X, _y, _domain, _distributions):
    __y_ = []
    __X_ = []
    __domain_ = []

    for ii in range(0, np.max(_distributions) + 1):
        sub = _distributions == ii
        __y = _y[sub]
        __X = _X[sub, :]
        __domain = _domain[sub]
        _idx = np.argsort(__X[:, 0])  # sorting w.r.t. first dimension
        __X = np.take_along_axis(__X, _idx.reshape(_idx.shape[0], 1), axis=0)
        __y = np.take_along_axis(__y, _idx, axis=0)
        __domain = np.take_along_axis(__domain, _idx, axis=0)
        __y_.append(__y)
        __X_.append(__X)
        __domain_.append(__domain)

    _X = np.concatenate(__X_, axis=0)
    _y = np.concatenate(__y_)
    _domain = np.concatenate(__domain_)
    return _X, _y, _domain, _distributions


# UNUSED
def domain_incremental(_X, _y, _domain):
    _idx = np.argsort(_domain)
    _X = np.take_along_axis(_X, _idx.reshape(_idx.shape[0], 1), axis=0)
    _y = np.take_along_axis(_y, _idx, axis=0)
    _domain = np.take_along_axis(_domain, _idx, axis=0)
    _distributions = compute_distributions_markers_in_domain_and_class_incremental(None, _domain)
    return _X, _y, _domain, _distributions


# UNUSED
def task_incremental(_X, _y, _domain, incremental_domain_too=False):
    _X, _y, _domain, _ = class_incremental(_X, _y, _domain, incremental_domain_too=incremental_domain_too)

    __y_ = []
    __X_ = []
    __domain_ = []
    __distributions_ = []
    kk = 0

    if not incremental_domain_too:
        cc = np.unique(_y)
        c = cc.size
        for _c in range(0, c):
            _i = cc[_c]
            _i_next = cc[_c + 1 % c]
            idx1 = np.nonzero(_y == _i)[0]
            idx2 = np.nonzero(_y == _i_next)[0]
            idx1 = idx1[idx1.shape[0] // 2:]
            idx2 = idx2[0:idx2.shape[0] // 2]
            _idx = np.concatenate([idx1, idx2])
            __X = _X[_idx, :]
            __y = _y[_idx]
            __domain = _domain[_idx]
            __distributions = np.ones_like(__y) * kk
            kk += 1
            __y_.append(__y)
            __X_.append(__X)
            __domain_.append(__domain)
            __distributions_.append(__distributions)
    else:
        _idx = np.argsort(_domain)
        _X = np.take_along_axis(_X, _idx.reshape(_idx.shape[0], 1), axis=0)
        _y = np.take_along_axis(_y, _idx, axis=0)
        _domain = np.take_along_axis(_domain, _idx, axis=0)

        for ii in range(0, np.max(_domain) + 1):
            sub = _domain == ii
            __y = _y[sub]
            __X = _X[sub, :]
            __domain = _domain[sub]
            cc = np.unique(__y)
            c = cc.size
            for _c in range(0, c):
                _i = cc[_c]
                _i_next = cc[_c + 1 % c]
                idx1 = np.nonzero(__y == _i)[0]
                idx2 = np.nonzero(__y == _i_next)[0]
                idx1 = idx1[idx1.shape[0] // 2:]
                idx2 = idx2[0:idx2.shape[0] // 2]
                _idx = np.concatenate([idx1, idx2])
                __Xc = __X[_idx, :]
                __yc = __y[_idx]
                __domainc = __domain[_idx]
                __distributionsc = np.ones_like(__yc) * kk
                kk += 1
                __y_.append(__yc)
                __X_.append(__Xc)
                __domain_.append(__domainc)
                __distributions_.append(__distributionsc)

    _X = np.concatenate(__X_, axis=0)
    _y = np.concatenate(__y_)
    _domain = np.concatenate(__domain_)
    _distributions = np.concatenate(__distributions_)
    return _X, _y, _domain, _distributions


# generating data
for d in datasets:
    XX = []
    yy = []
    domain = []
    n2 = n * 2
    nn = 0

    if d['name'] == 'single-modes':  # UNUSED
        for i in range(0, len(d['classes_ratio'])):
            nc = int(n2 * d['classes_ratio'][i]) if i < len(d['classes_ratio']) - 1 else n2 - nn
            nn += nc
            X, _ = dt.make_blobs(n_samples=nc, n_features=2,
                                 centers=d['centers'][None, i,:], cluster_std=d['std'][i],
                                 return_centers=False, shuffle=False, random_state=d['seeds'][i])
            XX.append(X)
            domain.append(np.zeros(XX[-1].shape[0], dtype=np.int64))
            yy.append(np.ones(nc, dtype=np.int64) * i)

    if d['name'] == 'bi-modals':
        for i in range(0, len(d['classes_ratio'])):
            nc = int(n2 * d['classes_ratio'][i]) if i < len(d['classes_ratio']) - 1 else n2 - nn
            nn += nc
            X, _ = dt.make_blobs(n_samples=nc // 2, n_features=2,
                                 centers=d['centers1'][None, i, :], cluster_std=d['std1'][i],
                                 return_centers=False, shuffle=False, random_state=d['seeds1'][i])
            XX.append(X)
            domain.append(np.zeros(XX[-1].shape[0], dtype=np.int64))
            X, _ = dt.make_blobs(n_samples=nc - (nc // 2), n_features=2,
                                 centers=d['centers2'][None, i, :], cluster_std=d['std2'][i],
                                 return_centers=False, shuffle=False, random_state=d['seeds2'][i])
            XX.append(X)
            domain.append(np.ones(XX[-1].shape[0], dtype=np.int64))
            yy.append(np.ones(nc, dtype=np.int64) * i)

    if d['name'] == 'single-moons':  # UNUSED
        for i in range(0, len(d['classes_ratio']), 2):
            nc1 = int(n2 * d['classes_ratio'][i])
            nc2 = int(n2 * d['classes_ratio'][i + 1]) if i < len(d['classes_ratio']) - 1 else n2 - nn - nc1
            nn += nc1 + nc2
            X, _ = dt.make_moons(n_samples=(nc1, nc2), noise=d['std'][i // 2],
                                 shuffle=False, random_state=d['seeds'][i // 2])
            XX.append(X + d['offsets'][None, i // 2, :])
            domain.append(np.zeros(XX[-1].shape[0], dtype=np.int64))
            yy.append(np.ones(nc1, dtype=np.int64) * i)
            yy.append(np.ones(nc2, dtype=np.int64) * (i + 1))

    if d['name'] == 'bi-moons':
        for i in range(0, len(d['classes_ratio']), 2):
            nc1 = int(n2 * d['classes_ratio'][i])
            nc2 = int(n2 * d['classes_ratio'][i + 1]) if i < len(d['classes_ratio']) - 1 else n2 - nn - nc1
            X, _ = dt.make_moons(n_samples=(nc1 // 2, nc2 // 2), noise=d['std1'][i // 2],
                                 shuffle=False, random_state=d['seeds1'][i // 2])
            XX.append(X + d['offsets1'][None, i // 2, :])
            domain.append(np.zeros(XX[-1].shape[0], dtype=np.int64))
            X, _ = dt.make_moons(n_samples=(nc1 - (nc1 // 2), nc2 - (nc2 // 2)), noise=d['std2'][i // 2],
                                 shuffle=False, random_state=d['seeds2'][i // 2])
            XX.append(X + d['offsets2'][None, i // 2, :])
            domain.append(np.ones(XX[-1].shape[0], dtype=np.int64))
            yy.append(np.ones(nc1 // 2, dtype=np.int64) * i)
            yy.append(np.ones(nc2 // 2, dtype=np.int64) * (i + 1))
            yy.append(np.ones(nc1 - (nc1 // 2), dtype=np.int64) * i)
            yy.append(np.ones(nc2 - (nc2 // 2), dtype=np.int64) * (i + 1))

    # merging and eventually rescaling
    X = np.concatenate(XX, axis=0)
    if scale_to_m1p1:
        X = X - np.min(X, axis=0)
        X = X / np.max(X, axis=0)
        X = 2.*X - 1.
    domain = np.concatenate(domain)
    y = np.concatenate(yy)

    # splitting into training, validation, and test sets (interleaved data - the validation set will be reduced later)
    X_train_original = X[0:2*n-1:2, :]
    y_train_original = y[0:2*n-1:2]
    domain_train_original = domain[0:2*n-1:2]
    X_val_test = X[1:2*n:2, :]
    y_val_test = y[1:2*n:2]
    domain_val_test = domain[1:2*n:2]
    m = X_val_test.shape[0]
    X_val_original = X_val_test[0:m-1:2, :]
    y_val_original = y_val_test[0:m-1:2]
    domain_val_original = domain_val_test[0:m-1:2]
    X_test_original = X_val_test[1:m:2, :]
    y_test_original = y_val_test[1:m:2]
    domain_test_original = domain_val_test[1:m:2]

    # shuffling (not so needed...anyhow...)
    X_train_original, y_train_original, domain_train_original = \
        shuffle(X_train_original, y_train_original, domain_train_original)
    X_val_original, y_val_original, domain_val_original = \
        shuffle(X_val_original, y_val_original, domain_val_original)
    X_test_original, y_test_original, domain_test_original = \
        shuffle(X_test_original, y_test_original, domain_test_original)

    # CI, CDI, CDID
    CDID_done = False
    for _incr_domain in [False, True, True]:
        X_train, y_train, domain_train, distributions_train = \
            class_incremental(X_train_original, y_train_original, domain_train_original,
                              incremental_domain_too=_incr_domain)
        X_val, y_val, domain_val, distributions_val = \
            class_incremental(X_val_original, y_val_original, domain_val_original,
                              incremental_domain_too=_incr_domain)
        X_test, y_test, domain_test, distributions_test = \
            class_incremental(X_test_original, y_test_original, domain_test_original,
                              incremental_domain_too=_incr_domain)
        if _incr_domain:
            if not CDID_done:
                CDID_done = True
                X_train, y_train, domain_train, distributions_train = \
                    dependent_sampling_within_each_distribution(X_train, y_train, domain_train, distributions_train)
                X_val, y_val, domain_val, distributions_val = \
                    dependent_sampling_within_each_distribution(X_val, y_val, domain_val, distributions_val)
                X_test, y_test, domain_test, distributions_test = \
                    dependent_sampling_within_each_distribution(X_test, y_test, domain_test, distributions_test)
                order = 'CDID'
            else:
                order = 'CDI'
        else:
            order = 'CI'

        # cutting validation set over time
        idx = distributions_val == -1
        distributions_count = np.max(distributions_val) + 1
        for z in range(0, distributions_count // 2):
            idx = np.logical_or(idx, distributions_val == z)  # 50% of the distributions are used for validation
        X_val = X_val[idx, :]
        y_val = y_val[idx]
        domain_val = domain_val[idx]
        distributions_val = distributions_val[idx]

        # saving to disk
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        with open(save_to + "/" + d['name'] + '_' + order + '.npz', "wb+") as f:
            np.savez(f,
                     dataset=d,
                     order=order,
                     X_train=X_train, y_train=y_train, domain_train=domain_train,
                     distributions_train=distributions_train,
                     X_val=X_val, y_val=y_val, domain_val=domain_val,
                     distributions_val=distributions_val,
                     X_test=X_test, y_test=y_test, domain_test=domain_test,
                     distributions_test=distributions_test)

    # I.I.D.
    order = 'IID'
    X_train, y_train, domain_train, distributions_train = \
        iid(X_train_original, y_train_original, domain_train_original)
    X_val, y_val, domain_val, distributions_val = \
        iid(X_val_original, y_val_original, domain_val_original)
    X_test, y_test, domain_test, distributions_test = \
        iid(X_test_original, y_test_original, domain_test_original)

    # cutting validation set size to be coherent with the other validation sets
    idx = np.arange(0, (X_val.shape[0] // 2))  # 50% of random shuffled data
    X_val = X_val[idx, :]
    y_val = y_val[idx]
    domain_val = domain_val[idx]
    distributions_val = distributions_val[idx]

    # saving to disk
    with open(save_to + "/" + d['name'] + '_' + order + '.npz', "wb+") as f:
        np.savez(f,
                 dataset=d,
                 order=order,
                 X_train=X_train, y_train=y_train, domain_train=domain_train,
                 distributions_train=distributions_train,
                 X_val=X_val, y_val=y_val, domain_val=domain_val,
                 distributions_val=distributions_val,
                 X_test=X_test, y_test=y_test, domain_test=domain_test,
                 distributions_test=distributions_test)
