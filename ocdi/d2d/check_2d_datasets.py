import os
import matplotlib.pyplot as plt
import numpy as np
from utils_2d_datasets import load_toy_dataset

# --------------------
# CUSTOMIZABLE OPTIONS
# --------------------
path = "generated_datasets"
progressively_plot_samples_in_CDID = False
# --------------------


# creating a list of files that end with .npz
files = []
for file in os.listdir(path):
    if file.endswith('.npz'):
    # if file.endswith('.npz'):
        files.append(path + "/" + file)

# for each data file...
for file in files:
    print("File: " + file)

    # loading data from file
    data, _, _ = load_toy_dataset(file)

    # unpacking (not really useful, sometimes I am just too lazy...)
    dataset_properties = data['dataset']
    # hack
    dataset_properties['scale_to_01'] = False

    order = data['order']
    X_train, y_train, X_val, y_val, X_test, y_test = \
        data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test']
    domain_train, domain_val, domain_test = data['domain_train'], data['domain_val'], data['domain_test']
    distributions_train, distributions_val, distributions_test = \
        data['distributions_train'], data['distributions_val'], data['distributions_test']

    # printing and plotting
    print("[TRAIN] Samples: " + str(X_train.shape[0]))
    print("[TRAIN] Class labels: " + str(np.unique(y_train)))
    plt.subplots(3, 3, figsize=(14, 9))
    plt.tight_layout()
    ax = plt.subplot(3, 3, 1)
    if dataset_properties['scale_to_01']:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    ax.set_title(dataset_properties['name'] + ", " + order + " (train): CLASS LABELS")
    print("[TRAIN] Domains: " + str(np.unique(domain_train)))
    ax = plt.subplot(3, 3, 2)
    if dataset_properties['scale_to_01']:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=domain_train)
    ax.set_title(dataset_properties['name'] + ", " + order + " (train): DOMAINS")
    print("[TRAIN] Distributions: " + str(np.unique(distributions_train)))
    ax = plt.subplot(3, 3, 3)
    if dataset_properties['scale_to_01']:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=distributions_train)
    ax.set_title(dataset_properties['name'] + ", " + order + " (train): DISTRIBUTIONS")
    print("[VAL] Samples: " + str(X_val.shape[0]))
    print("[VAL] Class labels: " + str(np.unique(y_val)))
    ax = plt.subplot(3, 3, 4)
    if dataset_properties['scale_to_01']:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val)
    ax.set_title(dataset_properties['name'] + ", " + order + " (val): CLASS LABELS")
    print("[VAL] Domains: " + str(np.unique(domain_val)))
    ax = plt.subplot(3, 3, 5)
    if dataset_properties['scale_to_01']:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    ax.scatter(X_val[:, 0], X_val[:, 1], c=domain_val)
    ax.set_title(dataset_properties['name'] + ", " + order + " (val): DOMAINS")
    print("[VAL] Distributions: " + str(np.unique(distributions_val)))
    ax = plt.subplot(3, 3, 6)
    if dataset_properties['scale_to_01']:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    ax.scatter(X_val[:, 0], X_val[:, 1], c=distributions_val)
    ax.set_title(dataset_properties['name'] + ", " + order + " (val): DISTRIBUTIONS")
    print("[TEST] Samples: " + str(X_test.shape[0]))
    print("[TEST] Class labels: " + str(np.unique(y_test)))
    ax = plt.subplot(3, 3, 7)
    if dataset_properties['scale_to_01']:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    ax.set_title(dataset_properties['name'] + ", " + order + " (test): CLASS LABELS")
    print("[TEST] Domains: " + str(np.unique(domain_test)))
    ax = plt.subplot(3, 3, 8)
    if dataset_properties['scale_to_01']:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=domain_test)
    ax.set_title(dataset_properties['name'] + ", " + order + " (test): DOMAINS")
    print("[TEST] Distributions: " + str(np.unique(distributions_test)))
    ax = plt.subplot(3, 3, 9)
    if dataset_properties['scale_to_01']:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=distributions_test)
    ax.set_title(dataset_properties['name'] + ", " + order + " (test): DISTRIBUTIONS")
    plt.show()

    # in the case of CDID data, we also progressively plot the samples
    if progressively_plot_samples_in_CDID:
        if file.endswith("CDID.npz"):
            for j in range(25, X_train.shape[0], 25):
                plt.scatter(X_train[0:j, 0], X_train[0:j, 1], c=y_train[0:j])
                if dataset_properties['scale_to_01']:
                    plt.xlim(-0.05, 1.05)
                    plt.ylim(-0.05, 1.05)
                plt.title(dataset_properties['name'] + ", " + order + " (train): CLASS LABELS - DATA ORDER")
                plt.show()
            for j in range(25, X_val.shape[0], 25):
                plt.scatter(X_val[0:j, 0], X_val[0:j, 1], c=y_val[0:j])
                if dataset_properties['scale_to_01']:
                    plt.xlim(-0.05, 1.05)
                    plt.ylim(-0.05, 1.05)
                plt.title(dataset_properties['name'] + ", " + order + " (val): CLASS LABELS - DATA ORDER")
                plt.show()
            for j in range(25, X_test.shape[0], 25):
                plt.scatter(X_test[0:j, 0], X_test[0:j, 1], c=y_test[0:j])
                if dataset_properties['scale_to_01']:
                    plt.xlim(-0.05, 1.05)
                    plt.ylim(-0.05, 1.05)
                plt.title(dataset_properties['name'] + ", " + order + " (test): CLASS LABELS - DATA ORDER")
                plt.show()
