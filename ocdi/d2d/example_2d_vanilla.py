import torch
import torch.nn as nn
from utils_2d_datasets import load_toy_dataset, fill_row_in_continual_confusion_matrix, \
    print_continual_confusion_matrix, update_metrics, print_metrics, check_validity_of_distributions, \
    create_metrics_dictionary, convert_toy_dataset_to_pytorch_tensors, create_continual_confusion_matrix, \
    plot_2d_data_and_predictions, add_keys_from_continual_memory_neurons

# --------------------
# CUSTOMIZABLE OPTIONS
# --------------------
device = torch.device('cpu')
# file = "generated_datasets/bi-modals_IID.npz"
file = "generated_datasets/bi-modals_CI.npz"
# file = "generated_datasets/bi-modals_CDI.npz"
# file = "generated_datasets/bi-modals_CDID.npz"
# file = "generated_datasets/bi-moons_IID.npz"
# file = "generated_datasets/bi-moons_CI.npz"
# file = "generated_datasets/bi-moons_CDI.npz"
# file = "generated_datasets/bi-moons_CDID.npz"
# --------------------

# loading data
loaded_data, d, c = load_toy_dataset(file)  # loading data (returns: data dictionary, input size, num classes)

# converting data collections (train, val, test) to torch tensors, right format, right device
X_train, y_train, distributions_train, X_val, y_val, distributions_val, X_test, y_test, distributions_test = \
    convert_toy_dataset_to_pytorch_tensors(loaded_data, device)

# checking validity of the distribution indices (better be sure!)
check_validity_of_distributions(distributions_train, distributions_val, distributions_test)

# creating the neural network
net = nn.Sequential(
    nn.Linear(in_features=d, out_features=10, bias=True),
    nn.Tanh(),
    nn.Linear(in_features=10, out_features=c, bias=True)
).to(device)

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.09)

# continual confusion matrices (empty, -1 means "unfilled yet", important!)
CCM_train = create_continual_confusion_matrix(distributions_train)
CCM_val = create_continual_confusion_matrix(distributions_val)
CCM_test = create_continual_confusion_matrix(distributions_test)

# metrics containers (empty, -1 means "unfilled yet", important!)
metrics_train = create_metrics_dictionary(distributions_train)
metrics_val = create_metrics_dictionary(distributions_val)
metrics_test = create_metrics_dictionary(distributions_test)

# streaming loop
net.train()
for t, (x, y, distribution) in enumerate(zip(X_train, y_train, distributions_train)):

    # adding batch size and clearing up the tensor box
    x.unsqueeze_(0)
    y.unsqueeze_(0)
    distribution = distribution.item()

    # learning step
    o = net(x)
    loss = nn.functional.cross_entropy(o, y, reduction='mean')  # no real reductions are actually performed...
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # printing
    print('t=' + str(t) + ', loss=' + str(loss.item()))

    # metrics (computed when training ends or when the distribution is going to change on the next step)
    if t == X_train.shape[0] - 1 or distributions_train[t + 1] != distribution:

        # adding a new row to the continual confusion matrices (exploiting the current network)
        fill_row_in_continual_confusion_matrix(CCM_train, distribution, net, X_train, y_train, distributions_train)
        fill_row_in_continual_confusion_matrix(CCM_val, distribution, net, X_val, y_val, distributions_val)
        fill_row_in_continual_confusion_matrix(CCM_test, distribution, net, X_test, y_test, distributions_test)

        # updating metrics
        update_metrics(metrics_train, CCM_train, distribution)
        update_metrics(metrics_val, CCM_val, distribution)
        update_metrics(metrics_test, CCM_test, distribution)

        # printing continual confusion matrix
        print_continual_confusion_matrix('train', CCM_train)
        print_continual_confusion_matrix('val', CCM_val)
        print_continual_confusion_matrix('test', CCM_test)

        # printing metrics
        print_metrics('train', metrics_train, distribution)
        print_metrics('val', metrics_val, distribution)
        print_metrics('test', metrics_test, distribution)

# class dictionary
class_dict = {'Class ' + str(j): j for j in range(0, c)}

# getting keys, if any, adding them to the data, with an ad-hoc label
X_train, y_train, class_dict = add_keys_from_continual_memory_neurons(net, X_train, y_train, class_dict)

# plotting predictions at the end of training
plot_2d_data_and_predictions('train', X_train, y_train, class_dict, net).show()
