import torch
import torch.nn as nn
import d2d.utils_2d_datasets
from d2d.utils_2d_datasets import load_toy_dataset, fill_row_in_continual_confusion_matrix, \
    print_continual_confusion_matrix, update_metrics, print_metrics, check_validity_of_distributions, \
    create_metrics_dictionary, convert_toy_dataset_to_pytorch_tensors, create_continual_confusion_matrix, \
    plot_2d_data_and_predictions, plot_keys_on_pred, get_keys, plot_ever_baseblock, get_current_metrics, compute_loss
import matplotlib.pyplot as plt
import wandb
from d2d.utils_2d_datasets import D2dModelFactory
from torchinfo import summary
import argparse
from torch.optim import SGD, Adam
import time
import random
import numpy as np
from packaging import version
from tqdm import tqdm
import torch.nn.functional as F
from competitors.utils import CompetitorFactory

import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CMM experiments')
    parser.add_argument('--model', type=str, default="mlp", choices=["lp", "mlp", "mh"])
    parser.add_argument('--bias', type=str, default="true")
    parser.add_argument('--watch', type=str, default="false")
    parser.add_argument('--hidden', type=int, default=10)
    parser.add_argument('--benchmark', type=str, default="bi-modals_CDI", choices=[
                                                                                   "bi-modals_CDI",
                                                                                   "imagenet"
                                                                                   ])
    parser.add_argument('--beta_m', type=float, default=0.095)
    parser.add_argument('--beta_k', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--loss', type=str, default="xent", choices=["xent", "hinge"])
    parser.add_argument('--wandb', type=str, default="false")
    parser.add_argument('--key_mem_units', type=int, default=16)
    parser.add_argument('--delta', type=int, default=3)
    parser.add_argument('--psi', type=str, default="identity",
                        choices=["identity", "sign", ])
    parser.add_argument('--key_size', type=int, default=None)
    parser.add_argument('--gamma_alpha', type=float, default=1.0)
    parser.add_argument('--tau_alpha', type=float, default=1.0)
    parser.add_argument('--tau_mu', type=float, default=None)
    parser.add_argument('--tau_eta', type=float, default=None)
    parser.add_argument('--upd_m', type=str, default="vanilla", choices=["vanilla", "WTA"])
    parser.add_argument('--upd_k', type=str, default="ad_hoc_WTA", choices=["ad_hoc_WTA", "grad_WTA"])
    parser.add_argument('--scramble', type=str, default="false")
    parser.add_argument('--shared_keys', type=str, default="false")
    parser.add_argument('--draw_plots', type=str, default="false")
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--competitor', type=str,
                        default="None")  # ER_reservoir, ER-random, GDumb, Ensemble, MIR, AGEM, GSS
    parser.add_argument('--buffer_size', type=int, default=None)
    parser.add_argument('--buffer_batch_size', type=int, default=None)
    parser.add_argument('--gdumb_epochs', type=int, default=None)  # GDumb
    parser.add_argument('--ensembled_models', type=int, default=None)
    parser.add_argument('--eval_chunk_size', type=int, default=10000)

    args_cmd = parser.parse_args()
    args_cmd.wandb = args_cmd.wandb in ["true", "True"]
    args_cmd.bias = args_cmd.bias in ["true", "True"]
    args_cmd.scramble = args_cmd.scramble in ["true", "True"]
    args_cmd.draw_plots = args_cmd.draw_plots in ["true", "True"]
    args_cmd.shared_keys = args_cmd.shared_keys in ["true", "True"]
    args_cmd.watch = args_cmd.watch in ["true", "True"]
    if args_cmd.competitor in ["none", "None"]:
        args_cmd.competitor = None
    args_cmd.upd_m = None if args_cmd.upd_m == "vanilla" else "WTA"
    device = args_cmd.device

    # filtering out not admissible configurations (only mh)

    if "mh" in args_cmd.model:
        assert args_cmd.upd_m is None or (args_cmd.upd_m == 'WTA' and args_cmd.upd_k is not None), \
            "If upd_m is 'WTA', then upd_k must be ad_hoc_WTA or grad_WTA (it cannot be None)"
        # scramble == false when upd_k is not ad_hoc_WTA
        assert args_cmd.upd_k == "ad_hoc_WTA" or (
                args_cmd.scramble is False), "If upd_k is not ad_hoc_WTA, then scramble has not effect!"

    ##### setting seeds

    seed = int(time.time()) if args_cmd.seed < 0 else int(args_cmd.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # enforcing a deterministic behaviour, when possible
    py_version = torch.__version__
    if version.parse(py_version) > version.parse("1.11"):
        print(f"Pytorch version: {py_version};  compatible with deterministic algohrithms!")
        torch.use_deterministic_algorithms(True, warn_only=True)

    if args_cmd.benchmark != 'imagenet':

        file = f"d2d/generated_datasets/{args_cmd.benchmark}.npz"
        # loading data
        loaded_data, d, c = load_toy_dataset(file)  # loading data (returns: data dictionary, input size, num classes)

        # converting data collections (train, val, test) to torch tensors, right format, right device
        X_train, y_train, distributions_train, X_val, y_val, distributions_val, X_test, y_test, distributions_test = \
            convert_toy_dataset_to_pytorch_tensors(loaded_data, device)

    else:
        from datasets.mini_imagenet_CDID import Mini_ImageNet_CDID
        # Load data
        dataset_params = {'dataset_path': 'datasets/mini_imagenet/',
                          'perturbation_params': [None,
                                                  {'type': 'gaussian', 'noise_factor': 0.25, 'sig': 0.1},
                                                  {'type': 'gaussian', 'noise_factor': 0.5, 'sig': 0.2}]

                          }
        ds = Mini_ImageNet_CDID(dataset_params, debug=False, save_cache=True, load_cache=False, device=device,
                                quick_load=False)

        d = 512
        c = ds.num_classes

        loaded_data = ds.get_train_val_test(val_ratio=0.1, test_ratio=0.2, num_val_distributions=0.5, use_resnet=True)

        # flatten images
        X_train = loaded_data['X_train']
        X_val = loaded_data['X_val']
        X_test = loaded_data['X_test']
        y_train = loaded_data['y_train']
        y_val = loaded_data['y_val']
        y_test = loaded_data['y_test']
        distributions_train = loaded_data['distributions_train']
        distributions_val = loaded_data['distributions_val']
        distributions_test = loaded_data['distributions_test']

    # checking validity of the distribution indices (better be sure!)
    check_validity_of_distributions(distributions_train, distributions_val, distributions_test)

    net = D2dModelFactory.createModel(options=args_cmd, in_features=d, num_classes=c).to(device)
    if args_cmd.competitor is not None:
        net = CompetitorFactory.createCompetitor(args_cmd, net, input_size=(d,), num_classes=c)
    net.to(device)


    # Prepare for training & testing
    def set_optimizer(args_cmd):
        if args_cmd.optimizer == "sgd":
            optimizer = SGD(net.parameters(), lr=args_cmd.beta_m, weight_decay=args_cmd.weight_decay)
        elif args_cmd.optimizer == "adam":
            optimizer = Adam(net.parameters(), lr=args_cmd.beta_m, weight_decay=args_cmd.weight_decay)
        else:
            raise NotImplementedError
        return optimizer


    optimizer = set_optimizer(args_cmd)

    # continual confusion matrices (empty, -1 means "unfilled yet", important!)
    CCM_train = create_continual_confusion_matrix(distributions_train)
    CCM_val = create_continual_confusion_matrix(distributions_val)
    CCM_test = create_continual_confusion_matrix(distributions_test)

    # metrics containers (empty, -1 means "unfilled yet", important!)
    metrics_train = create_metrics_dictionary(distributions_train)
    metrics_val = create_metrics_dictionary(distributions_val)
    metrics_test = create_metrics_dictionary(distributions_test)

    summ = summary(net, input_size=(1, d), device=device)
    params = {"total_params": summ.total_params, "trainable_params": summ.trainable_params}
    options = vars(args_cmd)
    options.update(params)
    # setting loggers

    if args_cmd.wandb:
        WANDB_PROJ = "mh"
        WANDB_ENTITY = "test"
        wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, config=options)

        if args_cmd.watch:
            if args_cmd.competitor is not None:
                wandb.watch(net.net, log="all", log_freq=10)
            else:
                wandb.watch(net, log="all", log_freq=10)

    # streaming loop
    net.eval()

    # initial plot for keys - d2d plot for keys only possible when one layer
    if "lp" not in args_cmd.model and args_cmd.draw_plots:
        with torch.no_grad():
            # class dictionary
            class_dict = {'Class ' + str(j): j for j in range(0, c)}

            # getting keys, if any, adding them to the data, with an ad-hoc label
            X_train_K, y_train_K, class_dict_K = d2d.utils_toy_datasets.add_keys_from_mh(net, X_train,
                                                                                         y_train,
                                                                                         class_dict)
            # plotting predictions together with keys
            plt_preds_init = plot_2d_data_and_predictions('init_keys', X_train_K, y_train_K,
                                                          class_dict_K, net)

            if args_cmd.wandb:
                wandb.log({"plots/init_keys": wandb.Image(plt_preds_init)})
            else:
                plt.show()

    # for t, (x, y, distribution) in enumerate(zip(X_train, y_train, distributions_train)):
    net.train()
    with tqdm(zip(X_train, y_train, distributions_train), unit="sample") as tepoch:
        for t, (x, y, distribution) in enumerate(tepoch):
            tepoch.set_description(f"Sample - ")

            wandb_dict = {}

            # adding batch size and clearing up the tensor box
            x.unsqueeze_(0)
            y.unsqueeze_(0)
            distribution = distribution.item()

            # learning step
            if args_cmd.competitor is not None and "GDumb" in args_cmd.competitor:
                # in case of GDumb, we do not forward the current pattern/compute any loss
                pass
            else:
                o = net(x)
                loss = compute_loss(o, y, c, args_cmd.loss)

                optimizer.zero_grad()
                # compute immediatly loss gradients, needed for MIR!
                loss.backward()
            # the ensemble competitor does not need any memory handling
            if args_cmd.competitor is not None and "Ensemble" not in args_cmd.competitor:
                replay_x, replay_y = net.retrieve_buffer(num_retrieve=args_cmd.buffer_batch_size)
                # only if there is something in memory!

                if (replay_x is not None and replay_x.shape[0] > 0) and "GDumb" not in args_cmd.competitor:
                    if args_cmd.competitor == "AGEM" and t > 0:
                        net.agem_step(replay_x, replay_y, optimizer)
                    else:
                        replay_o = net(replay_x)
                        replay_loss = compute_loss(replay_o, replay_y, c, args_cmd.loss)
                        # compute backward also for the replay loss (gradients are accumulated)
                        replay_loss.backward()
                    # update the buffer!
                net.update_buffer(x, y)

            if args_cmd.competitor is not None and "GDumb" in args_cmd.competitor:
                pass
            else:
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
            # sleep(0.1)

            # metrics (computed when training ends or when the distribution is going to change on the next step)
            if t == X_train.shape[0] - 1 or distributions_train[t + 1] != distribution:

                # GDumb learning at the end of each distribution
                if args_cmd.competitor is not None and "GDumb" in args_cmd.competitor:
                    # reset optimizer
                    optimizer = set_optimizer(args_cmd)
                    net.train_on_memory(batch_size=args_cmd.buffer_batch_size, epochs=args_cmd.gdumb_epochs,
                                        optimizer=optimizer)

                # adding a new row to the continual confusion matrices (exploiting the current network)
                fill_row_in_continual_confusion_matrix(CCM_train, distribution, net, X_train, y_train,
                                                       distributions_train, chunk_size=args_cmd.eval_chunk_size)
                fill_row_in_continual_confusion_matrix(CCM_val, distribution, net, X_val, y_val, distributions_val,
                                                       chunk_size=args_cmd.eval_chunk_size)
                fill_row_in_continual_confusion_matrix(CCM_test, distribution, net, X_test, y_test, distributions_test,
                                                       chunk_size=args_cmd.eval_chunk_size)

                # updating metrics
                update_metrics(metrics_train, CCM_train, distribution)
                update_metrics(metrics_val, CCM_val, distribution)
                update_metrics(metrics_test, CCM_test, distribution)

                # printing continual confusion matrix
                if args_cmd.benchmark != "imagenet":
                    # imagenet contains too many classes to plot this!
                    print_continual_confusion_matrix('train', CCM_train)
                    print_continual_confusion_matrix('val', CCM_val)
                    print_continual_confusion_matrix('test', CCM_test)

                # printing metrics
                print_metrics('train', metrics_train, distribution)
                print_metrics('val', metrics_val, distribution)
                print_metrics('test', metrics_test, distribution)

                # print(net.label_buffer)
                if args_cmd.wandb:
                    wandb_dict.update(get_current_metrics('train', metrics_train, distribution))
                    wandb_dict.update(get_current_metrics('val', metrics_val, distribution))
                    wandb_dict.update(get_current_metrics('test', metrics_test, distribution))
                    wandb.log(wandb_dict)

    net.eval()

    if "lp" not in args_cmd.model and args_cmd.draw_plots:

        # class dictionary
        class_dict = {'Class ' + str(j): j for j in range(0, c)}

        # plotting predictions at the end of training
        plt_preds = plot_2d_data_and_predictions('train', X_train, y_train, class_dict, net)

        if args_cmd.wandb:
            wandb_dict_final = {}
            wandb_dict_final["plots/predictions"] = wandb.Image(plt_preds)
        else:
            plt.show()

        # getting keys, if any, adding them to the data, with an ad-hoc label
        X_train_K, y_train_K, class_dict_K = d2d.utils_toy_datasets.add_keys_from_mh(net, X_train,
                                                                                     y_train,
                                                                                     class_dict)

        # plotting predictions together with keys
        plt_pred_keys_complete = plot_2d_data_and_predictions('train+keys', X_train_K, y_train_K,
                                                              class_dict, net)

        if args_cmd.wandb:
            wandb_dict_final["plots/predictions_keys"] = wandb.Image(plt_pred_keys_complete)
        else:
            plt.show()

        normed_data = F.normalize(X_train, p=2.0, dim=1, eps=1e-12, out=None)

        # getting keys, if any, adding them to the data, with an ad-hoc label
        X_train_K, y_train_K, class_dict_K = d2d.utils_toy_datasets.add_keys_from_mh(net, normed_data,
                                                                                     y_train,
                                                                                     class_dict)

        plt_preds_limits_norm1 = plot_2d_data_and_predictions('train+keys_norm1', X_train_K, y_train_K,
                                                              class_dict_K)

        if args_cmd.wandb:
            wandb_dict_final["plots/predictions_keys_norm1"] = wandb.Image(plt_preds_limits_norm1)
        else:
            plt.show()

        if args_cmd.wandb:
            for idx, i in enumerate(net):
                if hasattr(i, "scrambling_count"):
                    wandb_dict_final[f"scrambling_count_{idx}"] = net[idx].scrambling_count

            wandb.log(wandb_dict_final)

    else:
        # plotting predictions at the end of training

        if args_cmd.draw_plots:
            plt_preds = plot_2d_data_and_predictions('train', X_train, y_train,
                                                     {'Class ' + str(j): j for j in range(0, c)},
                                                     net)

            if args_cmd.wandb:
                wandb_dict_final = {}
                wandb_dict_final["plots/predictions"] = wandb.Image(plt_preds)
                wandb.log(wandb_dict_final)
            else:
                plt.show()
