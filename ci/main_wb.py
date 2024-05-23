import argparse
import json
import argparse
from trainer import train
from evaluator import test
import wandb
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--prefix', type=str, default='reproduce', help='Prefix for the command')
    parser.add_argument('--dataset', type=str, default='cifar100_224', help='Dataset to use')
    parser.add_argument('--memory_size', type=int, default=0, help='Memory size')
    parser.add_argument('--memory_per_class', type=int, default=0, help='Memory per class')
    parser.add_argument('--fixed_memory', action='store_true', help='Use fixed memory')
    parser.add_argument('--bcb_lrscale', type=float, default=0.0, help='BCB learning rate scale')
    parser.add_argument('--shuffle', type=str, default="True", help='Shuffle the data')
    parser.add_argument('--init_cls', type=int, default=10, help='Initial number of classes')
    parser.add_argument('--increment', type=int, default=10, help='Increment of classes')
    parser.add_argument('--model_postfix', type=str, default='20e', help='Model postfix')
    parser.add_argument('--convnet_type', type=str, default='vit-b-p16', help='Convnet type')
    parser.add_argument('--milestones', nargs='+', type=int, default=[18], help='Milestone values')
    parser.add_argument('--ca_epochs', type=int, default=0, help='CA epochs')
    parser.add_argument('--ca_with_logit_norm', type=float, default=0.1, help='CA with logit norm')
    ##################################### TUNABLE PARAMS ##################################################

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='mh_cifar', help='Name of the model')
    parser.add_argument('--device', nargs='+', default=['0'], help='Device(s) to use')
    parser.add_argument('--seed', nargs='+', type=int, default=[1993], help='Seed values')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--virtual_batch', type=int, default=128, help='Virtual batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Num workers')

    ##################################### MH PARAMS ##################################################
    parser.add_argument('--key_mem_units', type=int, default=10, help='Key memory units')
    parser.add_argument('--psi_fn', type=str, default='identity', help='Psi function')
    parser.add_argument('--upd_m', type=str, default='WTA', help='Update m')
    parser.add_argument('--upd_k', type=str, default='ad_hoc_WTA', help='Update k')
    parser.add_argument('--beta_k', type=float, default=0.001, help='Beta k')
    parser.add_argument('--gamma_alpha', type=float, default=25.0, help='Gamma alpha')
    parser.add_argument('--tau_alpha', type=float, default=0.95, help='Tau alpha')
    parser.add_argument('--tau_mu', type=int, default=50, help='Tau mu')
    parser.add_argument('--tau_eta', type=int, default=50, help='Tau eta')
    parser.add_argument('--scramble', type=str, default="True", help='Scramble data')
    parser.add_argument('--delta', type=int, default=2, help='Delta value')
    parser.add_argument('--layer_norm', type=str, default="True", help='Use layer normalization')
    parser.add_argument('--distance', type=str, default='cosine', help='Distance metric')
    parser.add_argument('--sigma_eu', type=float, default=1.0, help='Sigma eu')
    parser.add_argument('--wandb', type=str, default="false")
    parser.add_argument('--freeze_keys_ca', type=str, default="false")

    args = parser.parse_args()

    args.wandb = args.wandb in ["true", "True"]
    args.layer_norm = args.layer_norm in ["true", "True"]
    args.scramble = args.scramble in ["true", "True"]
    args.shuffle = args.shuffle in ["true", "True"]
    args.freeze_keys_ca = args.freeze_keys_ca in ["true", "True"]

    args.upd_m = None if args.upd_m == "vanilla" else "WTA"
    if args.upd_k == "grad_not_WTA":
        args.upd_k = None

    if args.wandb:
        WANDB_PROJ = "test"
        WANDB_ENTITY = "test"
        wandb.init(project=WANDB_PROJ, entity=WANDB_ENTITY, config=vars(args))

    return vars(args)


# Example usage:
if __name__ == '__main__':
    args = parse_arguments()

    if args['test_only']:
        test(args)
    else:
        train(args)
