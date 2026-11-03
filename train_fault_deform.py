import os
import argparse
import torch
import numpy as np
import random

from src.dataloaders.data_loader import get_training_dataloaders
from src.models.model_loader import load_model
from src.metrics_and_losses.photometric_metrics import realEPE
from src.metrics_and_losses.losses import IntermediateL1Loss
from src.training_inference.train import train
from src.configs.load_config import ConfigArgumentParser, YamlConfigAction

rng = 1


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """
    Parse command line arguments for the inference script.
    """
    parser = ConfigArgumentParser('training on fault deform')
    parser.add_argument('--train_config_name', default='train_microflow')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_dir = os.path.join(current_dir, 'data/configs/train_fault_deform/')
    parser.add_argument('--config_dir', default=default_config_dir, help='Directory containing the configs')
    parser.add_argument('--seed', default=1, type=int, help='Seed used for the random number generators')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (e.g. cuda or cpu)')
    
    # directories parameters
    parser.add_argument('--split_dir', type=str, help='directory containing the dataset splits')
    parser.add_argument('--dataset_dir', help='path to dataset dir')
    parser.add_argument('--inverse_dataset_dir', type=str, default="", help='path to inverse dataset dir')
    parser.add_argument('--save_offline', action='store_true', help='If true, save offline logs to be used for wandb')
    parser.add_argument('--offline_dir', type=str, default="", help='Directory where to save the offline logs')
    parser.add_argument('--checkpoints_dir', default='results/checkpoints_microflow/',
                        help='Directory where to save the checkpoints')
    
    # clean_pair dataset parameters
    parser.add_argument('--clean_pair_dataset_dir', type=str, default="", help='path to clean_pair dataset dir for augmentation')
    parser.add_argument('--clean_pair_prob', type=float, default=0.5, help='probability of using the clean_pair dataset')
    parser.add_argument('--clean_pair_add_noise', default=True, action=argparse.BooleanOptionalAction, help='apply noise to clean_pair dataset')
    parser.add_argument('--run_clean_pair_val', default=False, action=argparse.BooleanOptionalAction, help='run separate validation on clean_pair data')
    
    # pretraining parameters
    parser.add_argument('--training_from_pretrain', action='store_true',
        help='if true, restart the training parameters from the pretrained model. Else start from scratch')
    parser.add_argument('--restart', action='store_true', help='if True, restart the training from where it stopped')
    parser.add_argument('--pretrained_model_filename', type=str, default="", help="Path to pretrained model. If restart is True, it is overwritten")

    # ---- ITERATIVE_MODEL params ----
    parser.add_argument('--normalization', default='minmax', type=str,
                        help='Normalization method: minmax or standard')
    parser.add_argument('--apply_noise', default=False, action=argparse.BooleanOptionalAction,
                        help='Apply noise during inference (test feature, keep False)')

    # ---- BASELINE_MODELS params ----
    parser.add_argument('--renormalize', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--detach_gradients', default=True, action=argparse.BooleanOptionalAction,
                        help='Detach gradients between iterative model steps')
    parser.add_argument('--searaft_freeze', default=False, action=argparse.BooleanOptionalAction,
                        help='Freeze SEA-RAFT backbone weights')

    # ---- SEA-RAFT specific params ----
    parser.add_argument('--searaft_block_dims', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--searaft_dim', default=128, type=int)
    parser.add_argument('--searaft_initial_dim', default=64, type=int)
    parser.add_argument('--searaft_max_iterations', default=4, type=int)
    parser.add_argument('--searaft_num_blocks', default=2, type=int)
    parser.add_argument('--searaft_pretrain', default='resnet34', type=str)
    parser.add_argument('--searaft_radius', default=4, type=int)
    parser.add_argument('--searaft_use_var', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--searaft_var_max', default=10, type=float)
    parser.add_argument('--searaft_var_min', default=0, type=float)
    parser.add_argument('--upsample_learned', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--repeat_first_iteration', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--repeat_model', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--resnet_imagenet_pretrained', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--resnet_input_dim', default=6, type=int)
    parser.add_argument('--gamma_scale', default=0.85, type=float)
    parser.add_argument('--max_scale', default=1, type=int)

    namespace, _ = parser.parse_known_args()

    config_filename = os.path.join(namespace.config_dir, f'{namespace.train_config_name}.yaml')
    if not os.path.exists(config_filename):
        config_filename = os.path.join(default_config_dir, f'{namespace.train_config_name}.yaml')
        
    parser.add_argument('--config', action=YamlConfigAction, default=[config_filename])
    args = parser.parse_args()
    import datetime

    if args.restart:
        args.training_from_pretrain = True
        if not args.pretrained_model_filename:
            print("WARNING: You are restarting but haven't specified --pretrained_model_filename.")
            print("Please specify the exact checkpoint file path since checkpoint folders are now timestamped.")
        else:
            # Resume in the same directory as the checkpoint
            run_dir = os.path.dirname(os.path.abspath(args.pretrained_model_filename))
            
    if not args.restart or not args.pretrained_model_filename:
        # Create a unique run directory based on the current date and time
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.checkpoints_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

    # Set paths to the run directory
    args.save_path = run_dir
    args.offline_dir = run_dir
    args.save_offline = True
    args.save_offline_filename = os.path.join(args.offline_dir, "offline_logs.txt")

    if not args.restart or not args.pretrained_model_filename:
        # Initialize the offline logs file
        save_epoch = [
            "epoch","train_loss", "val_loss", "train_mae",
            "val_mae", "train_epe", "val_epe", "lr", "scaling_factor_errors", "train_step_errors", "val_step_errors"]
        with open(args.save_offline_filename, "w+") as writer:
            writer.write(";".join(save_epoch) + "\n")

        # Print and save hyperparameters
        hyperparameters_str = "Input Arguments\n---------------\n"
        for k, v in sorted(dict(vars(args)).items()):
            hyperparameters_str += f"{k}: {str(v)}\n"
        
        print(f"\n{hyperparameters_str}")
        
        with open(os.path.join(run_dir, "hyperparameters.txt"), "w") as f:
            f.write(hyperparameters_str)

    return args


def init_optimizer(args, model):
    if args.solver == "AdamW": # used in RAFT paper
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.wdecay, eps=args.epsilon)
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    return optimizer


def get_loss_functions(args):
    if args.loss.lower() == "mse":
        loss_function = torch.nn.MSELoss()
    elif args.loss.lower() == "l1":
        loss_function = torch.nn.L1Loss()
    elif args.loss.lower() == "raft":
        loss_function = IntermediateL1Loss(args.gamma)
    elif args.loss.lower() == "intermediate_l1":
        loss_function = IntermediateL1Loss(args.gamma)
    else:
        raise Exception("Loss not implemented")
    return loss_function


def run_train(args):
    # Fix the seed
    fix_seed(args.seed)

    # Define device
    device = torch.device(args.device)

    # Load model
    model = load_model(args, device)

    # Get dataloaders
    train_loader, val_loader, clean_pair_val_loader = get_training_dataloaders(args)
    train_dataset_count = len(train_loader)
    val_dataset_count = len(val_loader)
    print(f"train examples={train_dataset_count} val examples={val_dataset_count}")

    # Loss and metrics
    regression_loss_function = get_loss_functions(args)
    epe = realEPE
    mae = torch.nn.L1Loss()
    metrics = [epe, mae]

    # Optimizer
    optimizer = init_optimizer(args, model)

    # Scheduler
    if args.scheduler_name.lower() == "one_cycle_lr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, args.lr, args.num_steps + 100, pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    elif args.scheduler_name.lower() == "multi_steps_lr":
        milestones = [20, 40, 80]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5) 
    else:
        scheduler = None
    print(f"scheduler {scheduler}")

    # Restore epoch, scheduler, and optimizer
    start_epoch = 0
    if args.pretrained_model_filename and args.training_from_pretrain:
        checkpoint = torch.load(args.pretrained_model_filename)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    train(
        args=args,
        device=device,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        clean_pair_val_loader=clean_pair_val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        regression_loss_function=regression_loss_function,
        metrics=metrics,
    )


if __name__ == '__main__':
    # Load parameters
    args = parse_args()

    run_train(args)
