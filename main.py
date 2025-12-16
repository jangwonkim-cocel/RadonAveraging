import argparse
import torch
from model import RA
from trainer import train
import random
import warnings
warnings.filterwarnings('ignore')

from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train & Test with configurable args")
    parser.add_argument("--num_train_data", type=int, default=10000, help="number of training samples")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--group", type=str, default="C8", help="group name for RA model (C4 or C8)")
    parser.add_argument("--n_seeds", type=int, default=10, help="number of seeds to run")

    return parser.parse_args()


def train_and_test(model, train_loader, val_loader, test_loader, epochs, device):
    score, f1 = train(
        model=model,
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=1e-4,
        wd=1e-4,
        device=device,
    )
    return score, f1


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_seed = random.randint(1, 999)

    accs = []
    f1s = []

    print_run_config(args, device, base_seed)

    for seed in range(base_seed, base_seed + args.n_seeds):
        train_dataset = MnistDataset(num_train_data=args.num_train_data, mode="train", seed=seed)
        validation_dataset = MnistDataset(mode="validation", seed=seed)
        test_dataset = MnistDataset(mode="test", seed=seed)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

        model = RA(group=args.group).to(device)
        score, f1 = train_and_test(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            device=device,
        )
        accs.append(score)
        f1s.append(f1)

    a_m, a_std = cal_mean_std(accs)
    print(f"[Acc] Mean: {a_m}  |  Std: {a_std}\n")

    f_m, f_std = cal_mean_std(f1s)
    print(f"[F-Score] Mean: {f_m}  |  Std: {f_std}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
