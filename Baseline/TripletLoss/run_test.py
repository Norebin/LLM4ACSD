import argparse
import sys

import numpy as np
import torch
from code_utils import set_seed

import run


def _build_smell_range(smell_str):
    ranges = smell_str.split(",")
    ranges = [int(i) for i in ranges]
    assert len(ranges) == 2 and ranges[0] < ranges[1]
    return tuple(ranges)


def _build_lr(lr_str):
    lr = int(lr_str)
    assert lr < 0
    return np.power(10., lr)


def _to_shuffle(sh_str):
    sh = int(sh_str)
    assert sh == 1 or sh == 0
    return bool(sh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ep", "--embeds_path", default='/model/lxj/Baseline/TripletLoss/data/triple/actionableSmell_bert_nli_mean_token_pooler_output.npy')
    parser.add_argument("-lp", "--label_path", default='/model/lxj/Baseline/TripletLoss/data/labels.csv')
    parser.add_argument("-op", "--optimizer", default='Adam')
    parser.add_argument("-bs", "--batch_size", type=int, default=256)
    parser.add_argument("-sr", "--smell_range", type=_build_smell_range, default=",".join(str(i) for i in (0, 7)))
    parser.add_argument("-lr", "--learning_rate", type=_build_lr,default='-4')
    parser.add_argument("-sh", "--shuffle", type=_to_shuffle, default='1')
    parser.add_argument("-sd", "--seed", default=42, type=int)
    parser.add_argument("-pt", "--patience", default=None, type=int)
    parser.add_argument("-tol", "--tolerance", default=None, type=float)
    parser.add_argument("-e", "--n_epochs", default=2000, type=int)
    parser.add_argument("-dev", "--device", default="cuda")
    parser.add_argument("-o", "--output_folder",default='/model/lxj/Baseline/TripletLoss/result')
    args = parser.parse_args()

    params = {
        "optimizer": args.optimizer,
        "lr": args.learning_rate,
        "train_batch_size": args.batch_size,
        "seed": args.seed,
        "num_epochs": args.n_epochs,
        "patience": args.patience,
        "tolerance": args.tolerance
    }

    if not args.tolerance:
        params.pop("tolerance")

    if not args.patience:
        params.pop("patience")

    set_seed(int(args.seed))
    device = torch.device(args.device)
    runner = run.Runner(args.label_path, args.embeds_path, 0.2, device=device, params=params, title=None, output_folder=args.output_folder)
    runner.run(args.smell_range, shuffle=args.shuffle)