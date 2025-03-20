import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
import random
import os
import time
from datetime import timedelta
from collections import deque
import matplotlib.pyplot as plt
import json
import argparse

from data.gas import GAS
from data.bsds300 import BSDS300
from data.hepmass import HEPMASS
from data.miniboone import MINIBOONE
from data.power import POWER

# Example command:
# python train.py --dataset=power --method=tnaf --seed=1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def print_total_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Total parameters {total_params}")

def load_dataset(args):
    # Other datasets
    if args.dataset == "gas":
        dataset = GAS(os.path.join(args.data_path, "data/gas/ethylene_CO.pickle"))
    elif args.dataset == "bsds300":
        dataset = BSDS300(os.path.join(args.data_path, "data/BSDS300/BSDS300.hdf5"))
    elif args.dataset == "hepmass":
        dataset = HEPMASS(os.path.join(args.data_path, "data/hepmass"))
    elif args.dataset == "miniboone":
        dataset = MINIBOONE(os.path.join(args.data_path, "data/miniboone/data.npy"))
    elif args.dataset == "power":
        dataset = POWER(os.path.join(args.data_path, "data/power/data.npy"))
    else:
        raise RuntimeError()

    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.trn.x).float().to(args.device)
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True
    )

    dataset_valid = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.val.x).float().to(args.device)
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, shuffle=False
    )

    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.tst.x).float().to(args.device)
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False
    )
    args.n_dims = dataset.n_dims
    return data_loader_train, data_loader_valid, data_loader_test

def test(args, my_net, data_loader):
    loss_list = list()
    my_net.eval()
    for iteration, data in enumerate(data_loader):
           x = data[0]
           x = x[:,:,None].to(args.device)
           _, loss = my_net(x)
           loss_list.append(-loss.cpu().detach()) # [batch]
    my_net.train()
    return torch.cat(loss_list).mean()

def main(args):
    set_seed(args.seed)
    print(f"[INFO] Training on: {args.dataset}")
    data_loader_train, data_loader_valid, data_loader_test = load_dataset(args)
    if args.method == "tnaf":
        from methods.tnaf import TNAF, TNAFSequential
        my_net = TNAFSequential(n_dims=args.n_dims,
                                num_tnafs=args.num_tnafs,
                                dropout=args.dropout,
                                num_layers=args.num_layers,
                                num_heads=args.num_heads,
                                embedding_dim=args.embedding_dim,
                                mlp_dim=args.mlp_dim,
                                bnaf_dim=args.bnaf_dim,
                                attention_dropout=args.attention_dropout,
                                out_dim=args.out_dim,
                                representation_size=args.representation_size,
                                bnaf_hidden_layers=args.bnaf_hidden_layers)
    else:
        print(f"The method {args.method} is not supported!")
        quit()

    my_net = my_net.to(args.device)
    print_total_params(my_net)
    optimizer = torch.optim.Adam(my_net.parameters(), lr=0.001, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=5e-4, cooldown=10,
                                                           verbose=True, threshold_mode="abs",)
    start_time = time.time()
    loss_deque = deque(maxlen=100)
    best_score = float("inf")
    start_epoch = 0

    if args.resume_training:
        path = f"./checkpoints/last_{args.method}_{args.dataset}_seed_{args.seed}.pt"
        if os.path.exists(path):
            training_dict = torch.load(path)
            start_epoch = training_dict['epoch']
            best_score = training_dict['best_score']
            loss_deque = training_dict['train_ll']
            my_net.load_state_dict(training_dict['model_state_dict'])
            optimizer.load_state_dict(training_dict['optimizer_state_dict'])
            scheduler.load_state_dict(training_dict['scheduler_state_dict'])
            print("[INFO] Resuming training from last checkpoint")
        else:
            print("[INFO] Start training from scratch")

    print("[INFO] Training started...", flush=True)

    for epoch in range(start_epoch, args.epochs):
        for iteration, data in enumerate(data_loader_train):
           x = data[0]
           x = x[:,:,None].to(args.device)
           optimizer.zero_grad()
           loss = list()     
           _, loss = my_net(x)
           loss = -torch.mean(loss)
           loss.backward()           
           nn.utils.clip_grad_norm_(my_net.parameters(), max_norm=0.1)
           optimizer.step()
           loss_deque.append(loss.item())
           if(iteration%500==0 and iteration!=0):
               print(f"[{epoch}|{args.epochs}][{iteration}|{len(data_loader_train)}] Log-Like.: {-np.mean(loss_deque)}+-{np.std(loss_deque)};")
        
        # Epoch gone
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_str = str(timedelta(seconds=elapsed_time))
        str_header = ""
        # Validating
        val_nll = test(args, my_net=my_net, data_loader=data_loader_valid)
        if(val_nll<best_score): 
            best_score = val_nll
            path = f"./checkpoints/best_{args.method}_{args.dataset}_seed_{args.seed}_num_tnafs_{args.num_tnafs}.pt"
            if not os.path.exists("./checkpoints"): os.makedirs("./checkpoints")
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": my_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_ll": -np.mean(loss_deque),
                "val_ll": -val_nll,
                }, path)

            str_header = "*" # If best result add the star header

        print(f"{str_header}[{epoch}|{args.epochs}] Train Log-Like: {-np.mean(loss_deque)}+-{np.std(loss_deque)}; Val. Log-Like.: {-val_nll}; Time: {time_str}", flush=True)
        scheduler.step(val_nll)
        start_time = time.time()

        if args.save_last_model:
            path = f"./checkpoints/last_{args.method}_{args.dataset}_seed_{args.seed}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": my_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_ll": loss_deque,
                "best_score": best_score
            }, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="tnaf",
        choices=["tnaf"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="miniboone",
        choices=["gas", "bsds300", "hepmass", "miniboone", "power"])
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_last_model", action='store_true')
    parser.add_argument("--resume_training", action='store_true')
    parser.add_argument("--num_tnafs", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--mlp_dim", type=int, default=64)
    parser.add_argument("--bnaf_dim", type=int, default=128)
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--representation_size", type=int, default=256)
    parser.add_argument("--bnaf_hidden_layers", type=int, default=1)

    args = parser.parse_args()
    main(args)
