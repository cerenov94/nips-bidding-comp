from email.policy import default
from sre_parse import parse

import numpy as np
import torch
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from run.run_bppo import run_bppo



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Global hyperparameters
    parser.add_argument("--seed",default = 42,type=int)
    parser.add_argument('--device',default = 'cuda' if torch.cuda.is_available() else 'cpu',type=str)

    # Value hyperparameters
    parser.add_argument("--value_steps",default=50000,type=int)
    parser.add_argument("--value_bs",default=64,type=int)
    parser.add_argument("--value_lr",default=1e-4,type=float)
    parser.add_argument("--value_hidden_dim",default=128,type=float)

    # QValue hyperparameters
    parser.add_argument("--qvalue_steps",default=50000,type = int)
    parser.add_argument("--qvalue_bs",default=64,type = int)
    parser.add_argument("--qvalue_hidden_dim",default=128,type = int)
    parser.add_argument("--qvalue_lr",default=1e-4,type = float)
    parser.add_argument("--qvalue_update_freq",default = 200,type = int)
    parser.add_argument("--qvalue_tau",default = 5e-3,type = float)
    parser.add_argument("--qvalue_gamma",default=0.99,type = float)

    # Behavior clone hyperparameters
    parser.add_argument("--bc_steps",default=10000,type = int)
    parser.add_argument("--bc_hidden_dim",default=128,type = int)
    parser.add_argument("--bc_lr",default=1e-4,type = float)
    parser.add_argument("--bc_bs",default=32,type = int)

    # BPPO learning hyperparameters
    parser.add_argument("--bppo_steps",default=10000,type = int)
    parser.add_argument("--bppo_hidden_dim",default=128,type = int,help="should be equal BC hidden dim")
    parser.add_argument("--bppo_lr",default=1e-4,type = float)
    parser.add_argument("--bppo_bs",default=32,type = int)
    parser.add_argument("--is_clip_decay",default=True,type = bool)
    parser.add_argument("--is_bppo_lr_decay",default=True,type = bool)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_bppo(
        device = args.device,
        value_steps = args.value_steps,
        value_bs = args.value_bs,
        value_lr = args.value_lr,
        value_hidden_dim = args.value_hidden_dim,
        qvalue_steps=args.qvalue_steps,
        qvalue_bs=args.qvalue_bs,
        qvalue_hidden_dim=args.qvalue_hidden_dim,
        qvalue_lr=args.qvalue_lr,
        qvalue_update_freq=args.qvalue_update_freq,
        qvalue_tau=args.qvalue_tau,
        qvalue_gamma=args.qvalue_gamma,
        bc_steps=args.bc_steps,
        bc_bs=args.bc_bs,
        bc_lr=args.bc_lr,
        bc_hidden_dim=args.bc_hidden_dim,
        bppo_steps=args.bppo_steps,
        bppo_bs=args.bppo_bs,
        bppo_lr=args.bppo_lr,
        bppo_hidden_dim=args.bppo_hidden_dim,
        is_bppo_lr_decay=args.is_bppo_lr_decay,
        is_clip_decay=args.is_clip_decay
    )