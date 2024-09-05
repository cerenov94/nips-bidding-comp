import numpy as np
import torch
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from run.run_edac import run_edac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Global hyperparameters
    parser.add_argument("--seed",default = 42,type=int)
    parser.add_argument('--device',default = 'cuda' if torch.cuda.is_available() else 'cpu',type=str)

    # Value hyperparameters
    parser.add_argument("--value_steps",default=20000,type=int)
    parser.add_argument("--value_bs",default=64,type=int)
    parser.add_argument("--value_lr",default=1e-4,type=float)
    parser.add_argument("--value_hidden_dim",default=64,type=float)

    # QValue hyperparameters
    parser.add_argument("--qvalue_steps",default=25000,type = int)
    parser.add_argument("--qvalue_bs",default=128,type = int)
    parser.add_argument("--qvalue_hidden_dim",default=128,type = int)
    parser.add_argument("--qvalue_lr",default=1e-4,type = float)
    parser.add_argument("--qvalue_update_freq",default = 200,type = int)
    parser.add_argument("--qvalue_tau",default = 5e-3,type = float)
    parser.add_argument("--qvalue_gamma",default=0.99,type = float)

    # Behavior clone hyperparameters
    parser.add_argument("--bc_steps",default=10000,type = int)
    parser.add_argument("--bc_hidden_dim",default=64,type = int)
    parser.add_argument("--bc_lr",default=1e-4,type = float)
    parser.add_argument("--bc_bs",default=32,type = int)

    # BPPO learning hyperparameters
    parser.add_argument("--bppo_steps",default=10000,type = int)
    parser.add_argument("--bppo_hidden_dim",default=64,type = int,help="should be equal BC hidden dim")
    parser.add_argument("--bppo_lr",default=1e-4,type = float)
    parser.add_argument("--bppo_bs",default=16,type = int)
    parser.add_argument("--is_clip_decay",default=True,type = bool)
    parser.add_argument("--is_bppo_lr_decay",default=True,type = bool)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_edac()