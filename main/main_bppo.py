
import numpy as np
import torch
import os
import sys
import argparse

from matplotlib import defaultParams

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distutils.util import strtobool
from run.run_bppo import run_bppo
import optuna
import joblib

def update_args(args,params):
    dargs = vars(args)
    dargs.update(params)


def main(trials = None):

    parser = argparse.ArgumentParser()
    # Global hyperparameters
    parser.add_argument("--seed",default = 2,type=int)
    parser.add_argument('--device',default = 'cuda' if torch.cuda.is_available() else 'cpu',type=str)
    parser.add_argument("--is_learn_value",default = True,type = lambda x: bool(strtobool(x)),nargs="?",const=True)
    parser.add_argument("--with_validation",default= False,type = lambda x: bool(strtobool(x)),nargs="?",const=True)

    # Value hyperparameters
    parser.add_argument("--value_steps",default=20000,type=int)
    parser.add_argument("--value_bs",default=256,type=int)
    parser.add_argument("--value_lr",default=0.0015900275801136736,type=float)
    parser.add_argument("--value_hidden_dim",default=128,type=float)
    parser.add_argument("--v_expectile",default=0.759927899472099,type=float)

    # QValue hyperparameters
    parser.add_argument("--qvalue_steps",default=17000,type = int)
    parser.add_argument("--qvalue_bs",default=128,type = int)
    parser.add_argument("--qvalue_hidden_dim",default=128,type = int)
    parser.add_argument("--qvalue_lr",default=0.0027256508922447946,type = float)
    parser.add_argument("--qvalue_update_freq",default = 200,type = int)
    parser.add_argument("--qvalue_tau",default = 5e-3,type = float)
    parser.add_argument("--qvalue_gamma",default=0.99,type = float)
    parser.add_argument("--n_critics",default=200,type= int)
    parser.add_argument("--eta",default=1.0,type=float)

    # Behavior clone hyperparameters
    parser.add_argument("--bc_steps",default=24000,type = int)
    parser.add_argument("--bc_bs", default=16, type=int)
    parser.add_argument("--bc_lr", default=0.00011685694955947773, type=float)
    parser.add_argument("--bc_hidden_dim",default=192,type = int)
    parser.add_argument("--bc_temp",default= 2.124025793483719,type=float)



    # BPPO learning hyperparameters
    parser.add_argument("--bppo_steps",default=9000,type = int)
    parser.add_argument("--bppo_lr",default= 0.00011237508032926629,type = float)
    parser.add_argument("--bppo_bs",default=32,type = int)
    parser.add_argument("--is_clip_decay",default = True,type = lambda x: bool(strtobool(x)),nargs="?",const=True)
    parser.add_argument("--is_bppo_lr_decay",default = True,type = lambda x: bool(strtobool(x)),nargs="?",const=True)
    parser.add_argument("--clip_ratio",default=0.2,type=float)
    parser.add_argument('--omega',default= 0.7,type=float)

    args = parser.parse_args()

    if trials is not None:
        params = {
            'seed': trials.suggest_categorical('seed',[1,2,3,4,5,6,7,8,9,42]),
            'with_validation':trials.suggest_categorical('with_validation',['True']),
            'value_bs':trials.suggest_categorical('value_bs',[64,128,256,512]),
            'value_lr': trials.suggest_float('value_lr',1e-4,3e-3,log=True),
            'value_hidden_dim': trials.suggest_categorical('value_hidden_dim',[64,128]),
            'v_expectile': trials.suggest_float('v_expectile',0.7,0.98,log=True),
            'qvalue_steps':trials.suggest_int('qvalue_steps',14000,20000,step = 1000),
            'qvalue_bs': trials.suggest_categorical('qvalue_bs', [64, 128, 256, 512]),
            'qvalue_hidden_dim': trials.suggest_categorical('qvalue_hidden_dim',[64,128]),
            'qvalue_lr': trials.suggest_float('qvalue_lr',1e-4,3e-3,log=True),
            #'qvalue_gamma': trials.suggest_categorical('qvalue_gamma',[0.96,0.97,0.98,0.99]),
            'n_critics':trials.suggest_categorical('n_critics',[20,50,100,200]),
            'eta': trials.suggest_categorical('eta',[1.0,3.0,5.0]),
            'bc_steps': trials.suggest_int('bc_steps',10000,30000, step = 1000),
            'bc_bs': trials.suggest_categorical('bc_bs',[16,32,64,128]),
            'bc_lr': trials.suggest_float('bc_lr', 1e-4, 5e-4,log=True),
            'bc_hidden_dim': trials.suggest_categorical('bc_hidden_dim',[128,192,256,384,512]),
            'bc_temp':trials.suggest_float('bc_temp',1.0,3.0,log=True),
            'bppo_steps': trials.suggest_int('bppo_steps',1000,12000,step=1000),
            'bppo_lr': trials.suggest_float('bppo_lr',1e-4,5e-4,log=True),
            'bppo_bs': trials.suggest_categorical('bppo_bs',[16,32,64,128]),
            'clip_ratio': trials.suggest_categorical('clip_ratio',[0.1,0.15,0.2,0.25]),
            'omega': trials.suggest_categorical('omega',[0.7,0.8,0.9])
        }
        update_args(args,params)


    score = run_bppo(
        seed = args.seed,
        with_validation = args.with_validation,
        learn_value = args.is_learn_value,
        device = args.device,
        value_steps = args.value_steps,
        value_bs = args.value_bs,
        value_lr = args.value_lr,
        value_hidden_dim = args.value_hidden_dim,
        v_expect = args.v_expectile,
        qvalue_steps=args.qvalue_steps,
        qvalue_bs=args.qvalue_bs,
        qvalue_hidden_dim=args.qvalue_hidden_dim,
        qvalue_lr=args.qvalue_lr,
        qvalue_update_freq=args.qvalue_update_freq,
        qvalue_tau=args.qvalue_tau,
        qvalue_gamma=args.qvalue_gamma,
        n_critics = args.n_critics,
        eta = args.eta,
        bc_steps=args.bc_steps,
        bc_bs=args.bc_bs,
        bc_lr=args.bc_lr,
        bc_hidden_dim=args.bc_hidden_dim,
        bc_temp=args.bc_temp,
        bppo_steps=args.bppo_steps,
        bppo_bs=args.bppo_bs,
        bppo_lr=args.bppo_lr,
        is_bppo_lr_decay=args.is_bppo_lr_decay,
        is_clip_decay=args.is_clip_decay,
        clip_ratio = args.clip_ratio,
        omega = args.omega
    )
    return score

if __name__ == '__main__':
    #study = optuna.create_study(direction='minimize',study_name='experiment1')
    #study.optimize(main,n_trials=10,show_progress_bar=True)
    #joblib.dump(study,'train_logs/experiment_4.pkl')
    main()