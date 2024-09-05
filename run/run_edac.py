import numpy as np
import torch
import pandas as pd
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.bppo.replay_buffer_upd import ReplayBuffer
from bidding_train_env.bppo.edac import EDAC
import ast
from tqdm import tqdm
import torch.nn.functional as F

from run.run_bppo import train_model


def run_edac():
    train_model()


def train_model():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val  # 如果解析出错，返回原值

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)

    rewards = training_data['reward_continuous'].values
    dones = training_data['done'].values
    dones = np.split(dones, 336)

    returns = []
    for chunk, time_step in enumerate(np.split(rewards, 336)):
        r = 0
        current_return = np.zeros((48, 1))
        for i in reversed(range(len(time_step))):
            current_return[i] = time_step[i] + 0.99 * r * (1 - dones[chunk][i])
            r = current_return[i]
        returns.append(current_return)

    returns = np.concatenate(returns)

    returns_range = returns.max() - returns.min() + 1e-8
    norm_returns = (returns - returns.min()) / returns_range

    training_data['returns'] = norm_returns

    actions = training_data['action'].values
    actions = np.split(actions, 336)
    next_actions = []
    for a in actions:
        next_actions.append(np.append(a[1:], 0.))
    training_data['next_action'] = np.concatenate(next_actions)

    replay_buffer = ReplayBuffer(device=device)

    normalize_dic = normalize_state(training_data, 16, normalize_indices=[13, 14, 15])
    training_data['reward'] = normalize_reward(training_data, 'reward_continuous')
    save_normalize_dict(normalize_dic, "saved_model/EDACtest")

    #action_range = training_data['action'].max() - training_data['action'].min()
    #action_min = training_data['action'].min()
    #np.save('saved_model/EDACtest/action_range.npy',action_range)
    #np.save('saved_model/EDACtest/min_action.npy',action_min)
    #training_data['action'] = (training_data['action'] - action_min)/action_range * 2 + 1
    add_to_replay_buffer(replay_buffer, training_data, True)

    edac = EDAC(device = device)
    for step in tqdm(range(20000)):
        alpha_loss,critic_loss,actor_loss,actor_batch_entropy,alpha,q_policy_std,q_random_std = edac.train(replay_buffer)

        print(f'alpha loss: {alpha_loss:.4f},critic loss:{critic_loss:.4f},actor_loss:{actor_loss},q_policy_std:{q_policy_std},q_random_std:{q_random_std}')
    chunks = np.split(training_data, 336)
    ids = np.array(chunks[305].index)
    state, action, _, _, _, _, _ = replay_buffer.sample(48, random_samples=False, ids=ids)
    pred_action = edac.get_action(state)

    tem = torch.cat([action.cpu(), pred_action], dim = 1)
    print("action VS pred action:", tem)
    edac.save_weights()


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state,next_action,done,G = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate,row.next_action, row.done,row.returns
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(
                np.array(state),
                np.array([action]),
                np.array([reward]),
                np.array(next_state),
                np.array([next_action]),
                np.array([done]),
                np.array([G]))
        else:
            replay_buffer.push(
                np.array(state),
                np.array([action]),
                np.array([reward]),
                np.zeros_like(state),
                np.array([next_action]),
                np.array([done]),
                np.array([G])
                )