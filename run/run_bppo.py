import numpy as np
import torch
import pandas as pd
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.bppo.replay_buffer_upd import ReplayBuffer
from bidding_train_env.bppo.bppo import Value,QLSarsa,QP,BC,BPPO
import ast
from tqdm import tqdm
import torch.nn.functional as F



def run_bppo(
        device,
        value_steps,
        value_bs,
        value_lr,
        value_hidden_dim,
        qvalue_steps,
        qvalue_bs,
        qvalue_hidden_dim,
        qvalue_lr,
        qvalue_update_freq,
        qvalue_tau,
        qvalue_gamma,
        bc_steps,
        bc_hidden_dim,
        bc_lr,
        bc_bs,
        bppo_steps,
        bppo_hidden_dim,
        bppo_lr,
        bppo_bs,
        is_clip_decay,
        is_bppo_lr_decay,
):

    train_model(
        device,
        value_steps,
        value_bs,
        value_lr,
        value_hidden_dim,
        qvalue_steps,
        qvalue_bs,
        qvalue_hidden_dim,
        qvalue_lr,
        qvalue_update_freq,
        qvalue_tau,
        qvalue_gamma,
        bc_steps,
        bc_hidden_dim,
        bc_lr,
        bc_bs,
        bppo_steps,
        bppo_hidden_dim,
        bppo_lr,
        bppo_bs,
        is_clip_decay,
        is_bppo_lr_decay,
    )




def train_model(
        device,
        value_steps,
        value_bs,
        value_lr,
        value_hidden_dim,
        qvalue_steps,
        qvalue_bs,
        qvalue_hidden_dim,
        qvalue_lr,
        qvalue_update_freq,
        qvalue_tau,
        qvalue_gamma,
        bc_steps,
        bc_hidden_dim,
        bc_lr,
        bc_bs,
        bppo_steps,
        bppo_hidden_dim,
        bppo_lr,
        bppo_bs,
        is_clip_decay,
        is_bppo_lr_decay,
):

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
    save_normalize_dict(normalize_dic, "saved_model/BPPOtest")


    add_to_replay_buffer(replay_buffer, training_data, True)


    VL = Value(hidden_dim=value_hidden_dim,lr=value_lr,batch_size=value_bs,device=device)

    SARSA = QLSarsa(
        hidden_dim=qvalue_hidden_dim,
        lr = qvalue_lr,
        update_freq=qvalue_update_freq,
        tau=qvalue_tau,
        gamma=qvalue_gamma,
        batch_size=qvalue_bs,
        device = device
    )
    QPL = QP(
        hidden_dim=qvalue_hidden_dim,
        lr=qvalue_lr,
        update_freq=qvalue_update_freq,
        tau=qvalue_tau,
        gamma=qvalue_gamma,
        batch_size=qvalue_bs,
        device=device
    )
    BCL = BC(
        hidden_dim=bc_hidden_dim,
        lr = bc_lr,
        batch_size=bc_bs,
        device = device
    )
    BPPOL = BPPO(hidden_dim=bppo_hidden_dim,lr=bppo_lr,batch_size = bppo_bs,device = device)

    VALUE_STEPS = value_steps
    SARSA_STEPS = qvalue_steps
    BC_STEPS = bc_steps
    BPPO_STEPS = bppo_steps

    # sarsa learn
    for step in tqdm(range(SARSA_STEPS)):
        loss = SARSA.train(replay_buffer)

        if step % 200 == 0:
            print(f'Step: {step},Loss: {loss:.4f}')

    # value learn
    for step in tqdm(range(VALUE_STEPS)):
        loss = VL.train(replay_buffer)

        if step % 200 == 0:
            print(f'Step: {step},Loss: {loss:.4f}')



    QPL.target_Q.load_state_dict(SARSA.Q.state_dict())
    QPL.Q.load_state_dict(SARSA.Q.state_dict())

    # behaviour clone learn
    for step in tqdm(range(BC_STEPS)):
        loss = BCL.train(replay_buffer)

        if step % 200 == 0:
            print(f'Step: {step},Loss: {loss:.4f}')

    # BPPO learn

    BPPOL.policy.load_state_dict(BCL.policy.state_dict())
    BPPOL.old_policy.load_state_dict(BCL.policy.state_dict())
    QQ = SARSA
    score = float('inf')
    for step in tqdm(range(BPPO_STEPS)):
        if step > 200:
            is_clip_decay = False
            is_bppo_lr_decay = False
        loss = BPPOL.train(replay_buffer, QQ, VL, is_clip_decay, is_bppo_lr_decay)
        state, action, _, _, _, _, _ = replay_buffer.sample(16128)
        pred_action = BPPOL.get_action(state)
        current_score = F.l1_loss(pred_action.to(device), action).detach().cpu().numpy()
        if current_score < score:
            score = current_score
            BPPOL.old_policy.load_state_dict(BPPOL.policy.state_dict())
        for i in range(20):
            q_loss = QPL.train(replay_buffer, BPPOL)
        QQ = QPL

        print(f'Step: {step},loss: {loss:.4f},best score : {score:.4f}')

    BPPOL.save_weights()


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

if __name__ == "__main__":
    run_bppo()