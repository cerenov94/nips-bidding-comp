import numpy as np
import torch
import pandas as pd


from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.bppo.replay_buffer_upd import ReplayBuffer,NstepRB
from bidding_train_env.bppo.cql import CQL
import ast
from tqdm import tqdm
import torch.nn.functional as F
import os
import random



def run_cql():

    train_model()




def train_model():

    validation_indexes = np.array([   2,    3,    5,    7,    9,   13,   16,   17,   18,   19,   21,
         28,   29,   30,   31,   32,   33,   35,   36,   39,   41,   57,
         59,   60,   64,   66,   67,   76,   77,   78,   80,   81,   83,
         84,   85,   87,   90,   91,   93,   94,   95,   96,   97,   98,
         99,  101,  102,  103,  108,  112,  116,  117,  120,  125,  126,
        132,  133,  136,  137,  138,  142,  144,  146,  147,  149,  150,
        151,  153,  155,  156,  157,  160,  165,  172,  174,  176,  177,
        184,  185,  186,  189,  190,  191,  195,  198,  199,  201,  205,
        207,  208,  211,  212,  213,  216,  218,  220,  223,  224,  225,
        239,  240,  242,  243,  246,  249,  251,  252,  256,  257,  258,
        261,  264,  266,  269,  270,  271,  272,  275,  277,  282,  286,
        288,  290,  291,  293,  295,  297,  300,  301,  308,  314,  316,
        318,  320,  324,  325,  327,  329,  334,  335,  345,  347,  349,
        351,  352,  354,  355,  357,  360,  362,  365,  366,  368,  369,
        371,  372,  373,  375,  376,  377,  378,  381,  382,  383,  393,
        395,  396,  402,  403,  405,  408,  412,  413,  414,  415,  416,
        417,  419,  421,  423,  424,  426,  430,  431,  432,  434,  438,
        441,  445,  448,  450,  451,  453,  456,  458,  460,  461,  462,
        465,  467,  468,  471,  473,  474,  478,  479,  486,  487,  489,
        491,  492,  496,  498,  499,  501,  510,  513,  514,  515,  516,
        517,  519,  526,  527,  528,  537,  539,  544,  545,  546,  551,
        552,  554,  558,  559,  560,  563,  564,  565,  567,  568,  570,
        574,  576,  578,  582,  585,  592,  594,  595,  597,  600,  604,
        608,  609,  611,  612,  617,  618,  626,  630,  636,  642,  644,
        645,  648,  650,  653,  657,  659,  661,  663,  665,  666,  670,
        672,  674,  675,  677,  678,  681,  682,  683,  684,  685,  688,
        691,  693,  700,  711,  712,  713,  714,  717,  718,  720,  722,
        724,  725,  726,  727,  732,  736,  738,  740,  744,  748,  749,
        757,  759,  761,  766,  767,  768,  769,  771,  774,  775,  777,
        779,  784,  786,  787,  788,  789,  794,  796,  797,  798,  800,
        801,  807,  809,  816,  819,  821,  822,  823,  825,  827,  828,
        829,  832,  835,  837,  844,  849,  852,  863,  866,  870,  871,
        872,  873,  875,  876,  880,  881,  882,  883,  885,  887,  894,
        897,  898,  899,  900,  901,  903,  905,  906,  910,  911,  912,
        914,  915,  917,  918,  920,  923,  924,  925,  928,  933,  939,
        940,  941,  942,  943,  949,  951,  953,  954,  958,  959,  960,
        962,  969,  970,  971,  972,  973,  976,  977,  978,  979,  981,
        984,  993,  997,  999, 1001, 1002, 1006, 1007])


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 16

    train_data_path = "./data/traffic/final_rounds/training_data_rlData_folder/training_data_all-rlData.csv"
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
    #training_data['reward_continuous'] = training_data['reward_continuous'] * 0.5 + training_data['reward']


    rewards = training_data['reward_continuous'].values
    dones = training_data['done'].values
    dones = np.split(dones, 1008)

    returns = []
    for chunk, time_step in enumerate(np.split(rewards, 1008)):
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
    actions = np.split(actions, 1008)
    next_actions = []
    for a in actions:
        next_actions.append(np.append(a[1:], 0.))
    training_data['next_action'] = np.concatenate(next_actions)

    chunks = np.split(training_data,1008)
    valid_indexes = []
    for index in validation_indexes:
        valid_indexes.append(chunks[index].index.to_numpy())
    valid_indexes = np.concatenate(valid_indexes)
    del chunks

    valid_data = training_data.loc[valid_indexes].reset_index(drop=True)

    valid_replay_buffer = ReplayBuffer(device)

    replay_buffer = ReplayBuffer(device=device)


    normalize_dic = normalize_state(training_data, state_dim, normalize_indices=[13, 14, 15],train=True)
    training_data['reward'],min_reward_stat,reward_range_stat = normalize_reward(training_data, 'reward_continuous')
    save_normalize_dict(normalize_dic, "saved_model/CQLtest")
    add_to_replay_buffer(replay_buffer, training_data, True)

    valid_data['normalize_reward'] = (valid_data['reward_continuous'] - min_reward_stat) / reward_range_stat
    stats = normalize_state(valid_data, state_dim, normalize_indices=[13, 14, 15], train=False,normalize_dict=normalize_dic)
    add_to_replay_buffer(valid_replay_buffer, valid_data, True)

    #replay_buffer.split_memory(flag=True)
    print(f'train size: {len(replay_buffer)}, valid size: {len(valid_replay_buffer)}')

    cql_model = CQL(state_dim,1,5e-3,256,2e-4,1.,True,1.0,10.,device,32)
    best_score = -float('inf')
    for step in tqdm(range(1,10001)):
        d  = cql_model.learn(replay_buffer)
        print(f"\n\npolicy loss: {d['policy_loss']:.4f}, alpha loss: {d['alpha_loss']:.4f},\ncritic1 loss: {d['q1_loss']:.4f}"
              f" critic2 loss: {d['q2_loss']:4f}, cql1 loss: {d['cql1_loss']:.4f}, cql2 loss: {d['cql2_loss']:.4f}\n"
              f"current alpha: {d['current_alpha']:.4f}, cql alpha loss: {d['cql_alpha_loss']:.4f}, cql alpha: {d['cql_alpha']:.4f}\n"
              )


    cql_model.save_weights()


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
    run_cql()