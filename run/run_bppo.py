import numpy as np
import torch
import pandas as pd


from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.bppo.replay_buffer_upd import ReplayBuffer
from bidding_train_env.bppo.bppo import Value,QLSarsa,QP,BC,BPPO,QLVect
import ast
from tqdm import tqdm
import torch.nn.functional as F
import os
import random
from sklearn.model_selection import KFold



def run_bppo(
        seed,
        with_validation,
        state_dim,
        learn_value,
        learn_policy,
        device,
        value_steps,
        value_bs,
        value_lr,
        value_hidden_dim,
        v_expect,
        qvalue_steps,
        qvalue_bs,
        qvalue_hidden_dim,
        qvalue_lr,
        qvalue_update_freq,
        qvalue_tau,
        qvalue_gamma,
        n_critics,
        eta,
        bc_steps,
        bc_hidden_dim,
        bc_lr,
        bc_bs,
        bc_temp,
        activation,
        n_layers,
        dropout,
        bppo_steps,
        bppo_lr,
        bppo_bs,
        is_clip_decay,
        is_bppo_lr_decay,
        clip_ratio,
        omega
):

    # scores = []
    # kf = KFold(3,shuffle=True,random_state=42)
    # for _,validation_indexes in kf.split(np.arange(1008)):
    #     score = train_model(
    #         seed,
    #         with_validation,
    #         state_dim,
    #         learn_value,
    #         learn_policy,
    #         device,
    #         value_steps,
    #         value_bs,
    #         value_lr,
    #         value_hidden_dim,
    #         v_expect,
    #         qvalue_steps,
    #         qvalue_bs,
    #         qvalue_hidden_dim,
    #         qvalue_lr,
    #         qvalue_update_freq,
    #         qvalue_tau,
    #         qvalue_gamma,
    #         n_critics,
    #         eta,
    #         bc_steps,
    #         bc_hidden_dim,
    #         bc_lr,
    #         bc_bs,
    #         bc_temp,
    #         activation,
    #         n_layers,
    #         dropout,
    #         bppo_steps,
    #         bppo_lr,
    #         bppo_bs,
    #         is_clip_decay,
    #         is_bppo_lr_decay,
    #         clip_ratio,
    #         omega,
    #         validation_indexes
    #     )
    #     scores.append(score)
    #
    #validation_indexes = np.random.choice(1008, 500, replace=False, )
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
    scores = train_model(
        seed,
        with_validation,
        state_dim,
        learn_value,
        learn_policy,
        device,
        value_steps,
        value_bs,
        value_lr,
        value_hidden_dim,
        v_expect,
        qvalue_steps,
        qvalue_bs,
        qvalue_hidden_dim,
        qvalue_lr,
        qvalue_update_freq,
        qvalue_tau,
        qvalue_gamma,
        n_critics,
        eta,
        bc_steps,
        bc_hidden_dim,
        bc_lr,
        bc_bs,
        bc_temp,
        activation,
        n_layers,
        dropout,
        bppo_steps,
        bppo_lr,
        bppo_bs,
        is_clip_decay,
        is_bppo_lr_decay,
        clip_ratio,
        omega,
        validation_indexes
    )
    #return np.mean(scores)
    return scores




def train_model(
        seed,
        with_validation,
        state_dim,
        learn_value:bool,
        learn_policy,
        device,
        value_steps,
        value_bs,
        value_lr,
        value_hidden_dim,
        v_expect,
        qvalue_steps,
        qvalue_bs,
        qvalue_hidden_dim,
        qvalue_lr,
        qvalue_update_freq,
        qvalue_tau,
        qvalue_gamma,
        n_critics,
        eta,
        bc_steps,
        bc_hidden_dim,
        bc_lr,
        bc_bs,
        bc_temp,
        activation,
        n_layers,
        dropout,
        bppo_steps,
        bppo_lr,
        bppo_bs,
        is_clip_decay,
        is_bppo_lr_decay,
        clip_ratio,
        omega,
        validation_indexes,

):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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



    valid_data = training_data.loc[valid_indexes].reset_index(drop = True)
    WITH_VALIDATION = with_validation
    if WITH_VALIDATION:
        training_data = training_data.drop(valid_indexes).reset_index(drop = True)
    training_data = training_data.loc[valid_indexes].reset_index(drop=True)

    valid_replay_buffer = ReplayBuffer(device=device)

    replay_buffer = ReplayBuffer(device=device)


    normalize_dic = normalize_state(training_data, state_dim, normalize_indices=[13, 14, 15],train=True)
    training_data['reward'],min_reward_stat,reward_range_stat = normalize_reward(training_data, 'reward_continuous')
    save_normalize_dict(normalize_dic, "saved_model/BPPOtest")
    add_to_replay_buffer(replay_buffer, training_data, True)
    valid_data['normalize_reward'] = (valid_data['reward_continuous'] - min_reward_stat)/reward_range_stat
    stats = normalize_state(valid_data,state_dim,normalize_indices=[13,14,15],train=False,normalize_dict=normalize_dic)
    add_to_replay_buffer(valid_replay_buffer,valid_data,True)

    #replay_buffer.split_memory(flag=True)
    print(f'train size: {len(replay_buffer)}, valid size: {len(valid_replay_buffer)}')

    VALUE_STEPS = value_steps
    SARSA_STEPS = qvalue_steps
    BC_STEPS = bc_steps
    BPPO_STEPS = bppo_steps

    VL = Value(dim_obs=state_dim,hidden_dim=value_hidden_dim,lr=value_lr,batch_size=value_bs,device=device,expectile = v_expect)
    # QEnsemble = QLVect(
    #     dim_obs=state_dim,
    #     hidden_dim=qvalue_hidden_dim,
    #     lr=qvalue_lr,
    #     update_freq=qvalue_update_freq,
    #     tau=qvalue_tau,
    #     gamma=qvalue_gamma,
    #     batch_size=qvalue_bs,
    #     num_critics=n_critics,
    #     device=device,
    #     eta = eta
    # )
    BCL = BC(
        dim_obs=state_dim,
        hidden_dim=bc_hidden_dim,
        lr = bc_lr,
        batch_size=bc_bs,
        device = device,
        temp = bc_temp,
        activation=activation,
        n_layers=n_layers,
        dropout=dropout
    )
    BPPOL = BPPO(
        dim_obs=state_dim,
        hidden_dim=bc_hidden_dim,
        lr=bppo_lr,
        batch_size = bppo_bs,
        device = device,
        clip_ratio=clip_ratio,
        n_steps=bppo_steps,
        omega=omega,
        activation=activation,
        n_layers=n_layers,
        dropout=dropout
    )

    SARSA1 = QLSarsa(
        dim_obs=state_dim,
        hidden_dim=qvalue_hidden_dim,
        lr=qvalue_lr,
        update_freq=qvalue_update_freq,
        tau=qvalue_tau,
        gamma=qvalue_gamma,
        batch_size=qvalue_bs,
        device=device
    )
    SARSA2 = QLSarsa(
        dim_obs=state_dim,
        hidden_dim=qvalue_hidden_dim,
        lr=qvalue_lr,
        update_freq=qvalue_update_freq,
        tau=qvalue_tau,
        gamma=qvalue_gamma,
        batch_size=qvalue_bs,
        device=device
    )



    # learn value functions
    if learn_value:
        # sarsa learn
        for step in tqdm(range(SARSA_STEPS)):

            value_loss = VL.train(replay_buffer=replay_buffer,Q1=SARSA1,Q2=SARSA2)
            loss1 = SARSA1.train(replay_buffer, V=VL)
            loss2 = SARSA2.train(replay_buffer, V=VL)

            if step % 200 == 0:
                print(f'Step: {step},Value loss: {value_loss:.4f},Loss1: {loss1:.4f},Loss2: {loss2:.4f}')
    else:
        value_path = os.path.join("saved_model", "BPPOtest", "value_model_freezed.pth")
        VL.load_weights(value_path)
        q_path = os.path.join('saved_model','BPPOtest','Q1_freezed.pth')
        SARSA1.load_weights(q_path)
        SARSA2.load_weights(q_path)


    # BPPO learn
    if learn_policy:
        # behaviour clone learn
        for step in tqdm(range(BC_STEPS)):
            loss = BCL.train(replay_buffer,SARSA1,SARSA2,VL)

            if step % 200 == 0:
                print(f'Step: {step},Loss: {loss:.4f}')

        BPPOL.policy.load_state_dict(BCL.policy.state_dict())
        BPPOL.old_policy.load_state_dict(BCL.policy.state_dict())

        QQ1 = SARSA1
        QQ2 = SARSA2

        best_score = float('inf')
        current_score = 0
        for step in tqdm(range(1,BPPO_STEPS + 1)):
            if step > 200:
                is_clip_decay = False
                #is_bppo_lr_decay = False

            loss,approx_kl = BPPOL.train(replay_buffer, QQ1,QQ1, VL, is_clip_decay, is_bppo_lr_decay)
            if step % 200 == 0:
                BPPOL.policy.eval()
                state, action, reward, next_state, next_action, done, G = valid_replay_buffer.sample(1024,
                                                                                                     random_samples=False)
                with torch.no_grad():
                    q1 = SARSA1(state, action)
                    q2 = SARSA2(state, action)
                    min_Q = torch.min(q1, q2)
                    pdf = BPPOL.policy.get_pdf(next_state)
                    new_action = pdf.rsample()
                    q1 = SARSA1(state, new_action)
                    q2 = SARSA2(state, new_action)
                    target_Q = torch.min(q1, q2)
                current_score = ((min_Q - reward - qvalue_gamma * target_Q).pow(2)).mean()
                if current_score < best_score:
                    best_score = current_score
                    BPPOL.old_policy.load_state_dict(BPPOL.policy.state_dict())

            #for i in range(2):
                #q_loss1 = QPL1.train(replay_buffer, BPPOL)
                #q_loss2 = QPL2.train(replay_buffer, BPPOL)
            #QQ1 = QPL1
            #QQ2 = QPL2

            print(f'Step: {step},loss: {loss:.4f},current score: {current_score:.4f}, best score : {best_score:.4f},approx kl:{approx_kl:.4f}')
            #if approx_kl < delta:
            #    break

        BPPOL.policy.eval()
        state, action, reward, next_state, next_action, done, G = valid_replay_buffer.sample(1024, random_samples=False)
        with torch.no_grad():
            q1 = SARSA1(state, action)
            q2 = SARSA2(state, action)
            min_Q = torch.min(q1, q2)
            pdf = BPPOL.policy.get_pdf(next_state)
            new_action = pdf.rsample()
            q1 = SARSA1(state, new_action)
            q2 = SARSA2(state, new_action)
            target_Q = torch.min(q1, q2)
        score = ((min_Q - reward - qvalue_gamma * target_Q).pow(2)).mean()
        pred_action,_ = BPPOL.get_action(state)
        score_action = np.abs(action.cpu().numpy() - pred_action.numpy()).mean()
        with torch.no_grad():
            q1 = QQ1(state, action)
            q2 = QQ2(state, action)
            min_Q = torch.min(q1, q2)
            q1 = QQ1(state,pred_action.to(device))
            q2 = QQ2(state,pred_action.to(device))
            min_Q_pred = torch.min(q1, q2)
        tem = np.concatenate((action.cpu().numpy(), pred_action.numpy(),reward.cpu().numpy()), axis=1)
        print("DETERMINISTIC POLICY")
        print("action | pred action | reward:", tem)




        logs = torch.load('train_logs/logs.pth')
        best_score = float('inf')

        print(f'value score: {score:.5f}, mean deviation action: {score_action:.4f}')
        if score < best_score:
            best_score = score
            BPPOL.save_weights()
            VL.save_weights()
            QQ1.save_weights(name = 'Q1')
            QQ2.save_weights(name = 'Q2')
            torch.save({
                'min_valid_action_deviation': best_score,
                'results':tem,
                'Qvalues':torch.stack([min_Q,min_Q_pred]).detach().cpu().numpy()
            },'train_logs/logs1.pth')


        return score

    else:
        # FQE
        BPPOL = BPPO(
            dim_obs=state_dim,
            hidden_dim=256,
            lr=bppo_lr,
            batch_size=bppo_bs,
            device=device,
            clip_ratio=clip_ratio,
            n_steps=bppo_steps,
            omega=omega
        )
        p_path = os.path.join('saved_model', 'BPPOtest', 'bppo_model_freezed.pth')
        BPPOL.load_weights(p_path)
        BPPOL.policy.to(device)
        state, action, reward, next_state, next_action, done, G = valid_replay_buffer.sample(1024, random_samples=False)
        with torch.no_grad():
            q1 = SARSA1(state, action)
            q2 = SARSA2(state, action)
            min_Q = torch.min(q1, q2)
            pdf = BPPOL.policy.get_pdf(next_state)
            new_action = pdf.rsample()
            q1 = SARSA1(state, new_action)
            q2 = SARSA2(state, new_action)
            target_Q = torch.min(q1, q2)
        score = ((min_Q - reward - qvalue_gamma * target_Q).pow(2)).mean()
        print(f'Score: {score:.4f}')
        SARSA1.save_weights(name = 'Q1')
        SARSA2.save_weights(name = 'Q2')
        VL.save_weights()
        return score.item()

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
    score = run_bppo()