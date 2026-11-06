from preprocessor_cfsmdm_qrnn import MSMDMPreprocessorAdd
from preprocessor_cfsdp import SequentialPreprocessor
from preprocessor_flap import FLAP
from postprocessor_ecocf import ECOCF
from preprocessor_baselines import ConcatenatePreprocessor, UnawarenessPreprocessor
from agents import FQI, RandomAgent
from environment_simulated import SimulatedEnvironment, sample_simulated_env_trajectory
from environment_simulated import train_baseline_characteristics_synthesizer, generate_baseline_characteristics
from evaluation import evaluate_reward_through_simulated_environment, evaluate_fairness_through_simulated_environment
from reader import read_trajectory_from_dataframe
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import copy
import torch

def train_simulated_env(z_label, seed=1):
    impute_data = pd.read_csv('./imputed_ihs_data_avgreward.csv') # IHS dataset
    impute_data["msg_type"] = impute_data["msg_type"].replace("none", 0)
    impute_data["msg_type"] = impute_data["msg_type"].replace("mood", 1)
    impute_data["msg_type"] = impute_data["msg_type"].replace("sleep", 1)
    impute_data["msg_type"] = impute_data["msg_type"].replace("step", 1)
    impute_data["Sex"] = impute_data["Sex"].replace(1, 0)
    impute_data["Sex"] = impute_data["Sex"].replace(2, 1)
    impute_data["Child"] = impute_data["Child"].replace(1, 0)
    impute_data["Child"] = impute_data["Child"].replace(2, 1)
    impute_data = impute_data.sort_values(["STUDY_PRTCPT_ID", "week"])
    zs, xs, actions, rewards, ids = read_trajectory_from_dataframe(data=impute_data, 
                                                            z_labels=[z_label], 
                                                            state_labels=["step_count", "sleep_count", "mood"], 
                                                            action_label='msg_type', 
                                                            reward_label='mood', 
                                                            id_label='STUDY_PRTCPT_ID', 
                                                            T=25, 
                                                            )

    torch.manual_seed(seed)
    np.random.seed(seed)
    sim_env = SimulatedEnvironment(action_space=np.array([[0], [1]]), 
                                   epochs=1000)
    sim_env.fit(zs=zs, states=xs, actions=actions, rewards=rewards)
    synthesizer = train_baseline_characteristics_synthesizer(epochs=100, seed=seed)

    return sim_env, synthesizer



def run_exp1_one(methods, method_policy, z_label, N, T, N_eval, T_eval, z_coef, v_coef, env, synthesizer, seed):
    torch.set_num_threads(1)
    T_eval = T_eval
    N_eval = N_eval
    eval_seed = seed + 200
    IS_CF_LOGITS = False
    state_dim = 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    Z = generate_baseline_characteristics(n=N, zvars_cat=[z_label], synthesizer=synthesizer, seed=seed)
    Z = Z[z_label].to_numpy().reshape(-1, 1)
    Z_pre = generate_baseline_characteristics(n=N, zvars_cat=[z_label], synthesizer=synthesizer, seed=seed+100)
    Z_pre = Z_pre[z_label].to_numpy().reshape(-1, 1)
    working_policy = RandomAgent(2)
    qr_iters = 2000

    fqi_model = None
    if method_policy == "FQI_LM":
        learning_algorithm = FQI
        fqi_model = "lm"
        max_iters = 500 
    elif method_policy == "FQI_NN":
        learning_algorithm = FQI
        fqi_model = "nn"
        max_iters = 200
    else:
        raise ValueError("Method policy not found")
    
    # generate trajectory data used for policy learning
    (
        zs,
        xs,
        actions,
        rewards,
    ) = sample_simulated_env_trajectory(
        env, Z, state_dim, N, T, sigma=1, seed=seed, policy=working_policy 
    )

    # generate trajectory data used for preprocessor training
    (
        zs_pre,
        xs_pre,
        actions_pre,
        rewards_pre,
    ) = sample_simulated_env_trajectory(
        env, Z_pre, state_dim, N, T, sigma=1, seed=100+seed, policy=working_policy 
    )

    policies = []
    for method in methods:
        if method == "random":
            np.random.seed(seed)
            torch.manual_seed(seed)
            policies.append(RandomAgent(2))
        elif method == "full":
            np.random.seed(seed)
            torch.manual_seed(seed)
            preprocessor = ConcatenatePreprocessor(
                z_space=np.array([[0], [1]]), 
                action_space=np.array([[0], [1]])
            )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                action_space=np.array([[0], [1]]),
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                seed=seed,
            )
            policies.append(agent)
        elif method == "unaware":
            np.random.seed(seed)
            torch.manual_seed(seed)
            agent = learning_algorithm(
                model_type=fqi_model,
                action_space=np.array([[0], [1]]),
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                seed=seed,
            )
            policies.append(agent)
        elif method == "flap_m":
            np.random.seed(seed)
            torch.manual_seed(seed)
            preprocessor = FLAP(
                z_space=np.array([[0], [1]]), 
                action_space=np.array([[0], [1]]), 
                mode='m'
            )
            preprocessor.train_preprocessor(xs=copy.deepcopy(xs_pre),
                                            zs=copy.deepcopy(zs_pre),
                                            actions=copy.deepcopy(actions_pre),
                                            rewards=copy.deepcopy(rewards_pre))
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                action_space=np.array([[0], [1]]),
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                seed=seed,
            )
            policies.append(agent)
        elif method == "ecocf_m":
            np.random.seed(seed)
            torch.manual_seed(seed)
            agent = ECOCF(
                model_type=fqi_model, 
                z_space=np.array([[0], [1]]), 
                action_space=np.array([[0], [1]]), 
                mode='m'
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                seed=seed,
            )
            policies.append(agent)
        elif method == "cfsdp":
            np.random.seed(seed)
            torch.manual_seed(seed)
            preprocessor = SequentialPreprocessor(
                z_space=np.array([[0], [1]]), 
                action_space=np.array([[0], [1]]), 
                reg_model="nn",
                is_normalized=False,  
                hidden_dims=[256, 128], 
                learning_rate=0.01, 
                batch_size=128, 
                is_action_onehot=True, 
                is_early_stopping=False, 
                early_stopping_patience=20, 
                epochs=200
            )
            preprocessor.train_preprocessor(xs=copy.deepcopy(xs_pre),
                                            zs=copy.deepcopy(zs_pre),
                                            actions=copy.deepcopy(actions_pre),
                                            rewards=copy.deepcopy(rewards_pre), 
                                            seed=seed)
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                action_space=np.array([[0], [1]]),
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                seed=seed,
            )
            policies.append(agent)
        elif method == 'cfsmdm_qrnn':
            np.random.seed(seed)
            torch.manual_seed(seed)
            preprocessor = MSMDMPreprocessorAdd(
                initial_model_specs=[], 
                state_model_specs=[], 
                reward_model_spec=[], 
                z_space=np.array([[0], [1]]), 
                action_space=np.array([[0], [1]]), 
                max_iter=2000,  
                reg_model='qrnn_joint', 
                hidden_dims=[256, 128], 
                learning_rate=0.01, 
                batch_size=128, 
                is_action_onehot=True, 
                is_early_stopping=False, 
                early_stopping_patience=20, 
                epochs=200
            )
            preprocessor.train_preprocessor(xs=copy.deepcopy(xs_pre),
                                            zs=copy.deepcopy(zs_pre),
                                            actions=copy.deepcopy(actions_pre),
                                            rewards=copy.deepcopy(rewards_pre), )
            agent = learning_algorithm(
                preprocessor=preprocessor,
                model_type=fqi_model,
                action_space=np.array([[0], [1]]),
            )
            agent.train(
                xs=copy.deepcopy(xs),
                zs=copy.deepcopy(zs),
                actions=copy.deepcopy(actions),
                rewards=copy.deepcopy(rewards),
                max_iter=max_iters,
                seed=seed,
            )
            policies.append(agent)
        else:
            raise ValueError("Method not found")

    # evaluate reward and fairness
    df_n = pd.DataFrame()
    df_t = pd.DataFrame()

    Z_eval = generate_baseline_characteristics(n=N_eval, zvars_cat=[z_label], synthesizer=synthesizer, seed=eval_seed)
    Z_eval = Z_eval[z_label].to_numpy().reshape(-1, 1)
    for i, policy in enumerate(policies):
        np.random.seed(seed)
        torch.manual_seed(seed)
        discounted_cumulative_reward = evaluate_reward_through_simulated_environment(
            env=env, zs=Z_eval, z_levels=np.array([[0], [1]]), state_dim=state_dim, N=N_eval, T=T_eval, policy=policy, seed=eval_seed
        )
        cf_outs = evaluate_fairness_through_simulated_environment(
            env=env, zs=Z_eval, z_eval_levels=np.array([[0], [1]]), state_dim=state_dim, N=N_eval, T=T_eval, policy=policy, seed=eval_seed
        )

        df_n = pd.concat(
            [
                df_n,
                pd.DataFrame(
                    {
                        "N": [N],
                        "T": [T],
                        "cf": [cf_outs],
                        "reward": [discounted_cumulative_reward],
                        "method": [methods[i]],
                        "zcoef": [z_coef],
                        "vcoef": [v_coef], 
                        "seed": [seed], 
                        "fqi_iters": [max_iters], 
                        "qr_iters": [qr_iters]
                    }
                ),
            ]
        )

    return df_n



def run_exp(rep, env_name, z_label, start_seed=1, export=False, print_result=True, path='./results.csv'):
    NREP = rep
    Ns = [1000]
    Ts = [25]
    N_eval = 10000
    z_coefs = [0.5]
    v_coefs = [0]
    method_policy = "FQI_NN"

    methods = [
        "random",
        "full",
        "unaware",
        "cfsmdm_qrnn", 
        "cfsdp", 
        "flap_m", 
        "ecocf_m"
    ]

    df_n = pd.DataFrame()

    env, synthesizer = train_simulated_env(z_label=z_label, seed=100)

    for N in Ns:
        for T in Ts:
            for z_coef in z_coefs:
                for v_coef in v_coefs:
                    for _ in range(NREP):
                        df_n_one = run_exp1_one(methods=methods, z_label=z_label, 
                            method_policy=method_policy, N=N, T=T, N_eval=N_eval, 
                            T_eval=T, z_coef=z_coef, v_coef=v_coef, env=env, 
                            synthesizer=synthesizer, seed=_+start_seed)
                        df_n = pd.concat([df_n, df_n_one])
                        if print_result:
                            print(df_n)
    
    if export:
        df_n.to_csv(path)
    
    return df_n



df_n = run_exp(rep=50, env_name='m1', z_label='Sex', start_seed=1, export=False, path='./results/results_sex.csv')
