from preprocessor_cfsmdm_vecz import MSMDMPreprocessor
from preprocessor_cfsdp import SequentialPreprocessor
from preprocessor_flap import FLAP
from postprocessor_ecocf import ECOCF
from preprocessor_baselines import ConcatenatePreprocessor, UnawarenessPreprocessor
from agents import FQI, RandomAgent
from environment import SyntheticEnvironment, sample_counterfactual_trajectories, sample_trajectory
from evaluation import evaluate_fairness_through_simulation, evaluate_reward_through_simulation
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import copy
import torch

def run_exp1_one(methods, method_policy, N, T, N_eval, T_eval, z_coef, env_name, seed):
    torch.set_num_threads(1)
    T_eval = T_eval
    N_eval = N_eval
    eval_seed = seed + 200
    if env_name == 'm1':
        state_dim = 2
    elif env_name == 'm2':
        state_dim = 1
    elif env_name == 'm3':
        state_dim = 20
    else:
        raise ValueError('Undefined synthetic environment.')
    env = SyntheticEnvironment(z_coef=z_coef, env_name=env_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Z = np.random.binomial(1, 0.5, size=[N, 2]).reshape(N, -1)
    Z_pre = np.random.binomial(1, 0.5, size=[N, 2]).reshape(N, -1)
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
    ) = sample_trajectory(
        env, Z, state_dim, N, T, sigma=1, seed=seed, policy=working_policy 
    )

    # generate trajectory data used for preprocessor training
    (
        zs_pre,
        xs_pre,
        actions_pre,
        rewards_pre,
    ) = sample_trajectory(
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
                z_space=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 
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
                z_space=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 
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
                z_space=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 
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
                z_space=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 
                action_space=np.array([[0], [1]]), 
                reg_model="nn",
                is_normalized=False,
                hidden_dims=[512, 256], 
                learning_rate=1e-4, 
                batch_size=128, 
                is_early_stopping=True, 
                early_stopping_patience=10
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
        elif method == 'cfsmdm_qrnn_joint':
            np.random.seed(seed)
            torch.manual_seed(seed)
            preprocessor = MSMDMPreprocessor(
                initial_model_specs=[], 
                state_model_specs=[], 
                reward_model_spec=[], 
                z_space=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 
                max_iter=qr_iters,  
                reg_model='qrnn_joint', 
                hidden_dims=[512, 256], 
                learning_rate=1e-4, 
                batch_size=128, 
                early_stopping_patience=10
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
    
    for i, policy in enumerate(policies):
        np.random.seed(seed)
        torch.manual_seed(seed)
        discounted_cumulative_reward = evaluate_reward_through_simulation(
            env=env, z_levels=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), state_dim=state_dim, N=N_eval, T=T_eval, policy=policy, sigma=1, seed=eval_seed
        )
        cf_outs = evaluate_fairness_through_simulation(
            env, np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), state_dim, N_eval, T_eval, policy, sigma=1, seed=eval_seed
        )

        df_n = pd.concat(
            [
                df_n,
                pd.DataFrame(
                    {
                        "env_name": [env_name], 
                        "N": [N],
                        "T": [T],
                        "cf": [cf_outs],
                        "reward": [discounted_cumulative_reward],
                        "method": [methods[i]],
                        "zcoef": [z_coef],
                        "seed": [seed], 
                        "fqi_iters": [max_iters], 
                        "qr_iters": [qr_iters]
                    }
                ),
            ]
        )

    return df_n



def run_exp(rep, env_name, start_seed=1, export=False, print_result=True, path='./results.csv'):
    NREP = rep
    Ns = [500, 1000, 2000, 3000, 5000]
    Ts = [20]
    N_eval = 10000
    z_coefs = [1.0]
    method_policy = "FQI_NN"

    methods = [ 
        "random",
        "full",
        "unaware",
        "cfsdp", 
        "flap_m", 
        "ecocf_m", 
        "cfsmdm_qrnn_joint", 
    ]

    df_n = pd.DataFrame()

    for N in Ns:
        for T in Ts:
            for z_coef in z_coefs:
                for _ in range(NREP):
                    df_n_one = run_exp1_one(methods=methods, 
                          method_policy=method_policy, N=N, T=T, N_eval=N_eval, 
                          T_eval=T, z_coef=z_coef, env_name=env_name, seed=_+start_seed)
                    df_n = pd.concat([df_n, df_n_one])
                    if print_result:
                        print(df_n)
    
    if export:
        df_n.to_csv(path)
    
    return df_n



df_n = run_exp(rep=50, env_name='m3', start_seed=1, export=True, path='./results/results_m3_N.csv')