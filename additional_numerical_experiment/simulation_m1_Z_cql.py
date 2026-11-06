from preprocessor_cfsmdm import MSMDMPreprocessor
from preprocessor_cfsdp import SequentialPreprocessor
from preprocessor_flap import FLAP
from postprocessor_ecocf_cql import ECOCFCQL
from preprocessor_baselines import ConcatenatePreprocessor, UnawarenessPreprocessor
from agents import FQI, RandomAgent
from agents_ddqn import OfflineDoubleDQN
from agents_cql import CQL
from environment import SyntheticEnvironment, sample_counterfactual_trajectories, sample_trajectory
from evaluation import evaluate_fairness_through_simulation, evaluate_reward_through_simulation
from evaluation import evaluate_demographic_parity_through_simulation
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import copy
import torch
import random

def run_exp1_one(methods, method_policy, N, T, N_eval, T_eval, z_coef, v_coef, env_name, seed):
    torch.set_num_threads(1)
    T_eval = T_eval
    N_eval = N_eval
    eval_seed = seed + 200
    if env_name == 'm1' or env_name == 'r1':
        state_dim = 2
    elif env_name == 'm2' or env_name == 'r2':
        state_dim = 1
    else:
        raise ValueError('Undefined synthetic environment.')
    env = SyntheticEnvironment(z_coef=z_coef, v_coef=v_coef, env_name=env_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Z = np.random.binomial(1, 0.5, size=[N]).reshape(N, -1)
    Z_pre = np.random.binomial(1, 0.5, size=[N]).reshape(N, -1)
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
    elif method_policy == "CQL":
        learning_algorithm = CQL
        fqi_model = 'nn'
        max_iters = N
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
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            policies.append(RandomAgent(2))
        elif method == "full":
            random.seed(seed)
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
            random.seed(seed)
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
            random.seed(seed)
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
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            agent = ECOCFCQL(
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
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            preprocessor = SequentialPreprocessor(
                z_space=np.array([[0], [1]]), 
                action_space=np.array([[0], [1]]), 
                reg_model="nn",
                is_normalized=False,
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
        elif method == "cfsmdm":
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if env_name == 'm1' or env_name == 'r1':
                specs_init = ['state1 ~ 1 + zs', 'state2 ~ 1 + zs']
                specs_state = ['next_state1 ~ 1 + zs + state1 + I(state1**2) + I(state1**3) + actions + I(zs*state1) + I(zs*actions) + I(state1*actions)', 
                            'next_state2 ~ 1 + zs + state1 + I(state1**2) + I(state1**3) + state2 + I(state2**2) + I(state2**3) + actions + I(zs*state1) + I(zs*state2) + I(zs*actions) + I(state1*actions) + I(state2*actions)']
                spec_reward = 'rewards ~ 1 + zs + state1 + I(state1**2) + I(state1**3) + state2 + I(state2**2) + I(state2**3) + actions + I(zs*state1) + I(zs*state2) + I(zs*actions) + I(state1*actions) + I(state2*actions)'
            elif env_name == 'm2' or env_name == 'r2':
                specs_init = ['state1 ~ 1 + zs']
                specs_state = ['next_state1 ~ 1 + zs + state1 + actions + I(zs*state1) + I(zs*actions) + I(state1*actions)']
                spec_reward = 'rewards ~ 1 + zs + state1 + actions + I(zs*state1) + I(zs*actions) + I(state1*actions)'
            else:
                raise ValueError('Undefined synthetic environment.')

            preprocessor = MSMDMPreprocessor(
                initial_model_specs=specs_init, 
                state_model_specs=specs_state, 
                reward_model_spec=spec_reward, 
                max_iter=qr_iters
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
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        discounted_cumulative_reward = evaluate_reward_through_simulation(
            env=env, z_levels=np.array([[0], [1]]), state_dim=state_dim, N=N_eval, T=T_eval, policy=policy, sigma=1, seed=eval_seed
        )
        cf_outs = evaluate_fairness_through_simulation(
            env, np.array([[0], [1]]), state_dim, N_eval, T_eval, policy, sigma=1, seed=eval_seed
        )
        dp_outs = evaluate_demographic_parity_through_simulation(
            env=env, 
            z_levels=np.array([[0], [1]]), 
            state_dim=state_dim, 
            N=N_eval, T=T_eval, policy=policy, sigma=1, seed=eval_seed)

        df_n = pd.concat(
            [
                df_n,
                pd.DataFrame(
                    {
                        "env": [env_name], 
                        "policy": [method_policy], 
                        "N": [N],
                        "T": [T],
                        "cf": [cf_outs],
                        "dp": [dp_outs], 
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



def run_exp(rep, env_name, start_seed=1, export=False, print_result=True, path='./results.csv'):
    NREP = rep
    Ns = [500]
    Ts = [20]
    N_eval = 10000
    z_coefs = [0.0, 1.0, 2.0, 3.0, 4.0]
    v_coefs = [0]
    method_policy = "CQL"

    methods = [
        "random",
        "full",
        "unaware",
        "cfsmdm",
        "cfsdp", 
        "flap_m", 
        "ecocf_m"
    ]

    df_n = pd.DataFrame()

    for N in Ns:
        for T in Ts:
            for z_coef in z_coefs:
                for v_coef in v_coefs:
                    for _ in range(NREP):
                        df_n_one = run_exp1_one(methods=methods, 
                            method_policy=method_policy, N=N, T=T, N_eval=N_eval, 
                            T_eval=T, z_coef=z_coef, v_coef=v_coef, env_name=env_name, 
                            seed=_+start_seed)
                        df_n = pd.concat([df_n, df_n_one])
                        if print_result:
                            print(df_n)
    
    if export:
        df_n.to_csv(path)
    
    return df_n



df_n = run_exp(rep=50, env_name='m1', start_seed=1, export=True, path='./results/results_m1_Z_cql.csv')