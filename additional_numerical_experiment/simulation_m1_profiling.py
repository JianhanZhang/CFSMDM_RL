from preprocessor_cfsmdm import MSMDMPreprocessor
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
import time

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

    wallclock_times_train = {'type': ['wallclock_times_train'], 'seed': [seed]}
    cpu_times_train = {'type': ['cpu_times_train'], 'seed': [seed]}
    policies = []
    for method in methods:
        if method == "random":
            np.random.seed(seed)
            torch.manual_seed(seed)
            cpu_start = time.process_time()
            wallclock_start = time.perf_counter()
            agent = RandomAgent(2)
            wallclock_elapsed = time.perf_counter() - wallclock_start
            cpu_elapsed = time.process_time() - cpu_start
            wallclock_times_train[method] = wallclock_elapsed
            cpu_times_train[method] = cpu_elapsed
            policies.append(agent)
        elif method == "full":
            np.random.seed(seed)
            torch.manual_seed(seed)
            cpu_start = time.process_time()
            wallclock_start = time.perf_counter()
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
            wallclock_elapsed = time.perf_counter() - wallclock_start
            cpu_elapsed = time.process_time() - cpu_start
            wallclock_times_train[method] = wallclock_elapsed
            cpu_times_train[method] = cpu_elapsed
            policies.append(agent)
        elif method == "unaware":
            np.random.seed(seed)
            torch.manual_seed(seed)
            cpu_start = time.process_time()
            wallclock_start = time.perf_counter()
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
            wallclock_elapsed = time.perf_counter() - wallclock_start
            cpu_elapsed = time.process_time() - cpu_start
            wallclock_times_train[method] = [wallclock_elapsed]
            cpu_times_train[method] = [cpu_elapsed]
            policies.append(agent)
        elif method == "flap_m":
            np.random.seed(seed)
            torch.manual_seed(seed)
            cpu_start = time.process_time()
            wallclock_start = time.perf_counter()
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
            wallclock_elapsed = time.perf_counter() - wallclock_start
            cpu_elapsed = time.process_time() - cpu_start
            wallclock_times_train[method] = wallclock_elapsed
            cpu_times_train[method] = cpu_elapsed
            policies.append(agent)
        elif method == "ecocf_m":
            np.random.seed(seed)
            torch.manual_seed(seed)
            cpu_start = time.process_time()
            wallclock_start = time.perf_counter()
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
            wallclock_elapsed = time.perf_counter() - wallclock_start
            cpu_elapsed = time.process_time() - cpu_start
            wallclock_times_train[method] = wallclock_elapsed
            cpu_times_train[method] = cpu_elapsed
            policies.append(agent)
        elif method == "cfsdp":
            np.random.seed(seed)
            torch.manual_seed(seed)
            cpu_start = time.process_time()
            wallclock_start = time.perf_counter()
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
            wallclock_elapsed = time.perf_counter() - wallclock_start
            cpu_elapsed = time.process_time() - cpu_start
            wallclock_times_train[method] = wallclock_elapsed
            cpu_times_train[method] = cpu_elapsed
            policies.append(agent)
        elif method == "cfsmdm":
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

            cpu_start = time.process_time()
            wallclock_start = time.perf_counter()
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
            wallclock_elapsed = time.perf_counter() - wallclock_start
            cpu_elapsed = time.process_time() - cpu_start
            wallclock_times_train[method] = wallclock_elapsed
            cpu_times_train[method] = cpu_elapsed
            policies.append(agent)
        else:
            raise ValueError("Method not found")

    # evaluate runtime for taking actions
    wallclock_times_act = {'type': ['wallclock_times_act'], 'seed': [seed]}
    cpu_times_act = {'type': ['cpu_times_act'], 'seed': [seed]}
    for i, policy in enumerate(policies):
        np.random.seed(seed)
        torch.manual_seed(seed)
        Z_eval = np.random.binomial(1, p=0.5, size=[N, 1])
        cpu_start = time.process_time()
        wallclock_start = time.perf_counter()
        _, _, _, _ = sample_trajectory(env, Z_eval, state_dim, N, T, 1, seed, policy)
        wallclock_elapsed = time.perf_counter() - wallclock_start
        cpu_elapsed = time.process_time() - cpu_start
        wallclock_times_act[methods[i]] = [wallclock_elapsed]
        cpu_times_act[methods[i]] = [cpu_elapsed]

    return wallclock_times_train, cpu_times_train, wallclock_times_act, cpu_times_act



def run_exp(rep, env_name, start_seed=1, export=False, print_result=True, path='./results.csv'):
    NREP = rep
    N = 500
    T = 20
    N_eval = 1000 
    z_coef = 2.0
    v_coef = 0
    method_policy = "FQI_NN"

    methods = [
        "random",
        "full",
        "unaware",
        "cfsmdm",
        "cfsdp", 
        "flap_m", 
        "ecocf_m", 
    ]

    runtimes = pd.DataFrame()
    for _ in range(NREP):
        wtt_one, ctt_one, wta_one, cta_one = run_exp1_one(
            methods=methods, method_policy=method_policy, N=N, T=T, N_eval=N_eval, 
            T_eval=T, z_coef=z_coef, v_coef=v_coef, env_name=env_name, 
            seed=_+start_seed) # wtt = wallclock_times_train synonymized etc.
        runtimes = pd.concat([runtimes, pd.DataFrame(wtt_one)])
        runtimes = pd.concat([runtimes, pd.DataFrame(ctt_one)])
        runtimes = pd.concat([runtimes, pd.DataFrame(wta_one)])
        runtimes = pd.concat([runtimes, pd.DataFrame(cta_one)])
        if print_result:
            print(runtimes)
    
    if export:
        runtimes.to_csv(path)
    
    return runtimes



runtimes = run_exp(rep=40, env_name='m1', start_seed=11, export=True, path='./results/results_m1_profiling.csv')