import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from environment_simulated import sample_simulated_env_trajectory, sample_counterfactual_simulated_env_trajectories


# REQUIRES: z_eval_levels must be the one used to generate the counterfactual trajectories
def _compute_cf_metric(trajectories, z_eval_levels, N, T):
    z_eval_levels = np.array(z_eval_levels)

    max_cf_metric = 0
    for i in range(len(z_eval_levels)):
        for j in range(i, len(z_eval_levels)):
            ai = trajectories[tuple(z_eval_levels[i].flatten())]['A']
            aj = trajectories[tuple(z_eval_levels[j].flatten())]['A']
            cf_metric = np.mean(np.mean(np.abs(ai != aj), axis=0))
            if cf_metric > max_cf_metric:
                max_cf_metric = cf_metric

    return max_cf_metric



def evaluate_fairness_through_simulated_environment(env, zs, z_eval_levels, state_dim, N, T, 
                                    policy, sigma_a=1, sigma_x=None, sigma_r=1, seed=1):
    z_eval_levels = np.array(z_eval_levels)
    np.random.seed(seed)

    # generate the simulated counterfactual trajectories
    trajectories = sample_counterfactual_simulated_env_trajectories(env, zs, z_eval_levels, state_dim, 
                                                                    N, T, policy, sigma_a, sigma_x, 
                                                                    sigma_r, seed)
    # compute the CF metric
    cf_metric = _compute_cf_metric(trajectories, z_eval_levels, N, T)

    return cf_metric



def evaluate_reward_through_simulated_environment(env, zs, z_levels, state_dim, N, T, policy, 
                                       sigma=1, seed=1, z_probs=None, gamma=0.9):
    z_levels = np.array(z_levels)
    np.random.seed(seed)
    _, _, _, rewards = sample_simulated_env_trajectory(env, zs, state_dim, N, T, sigma, seed, policy)

    discounted_factor = np.repeat(
        np.array([[gamma**i for i in range(T)]]), repeats=N, axis=0
    )
    discounted_cumulative_reward = np.mean(discounted_factor * rewards) * T
    
    return discounted_cumulative_reward