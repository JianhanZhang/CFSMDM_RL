import numpy as np
import numpy as np
from environment import sample_trajectory, sample_counterfactual_trajectories

def evaluate_reward_through_simulation(env, z_levels, state_dim, N, T, policy, 
                                       sigma=1, seed=1, z_probs=None, gamma=0.9):
    z_levels = np.array(z_levels)
    np.random.seed(seed)
    Z = np.random.binomial(1, p=0.5, size=[N, 1])
    _, _, _, rewards = sample_trajectory(env, Z, state_dim, N, T, sigma, seed, policy)

    discounted_factor = np.repeat(
        np.array([[gamma**i for i in range(T)]]), repeats=N, axis=0
    )
    discounted_cumulative_reward = np.mean(discounted_factor * rewards) * T
    
    return discounted_cumulative_reward



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



def evaluate_fairness_through_simulation(env, z_eval_levels, state_dim, N, T, 
                                         policy, sigma=1, z_probs=None, seed=1):
    z_eval_levels = np.array(z_eval_levels)
    np.random.seed(seed)

    zs = np.random.binomial(n=1, p=1/2, size=[N, z_eval_levels.shape[1]])

    # generate the simulated counterfactual trajectories
    trajectories = sample_counterfactual_trajectories(env, zs, z_eval_levels, state_dim, N, T, 
                                                    policy, sigma, seed)
    # compute the CF metric
    cf_metric = _compute_cf_metric(trajectories, z_eval_levels, N, T)

    return cf_metric


