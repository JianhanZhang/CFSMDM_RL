import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from environment import sample_trajectory, sample_counterfactual_trajectories

def evaluate_reward_through_simulation(env, z_levels, state_dim, N, T, policy, 
                                       sigma=1, seed=1, z_probs=None, gamma=0.9):
    z_levels = np.array(z_levels)
    np.random.seed(seed)
    Z = np.random.binomial(1, p=0.5, size=[N, z_levels.shape[-1]])
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



def evaluate_demographic_parity_through_simulation(
    env,
    z_levels,
    state_dim,
    N,
    T,
    policy,
    sigma=1,
    seed=1,
    z_probs=None,
):
    np.random.seed(seed)

    # Sample observed sensitive attributes.
    if z_probs is None:
        p_z1 = 0.5
    else:
        z_probs = np.asarray(z_probs, dtype=float)
        if len(z_probs) != 2:
            raise ValueError("z_probs must contain probabilities for Z=0 and Z=1.")
        if np.any(z_probs < 0) or not np.isclose(z_probs.sum(), 1.0):
            raise ValueError("z_probs must be nonnegative and sum to 1.")
        p_z1 = z_probs[1]

    Z = np.random.binomial(n=1, p=p_z1, size=(N, 1))

    # Generate trajectories using the given policy
    _, _, actions, _ = sample_trajectory(env=env, zs=Z, state_dim=state_dim, N=N, T=T, 
                                         sigma=sigma, seed=seed, policy=policy)

    group_0 = Z[:, 0] == 0
    group_1 = Z[:, 0] == 1

    if not np.any(group_0) or not np.any(group_1):
        raise ValueError(
            "Both sensitive groups must be represented in the evaluation sample."
        )

    # Estimate probability of taking action 1 cnditional on each z for each time step.
    prob_action_1_z0 = np.mean(actions[group_0] == 1, axis=0)
    prob_action_1_z1 = np.mean(actions[group_1] == 1, axis=0)

    # Average the absolute group disparity over time.
    dp_metric = np.mean(
        np.abs(prob_action_1_z1 - prob_action_1_z0)
    )

    return float(dp_metric)