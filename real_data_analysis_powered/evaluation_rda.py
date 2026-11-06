import numpy as np
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from environment_simulated_rda import estimate_counterfactual_trajectories_from_data
from fqe_rda import FQE

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



def evaluate_fairness_through_model(preprocessor, zs, states, actions, 
                                    policy, sigma_a=1, seed=1):
    z_eval_levels = np.unique(zs, axis=0)
    z_eval_levels = np.array(z_eval_levels)

    # generate the simulated counterfactual trajectories
    trajectories = estimate_counterfactual_trajectories_from_data(preprocessor, zs, states, 
                                                                  actions, 
                                                                  policy, sigma_a, seed)
    # compute the CF metric
    cf_metric = _compute_cf_metric(trajectories, z_eval_levels, states.shape[0], states.shape[1])

    return cf_metric



def evaluate_reward_through_fqe(
    zs, states, actions, rewards, zs_eval, states_eval, actions_eval, rewards_eval, 
    model_type, policy, uat=None, 
    hidden_dims=[32], lr=0.1, epochs=500, gamma=0.9, max_iter=200, seed=1, **kwargs
):
    # FQE estimate
    num_tries = 0
    while num_tries < 5:
        fqe = FQE(
            action_size=3,
            policy=policy,
            n_actions=3,
            hidden_dims=hidden_dims,
        )
        fqe.fit(
            xs=states,
            zs=zs,
            actions=actions,
            rewards=rewards,
            inner_iter=epochs,
            max_iter=max_iter,
            learning_rate=lr,
        )
        value, fqe_final_loss = fqe.evaluate(
            xs=states_eval, zs=zs_eval, actions=actions_eval
        )
        if fqe_final_loss < 3:
            break
        else:
            print(
                "FQE final loss too large, try to refit the FQE model. fqe_final_loss: {}, num_tries: {}",
                fqe_final_loss,
                num_tries,
            )
        num_tries += 1

    return np.mean(value), fqe_final_loss