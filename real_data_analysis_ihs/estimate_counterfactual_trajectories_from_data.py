import numpy as np
import torch
import warnings
import sys
import copy

sys.path.append("../../core")

warnings.simplefilter(action="ignore", category=FutureWarning)
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from base_models_rda import NeuralNetRegressor, LinearRegressor
from agents_rda import RandomAgent

def estimate_counterfactual_trajectories_from_data(preprocessor, zs, states, 
                                                   actions, policy, 
                                                   sigma_a=1, seed=1):
    N = actions.shape[0]
    T = actions.shape[1]
    state_dim = states.shape[2]
    np.random.seed(seed)
    torch.manual_seed(seed)
    z_eval_levels = np.unique(zs, axis=0)
    z_eval_levels = np.array(z_eval_levels)

    # dictionary to contain the counterfactual trajectories; key = z and value = trajectory
    trajectories = {}
    for z_level in z_eval_levels:
        preprocessor_z = copy.deepcopy(preprocessor)
        policy_z = copy.deepcopy(policy)
        Z = np.repeat(z_level, repeats=N, axis=0).reshape(N, -1)
        X = np.zeros([N, T + 1, state_dim], dtype=float)
        A = np.zeros([N, T + 1], dtype=int)
        R = np.zeros([N, T], dtype=float)
        trajectories[tuple(z_level.flatten())] = {'Z': Z, 'X': X, 'A': A, 'R': R, 
                                        'preprocessor_z': preprocessor_z, 'policy_z': policy_z}

    # obtain the estimated counterfactual states using MSMDM
    xs_tilde = preprocessor.preprocess_multiple_steps(zs=zs, 
                                                      xs=states, 
                                                      actions=actions)
    
    # organize the counterfactual states into the disctionary format
    for i, z_level in enumerate(z_eval_levels):
        for t in range(T + 1):
            trajectories[tuple(z_level.flatten())]['X'][:, t] = xs_tilde[:, t, i*state_dim:(i+1)*state_dim]

    # take the first step
    ua0 = np.random.uniform(0, sigma_a, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, 0] = p.act(
                t=0, 
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, 0],
                xtm1=None,
                atm1=None,
                uat=ua0,
            )

    # take subsequent steps
    for t in range(1, T):
        uat = np.random.uniform(0, sigma_a, size=[N])
        for z_level in z_eval_levels:
            p = trajectories[tuple(z_level.flatten())]['policy_z']
            trajectories[tuple(z_level.flatten())]['A'][:, t] = p.act(
                    t=t, 
                    z=trajectories[tuple(z_level.flatten())]['Z'],
                    xt=trajectories[tuple(z_level.flatten())]['X'][:, t],
                    xtm1=trajectories[tuple(z_level.flatten())]['X'][:, t - 1],
                    #atm1=trajectories[tuple(z_level.flatten())]['A'][:, t - 1],
                    atm1=actions[:, t - 1],
                    uat=uat,
                )
    
    # take the final step
    ua0 = np.random.uniform(0, sigma_a, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, T] = p.act(
                t=T, 
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, T],
                xtm1=trajectories[tuple(z_level.flatten())]['X'][:, T - 1],
                atm1=actions[:, T - 1],
                uat=ua0,
            )
    
    return trajectories