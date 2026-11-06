import gymnasium as gym
import numpy as np
import copy
from get_env import get_env

class SyntheticEnvironment(gym.Env):
    def __init__(self, env_name='m1', state_dim=1, z_coef=1):
        # initialize member variables from the parameter list
        self.state_dim = state_dim
        self.z_coef = z_coef
        self.f_x0, self.f_xt, self.f_rt = get_env(name=env_name)

        # temporarily store the state, action, and reward data of the environment
        self.xt = np.zeros(state_dim)
        self.atm1 = None
        self.rtm1 = 0
    
    def get_z_coef(self):
        return self.z_coef
    
    def set_z_coef(self, new_z_coef):
        self.z_coef = new_z_coef
    
    def reset(self, zs, ux0):
        self.N = zs.shape[0] # number of samples/individuals
        self.zs = zs.reshape(self.N, -1)

        # generate initial state
        self.xt = self.f_x0(zs=self.zs, ux0=ux0, z_coef=self.z_coef)

        # form the current observation
        observation = self.xt

        return observation, None
    
    def step(self, action, uxt, urt):
        self.atm1 = action

        # generate next state and reward
        self.rtm1 = self.f_rt(zs=self.zs, xt=self.xt, at=self.atm1, urt=urt, z_coef=self.z_coef)
        self.xt = self.f_xt(zs=self.zs, xtm1=self.xt, atm1=self.atm1, uxt=uxt, z_coef=self.z_coef)
        
        observation = self.xt
        reward = self.rtm1

        # the synthetic enviornment never terminates or gets truncated
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated



# REQUIRES: zs should be the same as the zs passed to the environment
def sample_trajectory(env, zs, state_dim, N, T, sigma=1, seed=1, policy=None):
    # initialize containers to store the trajectory
    Z = zs.reshape(N, -1)
    X = np.zeros([N, T + 1, state_dim], dtype=float)
    A = np.zeros([N, T], dtype=int)
    R = np.zeros([N, T], dtype=float)
    np.random.seed(seed)

    policy_c = copy.deepcopy(policy)
    X_c = np.zeros([N, T + 1, state_dim], dtype=float)
    A_c = np.zeros([N, T], dtype=int)
    R_c = np.zeros([N, T], dtype=float)
    L_c = np.zeros([N, T, 2], dtype=float)

    # generate the initial state
    ux0 = np.random.normal(0, sigma, [N, state_dim + 1])
    X[:, 0], _ = env.reset(Z, ux0)
    
    # take the first step
    ua0 = np.random.uniform(0, 1, size=[N])
    A[:, 0] = policy.act(
            z=Z,
            xt=X[:, 0],
            xtm1=None,
            atm1=None,
            uat=ua0,
        )
    ur0 = np.random.normal(0, sigma, [N, 1])
    ux1 = np.random.normal(0, sigma, [N, state_dim + 1])
    X[:, 1], R[:, 0], _, _ = env.step(A[:, 0], ux1, ur0)

    # take subsequent steps
    for t in range(1, T):
        uat = np.random.uniform(0, 1, size=[N])
        A[:, t] = policy.act(
                z=Z,
                xt=X[:, t],
                xtm1=X[:, t - 1],
                atm1=A[:, t - 1],
                uat=uat,
            )
        urtm1 = np.random.normal(0, sigma, [N, 1])
        uxt = np.random.normal(0, sigma, [N, state_dim + 1])
        X[:, t + 1], R[:, t], _, _ = env.step(A[:, t], uxt, urtm1)

    # prepare and return the output
    out = [Z, X, A, R]
    return out



def sample_counterfactual_trajectories(env, zs, z_eval_levels, state_dim, N, T, 
                                       policy, sigma=1, seed=1):
    np.random.seed(seed)
    z_eval_levels = np.array(z_eval_levels)

    trajectories = {}
    for z_level in z_eval_levels:
        env_z = copy.deepcopy(env)
        policy_z = copy.deepcopy(policy)
        Z = np.repeat(z_level, repeats=N, axis=0).reshape(N, -1)
        X = np.zeros([N, T + 1, state_dim], dtype=float)
        A = np.zeros([N, T], dtype=int)
        R = np.zeros([N, T], dtype=float)
        trajectories[tuple(z_level.flatten())] = {'Z': Z, 'X': X, 'A': A, 'R': R, 
                                        'env_z': env_z, 'policy_z': policy_z}

    # generate the initial state
    ux0 = np.random.normal(0, sigma, [N, state_dim + 1])
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        trajectories[tuple(z_level.flatten())]['X'][:, 0], _ = e.reset(np.tile(z_level, (N, 1)), ux0)
    
    # take the first step
    ua0 = np.random.uniform(0, 1, size=[N])
    for z_level in z_eval_levels:
        p = trajectories[tuple(z_level.flatten())]['policy_z']
        trajectories[tuple(z_level.flatten())]['A'][:, 0] = p.act(
                z=trajectories[tuple(z_level.flatten())]['Z'],
                xt=trajectories[tuple(z_level.flatten())]['X'][:, 0],
                xtm1=None,
                atm1=None,
                uat=ua0,
            )
    ur0 = np.random.normal(0, sigma, [N, 1])
    ux1 = np.random.normal(0, sigma, [N, state_dim + 1])
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, 0] for i in range(N)]) 
        (
            trajectories[tuple(z_level.flatten())]['X'][:, 1], 
            trajectories[tuple(z_level.flatten())]['R'][:, 0], 
            _, 
            _
        ) = e.step(actions, ux1, ur0)

    # take subsequent steps
    for t in range(1, T):
        uat = np.random.uniform(0, 1, size=[N])
        for z_level in z_eval_levels:
            p = trajectories[tuple(z_level.flatten())]['policy_z']
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t - 1] for i in range(N)]) 
            trajectories[tuple(z_level.flatten())]['A'][:, t] = p.act(
                    z=trajectories[tuple(z_level.flatten())]['Z'],
                    xt=trajectories[tuple(z_level.flatten())]['X'][:, t],
                    xtm1=trajectories[tuple(z_level.flatten())]['X'][:, t - 1],
                    atm1=actions,
                    uat=uat,
                )
        urtm1 = np.random.normal(0, sigma, [N, 1])
        uxt = np.random.normal(0, sigma, [N, state_dim + 1])
        for z_level in z_eval_levels:
            e = trajectories[tuple(z_level.flatten())]['env_z']
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t] for i in range(N)]) 
            (
                trajectories[tuple(z_level.flatten())]['X'][:, t + 1], 
                trajectories[tuple(z_level.flatten())]['R'][:, t], 
                _, 
                _
            ) = e.step(actions, uxt, urtm1)

    # return the output
    return trajectories