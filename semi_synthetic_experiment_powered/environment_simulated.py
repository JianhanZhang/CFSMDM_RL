import numpy as np
import torch
import warnings
import sys
import copy
import pandas as pd
import os
import gymnasium as gym
from sklearn.preprocessing import OneHotEncoder

warnings.simplefilter(action="ignore", category=FutureWarning)
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from base_models import NeuralNetRegressor, LinearRegressor
from agents import RandomAgent

class SimulatedEnvironment(gym.Env):
    def __init__(
        self,
        action_space, 
        reward_multiplication_factor=[1.0, 1.0, 1.0],
        state_variance_factor=1.0,
        z_factor=0.0,
        trans_model_type="nn",
        trans_model_hidden_dims=[128, 128],
        reward_model_type="nn",
        reward_model_hidden_dims=[128, 128], 
        is_action_onehot=True, 
        epochs=1000,
        batch_size=128,
        learning_rate=0.001,
        is_early_stopping=True,
        test_size=0.2,
        early_stopping_patience=10,
        early_stopping_min_delta=0.01,
        enforce_min_max=False,
        ):
        self.action_space = action_space
        self.reward_multiplication_factor = reward_multiplication_factor
        self.state_variance_factor = state_variance_factor
        self.z_factor = z_factor
        self.trans_model_type = trans_model_type
        self.trans_model_hidden_dims = trans_model_hidden_dims
        self.reward_model_type = reward_model_type
        self.reward_model_hidden_dims = reward_model_hidden_dims
        self.is_action_onehot = is_action_onehot
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_early_stopping = is_early_stopping
        self.test_size = test_size 
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.enforce_min_max = enforce_min_max
        self.is_trained = False
    
    @staticmethod
    def standardize(x, mean=None, std=None):
        if mean is None and std is None:
            mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            return (x - mean) / std, mean, std
        else:
            return (x - mean) / std

    @staticmethod
    def destandardize(x, mean, std):
        return x * std + mean
    
    def encode_a(self, a):
        enc = OneHotEncoder(categories=[self.action_space.flatten()], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()

    def fit(self, zs, states, actions, rewards):
        z = zs
        xt = states
        at = actions
        rt = rewards
        N, T, state_dim = xt.shape
        self.N = N
        self.T = T
        self.state_dim = state_dim

        # learn the initial state model i.e. t = 0
        # prepare data
        training_y = xt[:, 0, :]
        training_x_cat = zs

        training_x = training_x_cat
        self.initial_y_min = np.min(training_y, axis=0)
        self.initial_y_max = np.max(training_y, axis=0)

        self.initial_model_mean = {}
        self.initial_model_var = {}
        for z_ in np.unique(zs, axis=0):
            idx = np.all(training_x == z_, axis=1).flatten()
            training_y_z = training_y[idx]
            self.initial_model_mean[tuple(z_)] = np.mean(training_y_z, axis=0)
            self.initial_model_var[tuple(z_)] = np.var(training_y_z, axis=0)

        # learn the transition and reward models for t > 0
        # prepare input
        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(self.N, T - 1, -1)
            dim_a = actions.shape[-1]
        else:
            actions = at
            dim_a = 1 # we require actions to be 1-dimensional w/out one-hot encoding
        actions = actions.reshape(-1, dim_a)
        next_states = xt[:, 1:T, :].reshape(N * (T - 1), -1)
        states = xt[:, 0 : (T - 1), :].reshape(N * (T - 1), -1)
        sensitives = np.repeat(z[:, np.newaxis, :], repeats=T - 1, axis=1).reshape(
            N * (T - 1), -1
        )
        rewards = rt.reshape(-1, 1)

        # fit the transition model
        if self.trans_model_type == "lm":
            self.trans_models = LinearRegressor()
            X = np.concatenate([states, sensitives, actions.reshape(-1, dim_a)], axis=1)
            y = next_states
            self.trans_models.fit(
                X,
                y, 
            )

        elif self.trans_model_type == "nn":

            # distinct models for different sensitive groups
            self.trans_models = {}
            for z_ in np.unique(zs, axis=0):
                idxs = np.all(sensitives == z_, axis=1).flatten()
                states_z_ = states[idxs]
                actions_z_ = actions[idxs]

                next_states_z_ = next_states[idxs]
                self.trans_models[tuple(z_)] = NeuralNetRegressor(
                    in_dim=states_z_.shape[1] + dim_a,
                    out_dim=next_states_z_.shape[1],
                    hidden_dims=self.trans_model_hidden_dims,
                )
                X = np.concatenate([states_z_, actions_z_.reshape(-1, dim_a)], axis=1)
                y = next_states_z_

                self.trans_models[tuple(z_)].fit(
                    X,
                    y,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    is_early_stopping=self.is_early_stopping,
                    test_size=self.test_size,
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_min_delta=self.early_stopping_min_delta,
                )

            self.transition_var = {}
            for z_ in np.unique(zs, axis=0):
                key = tuple(z_)
                self.transition_var[key] = (
                    self.trans_models[key].var.copy()
                    * self.state_variance_factor
                )

        # reward model
        # input: x_t (weekly_pain_score, weekly_pain_interference), z (baseline characteristic), a_t
        # output: r_t
        if self.reward_model_type == "lm":
            self.reward_models = LinearRegressor()
            X = np.concatenate([states, sensitives, actions.reshape(-1, dim_a)], axis=1)
            y = rewards.reshape(-1, 1)
            self.reward_models.fit(
                X, 
                y, 
            )
        elif self.reward_model_type == "nn":
            self.reward_models = {}
            for z_ in np.unique(zs, axis=0):
                idxs = np.all(sensitives == z_, axis=1).flatten()
                states_z_ = states[idxs]
                actions_z_ = actions[idxs]
                rewards_z_ = rewards[idxs]
                self.reward_models[tuple(z_)] = NeuralNetRegressor(
                    in_dim=states_z_.shape[1] + dim_a,
                    out_dim=1,
                    hidden_dims=self.reward_model_hidden_dims,
                )
                X = np.concatenate([states_z_, actions_z_.reshape(-1, dim_a)], axis=1)
                y = rewards_z_.reshape(-1, 1)
                self.reward_models[tuple(z_)].fit(
                    X,
                    y,
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    is_early_stopping=self.is_early_stopping,
                    test_size=self.test_size,
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_min_delta=self.early_stopping_min_delta,
                )
            self.reward_var = {}

            for z_ in np.unique(zs, axis=0):
                key = tuple(z_)
                self.reward_var[key] = float(
                    self.reward_models[key].var.item()
                )

        # force min and max for states and rewards
        if self.enforce_min_max:
            self.states_max = np.max(states, axis=0)
            self.states_min = np.min(states, axis=0)
            self.rewards_max = np.max(rewards)
            self.rewards_min = np.min(rewards)
            self.next_states_max = np.max(next_states, axis=0)
            self.next_states_min = np.min(next_states, axis=0)
        
        self.is_trained = True
        # temporarily store the state, action, and reward data of the environment
        self.xt = np.zeros(xt.shape[-1])
        self.atm1 = None
        self.rtm1 = 0
    
    def reset(
        self, 
        zs, 
        seed=1,
        enforce_min_max=False,
        errors=None, 
    ):
        if not self.is_trained:
            print('Cannot make transitions because the environment is not yet trained.')
            exit(1)

        testing_x_cat = zs
        testing_x = testing_x_cat
        self.zs = zs

        testing_y = np.zeros([testing_x.shape[0], self.state_dim])

        if errors is None:
            errors = np.random.normal(
                size=(testing_x.shape[0], self.state_dim)
            )

        for z_ in np.unique(zs, axis=0):
            idx = np.all(testing_x == z_, axis=1).flatten()
            key = tuple(z_)
            testing_y[idx] = (
                self.initial_model_mean[key]
                + errors[idx] * np.sqrt(self.initial_model_var[key])
            )

        # cliping
        if enforce_min_max:
            for i in range(self.state_dim):
                testing_y[:, i] = np.clip(
                    testing_y[:, i], a_min=self.initial_y_min[i], a_max=self.initial_y_max[i]
                )
        
        self.xt = testing_y
        observation = self.xt

        return observation, None
    
    # helper function
    def _next_state_reward_mean(self, zs, xt, at):
        N = xt.shape[0]

        states = xt.reshape(N, -1)
        actions = at.flatten()
        next_states = np.zeros([N, xt.shape[-1]])
        rewards = np.zeros([N])

        # standarize
        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(N, -1)
            dim_a = actions.shape[-1]
        else:
            dim_a = 1

        # next state
        if self.trans_model_type == "lm":
            X = np.concatenate(
                    [states, zs, actions.reshape(-1, dim_a)], axis=1
                )
            next_states = self.trans_models.predict(X)

        elif self.trans_model_type == "nn":
            for z_ in np.unique(zs, axis=0):
                #idx = (sensitives == z_).flatten()
                idx = np.all(zs == z_, axis=1).flatten()
                states_z_ = states[idx]
                actions_z_ = actions[idx]
                X = np.concatenate(
                    [states_z_, actions_z_.reshape(-1, dim_a)], axis=1
                )
                next_states_z_mean = self.trans_models[tuple(z_)].predict(X)
                next_states_z_ = next_states_z_mean
                next_states[idx] = next_states_z_

        # reward
        if self.reward_model_type == "lm":
            X = np.concatenate(
                    [states, zs, actions.reshape(-1, dim_a)], axis=1
                )
            rewards = self.reward_models.predict(X).flatten()
        else:
            for z_ in np.unique(zs, axis=0):
                idx = np.all(zs == z_, axis=1).flatten()
                states_z_ = states[idx]
                actions_z_ = actions[idx]
                X = np.concatenate(
                    [states_z_, actions_z_.reshape(-1, dim_a)], axis=1
                )
                rewards_z_ = self.reward_models[tuple(z_)].predict(X).flatten()
                rewards[idx] = rewards_z_

        if self.enforce_min_max:
            next_states = np.clip(
                next_states, a_min=self.next_states_min, a_max=self.next_states_max
            )
            rewards = np.clip(rewards, a_min=self.rewards_min, a_max=self.rewards_max)

        return next_states, rewards
    
    def step(self, action, errors_next_states=None, errors_rewards=None, seed=1):
        if not self.is_trained:
            print('Cannot make transitions because the environment is not yet trained.')
            exit(1)

        at = action
        xt = self.xt
        next_states_mean, rewards_mean = self._next_state_reward_mean(
            self.zs, xt, at
        )

        if errors_next_states is None:
            errors_next_states = np.random.normal(
                size=next_states_mean.shape
            )

        if errors_rewards is None:
            errors_rewards = np.random.normal(
                size=rewards_mean.shape
            )

        transition_std = np.zeros_like(next_states_mean)
        reward_std = np.zeros_like(rewards_mean)

        for z_ in np.unique(self.zs, axis=0):
            idx = np.all(self.zs == z_, axis=1)
            key = tuple(z_)

            transition_std[idx] = np.sqrt(self.transition_var[key])
            reward_std[idx] = np.sqrt(self.reward_var[key])

        # Generate next states and rewards
        next_states = (
            next_states_mean
            + errors_next_states * transition_std
        )

        rewards = (
            rewards_mean
            + errors_rewards * reward_std
        )

        self.atm1 = action
        self.xt = next_states
        self.rtm1 = rewards

        # form the current observation and compute the reward
        observation = self.xt
        reward = self.rtm1

        # the synthetic enviornment never terminates or gets truncated
        terminated = False
        truncated = False

        return observation, reward, terminated, truncated



def train_baseline_characteristics_synthesizer(epochs=100, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    metadata = SingleTableMetadata()
    impute_data = pd.read_csv('./impute_data.csv')
    demo_data = impute_data.copy()
    demo_data = demo_data[
        [
            "num_substances",
            "baseline_pain",
            "age",
            "sex",
            "ethnicity",
            "employed",
            "depression",
            "race",
            "education",
            "comm_score",
        ]
    ]

    metadata.detect_from_dataframe(data=demo_data)
    metadata.update_column(column_name="sex", sdtype="categorical")
    metadata.update_column(column_name="ethnicity", sdtype="categorical")
    metadata.update_column(column_name="employed", sdtype="categorical")
    metadata.update_column(column_name="depression", sdtype="categorical")
    metadata.update_column(column_name="race", sdtype="categorical")
    metadata.update_column(column_name="education", sdtype="categorical")

    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        epochs=epochs,
    )
    synthesizer.fit(demo_data)

    return synthesizer



# baseline characteristic model
def generate_baseline_characteristics(
    n,
    zvars_cat,
    synthesizer,
    seed=1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    synthetic_data = synthesizer.sample(num_rows=n, output_file_path="disable")
    synthetic_data["id"] = range(n)
    #
    synthetic_data["race_b"] = (synthetic_data["race"] == "white").astype(int)
    synthetic_data["age_b"] = (synthetic_data["age"] >= 40).astype(int)
    synthetic_data["sex_b"] = (synthetic_data["sex"] == 2).astype(int)
    synthetic_data["education_b"] = (synthetic_data["education"] == "college").astype(
        int
    )
    synthetic_data["depression_b"] = (
        synthetic_data["depression"] == "mod/severe"
    ).astype(int)
    synthetic_data["ethnicity_b"] = (synthetic_data["ethnicity"] == 2).astype(int)
    return synthetic_data[["id"] + zvars_cat]



# REQUIRES: zs should be the same as the zs passed to the environment
def sample_simulated_env_trajectory(env, zs, state_dim, N, T, sigma=1, seed=1, policy=None):
    # initialize containers to store the trajectory
    Z = zs.reshape(N, -1)
    X = np.zeros([N, T + 1, state_dim], dtype=float)
    A = np.zeros([N, T], dtype=int)
    R = np.zeros([N, T], dtype=float)
    np.random.seed(seed)

    # generate the initial state
    X[:, 0], _ = env.reset(Z)
    
    # take the first step
    ua0 = np.random.uniform(0, 1, size=[N])
    #ua0 = np.zeros(N)
    A[:, 0] = policy.act(
            t=0, 
            z=Z,
            xt=X[:, 0],
            xtm1=None,
            atm1=None,
            uat=ua0,
        )
    X[:, 1], R[:, 0], _, _ = env.step(A[:, 0])

    # take subsequent steps
    for t in range(1, T):
        uat = np.random.uniform(0, 1, size=[N])
        A[:, t] = policy.act(
                t=t, 
                z=Z,
                xt=X[:, t],
                xtm1=X[:, t - 1],
                atm1=A[:, t - 1],
                uat=uat,
            )
        X[:, t + 1], R[:, t], _, _ = env.step(A[:, t])

    # prepare and return the output
    out = [Z, X, A, R]
    return out



def sample_counterfactual_simulated_env_trajectories(env, zs, z_eval_levels, state_dim, N, T, 
                                       policy, sigma_a=1, sigma_x=None, sigma_r=1, seed=1):
    np.random.seed(seed)
    if sigma_x is None:
        sigma_x = np.ones(env.state_dim)
    z_eval_levels = np.array(z_eval_levels)

    # dictionary to contain the counterfactual trajectories; key = z and value = trajectory
    trajectories = {}
    for z_level in z_eval_levels:
        env_z = copy.deepcopy(env)
        policy_z = copy.deepcopy(policy)
        Z = np.repeat(z_level, repeats=N, axis=0) .reshape(N, -1)
        X = np.zeros([N, T + 1, state_dim], dtype=float)
        A = np.zeros([N, T], dtype=int)
        R = np.zeros([N, T], dtype=float)
        trajectories[tuple(z_level.flatten())] = {'Z': Z, 'X': X, 'A': A, 'R': R, 
                                        'env_z': env_z, 'policy_z': policy_z}

    # generate the initial state
    ux0 = np.random.multivariate_normal(
                mean=np.zeros(env.state_dim), 
                cov=np.diag(sigma_x), 
                size=N
            )
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        trajectories[tuple(z_level.flatten())]['X'][:, 0], _ = e.reset(np.tile(z_level, (N, 1)), errors=ux0)
    
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
    ux1 = np.random.multivariate_normal(
                mean=np.zeros(env.state_dim),
                cov=np.diag(sigma_x),
                size=N,
            )
    ur0 = np.random.normal(
                loc=0, scale=sigma_r, size=N
            )
    for z_level in z_eval_levels:
        e = trajectories[tuple(z_level.flatten())]['env_z']
        actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, 0] for i in range(N)]) 
        (
            trajectories[tuple(z_level.flatten())]['X'][:, 1], 
            trajectories[tuple(z_level.flatten())]['R'][:, 0], 
            _, 
            _
        ) = e.step(actions, errors_next_states=ux1, errors_rewards=ur0)

    # take subsequent steps
    for t in range(1, T):
        uat = np.random.uniform(0, sigma_a, size=[N])
        for z_level in z_eval_levels:
            p = trajectories[tuple(z_level.flatten())]['policy_z']
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t - 1] for i in range(N)]) 
            trajectories[tuple(z_level.flatten())]['A'][:, t] = p.act(
                    t=t, 
                    z=trajectories[tuple(z_level.flatten())]['Z'],
                    xt=trajectories[tuple(z_level.flatten())]['X'][:, t],
                    xtm1=trajectories[tuple(z_level.flatten())]['X'][:, t - 1],
                    atm1=actions,
                    uat=uat,
                )
        uxt = np.random.multivariate_normal(
                mean=np.zeros(env.state_dim),
                cov=np.diag(sigma_x),
                size=N,
            )
        urtm1 = np.random.normal(
                    loc=0, scale=sigma_r, size=N
            )
        for z_level in z_eval_levels:
            e = trajectories[tuple(z_level.flatten())]['env_z']
            actions = np.array([trajectories[tuple(zs[i].flatten())]['A'][i, t] for i in range(N)]) 
            (
                trajectories[tuple(z_level.flatten())]['X'][:, t + 1], 
                trajectories[tuple(z_level.flatten())]['R'][:, t], 
                _, 
                _
            ) = e.step(actions, errors_next_states=uxt, errors_rewards=urtm1)

    # return the output
    return trajectories