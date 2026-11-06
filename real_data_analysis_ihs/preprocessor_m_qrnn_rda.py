import numpy as np
import pandas as pd
import sys

sys.path.append("../")

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import torch
from statsmodels.regression.quantile_regression import QuantReg
import patsy
from base_models_rda import QRNN, QRNNJoint



class Preprocessor:
    def __init__(self) -> None:
        pass

    def preprocess(self):
        raise NotImplementedError

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

    def reset_buffer(self, n):
        self.buffer = np.zeros((n, self.xdim * self.num_z_levels))



# preprocessor for univariate states and binary sensitive attribute
class MSMDMPreprocessorAdd(Preprocessor):
    def __init__(self, 
                 initial_model_specs, 
                 state_model_specs, 
                 reward_model_spec, 
                 z_space=np.array([[0], [1]]), 
                 action_space=np.array([[0], [1]]), 
                 cross_folds=1, 
                 reg_model='qr', 
                 is_action_onehot=False, 
                 max_iter=1000, 
                 hidden_dims=[64, 64], 
                 epochs=500, 
                 learning_rate=0.01, 
                 batch_size=128, 
                 is_early_stopping=True, 
                 early_stopping_patience=10, 
                 early_stopping_min_delta=0.01, 
                 test_size=0.2):
        self.initial_model_specs = initial_model_specs
        self.state_model_specs = state_model_specs
        self.reward_model_spec = reward_model_spec
        self.z_space = np.array(z_space)
        self.action_space = np.array(action_space)
        self.cross_folds = cross_folds
        self.reg_model = reg_model
        self.is_action_onehot = is_action_onehot
        self.max_iter = max_iter
        self.num_z_levels = z_space.shape[0]
        self.is_trained = False
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.is_early_stopping = is_early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.test_size = test_size

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            # Shallow copy the massive lists of statsmodels to prevent memory explosions
            if k in ['models0', 'state_models', 'reward_models']:
                setattr(result, k, v)
            else:
                # Deepcopy everything else (like self.buffer and basic parameters)
                setattr(result, k, copy.deepcopy(v, memo))
                
        return result

    def encode_a(self, a):
        enc = OneHotEncoder(categories=[self.action_space.flatten()], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()
    
    def _learn_initial_models(self, zs, xs):
        if self.reg_model == 'qr':
            models0 = self._learn_initial_qr_models(zs=zs, xs=xs)
        elif self.reg_model == 'qrnn':
            models0 = self._learn_initial_qrnn_models(zs=zs, xs=xs)
        elif self.reg_model == 'qrnn_joint':
            models0 = self._learn_initial_qrnn_joint_models(zs=zs, xs=xs)
        else:
            print("Model type is undefined. Please specify \"qr\" or \"qrnn\".")
            exit(1)

        return models0

    # specs should be a 1D list whose i-th entry is the model spec for the i-th state component
    def _learn_initial_qr_models(self, zs, xs):
        xs = np.array(xs)
        zs = np.array(zs)#.flatten()

        # prepare the training data
        data = pd.DataFrame()
        for i in range(self.state_dim):
            data['state'+str(i+1)] = xs[:, 0, i].flatten()
        for i in range(zs.shape[-1]):
            data['zs'+str(i+1)] = zs[: , i].flatten()
        exog_index = data.index
        models0 = [] # one row for each dimension; one column for each tau
        for i in range(self.state_dim):
            models0_i = [] # one QR model for each of tau = 0.01, 0.02, ..., 0.99
            y, X = patsy.dmatrices(self.initial_model_specs[i], data, return_type="dataframe", NA_action='drop')
            y = y.reindex(exog_index)
            X = X.reindex(exog_index)
            for tau in range(1, 100):
                tau = tau / 100
                model = QuantReg(y, X)
                res = model.fit(q=tau, max_iter=self.max_iter)
                models0_i.append(res)
            models0.append(models0_i)
        return models0
    
    def _learn_initial_qrnn_models(self, zs, xs):
        xs = np.array(xs)
        zs = np.array(zs)#.flatten()

        # prepare the training data
        models0 = [] # one row for each dimension; one column for each tau
        for i in range(self.state_dim):
            models0_i = [] # one QR model for each of tau = 0.01, 0.02, ..., 0.99
            X = zs.reshape(-1, zs.shape[-1])
            y = xs[:, 0, i].reshape(-1, 1)
            for tau in range(1, 100):
                tau = tau / 100
                model = QRNN(in_dim=X.shape[-1], out_dim=1, hidden_dims=self.hidden_dims)
                model.fit(
                    X=X,
                    y=y,
                    q=tau, 
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    is_early_stopping=self.is_early_stopping,
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_min_delta=self.early_stopping_min_delta,
                    test_size=self.test_size,
                )
                models0_i.append(model)
            models0.append(models0_i)
        return models0
    
    def _learn_initial_qrnn_joint_models(self, zs, xs):
        xs = np.array(xs)
        zs = np.array(zs)#.flatten()

        # prepare the training data
        models0 = [] # one row for each dimension
        for i in range(self.state_dim):
            X = zs.reshape(-1, zs.shape[-1])
            y = xs[:, 0, i].reshape(-1, 1)
            taus = np.arange(1, 100) / 100
            model0_i = QRNNJoint(in_dim=X.shape[-1], out_dim=99, hidden_dims=self.hidden_dims)
            model0_i.fit(
                X=X,
                y=y,
                qs=taus, 
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                is_early_stopping=self.is_early_stopping,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_min_delta=self.early_stopping_min_delta,
                test_size=self.test_size,
            )
            models0.append(model0_i)
        return models0
    
    def _learn_quantile_models(self, zs, states, next_states, actions, rewards):
        # prepare the data
        data = pd.DataFrame({'rewards': rewards})
        if self.is_action_onehot:
            for i in range(self.dim_a):
                data['action'+str(i+1)] = actions[:, i]
        else:
            data['actions'] = actions
        for i in range(self.state_dim):
            data['state'+str(i+1)] = states[:, :, i].flatten()
            data['next_state'+str(i+1)] = next_states[:, :, i].flatten()
        for i in range(zs.shape[-1]):
            data['zs'+str(i+1)] = zs[: , : , i].flatten()
        exog_index = data.index
        
        # fit quantile regression models for states
        state_models = [] # one row for each dimension; one column for each tau
        for i in range(self.state_dim):
            state_models_i = [] # one QR model for each of tau = 0.01, 0.02, ..., 0.99
            y, X = patsy.dmatrices(self.state_model_specs[i], data, return_type="dataframe", NA_action='drop')
            y = y.reindex(exog_index)
            X = X.reindex(exog_index)
            for tau in range(1, 100):
                tau = tau / 100
                model = QuantReg(y, X)
                res = model.fit(q=tau, max_iter=self.max_iter)
                state_models_i.append(res)
            state_models.append(state_models_i)

        # fit quantile regression models for rewards
        reward_models = [] # one QR model for each of tau = 0.01, 0.02, ..., 0.99
        y, X = patsy.dmatrices(self.reward_model_spec, data, return_type="dataframe", NA_action='drop')
        y = y.reindex(exog_index)
        X = X.reindex(exog_index)
        for tau in range(1, 100):
            tau = tau / 100
            model = QuantReg(y, X)
            res = model.fit(q=tau, max_iter=self.max_iter)
            reward_models.append(res)
        
        return state_models, reward_models
    
    def _learn_qrnn_models(self, zs, states, next_states, actions, rewards):
        # fit quantile regression models for states
        state_models = [] # one row for each dimension; one column for each tau
        for i in range(self.state_dim):
            state_models_i = [] # one QR model for each of tau = 0.01, 0.02, ..., 0.99
            X = np.concatenate(
                [
                    zs.reshape(-1, zs.shape[-1]), 
                    actions.reshape(-1, self.dim_a),
                    states.reshape(-1, states.shape[-1]),
                ],
                axis=1,
            )
            y = next_states[:, :, i].reshape(-1, 1)
            for tau in range(1, 100):
                tau = tau / 100
                model = QRNN(in_dim=X.shape[1], out_dim=1, hidden_dims=self.hidden_dims)
                model.fit(
                    X=X,
                    y=y,
                    q=tau, 
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size,
                    is_early_stopping=self.is_early_stopping,
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_min_delta=self.early_stopping_min_delta,
                    test_size=self.test_size,
                )
                state_models_i.append(model)
            state_models.append(state_models_i)

        # fit quantile regression models for rewards
        reward_models = [] # one QR model for each of tau = 0.01, 0.02, ..., 0.99
        X = np.concatenate(
            [
                zs.reshape(-1, zs.shape[-1]), 
                actions.reshape(-1, self.dim_a),
                states.reshape(-1, states.shape[-1]),
            ],
            axis=1,
        )
        y = rewards.reshape(-1, 1)
        for tau in range(1, 100):
            tau = tau / 100
            model = QRNN(in_dim=X.shape[1], out_dim=1, hidden_dims=self.hidden_dims)
            model.fit(
                X=X,
                y=y,
                q=tau, 
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                is_early_stopping=self.is_early_stopping,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_min_delta=self.early_stopping_min_delta,
                test_size=self.test_size,
            )
            reward_models.append(model)
        
        return state_models, reward_models
    
    def _learn_qrnn_joint_models(self, zs, states, next_states, actions, rewards):
        # fit quantile regression models for states
        state_models = [] # one row for each dimension; one column for each tau
        for i in range(self.state_dim):
            X = np.concatenate(
                [
                    zs.reshape(-1, zs.shape[-1]), 
                    actions.reshape(-1, self.dim_a),
                    states.reshape(-1, states.shape[-1]),
                ],
                axis=1,
            )
            y = next_states[:, :, i].reshape(-1, 1)
            taus = np.arange(1, 100) / 100
            state_model_i = QRNNJoint(in_dim=X.shape[1], out_dim=99, hidden_dims=self.hidden_dims)
            state_model_i.fit(
                X=X,
                y=y,
                qs=taus, 
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                is_early_stopping=self.is_early_stopping,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_min_delta=self.early_stopping_min_delta,
                test_size=self.test_size,
            )
            state_models.append(state_model_i)

        # fit quantile regression models for rewards
        X = np.concatenate(
            [
                zs.reshape(-1, zs.shape[-1]), 
                actions.reshape(-1, self.dim_a),
                states.reshape(-1, states.shape[-1]),
            ],
            axis=1,
        )
        y = rewards.reshape(-1, 1)
        taus = np.arange(1, 100) / 100
        reward_models = QRNNJoint(in_dim=X.shape[1], out_dim=99, hidden_dims=self.hidden_dims)
        reward_models.fit(
            X=X,
            y=y,
            qs=taus, 
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            is_early_stopping=self.is_early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            test_size=self.test_size,
        )
        
        return state_models, reward_models
    
    def _learn_subsequent_models(self, zs, xs, actions, rewards):
        xs = np.array(xs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        zs = np.array(zs)

        xdim = xs.shape[-1]
        N = xs.shape[0]
        T = xs.shape[1]
        states = xs[:, : (self.T - 1), :]
        next_states = xs[:, 1:self.T, :]
        if self.is_action_onehot:
            T = xs.shape[1]
            actions = actions[:, : (T - 1), np.newaxis].reshape(-1, self.dim_a)
        else:
            actions = actions[:, :].flatten() 
        rewards = rewards[:, :].flatten()
        zs = np.repeat(zs[:, np.newaxis, :], repeats=self.T-1, axis=1)

        if self.reg_model == 'qr':
            return self._learn_quantile_models(zs, states, next_states, actions, rewards)
        elif self.reg_model == 'qrnn':
            return self._learn_qrnn_models(zs, states, next_states, actions, rewards)
        elif self.reg_model == 'qrnn_joint':
            return self._learn_qrnn_joint_models(zs, states, next_states, actions, rewards)
        else:
            print("Model type is undefined. Please specify \"qr\" or \"qrnn\".")
            exit(1)

    def _learn_transition_models(self, zs, xs, actions, rewards):
        N, T, _ = xs.shape
        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(N, T - 1, -1)
            self.dim_a = actions.shape[-1]
        else:
            self.dim_a = 1

        models0 = self._learn_initial_models(zs, xs)
        state_models, reward_models = self._learn_subsequent_models(zs, xs, actions, rewards)
        return models0, state_models, reward_models
    
    def _estimate_cf_initial_state_i(self, zs, xt, i, initial_state_models_i):
        zs = np.array(zs)#.flatten()
        xt = np.array(xt)#.flatten()
        N = xt.shape[0]

        # estimate the state quantiles for each individual under z = 0 and z = 1, respectively
        if self.reg_model == 'qr':
            qs_list_state = {}
            for z in self.z_space:
                new_data = pd.DataFrame()
                for j in range(self.state_dim):
                    new_data['state'+str(j+1)] = np.zeros(N) 
                for j in range(len(z)):
                    new_data['zs'+str(j+1)] = np.repeat(z[j], repeats=N)
                qs_state = np.zeros((N, 99))
                exog_index = new_data.index
                new_y, new_X = patsy.dmatrices(self.initial_model_specs[i], new_data, return_type="dataframe", NA_action='drop')
                new_X = new_X.reindex(exog_index) # we do not reindex new_y here because it is not used
                for tau in range(1, 100):  
                    q = initial_state_models_i[tau - 1].predict(new_X)
                    qs_state[:, tau - 1] = q
                qs_list_state[tuple(z.flatten())] = qs_state

            # map the observed states to counterfactual states for each individual
            cf_states = np.zeros((N, self.num_z_levels))
            fill_next = 0
            for z in self.z_space:
                for z_obs in self.z_space:
                    if np.array_equal(z, z_obs): # mapping states under the observed sensitive attribute
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        cf_states[idx_z, fill_next] = xt[idx_z]
                    else: # mapping states under counterfactual sensitive attributes
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        idx_obs = np.where(np.all(zs == z_obs, axis=1))[0]
                        idx_state = np.argmin((abs(qs_list_state[tuple(z_obs.flatten())][idx_obs, :] - xt[idx_obs].reshape(-1, 1))), 
                                            axis=1)
                        cf_states_z = qs_list_state[tuple(z.flatten())][idx_obs, idx_state]
                        cf_states[idx_obs, fill_next] = cf_states_z
                fill_next += 1
        elif self.reg_model == 'qrnn':
            qs_list_state = {}
            for z in self.z_space:
                new_X = np.repeat(z.reshape(1, -1), repeats=N, axis=0)
                qs_state = np.zeros((N, 99))
                for tau in range(1, 100):  
                    q = initial_state_models_i[tau - 1].predict(new_X)
                    qs_state[:, tau - 1] = q.flatten()
                qs_list_state[tuple(z.flatten())] = qs_state

            # map the observed states to counterfactual states for each individual
            cf_states = np.zeros((N, self.num_z_levels))
            fill_next = 0
            for z in self.z_space:
                for z_obs in self.z_space:
                    if np.array_equal(z, z_obs): # mapping states under the observed sensitive attribute
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        cf_states[idx_z, fill_next] = xt[idx_z]
                    else: # mapping states under counterfactual sensitive attributes
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        idx_obs = np.where(np.all(zs == z_obs, axis=1))[0]
                        idx_state = np.argmin((abs(qs_list_state[tuple(z_obs.flatten())][idx_obs, :] - xt[idx_obs].reshape(-1, 1))), 
                                            axis=1)
                        cf_states_z = qs_list_state[tuple(z.flatten())][idx_obs, idx_state]
                        cf_states[idx_obs, fill_next] = cf_states_z
                fill_next += 1
        elif self.reg_model == 'qrnn_joint':
            qs_list_state = {}
            for z in self.z_space:
                new_X = np.repeat(z.reshape(1, -1), repeats=N, axis=0)
                qs_state = initial_state_models_i.predict(new_X)
                qs_list_state[tuple(z.flatten())] = qs_state

            # map the observed states to counterfactual states for each individual
            cf_states = np.zeros((N, self.num_z_levels))
            fill_next = 0
            for z in self.z_space:
                for z_obs in self.z_space:
                    if np.array_equal(z, z_obs): # mapping states under the observed sensitive attribute
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        cf_states[idx_z, fill_next] = xt[idx_z]
                    else: # mapping states under counterfactual sensitive attributes
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        idx_obs = np.where(np.all(zs == z_obs, axis=1))[0]
                        idx_state = np.argmin((abs(qs_list_state[tuple(z_obs.flatten())][idx_obs, :] - xt[idx_obs].reshape(-1, 1))), 
                                            axis=1)
                        cf_states_z = qs_list_state[tuple(z.flatten())][idx_obs, idx_state]
                        cf_states[idx_obs, fill_next] = cf_states_z
                fill_next += 1
        else:
            print("Model type is undefined. Please specify \"qr\" or \"qrnn\".")
            exit(1)
        
        return cf_states
    
    def _estimate_cf_initial_state(self, zs, xt, initial_state_models):
        N = xt.shape[0]
        cf_states = np.zeros((N, self.state_dim * self.num_z_levels))
        for i in range(self.state_dim):
            cf_states_i = self._estimate_cf_initial_state_i(zs=zs, xt=xt[:, i], i=i, 
                                                            initial_state_models_i=initial_state_models[i])
            for z_idx in range(self.num_z_levels):
                cf_states[:, self.state_dim * z_idx + i] = cf_states_i[:, z_idx]
        return cf_states
    
    def _estimate_cf_next_state_i(self, zs, xt, xtp1_i, at, i, state_models_i):
        zs = np.array(zs)#.flatten()
        at = np.array(at)#.flatten()
        xt = np.array(xt)
        xtp1_i = np.array(xtp1_i).flatten()
        N = xt.shape[0]

        # estimate the state and reward quantiles for each individual under z = 0 and z = 1, respectively
        if self.reg_model == 'qr':
            qs_list_state = {}
            for z_idx, z in enumerate(self.z_space):
                new_data = pd.DataFrame()
                if self.is_action_onehot:
                    for j in range(self.dim_a):
                        new_data['action'+str(j+1)] = at.reshape(-1, self.dim_a)[:, j]
                else:
                    new_data['actions'] = at.flatten()
                for j in range(self.state_dim):
                    new_data['state'+str(j+1)] = xt[:, z_idx * self.state_dim + j].flatten()
                for j in range(self.state_dim):
                    new_data['next_state'+str(j+1)] = np.zeros(N)
                for j in range(len(z)):
                    new_data['zs'+str(j+1)] = np.repeat(z[j], repeats=N)
                qs_state = np.zeros((N, 99))
                exog_index = new_data.index
                new_y, new_X = patsy.dmatrices(self.state_model_specs[i], new_data, return_type="dataframe", NA_action='drop')
                new_X = new_X.reindex(exog_index) # we do not reindex new_y here because it is not used
                for tau in range(1, 100):
                    q = state_models_i[tau - 1].predict(new_X)
                    qs_state[:, tau - 1] = q
                qs_list_state[tuple(z.flatten())] = qs_state

            # map the observed states to counterfactual states for each individual
            cf_states = np.zeros((N, self.num_z_levels))
            fill_next = 0
            for z in self.z_space:
                for z_obs in self.z_space:
                    if np.array_equal(z, z_obs): # mapping states under the observed sensitive attribute
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        cf_states[idx_z, fill_next] = xtp1_i[idx_z]
                    else: # mapping states under counterfactual sensitive attributes
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        idx_obs = np.where(np.all(zs == z_obs, axis=1))[0]
                        idx_state = np.argmin((abs(qs_list_state[tuple(z_obs.flatten())][idx_obs, :] - xtp1_i[idx_obs].reshape(-1, 1))), 
                                            axis=1)
                        cf_states_z = qs_list_state[tuple(z.flatten())][idx_obs, idx_state]
                        cf_states[idx_obs, fill_next] = cf_states_z
                fill_next += 1
        elif self.reg_model == 'qrnn':
            qs_list_state = {}
            for z_idx, z in enumerate(self.z_space):
                new_X = np.concatenate(
                    [
                        np.repeat(z.reshape(1, -1), repeats=N, axis=0), 
                        at.reshape(-1, self.dim_a),
                        xt[:, z_idx * self.state_dim:(z_idx + 1) * self.state_dim].reshape(-1, self.state_dim),
                    ],
                    axis=1,
                )
                qs_state = np.zeros((N, 99))
                for tau in range(1, 100):
                    q = state_models_i[tau - 1].predict(new_X)
                    qs_state[:, tau - 1] = q.flatten()
                qs_list_state[tuple(z.flatten())] = qs_state

            # map the observed states to counterfactual states for each individual
            cf_states = np.zeros((N, self.num_z_levels))
            fill_next = 0
            for z in self.z_space:
                for z_obs in self.z_space:
                    if np.array_equal(z, z_obs): # mapping states under the observed sensitive attribute
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        cf_states[idx_z, fill_next] = xtp1_i[idx_z]
                    else: # mapping states under counterfactual sensitive attributes
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        idx_obs = np.where(np.all(zs == z_obs, axis=1))[0]
                        idx_state = np.argmin((abs(qs_list_state[tuple(z_obs.flatten())][idx_obs, :] - xtp1_i[idx_obs].reshape(-1, 1))), 
                                            axis=1)
                        cf_states_z = qs_list_state[tuple(z.flatten())][idx_obs, idx_state]
                        cf_states[idx_obs, fill_next] = cf_states_z
                fill_next += 1
        elif self.reg_model == 'qrnn_joint':
            qs_list_state = {}
            for z_idx, z in enumerate(self.z_space):
                new_X = np.concatenate(
                    [
                        np.repeat(z.reshape(1, -1), repeats=N, axis=0), 
                        at.reshape(-1, self.dim_a),
                        xt[:, z_idx * self.state_dim:(z_idx + 1) * self.state_dim].reshape(-1, self.state_dim),
                    ],
                    axis=1,
                )
                qs_state = state_models_i.predict(new_X)
                qs_list_state[tuple(z.flatten())] = qs_state

            # map the observed states to counterfactual states for each individual
            cf_states = np.zeros((N, self.num_z_levels))
            fill_next = 0
            for z in self.z_space:
                for z_obs in self.z_space:
                    if np.array_equal(z, z_obs): # mapping states under the observed sensitive attribute
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        cf_states[idx_z, fill_next] = xtp1_i[idx_z]
                    else: # mapping states under counterfactual sensitive attributes
                        idx_z = np.where(np.all(zs == z, axis=1))[0]
                        idx_obs = np.where(np.all(zs == z_obs, axis=1))[0]
                        idx_state = np.argmin((abs(qs_list_state[tuple(z_obs.flatten())][idx_obs, :] - xtp1_i[idx_obs].reshape(-1, 1))), 
                                            axis=1)
                        cf_states_z = qs_list_state[tuple(z.flatten())][idx_obs, idx_state]
                        cf_states[idx_obs, fill_next] = cf_states_z
                fill_next += 1
        else:
            print("Model type is undefined. Please specify \"qr\" or \"qrnn\".")
            exit(1)
        
        return cf_states
    
    def _estimate_cf_next_state_reward(self, zs, xt, xtp1, at, rt, state_models, reward_models):
        zs = np.array(zs)#.flatten()
        at = np.array(at).flatten()
        xt = np.array(xt)
        xtp1 = np.array(xtp1)
        if rt is not None:
            rt = np.array(rt).flatten()
    
        N = xt.shape[0]
        if self.is_action_onehot:
            encoded = self.encode_a(at.reshape(-1, 1))
            at = encoded.reshape(N, -1)
            self.dim_a = at.shape[-1]
        else:
            self.dim_a = 1

        # estimate the counterfactual states
        cf_states = np.zeros((N, self.state_dim * self.num_z_levels))
        for i in range(self.state_dim):
            cf_states_i = self._estimate_cf_next_state_i(zs=zs, xt=xt, xtp1_i=xtp1[:, i], at=at, 
                                                         i=i, state_models_i=state_models[i])
            for z_idx in range(self.num_z_levels):
                cf_states[:, self.state_dim * z_idx + i] = cf_states_i[:, z_idx]

        # estimate the reward quantiles for each individual under z = 0 and z = 1, respectively
        if rt is not None:
            if self.reg_model == 'qr':
                qs_list_reward = {}
                for z_idx, z in enumerate(self.z_space):
                    new_data = pd.DataFrame()
                    if self.is_action_onehot:
                        for j in range(self.dim_a):
                            new_data['action'+str(j+1)] = at.reshape(-1, self.dim_a)[:, j]
                    else:
                        new_data['actions'] = at.flatten()
                    for j in range(self.state_dim):
                        new_data['state'+str(j+1)] = xt[:, z_idx * self.state_dim + j].flatten()
                    for j in range(len(z)):
                        new_data['zs'+str(j+1)] = np.repeat(z[j], repeats=N)
                    new_data['rewards'] = np.zeros(N)
                    qs_reward = np.zeros((N, 99))
                    exog_index = new_data.index
                    new_y, new_X = patsy.dmatrices(self.reward_model_spec, new_data, return_type="dataframe", NA_action='drop')
                    new_X = new_X.reindex(exog_index) # we do not reindex new_y here because it is not used
                    for tau in range(1, 100):
                        q = reward_models[tau - 1].predict(new_X)
                        qs_reward[:, tau - 1] = q
                    qs_list_reward[tuple(z.flatten())] = qs_reward

                # map the observed states to counterfactual states for each individual
                cf_rewards = np.zeros((N, self.num_z_levels))
                fill_next = 0
                for z in self.z_space:
                    for z_obs in self.z_space:
                        if np.array_equal(z, z_obs): # mapping states under the observed sensitive attribute
                            idx_z = np.where(np.all(zs == z, axis=1))[0]
                            cf_rewards[idx_z, fill_next] = rt[idx_z]
                        else: # mapping states under counterfactual sensitive attributes
                            idx_z = np.where(np.all(zs == z, axis=1))[0]
                            idx_obs = np.where(np.all(zs == z_obs, axis=1))[0]
                            idx_reward = np.argmin((abs(qs_list_reward[tuple(z_obs.flatten())][idx_obs, :] - rt[idx_obs].reshape(-1, 1))), 
                                                axis=1)
                            cf_reward_z = qs_list_reward[tuple(z.flatten())][idx_obs, idx_reward]
                            cf_rewards[idx_obs, fill_next] = cf_reward_z
                    fill_next += 1
            elif self.reg_model == 'qrnn':
                qs_list_reward = {}
                for z_idx, z in enumerate(self.z_space):
                    new_X = np.concatenate(
                        [
                            np.repeat(z.reshape(1, -1), repeats=N, axis=0), 
                            at.reshape(-1, self.dim_a),
                            xt[:, z_idx * self.state_dim:(z_idx + 1) * self.state_dim].reshape(-1, self.state_dim),
                        ],
                        axis=1,
                    )
                    qs_reward = np.zeros((N, 99))
                    for tau in range(1, 100):
                        q = reward_models[tau - 1].predict(new_X)
                        qs_reward[:, tau - 1] = q.flatten()
                    qs_list_reward[tuple(z.flatten())] = qs_reward

                # map the observed states to counterfactual states for each individual
                cf_rewards = np.zeros((N, self.num_z_levels))
                fill_next = 0
                for z in self.z_space:
                    for z_obs in self.z_space:
                        if np.array_equal(z, z_obs): # mapping states under the observed sensitive attribute
                            idx_z = np.where(np.all(zs == z, axis=1))[0]
                            cf_rewards[idx_z, fill_next] = rt[idx_z]
                        else: # mapping states under counterfactual sensitive attributes
                            idx_z = np.where(np.all(zs == z, axis=1))[0]
                            idx_obs = np.where(np.all(zs == z_obs, axis=1))[0]
                            idx_reward = np.argmin((abs(qs_list_reward[tuple(z_obs.flatten())][idx_obs, :] - rt[idx_obs].reshape(-1, 1))), 
                                                axis=1)
                            cf_reward_z = qs_list_reward[tuple(z.flatten())][idx_obs, idx_reward]
                            cf_rewards[idx_obs, fill_next] = cf_reward_z
                    fill_next += 1
            elif self.reg_model == 'qrnn_joint':
                qs_list_reward = {}
                for z_idx, z in enumerate(self.z_space):
                    new_X = np.concatenate(
                        [
                            np.repeat(z.reshape(1, -1), repeats=N, axis=0), 
                            at.reshape(-1, self.dim_a),
                            xt[:, z_idx * self.state_dim:(z_idx + 1) * self.state_dim].reshape(-1, self.state_dim),
                        ],
                        axis=1,
                    )
                    qs_reward = reward_models.predict(new_X)
                    qs_list_reward[tuple(z.flatten())] = qs_reward

                # map the observed states to counterfactual states for each individual
                cf_rewards = np.zeros((N, self.num_z_levels))
                fill_next = 0
                for z in self.z_space:
                    for z_obs in self.z_space:
                        if np.array_equal(z, z_obs): # mapping states under the observed sensitive attribute
                            idx_z = np.where(np.all(zs == z, axis=1))[0]
                            cf_rewards[idx_z, fill_next] = rt[idx_z]
                        else: # mapping states under counterfactual sensitive attributes
                            idx_z = np.where(np.all(zs == z, axis=1))[0]
                            idx_obs = np.where(np.all(zs == z_obs, axis=1))[0]
                            idx_reward = np.argmin((abs(qs_list_reward[tuple(z_obs.flatten())][idx_obs, :] - rt[idx_obs].reshape(-1, 1))), 
                                                axis=1)
                            cf_reward_z = qs_list_reward[tuple(z.flatten())][idx_obs, idx_reward]
                            cf_rewards[idx_obs, fill_next] = cf_reward_z
                    fill_next += 1
            else:
                print("Model type is undefined. Please specify \"qr\" or \"qrnn\".")
                exit(1)
        
        return (cf_states, cf_rewards) if rt is not None else cf_states
    
    def _learn_marginal_dist_z(self, zs):
        marginal_dist_z = []
        for z in self.z_space:
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z.append(sum(z_idx) / len(z_idx))
        return np.array(marginal_dist_z)
    
    def train_preprocessor(self, zs, xs, actions, rewards):
        zs = np.array(zs)
        xs = np.array(xs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        self.state_dim = xs.shape[-1]

        # learn the transition models
        self.N, self.T, self.xdim = xs.shape
        self.xdim = xs.shape[-1]
        self.marginal_dist_z = self._learn_marginal_dist_z(zs)
        self.models0, self.state_models, self.reward_models = self._learn_transition_models(zs=zs, 
                                                                                            xs=xs, 
                                                                                            actions=actions, 
                                                                                            rewards=rewards)
         
        # preprocess the training data
        if self.cross_folds == 1:
            self.reset_buffer(n=self.N)
            xs_tilde = np.zeros([self.N, self.T, self.xdim * len(self.marginal_dist_z)])
            rs_tilde = np.zeros([self.N, self.T - 1])
            for t in range(self.T):
                if t == 0:
                    xt_tilde = self._estimate_cf_initial_state(zs=zs, xt=xs[:, 0, :], initial_state_models=self.models0)
                else:
                    xt_tilde, rt_tilde = self._estimate_cf_next_state_reward(zs=zs, 
                                                                             xt=self.buffer, 
                                                                             xtp1=xs[:, t, :], 
                                                                             at=actions[:, t - 1], 
                                                                             rt=rewards[:, t - 1], 
                                                                             state_models=self.state_models, 
                                                                             reward_models=self.reward_models)
                    rs_tilde[:, t - 1] = np.matmul(rt_tilde, self.marginal_dist_z)
                self.buffer = xt_tilde.copy()
                xs_tilde[:, t, :] = xt_tilde
        else:
            kf = KFold(n_splits=self.cross_folds)
            models0_folds = [None for _ in range(self.cross_folds)]
            state_models_folds = [None for _ in range(self.cross_folds)]
            reward_models_folds = [None for _ in range(self.cross_folds)]
            xs_tilde = np.zeros([self.N, self.T, self.xdim * len(self.marginal_dist_z)])
            rs_tilde = np.zeros([self.N, self.T - 1])
            for i, (train_index, test_index) in enumerate(kf.split(xs)):
                xs_train, xs_test = xs[train_index], xs[test_index]
                zs_train, zs_test = zs[train_index], zs[test_index]
                actions_train, actions_test = actions[train_index], actions[test_index]
                rewards_train, rewards_test = rewards[train_index], rewards[test_index]
                models0_folds[i], state_models_folds[i], reward_models_folds[i] = self._learn_transition_models(zs=zs_train, 
                                                                                                                xs=xs_train, 
                                                                                                                actions=actions_train, 
                                                                                                                rewards=rewards_train)
                self.reset_buffer(n=xs_test.shape[0])
                for t in range(self.T):
                    if t == 0:
                        xt_tilde_test = self._estimate_cf_initial_state(zs=zs_test, xt=xs_test[:, 0, :], initial_state_models=models0_folds[i])
                    else:
                        xt_tilde_test, rt_tilde_test = self._estimate_cf_next_state_reward(zs=zs_test, 
                                                                                           xt=self.buffer, 
                                                                                           xtp1=xs_test[:, t, :], 
                                                                                           at=actions_test[:, t - 1], 
                                                                                           rt=rewards_test[:, t - 1], 
                                                                                           state_models=state_models_folds[i], 
                                                                                           reward_models=reward_models_folds[i])
                        rs_tilde[test_index, t - 1] = np.matmul(rt_tilde_test, self.marginal_dist_z)
                    self.buffer = xt_tilde_test.copy()
                    xs_tilde[test_index, t, :] = xt_tilde_test
        
        self.is_trained = True
        return xs_tilde, rs_tilde
    
    def preprocess_single_step(self, t, z, xt, xtm1=None, atm1=None, rtm1=None, **kwargs):
        if self.is_trained == False: # sanity check to make sure the preprocessor has already been trained
            print('Preprocessor not trained yet. Please train the preprocessor before data preprocessing.')
            exit(1)
        N = z.shape[0]

        # t = 0
        if atm1 is None:
            self.reset_buffer(n=N)
            xt_tilde = self._estimate_cf_initial_state(zs=z, xt=xt, initial_state_models=self.models0)
            self.buffer = xt_tilde
        # t > 0 & rtm1 is None
        elif rtm1 is None:
            xt_tilde = self._estimate_cf_next_state_reward(zs=z, 
                                                           xt=self.buffer, 
                                                           xtp1=xt, 
                                                           at=atm1, 
                                                           rt=rtm1, 
                                                           state_models=self.state_models, 
                                                           reward_models=self.reward_models)
            self.buffer = xt_tilde
        else:
            xt_tilde, rtm1_tilde = self._estimate_cf_next_state_reward(zs=z, 
                                                                       xt=self.buffer, 
                                                                       xtp1=xt, 
                                                                       at=atm1, 
                                                                       rt=rtm1, 
                                                                       state_models=self.state_models, 
                                                                       reward_models=self.reward_models)
            rtm1_tilde = np.matmul(rtm1_tilde, self.marginal_dist_z)
            self.buffer = xt_tilde
        
        return (xt_tilde, rtm1_tilde) if rtm1 is not None else xt_tilde
    
    def preprocess_multiple_steps(self, zs, xs, actions, rewards=None):
        # some convenience variables
        N, T, xdim = xs.shape

        # check if state dimension of the input is the same as state dimension of the training data
        if xdim != self.xdim:
            raise ValueError('The state dimension of the input does not match that of the training data.')
        
        # define the returned arrays; the arrays will be filled later
        xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z)])
        rs_tilde = np.zeros([N, T - 1])

        # preprocess the initial step
        xs_tilde[:, 0, :] = self.preprocess_single_step(t=0, z=zs, xt=xs[:, 0, :])

        # preprocess subsequent steps
        if rewards is not None:
            for t in range (1, T):
                xs_tilde[:, t, :], rs_tilde[:, t - 1] = self.preprocess_single_step(t=t, 
                                                                                    z=zs, 
                                                                                    xt=xs[:, t, :], 
                                                                                    xtm1=xs[:, t - 1, :], 
                                                                                    atm1=actions[:, t - 1], 
                                                                                    rtm1=rewards[:, t - 1]
                                                                                    )
            return xs_tilde, rs_tilde                
        else:
            for t in range (1, T):
                xs_tilde[:, t, :] = self.preprocess_single_step(t=t, 
                                                                z=zs, 
                                                                xt=xs[:, t, :], 
                                                                xtm1=xs[:, t - 1, :], 
                                                                atm1=actions[:, t - 1]
                                                                )
            return xs_tilde




            

            
        
        



