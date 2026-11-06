import numpy as np
import pandas as pd
import sys

sys.path.append("../")

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import torch
from statsmodels.regression.quantile_regression import QuantReg
import patsy



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
        self.buffer = np.zeros((n, self.xdim * 2))



# preprocessor for univariate states and binary sensitive attribute
class MSMDMPreprocessor(Preprocessor):
    def __init__(self, 
                 initial_model_specs, 
                 state_model_specs, 
                 reward_model_spec, 
                 z_space=np.array([[0], [1]]), 
                 action_space=np.array([[0], [1]]), 
                 cross_folds=1, 
                 reg_model='qr', 
                 is_action_onehot=False, 
                 max_iter=1000):
        self.initial_model_specs = initial_model_specs
        self.state_model_specs = state_model_specs
        self.reward_model_spec = reward_model_spec
        self.z_space = np.array(z_space)
        self.action_space = np.array(action_space)
        self.cross_folds = cross_folds
        self.reg_model = reg_model
        self.is_action_onehot = is_action_onehot
        self.max_iter = max_iter

    def encode_a(self, a):
        enc = OneHotEncoder(categories=[self.action_space.flatten()], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()

    # specs should be a 1D list whose i-th entry is the model spec for the i-th state component
    def _learn_initial_models(self, zs, xs):
        xs = np.array(xs)
        zs = np.array(zs).flatten()

        # prepare the training data
        data = pd.DataFrame({'zs': zs})
        for i in range(self.state_dim):
            data['state'+str(i+1)] = xs[:, 0, i].flatten()
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
    
    def _learn_quantile_models(self, zs, states, next_states, actions, rewards):
        # prepare the data
        data = pd.DataFrame({'zs': zs, 'rewards': rewards})
        if self.is_action_onehot:
            for i in range(self.dim_a):
                data['action'+str(i+1)] = actions[:, i]
        else:
            data['actions'] = actions
        for i in range(self.state_dim):
            data['state'+str(i+1)] = states[:, :, i].flatten()
            data['next_state'+str(i+1)] = next_states[:, :, i].flatten()
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
        zs = np.repeat(zs.flatten(), repeats=self.T-1)

        if self.reg_model == 'qr':
            return self._learn_quantile_models(zs, states, next_states, actions, rewards)
        else:
            print("Model type is undefined. Please specify \"qr\".")
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
        zs = np.array(zs).flatten()
        xt = np.array(xt).flatten()
        N = xt.shape[0]

        # estimate the state quantiles for each individual under z = 0 and z = 1, respectively
        qs_list_state = []
        for z in [0, 1]:
            new_data = pd.DataFrame({'zs': np.repeat(z, repeats=N)})
            for j in range(self.state_dim):
                new_data['state'+str(j+1)] = np.zeros(N) 
            qs_state = np.zeros((N, 99))
            exog_index = new_data.index
            new_y, new_X = patsy.dmatrices(self.initial_model_specs[i], new_data, return_type="dataframe", NA_action='drop')
            new_X = new_X.reindex(exog_index) # we do not reindex new_y here because it is not used
            for tau in range(1, 100):  
                q = initial_state_models_i[tau - 1].predict(new_X)
                qs_state[:, tau - 1] = q
            qs_list_state.append(qs_state)

        # map the observed states to counterfactual states for each individual
        cf_states = np.zeros((N, 2))
        for z in [0, 1]:
            idx_z = np.where(zs == z)[0]
            cf_states[idx_z, z] = xt[idx_z]

            idx_c = np.where(zs != z)[0]
            idx_state = np.argmin((abs(qs_list_state[1 - z][idx_c, :] - xt[idx_c].reshape(-1, 1))), 
                                  axis=1)
            cf_states_z = qs_list_state[z][idx_c, idx_state]
            cf_states[idx_c, z] = cf_states_z
        
        return cf_states
    
    def _estimate_cf_initial_state(self, zs, xt, initial_state_models):
        N = xt.shape[0]
        cf_states = np.zeros((N, self.state_dim * 2))
        for i in range(self.state_dim):
            cf_states_i = self._estimate_cf_initial_state_i(zs=zs, xt=xt[:, i], i=i, 
                                                            initial_state_models_i=initial_state_models[i])
            cf_states[:, i] = cf_states_i[:, 0] # the first state_dim columns are for z = 0
            cf_states[:, self.state_dim + i] = cf_states_i[:, 1] # the second state_dim columns are for z = 1
        return cf_states
    
    def _estimate_cf_next_state_i(self, zs, xt, xtp1_i, at, i, state_models_i):
        zs = np.array(zs).flatten()
        at = np.array(at)#.flatten()
        xt = np.array(xt)
        xtp1_i = np.array(xtp1_i).flatten()
        N = xt.shape[0]

        # estimate the state and reward quantiles for each individual under z = 0 and z = 1, respectively
        qs_list_state = []
        for z in [0, 1]:
            new_data = pd.DataFrame({'zs': np.repeat(z, repeats=N)})
            if self.is_action_onehot:
                for j in range(self.dim_a):
                    new_data['action'+str(j+1)] = at.reshape(-1, self.dim_a)[:, j]
            else:
                new_data['actions'] = at.flatten()
            for j in range(self.state_dim):
                new_data['state'+str(j+1)] = xt[:, z * self.state_dim + j].flatten()
            for j in range(self.state_dim):
                new_data['next_state'+str(j+1)] = np.zeros(N)
            qs_state = np.zeros((N, 99))
            exog_index = new_data.index
            new_y, new_X = patsy.dmatrices(self.state_model_specs[i], new_data, return_type="dataframe", NA_action='drop')
            new_X = new_X.reindex(exog_index) # we do not reindex new_y here because it is not used
            for tau in range(1, 100):
                q = state_models_i[tau - 1].predict(new_X)
                qs_state[:, tau - 1] = q
            qs_list_state.append(qs_state)

        # map the observed states to counterfactual states for each individual
        cf_states = np.zeros((N, 2))
        for z in [0, 1]:
            idx_z = np.where(zs == z)[0]
            cf_states[idx_z, z] = xtp1_i[idx_z]

            idx_c = np.where(zs != z)[0]
            idx_state = np.argmin((abs(qs_list_state[1 - z][idx_c, :] - xtp1_i[idx_c].reshape(-1, 1))), 
                                  axis=1)
            cf_states_z = qs_list_state[z][idx_c, idx_state]
            cf_states[idx_c, z] = cf_states_z
        
        return cf_states
    
    def _estimate_cf_next_state_reward(self, zs, xt, xtp1, at, rt, state_models, reward_models):
        zs = np.array(zs).flatten()
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
        cf_states = np.zeros((N, self.state_dim * 2))
        for i in range(self.state_dim):
            cf_states_i = self._estimate_cf_next_state_i(zs=zs, xt=xt, xtp1_i=xtp1[:, i], at=at, 
                                                         i=i, state_models_i=state_models[i])
            cf_states[:, i] = cf_states_i[:, 0] # the first state_dim columns are for z = 0
            cf_states[:, self.state_dim + i] = cf_states_i[:, 1] # the second state_dim columns are for z = 1

        # estimate the reward quantiles for each individual under z = 0 and z = 1, respectively
        if rt is not None:
            qs_list_reward = []
            for z in [0, 1]:
                new_data = pd.DataFrame({'zs': np.repeat(z, repeats=N)})
                if self.is_action_onehot:
                    for j in range(self.dim_a):
                        new_data['action'+str(j+1)] = at.reshape(-1, self.dim_a)[:, j]
                else:
                    new_data['actions'] = at.flatten()
                for j in range(self.state_dim):
                    new_data['state'+str(j+1)] = xt[:, z * self.state_dim + j].flatten()
                new_data['rewards'] = np.zeros(N)
                qs_reward = np.zeros((N, 99))
                exog_index = new_data.index
                new_y, new_X = patsy.dmatrices(self.reward_model_spec, new_data, return_type="dataframe", NA_action='drop')
                new_X = new_X.reindex(exog_index) # we do not reindex new_y here because it is not used
                for tau in range(1, 100):
                    q = reward_models[tau - 1].predict(new_X)
                    qs_reward[:, tau - 1] = q
                qs_list_reward.append(qs_reward)

            # map the observed states to counterfactual states for each individual
            cf_rewards = np.zeros((N, 2))
            for z in [0, 1]:
                idx_z = np.where(zs == z)[0]
                cf_rewards[idx_z, z] = rt[idx_z]

                idx_c = np.where(zs != z)[0]
                idx_reward = np.argmin((abs(qs_list_reward[1 - z][idx_c, :] - rt[idx_c].reshape(-1, 1))), 
                                    axis=1)
                cf_rewards_z = qs_list_reward[z][idx_c, idx_reward]
                cf_rewards[idx_c, z] = cf_rewards_z
        
        return (cf_states, cf_rewards) if rt is not None else cf_states
    
    def _learn_marginal_dist_z(self, zs):
        marginal_dist_z = []
        for z in [0, 1]:
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
        
        return xs_tilde, rs_tilde
    
    def preprocess_single_step(self, t, z, xt, xtm1=None, atm1=None, rtm1=None, **kwargs):
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




            

            
        
        



