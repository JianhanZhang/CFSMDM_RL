import numpy as np
import sys

sys.path.append("../")

from sklearn.linear_model import LinearRegression
from base_models import NeuralNetRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


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
        self.buffer = {}
        for key, prob in self.marginal_dist_z.items():
            self.buffer[key] = np.zeros((n, self.xdim))

    def _hash_tuples(self, tuples):
        if tuples.ndim == 1:
            hashing_tuples = "_".join(np.array(tuples).astype("str"))
        else:
            hashing_tuples = np.apply_along_axis(
                lambda x: "_".join(np.array(x).astype("str")), axis=1, arr=tuples
            )
        return hashing_tuples

    def _reverse_hash_tuples(self, hashing_tuples):
        if isinstance(hashing_tuples, str):
            tuples = np.array(hashing_tuples.split("_")).astype("float")
        else:
            tuples = np.apply_along_axis(
                lambda x: np.array(x.split("_")).astype("float"),
                axis=0,
                arr=hashing_tuples,
            )
        return tuples


class SequentialPreprocessor(Preprocessor):
    def __init__(
        self,
        z_space,
        action_space,
        reg_model="nn",
        hidden_dims=[64, 64], 
        epochs=1000,
        learning_rate=0.005,
        batch_size=512,
        is_action_onehot=True,
        is_normalized=False,
        is_early_stopping=False,
        test_size=0.2,
        early_stopping_patience=10,
        early_stopping_min_delta=0.01,
        cross_folds=1,
        mode="single", 
    ) -> None:
        z_space = np.array(z_space)
        action_space = np.array(action_space)

        self.reg_model = reg_model
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.is_action_onehot = is_action_onehot
        self.is_normalized = is_normalized
        self.n_actions = len(np.unique(action_space.flatten()))
        self.action_space = action_space
        self.z_space = z_space
        self.zdim = z_space.shape[-1]
        self.__name__ = 'SequentialPreprocessor'

        self.is_early_stopping = is_early_stopping
        self.test_size = test_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.cross_folds = cross_folds
        self.mode = mode

    def encode_a(self, a):
        enc = OneHotEncoder(categories=[self.action_space.flatten()], drop=None)
        return enc.fit_transform(a.reshape(-1, 1)).toarray()

    def _learn_initial_model(self, xs, zs):
        # learn model at time 0
        model0 = {}
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            model0[tuple(z)] = np.mean(xs[idx_z, 0, :], axis=0)
        return model0
        

    def _learn_transition_models(self, xs, zs, actions, rewards):
        N, T, _ = xs.shape
        if self.is_normalized:
            states, self.states_mean, self.states_std = self.standardize(
                xs[:, : (T - 1), :].reshape(-1, self.xdim)
            )
            next_states, self.next_states_mean, self.next_states_std = self.standardize(
                np.concatenate(
                    [
                        xs[:, 1:T, :].reshape(-1, self.xdim),
                        rewards[:, :].reshape(-1, 1),
                    ],
                    axis=1,
                )
            )
        else:
            states = xs[:, : (T - 1), :].reshape(-1, self.xdim)
            next_states = np.concatenate(
                [xs[:, 1:T, :].reshape(-1, self.xdim), rewards[:, :].reshape(-1, 1)],
                axis=1,
            )

        if self.is_action_onehot:
            actions = self.encode_a(actions.reshape(-1, 1)).reshape(N, T - 1, -1)
            self.dim_a = actions.shape[-1]
        else:
            self.dim_a = 1

        if self.reg_model == "lm":
            return self._learn_linear_model(states, next_states, zs, actions, T)
        elif self.reg_model == 'nn':
            return self._learn_neural_model(states, next_states, zs, actions, T)
        elif self.reg_model == 'qr':
            return self._learn_quantile_model(states, next_states, zs, actions, T)
        else:
            print("Model type is undefined. Please specify either \"lm\" or \"nn\".")
            exit(1)

    def _learn_quantile_model(self, states, next_states, zs, actions, T):
        pass

    def _learn_linear_model(self, states, next_states, zs, actions, T):
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions[:, : (T - 1), np.newaxis].reshape(-1, self.dim_a),
                states,
            ],
            axis=1,
        )
        Y = next_states
        return LinearRegression().fit(X, Y)

    def _learn_neural_model(self, states, next_states, zs, actions, T):
        if self.mode == "single":
            return self._learn_single_neural_model(states, next_states, zs, actions, T)
        elif self.mode == "sensitive":
            return self._learn_sensitive_neural_model(
                states, next_states, zs, actions, T
            )
        else: 
            print("Model mode is undefined. Please specify either \"single\" or \"sensitive\".")
            exit(1)


    def _learn_single_neural_model(self, states, next_states, zs, actions, T):
        X = np.concatenate(
            [
                np.repeat(zs.reshape(-1, 1, self.zdim), repeats=T - 1, axis=1).reshape(
                    -1, self.zdim
                ),
                actions[:, : (T - 1), np.newaxis].reshape(-1, self.dim_a),
                states,
            ],
            axis=1,
        )
        Y = next_states
        model = NeuralNetRegressor(
            in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dims=self.hidden_dims
        )
        model.fit(
            X,
            Y,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            is_early_stopping=self.is_early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            test_size=self.test_size,
        )
        return model

    def _learn_sensitive_neural_model(self, states, next_states, zs, actions, T):
        model = {}
        zs = np.repeat(zs[:, np.newaxis, :], axis=1, repeats=T - 1).reshape(
            -1, self.zdim
        )
        for z in np.unique(zs, axis=0):
            idx_z = np.all(zs == z, axis=1)
            states_z = states[idx_z].reshape(-1, self.xdim)
            next_states_z = next_states[idx_z].reshape(-1, next_states.shape[-1])
            actions_z = actions[:, : (T - 1)].reshape(-1, self.dim_a)[idx_z]
            X = np.concatenate([actions_z, states_z], axis=1)
            Y = next_states_z

            model[tuple(z)] = NeuralNetRegressor(
                in_dim=X.shape[1], out_dim=Y.shape[1], hidden_dims=self.hidden_dims
            )
            model[tuple(z)].fit(
                X,
                Y,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                is_early_stopping=self.is_early_stopping,
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_min_delta=self.early_stopping_min_delta,
                test_size=self.test_size,
            )
        return model

    def learn_transition_models(self, xs, zs, actions, rewards):
        model0 = self._learn_initial_model(xs, zs)
        model = self._learn_transition_models(xs, zs, actions, rewards)
        return model0, model
    

    def learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _estimate_cf_next_state_reward_mean_tg1(self, model, z, at, xt):
        N = xt.shape[0]
        if self.is_normalized:
            state = self.standardize(xt, self.states_mean, self.states_std)
        else:
            state = xt

        if self.is_action_onehot:
            encoded = self.encode_a(at.reshape(-1, 1))
            at = encoded.reshape(N, -1)
            dim_a = at.shape[-1]
        else:
            dim_a = 1

        if self.mode == "single":
            m = model.predict(
                np.concatenate(
                    [
                        z,
                        at.reshape(-1, dim_a),
                        state,
                    ],
                    axis=1,
                )
            )
        elif self.mode == "sensitive":
            m = np.zeros((xt.shape[0], self.xdim + 1))
            for z_ in self.z_space:
                idx_z = np.all(z == z_, axis=1)
                states_z = state[idx_z]
                actions_z = at[idx_z].reshape(-1, dim_a)
                X = np.concatenate([actions_z, states_z], axis=1)
                m[idx_z] = model[tuple(z_)].predict(X)
        if self.is_normalized:
            m = self.destandardize(m, self.next_states_mean, self.next_states_std)
        return m

    def _process_initial_state(self, initial_model, xt, zs):
        N = xt.shape[0]
        epsilon_hat = xt - np.array([initial_model[tuple(z.flatten())] for z in zs])
        xt_c = {
            key: np.broadcast_to(initial_model[key], (N, self.xdim)) + epsilon_hat
            for key in self.marginal_dist_z
        }
        xt_tilde = np.hstack([xt_c[key] for key in self.marginal_dist_z])
        return xt_tilde, xt_c

    def _process_subsequent_states(
        self, transition_model, xt, xtm1, zs, atm1, rtm1=None
    ):
        N = xt.shape[0]
        m = self._estimate_cf_next_state_reward_mean_tg1(
            transition_model, zs, atm1, xtm1
        )
        xt_mean = m[:, :-1]
        epsilon_hat = xt - xt_mean

        if rtm1 is not None:
            rtm1_mean = m[:, -1]
            epsilon_hat_reward = rtm1 - rtm1_mean

        xt_c = {}
        rtm1_c = {}
        for key, prob in self.marginal_dist_z.items():
            key_vec = np.repeat(np.array(key).reshape(1, -1), repeats=N, axis=0)
            m = self._estimate_cf_next_state_reward_mean_tg1(
                transition_model, key_vec, atm1, self.buffer[key]
            )
            xt_mean = m[:, :-1]
            xt_c[key] = xt_mean + epsilon_hat
            if rtm1 is not None:
                rtm1_mean = m[:, -1]
                rtm1_c[key] = rtm1_mean + epsilon_hat_reward

        xt_tilde = np.hstack(
            [xt_c[key] * 1 for key, prob in self.marginal_dist_z.items()]
        )
        if rtm1 is not None:
            rtm1_tilde = np.sum(
                [rtm1_c[key] * prob for key, prob in self.marginal_dist_z.items()],
                axis=0,
            )
        else:
            rtm1_tilde = None

        return xt_tilde, rtm1_tilde, xt_c

    def train_preprocessor(self, zs, xs, actions, rewards, seed=None):
        # some convenience variables
        N, T, xdim = xs.shape
        self.N, self.T, self.xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self.learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        self.model0 = [None for _ in range(self.cross_folds)]
        self.model = [None for _ in range(self.cross_folds)]

        if self.cross_folds == 1:
            self.model0[0], self.model[0] = self.learn_transition_models(
                xs=xs, zs=zs, actions=actions, rewards=rewards
            )
            self.reset_buffer(n=N)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for t in range(T):
                if t == 0:
                    xt_tilde, xt_c = self._process_initial_state(
                        self.model0[0], xs[:, 0, :], zs
                    )
                else:
                    xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                        self.model[0],
                        xs[:, t, :],
                        xs[:, t - 1, :],
                        zs,
                        actions[:, t - 1],
                        rewards[:, t - 1],
                    )
                    rs_tilde[:, t - 1] = rtm1_tilde
                self.buffer = xt_c.copy()
                xs_tilde[:, t, :] = xt_tilde
        else:
            kf = KFold(n_splits=self.cross_folds)
            xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
            rs_tilde = np.zeros([N, T - 1])
            for i, (train_index, test_index) in enumerate(kf.split(xs)):

                xs_train, xs_test = xs[train_index], xs[test_index]
                zs_train, zs_test = zs[train_index], zs[test_index]
                actions_train, actions_test = actions[train_index], actions[test_index]
                rewards_train, rewards_test = rewards[train_index], rewards[test_index]
                self.model0[i], self.model[i] = self.learn_transition_models(
                    xs=xs_train,
                    zs=zs_train,
                    actions=actions_train,
                    rewards=rewards_train,
                )
                n = xs_test.shape[0]
                self.reset_buffer(n=n)
                for t in range(T):
                    if t == 0:
                        xt_tilde_test, xt_c_test = self._process_initial_state(
                            self.model0[i], xs_test[:, 0, :], zs_test
                        )
                    else:
                        xt_tilde_test, rtm1_tilde_test, xt_c_test = (
                            self._process_subsequent_states(
                                self.model[i],
                                xs_test[:, t, :],
                                xs_test[:, t - 1, :],
                                zs_test,
                                actions_test[:, t - 1],
                                rewards_test[:, t - 1],
                            )
                        )
                        rs_tilde[test_index, t - 1] = rtm1_tilde_test
                    xs_tilde[test_index, t, :] = xt_tilde_test
                    self.buffer = xt_c_test.copy()
        return xs_tilde, rs_tilde

    def preprocess_single_step(self, t, z, xt, xtm1=None, atm1=None, rtm1=None, **kwargs):
        # some convenience variables
        N, xdim = xt.shape
        cross_folds = len(self.model)

        # check if state dimension of the input is the same as state dimension of the training data
        if xdim != self.xdim:
            raise ValueError('The state dimension of the input does not match that of the training data.')

        # t = 0
        if xtm1 is None and atm1 is None:
            self.reset_buffer(n=N) 

            if self.cross_folds == 1:
                xt_tilde, xt_c = self._process_initial_state(self.model0[0], xt, z)
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                for k in range(cross_folds):
                    xt_tilde_k, xt_c_k = self._process_initial_state(
                        self.model0[k], xt, z
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
        else:
            if cross_folds == 1:
                xt_tilde, rtm1_tilde, xt_c = self._process_subsequent_states(
                    self.model[0], xt, xtm1, z, atm1, rtm1
                )
                self.buffer = xt_c.copy()
            else:
                buffer_tmp = {key: np.zeros_like(xt) for key in self.marginal_dist_z}
                xt_tilde = np.zeros((N, self.xdim * len(self.marginal_dist_z.keys())))
                if rtm1 is not None:
                    rtm1_tilde = np.zeros_like(rtm1)
                for k in range(cross_folds):
                    xt_tilde_k, rtm1_tilde_k, xt_c_k = self._process_subsequent_states(
                        self.model[k], xt, xtm1, z, atm1, rtm1
                    )
                    buffer_tmp = {
                        key: buffer_tmp[key] + xt_c_k[key] / cross_folds
                        for key in self.marginal_dist_z
                    }
                    xt_tilde += xt_tilde_k / cross_folds
                    if rtm1 is not None:
                        rtm1_tilde += rtm1_tilde_k / cross_folds
                self.buffer = buffer_tmp.copy()
        return (xt_tilde, rtm1_tilde) if rtm1 is not None else xt_tilde
    
    def preprocess_multiple_steps(self, zs, xs, actions, rewards=None):
        # some convenience variables
        N, T, xdim = xs.shape

        # check if state dimension of the input is the same as state dimension of the training data
        if xdim != self.xdim:
            raise ValueError('The state dimension of the input does not match that of the training data.')
        
        # define the returned arrays; the arrays will be filled later
        xs_tilde = np.zeros([N, T, xdim * len(self.marginal_dist_z.keys())])
        rs_tilde = np.zeros([N, T - 1])

        # preprocess the initial step
        xs_tilde[:, 0, :] = self.preprocess_single_step(t=0, z=zs, xt=xs[:, 0, :])

        # preprocess subsequent steps
        if rewards is not None:
            for t in range (1, T):
                xs_tilde[:, t, :], rs_tilde[:, t-1] = self.preprocess_single_step(t=t, 
                                                                                  z=zs, 
                                                                                     xt=xs[:, t, :], 
                                                                                     xtm1=xs[:, t-1, :], 
                                                                                     atm1=actions[:, t-1], 
                                                                                     rtm1=rewards[:, t-1]
                                                                                     )
            return xs_tilde, rs_tilde                
        else:
            for t in range (1, T):
                xs_tilde[:, t, :] = self.preprocess_single_step(t=t, 
                                                                z=zs, 
                                                                xt=xs[:, t, :], 
                                                                xtm1=xs[:, t-1, :], 
                                                                atm1=actions[:, t-1]
                                                                )
            return xs_tilde
