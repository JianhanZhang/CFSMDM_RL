import numpy as np
from base_models_rda import NeuralNet
from agents_rda import FQI
from utils_rda import EmpiricalCDF

class ECOCF:
    def __init__(
        self,
        model_type,
        z_space, 
        action_space,
        mode='m', 
        hidden_dims=[32],
        gamma=0.9,
        learning_rate=0.1,
        epochs=500,
        device='cpu'
    ):
        z_space = np.array(z_space)
        action_space = np.array(action_space)
        self.z_space = z_space
        self.action_size = action_space.shape[0]
        if not (mode == 'o' or mode == 'm'):
            raise ValueError('Invalid mode for ECOCF. Must be \'m\' or \'o\'.')
        self.mode = mode

        self.agent = FQI(
            model_type='nn', 
            action_space=action_space, 
            hidden_dims=hidden_dims, 
            gamma=gamma, 
            learning_rate=learning_rate, 
            epochs=epochs, 
            device=device
        )

    def _train_Q(
        self,
        xs,
        z,
        actions,
        rewards,
        max_iter,
    ):
        N, T = xs.shape[:2]

        z_expanded = np.repeat(z[:, np.newaxis, :], T, axis=1)
        states_p = np.concatenate((xs, z_expanded), axis=-1)

        self.final_fqi_loss = self.agent.train(
            zs=z, 
            xs=states_p,
            actions=actions,
            rewards=rewards,
            max_iter=max_iter,
        )

    def _train_postprocess_o(self, xs, zs):
        # learn marginal distribution of z
        self.marginal_dist_z = {}
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            self.marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)

        # learn conditional expectation E(S_t | Z=z)
        self.model = {}  # {time: {z: mean_state}}
        N, T, xdim = xs.shape
        for t in range(T):
            self.model[t] = {}
            xs_t = xs[:, t, :]
            for z in np.unique(zs, axis=0):
                idx_z = np.all(zs == z, axis=1)
                self.model[t][tuple(z)] = np.mean(xs_t[idx_z, :], axis=0)

    def _train_postprocess_m(self, xs, zs):
        # learn marginal distribution of z
        self.marginal_dist_z = {}
        #self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            self.marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)

        # learn empirical CDF of P(S_t | Z=z)
        self.model = {}
        N, T, xdim = xs.shape
        for t in range(T):
            self.model[t] = {}
            xs_t = xs[:, t, :]
            for z in np.unique(zs, axis=0):
                idx_z = np.all(zs == z, axis=1)
                self.model[t][tuple(z)] = {}
                for i in range(xdim):
                    self.model[t][tuple(z)][i] = EmpiricalCDF(xs_t[idx_z, i].flatten())

    def train(
        self,
        zs, 
        xs,
        actions,
        rewards,
        max_iter,
        seed=None, 
        preprocess=True
    ):
        self.state_size = xs.shape[-1]
        self._train_Q(
            xs=xs,
            z=zs,
            actions=actions,
            rewards=rewards,
            max_iter=max_iter,
        )
        if self.mode == 'o':
            self._train_postprocess_o(xs=xs, zs=zs)
        elif self.mode == 'm':
            self._train_postprocess_m(xs=xs, zs=zs)
        return self.final_fqi_loss

    def act(self, t, z, xt, xtm1=None, atm1=None, uat=None, is_return_prob=False, **kwargs):
        N, xdim = xt.shape

        if self.mode == 'o':
            Q_values = np.zeros((N, self.action_size))
            for z1 in self.z_space:
                p_z1 = self.marginal_dist_z[tuple(z1)]
                q_values_inner = np.zeros((N, self.action_size))
                s_z1 = (
                    xt
                    - np.array([self.model[t][tuple(zi)] for zi in z])
                    + np.repeat(np.array(self.model[t][tuple(z1)]), N, axis=0).reshape(
                        -1, xt.shape[-1]
                    )
                )
                for z2 in self.z_space:
                    p_z2 = self.marginal_dist_z[tuple(z2)]
                    states = np.concatenate(
                        (s_z1, np.repeat(z2.reshape(1, -1), N, axis=0)), axis=-1
                    )
                    q_values_z2_sz1 = self.agent.forward(states)
                    q_values_inner += p_z2 * q_values_z2_sz1
                Q_values += p_z1 * q_values_inner
            probs = np.zeros((N, self.action_size))
            actions = np.argmax(Q_values, axis=1)
            probs[np.arange(len(actions)), actions] = 1.0
        elif self.mode == 'm':
            Q_values = np.zeros((N, self.action_size))
            for z1 in self.z_space:
                p_z1 = self.marginal_dist_z[tuple(z1)]
                q_values_inner = np.zeros((N, self.action_size))
                s_z1 = np.zeros((N, xdim))
                for i in range(xdim):
                    qs = []
                    for j in range(N):
                        z_j = z[j, :]
                        qs.append(self.model[t][tuple(z_j)][i].get_quantile_levels(x=xt[j, i]))
                    s_z1[:, i] = self.model[t][tuple(z1)][i].get_quantiles(q=qs)
                for z2 in self.z_space:
                    p_z2 = self.marginal_dist_z[tuple(z2)]
                    states = np.concatenate(
                        (s_z1, np.repeat(z2.reshape(1, -1), N, axis=0)), axis=-1
                    )
                    q_values_z2_sz1 = self.agent.forward(states)
                    q_values_inner += p_z2 * q_values_z2_sz1
                Q_values += p_z1 * q_values_inner
            probs = np.zeros((N, self.action_size))
            actions = np.argmax(Q_values, axis=1)
            probs[np.arange(len(actions)), actions] = 1.0

        if is_return_prob:
            return probs
        else:
            return actions