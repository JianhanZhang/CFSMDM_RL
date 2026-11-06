import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from preprocessor_baselines import Preprocessor
from utils import EmpiricalCDF

class FLAP(Preprocessor):
    def __init__(
        self,
        z_space=np.array([[0], [1]]), 
        action_space=np.array([[0], [1]]), 
        mode='m'
    ) -> None:
        self.n_actions = len(np.unique(action_space.flatten()))
        self.z_space = z_space
        self.zdim = z_space.shape[-1]
        if mode == 'm' or mode == 'o':
            self.mode = mode
        else:
            raise ValueError('Invalid mode for FLAP. Must be \'m\' or \'o\'.')

    def learn_model_o(self, xs, zs):
        model = {}  # {time: {z: mean_state}}
        N, T, xdim = xs.shape
        for t in range(T):
            model[t] = {}
            xs_t = xs[:, t, :]
            for z in np.unique(zs, axis=0):
                idx_z = np.all(zs == z, axis=1)
                model[t][tuple(z)] = np.mean(xs_t[idx_z, :], axis=0)
        return model
    
    def learn_model_m(self, xs, zs):
        model = {}
        N, T, xdim = xs.shape
        for t in range(T):
            model[t] = {}
            xs_t = xs[:, t, :]
            for z in np.unique(zs, axis=0):
                idx_z = np.all(zs == z, axis=1)
                model[t][tuple(z)] = {}
                for i in range(xdim):
                    model[t][tuple(z)][i] = EmpiricalCDF(xs_t[idx_z, i].flatten())
        return model

    def _learn_marginal_dist_z(self, zs):
        marginal_dist_z = {}
        self.z_space = np.unique(zs, axis=0)
        for z in np.unique(zs, axis=0):
            z_idx = np.all(zs == z, axis=1)
            marginal_dist_z[tuple(z)] = sum(z_idx) / len(z_idx)
        return marginal_dist_z

    def _process_state_o(self, xt, zs, t):
        N, xdim = xt.shape
        epsilon_hat = xt - np.array([self.model[t][tuple(z)] for z in zs])
        xt_c = {
            key: np.broadcast_to(self.model[t][key], (N, xdim)) + epsilon_hat
            for key in self.marginal_dist_z
        }

        xt_tilde = np.zeros((N, xdim))
        for key in self.marginal_dist_z:
            xt_tilde += xt_c[key] * self.marginal_dist_z[key]
        return xt_tilde
    
    def _process_state_m(self, xt, zs, t):
        N, xdim = xt.shape
        xt_c = {}
        for key in self.marginal_dist_z:
            xt_c[key] = np.zeros((N, xdim))
            for i in range(xdim):
                qs = []
                for j in range(N):
                    z = zs[j, :]
                    qs.append(self.model[t][tuple(z)][i].get_quantile_levels(x=xt[j, i]))
                xt_c[key][:, i] = self.model[t][key][i].get_quantiles(q=qs)

        xt_tilde = np.zeros((N, xdim))
        for key in self.marginal_dist_z:
            xt_tilde += xt_c[key] * self.marginal_dist_z[key]
        return xt_tilde

    def train_preprocessor(self, zs, xs, actions, rewards):
        # some convenience variables
        N, T, xdim = xs.shape

        # learn marginal distribution of z
        self.marginal_dist_z = self._learn_marginal_dist_z(zs)

        # learn transition models, estimate residuals and estimate counterfactual outcome
        if self.mode == 'o':
            self.model = self.learn_model_o(xs, zs)  # {time: {z: mean_state}}
        elif self.mode == 'm':
            self.model = self.learn_model_m(xs, zs)

        xs_tilde = np.zeros([N, T, xdim])
        if self.mode == 'o':
            for t in range(T):
                xs_tilde[:, t, :] = self._process_state_o(xs[:, t, :], zs, t)
        elif self.mode == 'm':
            for t in range(T):
                xs_tilde[:, t, :] = self._process_state_m(xs[:, t, :], zs, t)

        rs_tilde = rewards
        return xs_tilde, rs_tilde

    def preprocess_single_step(self, t, z, xt, xtm1=None, rtm1=None, **kwargs):
        N, xdim = xt.shape
        if t not in self.model:
            raise ValueError(f"Time step {t} not found in the model.")
        if self.mode == 'o':
            xt_tilde = self._process_state_o(xt, z, t)
        elif self.mode == 'm':
            xt_tilde = self._process_state_m(xt, z, t)
        rtm1_tilde = rtm1

        return (xt_tilde, rtm1_tilde) if rtm1 is not None else xt_tilde
    
    def preprocess_multiple_steps(self, zs, xs, actions, rewards=None):
        # some convenience variables
        N, T, xdim = xs.shape
        
        # define the returned arrays; the arrays will be filled later
        xs_tilde = np.zeros([N, T, xdim])
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