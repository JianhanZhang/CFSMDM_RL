import numpy as np



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



# Preprocessor for the baseline "unaware"
class UnawarenessPreprocessor(Preprocessor):
    def __init__(self, z_space, action_space) -> None:
        self.action_space = action_space
        self.z_space = z_space

    def preprocess(self, xt, **kwargs):
        return xt   
    
    def preprocess_single_step(self, z, xt, xtm1=None, atm1=None, rtm1=None, set_seed=True, **kwargs):
        if rtm1 is None:
            return xt
        else:
            return xt, rtm1
        
    def preprocess_multiple_steps(self, zs, xs, actions, rewards=None):
        if rewards is not None:
            return xs, rewards
        else:
            return xs
        
    

# Preprocessor for the baseline "full"
class ConcatenatePreprocessor(Preprocessor):
    def __init__(self, z_space, action_space) -> None:
        self.action_space = action_space
        self.z_space = z_space

    def preprocess(self, z, xt, **kwargs):
        if xt.ndim == 1:
            xt = xt[np.newaxis, :]
            z = z[np.newaxis, :]
            xt_new = np.concatenate([xt, z], axis=1)
            return xt_new.flatten()
        elif xt.ndim == 2:
            xt_new = np.concatenate([xt, z], axis=1)
            return xt_new
        
    def preprocess_single_step(self, z, xt, xtm1=None, atm1=None, rtm1=None, set_seed=True, **kwargs):
        xt_new = self.preprocess(z, xt)
        if rtm1 is None:
            return xt_new
        else:
            return xt_new, rtm1      

    def preprocess_multiple_steps(self, zs, xs, actions, rewards=None):
        # some convenience variables
        N, T, xdim = xs.shape
        
        # define the returned arrays; the arrays will be filled later
        xs_tilde = np.zeros([N, T, xdim + zs.shape[-1]])
        rs_tilde = np.zeros([N, T - 1])

        # preprocess the initial step
        xs_tilde[:, 0, :] = self.preprocess_single_step(zs, xs[:, 0, :])

        # preprocess subsequent steps
        if rewards is not None:
            for t in range (1, T):
                xs_tilde[:, t, :], rs_tilde[:, t-1] = self.preprocess_single_step(zs, 
                                                                                  xs[:, t, :], 
                                                                                  xs[:, t-1, :], 
                                                                                  actions[:, t-1], 
                                                                                  rewards[:, t-1], 
                                                                                  set_seed=False
                                                                                  )
            return xs_tilde, rs_tilde                
        else:
            for t in range (1, T):
                xs_tilde[:, t, :] = self.preprocess_single_step(zs, 
                                                                xs[:, t, :], 
                                                                xs[:, t-1, :], 
                                                                actions[:, t-1], 
                                                                set_seed=False
                                                                )
            return xs_tilde