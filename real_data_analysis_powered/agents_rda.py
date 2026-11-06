import numpy as np
import torch
import copy
from base_models_rda import LinearRegressor, NeuralNetRegressor, NeuralNet

class RandomAgent:
    def __init__(self, num_action_levels) -> None:
        self.name = "random"
        self.state_size = 0  # not used
        self.num_action_levels = num_action_levels
        self.final_fqi_loss = None

    def act(self, uat=None, is_return_prob=False, **kargs):
        probs = np.ones([uat.shape[0], 3]) * 1 / 3
        cdf = np.cumsum(probs, axis=1)

        actions = np.zeros([uat.shape[0]], dtype=int)
        visits = np.zeros_like(actions)
        if uat is not None:
            for i in range(cdf.shape[1]):
                masks = (uat.flatten() <= cdf[:, i]).astype(int) - visits
                actions += masks * i
                visits += masks
        else:
            uat = np.random.uniform(size=uat.shape[0])
            for i in range(cdf.shape[1]):
                masks = (uat.flatten() <= cdf[:, i]).astype(int) - visits
                actions += masks * i
                visits += masks

        if is_return_prob:
            return probs
        else:
            return actions



class FQI:
    def __init__(
        self,
        model_type,
        action_space,
        hidden_dims=[32],
        preprocessor=None,
        gamma=0.9,
        learning_rate=0.1,
        epochs=500,
        device='cpu'
    ):
        self.model_type = model_type
        self.action_space = action_space
        self.action_size = len(action_space)
        self.hidden_dims = hidden_dims
        self.preprocessor = preprocessor
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.__name__ = 'FQI'
        self.device = device
        self._sanity_check()

    def _sanity_check(self):
        assert self.model_type in ["lm", "nn"], "Invalid model type"

    def _init_model(self, state_dim):
        if self.model_type == "lm":
            self.model = [
                LinearRegressor(
                    featurize_method="polynomial",
                    degree=2,
                    interaction_only=False,
                    is_standarized=False,
                )
                for _ in range(self.action_size)
            ]
        elif self.model_type == "nn":
            self.model = NeuralNet(
                in_dim=state_dim,
                out_dim=self.action_size,
                hidden_dims=self.hidden_dims,
            )

    def _fit_helper(self, states, actions, rewards, next_states, state_dim, max_iter):
        torch.set_num_threads(1)
        states = states.reshape(-1, state_dim)
        next_states = next_states.reshape(-1, state_dim)
        actions = actions.reshape(-1, 1)
        rewards = rewards.reshape(-1, 1)

        for iteration in range(max_iter):
            if self.model_type == "lm":
                self._fit_lm(states, actions, rewards, next_states, iteration)
                loss = None
            elif self.model_type == "nn":
                loss = self._fit_nn(states, actions, rewards, next_states, iteration, state_dim)
            if loss is not None and loss > 10:
                print("FQI diverged. Stop training.")
                break
        
        return loss

    def _fit_nn(self, states, actions, rewards, next_states, iteration, state_dim):
        if iteration == 0:
            Y = rewards
        else:
            self.model.eval()
            with torch.no_grad():
                next_q_values = self.model(torch.FloatTensor(next_states)).numpy()
            Y = (
                rewards + self.gamma * np.max(next_q_values, axis=1, keepdims=True)
            ).reshape(-1, 1)

        new_model = NeuralNet(
            in_dim=state_dim,
            out_dim=self.action_size,
            hidden_dims=self.hidden_dims,
        )

        X = torch.FloatTensor(states)
        Y = torch.FloatTensor(Y)
        actions_tensor = torch.LongTensor(actions.reshape(-1, 1))

        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()

        for _ in range(self.epochs):
            new_model.train()
            optimizer.zero_grad()
            Y_pred = new_model(X).gather(1, actions_tensor)
            loss = criterion(Y_pred, Y)
            loss.backward()
            optimizer.step()

        self.model = copy.deepcopy(new_model)
        return loss.item()

    def _fit_lm(self, states, actions, rewards, next_states, iteration):
        if iteration == 0:
            Y = rewards
        else:
            tmp = np.array(
                [model.predict(next_states) for model in self.model]
            ).T.squeeze()
            Y = (rewards + self.gamma * np.max(tmp, axis=1, keepdims=True)).reshape(
                -1, 1
            )

        new_model = [
            LinearRegressor(
                featurize_method="polynomial",
                degree=2,
                interaction_only=False,
                is_standarized=False,
            )
            for _ in range(self.action_size)
        ]

        for a in range(self.action_size):
            idx = actions.flatten() == a
            new_model[a].fit(states[idx], Y[idx])

        mse = np.mean([m.mse for m in new_model])
        self.model = copy.deepcopy(new_model)

    def _act_helper(self, states):
        if self.model_type == "lm":
            q_values = np.array(
                [model.predict(states) for model in self.model]
            ).T.squeeze()
        elif self.model_type == "nn":
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(states)).numpy()
        return np.argmax(q_values, axis=1)
    
    def train(self, zs, xs, actions, rewards, max_iter, seed=None, preprocess=True):
        N, T = xs.shape[:2]
        
        if self.preprocessor is not None and preprocess:
            states_p, rewards_p = self.preprocessor.preprocess_multiple_steps(
                    zs=zs, xs=xs, actions=actions, rewards=rewards
            )
        else:
            states_p = xs
            rewards_p = rewards
        state_dim = states_p.shape[-1]
        self._init_model(state_dim)

        next_states_p = states_p[:, 1:].copy()
        states_p = states_p[:, :-1].copy()

        num_retries = 0
        while num_retries < 5:
            self.final_fqi_loss = self._fit_helper(states_p, 
                                                   actions, 
                                                   rewards_p, 
                                                   next_states_p, 
                                                   state_dim, 
                                                   max_iter)
            if self.final_fqi_loss < 2:
                break
            print('FQI loss is:', self.final_fqi_loss, "This is too high; retraining FQI...")
            num_retries += 1
        return self.final_fqi_loss

    def act(self, t, z, xt, xtm1=None, atm1=None, uat=None, is_return_prob=False, preprocess=True, **kwargs):
        if self.preprocessor is not None and preprocess:
            states = self.preprocessor.preprocess_single_step(
                t=t, xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=None
            )
        else:
            states = xt
        actions = self._act_helper(states)
        probs = np.zeros((xt.shape[0], self.action_size))
        probs[np.arange(len(actions)), actions] = 1.0
        if is_return_prob:
            return probs
        else:
            return actions
        
    def forward(self, states):
        self.model.eval()
        return (
            self.model.forward(
                torch.tensor(states, dtype=torch.float32).to(self.device)
            )
            .detach()
            .cpu()
            .numpy()
        )
    