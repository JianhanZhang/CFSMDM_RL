import torch
import torch.nn as nn
import numpy as np
from agents_ddqn import OfflineDoubleDQNCore, OfflineDoubleDQN


class CQLCore(OfflineDoubleDQNCore):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=[64, 64],
        learning_rate=1e-3,
        gamma=0.9,
        batch_size=64,
        alpha=1.0,
    ):
        super().__init__(
            state_dim, action_dim, hidden_dims, learning_rate, gamma, batch_size
        )
        self.alpha = alpha

    def update(self, states, actions, rewards, next_states):

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + self.gamma * next_q_values

        td_loss = nn.MSELoss()(current_q_values, target_q_values)
        cql_loss = self.alpha * (
            self.policy_net(states).logsumexp(dim=1).mean() - current_q_values.mean()
        )
        loss = td_loss + cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def forward(self, states):
        self.policy_net.eval()
        return (
            self.policy_net.forward(
                torch.tensor(states, dtype=torch.float32)
            )
            .detach()
            .cpu()
            .numpy()
        )


class CQL:
    def __init__(
            self, 
            action_space, 
            preprocessor=None, 
            name="default", 
            hidden_dims=[64, 64],
            learning_rate=0.001,
            gamma=0.9,
            batch_size=256,
            **kwargs):
        self.preprocessor = preprocessor
        self.action_space = action_space
        self.action_size = len(action_space)
        self.name = "ddqn_" + name
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size

    def train(self, zs, xs, actions, rewards, max_iter, update_frequency=100, seed=None, preprocess=True):
        N, T = xs.shape[:2]

        if self.preprocessor is not None and preprocess:
            states_p, rewards_p = self.preprocessor.preprocess_multiple_steps(
                    zs=zs, xs=xs, actions=actions, rewards=rewards
            )
        else:
            states_p = xs
            rewards_p = rewards
        self.state_size = states_p.shape[-1]

        next_states_p = states_p[:, 1:].copy()
        states_p = states_p[:, :-1].copy()

        self.agent = CQLCore(
                    action_dim=self.action_size,
                    state_dim=self.state_size,
                    hidden_dims=self.hidden_dims,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    batch_size=self.batch_size,
                )

        self.agent.train(
            states_p, actions, rewards_p, next_states_p, max_iter, update_frequency=update_frequency
        )

    def act(self, t, z, xt, xtm1=None, atm1=None, uat=None, preprocess=True, is_return_prob=False, **kwargs):
        if self.preprocessor is not None and preprocess:
            states = self.preprocessor.preprocess_single_step(
                t=t, xt=xt, xtm1=xtm1, z=z, atm1=atm1, rtm1=None
            )
        else:
            states = xt
        probs = np.zeros((xt.shape[0], 2))
        actions = self.agent.act(states).cpu().numpy()
        probs[np.arange(len(actions)), actions] = 1.0
        if is_return_prob:
            return actions, probs
        else:
            return actions

    def forward(self, states):
        return self.agent.forward(states)