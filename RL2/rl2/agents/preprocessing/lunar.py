"""
Implements preprocessing for the Lunar Lander.
"""
import torch as tc
from rl2.agents.preprocessing.common import one_hot, Preprocessing


class LunarLanderPreprocessing(Preprocessing):
    def __init__(self, num_actions: int):
        super().__init__()
        self._num_actions = num_actions

    @property
    def output_dim(self):
        return 8 + self._num_actions + 2  # 8 for the continuous state space (Lunar Lander state dimensions)

    def forward(
        self,
        curr_obs: tc.FloatTensor,  # Continuous state representation
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Creates an input vector for a meta-learning agent.

        Args:
            curr_obs: tc.FloatTensor of shape [B, 8]; the continuous state from Lunar Lander.
            prev_action: tc.LongTensor of shape [B, ...]; previous action taken.
            prev_reward: tc.FloatTensor of shape [B, ...]; previous reward.
            prev_done: tc.FloatTensor of shape [B, ...]; indicates if the episode was done.

        Returns:
            tc.FloatTensor of shape [B, ..., 8 + A + 2]; concatenated state, action, reward, and done.
        """
        # Embedding action into one-hot vector for discrete action space
        # emb_a = tc.zeros(prev_action.shape[0], self._num_actions, device=prev_action.device)
        # emb_a.scatter_(1, prev_action.unsqueeze(-1), 1)
        emb_a = one_hot(prev_action, depth=self._num_actions)
        
        # Unsqueezing reward and done for concatenation
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)

        # Concatenate the state, action, reward, and done info
        vec = tc.cat((curr_obs, emb_a, prev_reward, prev_done), dim=-1).float()
        return vec
