"""
Implements preprocessing for the Lunar Lander.
"""
import torch as tc
import torch.nn as nn
from typing import Union, Tuple
from rl2.agents.preprocessing.common import one_hot, Preprocessing


class LunarPreprocessing(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        
        # Input dimensions for each component
        self.prev_obs_dim = obs_dim  # Previous observation dimension
        self.action_dim = 1  # Single discrete action
        self.reward_dim = 1  # Single reward value
        self.done_dim = 1  # Single termination flag
        
        # Total input dimension after concatenation
        self.total_input_dim = obs_dim + self.prev_obs_dim + self.action_dim + self.reward_dim + self.done_dim
        
        # Optional: Add normalization layers if needed
        self.obs_norm = nn.LayerNorm(obs_dim)
        self.prev_obs_norm = nn.LayerNorm(self.prev_obs_dim)
        
    def forward(
        self,
        curr_obs: Union[tc.LongTensor, tc.FloatTensor],
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Preprocess and concatenate inputs.
        
        Args:
            curr_obs: Current observation tensor [B, obs_dim]
            prev_action: Previous action tensor [B, 1]
            prev_reward: Previous reward tensor [B, 1]
            prev_done: Previous done flag tensor [B, 1]
            
        Returns:
            Concatenated and preprocessed tensor [B, total_input_dim]
        """
        batch_size = curr_obs.size(0)
        device = curr_obs.device
        
        # Normalize observations if using LayerNorm
        curr_obs = self.obs_norm(curr_obs)
        
        # Get previous observation from the current batch
        # If it's the first step, use zeros
        prev_obs = tc.zeros_like(curr_obs)
        if batch_size > 1:
            prev_obs[1:] = curr_obs[:-1].clone()
        prev_obs = self.prev_obs_norm(prev_obs)
        
        # Ensure all inputs have correct shape
        prev_action = prev_action.float().view(batch_size, 1)
        prev_reward = prev_reward.view(batch_size, 1)
        prev_done = prev_done.float().view(batch_size, 1)
        
        # Concatenate all inputs
        concatenated = tc.cat([
            curr_obs,           # Current observation
            prev_obs,          # Previous observation (st-1)
            prev_action,       # Previous action (at)
            prev_reward,       # Previous reward (rt)
            prev_done,         # Previous termination flag (dt)
        ], dim=1)
        
        return concatenated
    
    @property
    def output_dim(self) -> int:
        """Returns the dimension of the preprocessed output."""
        return self.total_input_dim


class LunarLanderPreprocessing(Preprocessing):
    def __init__(self, num_actions: int, use_state_history: bool = True):
        super().__init__()
        self._num_actions = num_actions
        self.use_state_history = use_state_history
        
        # Define dimensions
        self.obs_dim = 8  # Lunar Lander state dimensions
        
        # Optional normalization layers
        self.obs_norm = nn.LayerNorm(self.obs_dim)
        self.prev_obs_norm = nn.LayerNorm(self.obs_dim) if use_state_history else None

    @property
    def output_dim(self):
        if self.use_state_history:
            # Current obs + prev obs + one-hot action + reward + done
            return self.obs_dim + self.obs_dim + self._num_actions + 2
        else:
            # Just current obs + one-hot action + reward + done (original behavior)
            return self.obs_dim + self._num_actions + 2
    
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
            tc.FloatTensor concatenating current observation, previous observation (if enabled),
            one-hot encoded previous action, previous reward, and done flag.
        """
        batch_size = curr_obs.size(0)
        device = curr_obs.device
        
        # Normalize current observations
        curr_obs = self.obs_norm(curr_obs)
        
        # Process previous observation if using state history
        if self.use_state_history:
            # Create previous observation tensor (zeros for first step)
            prev_obs = tc.zeros_like(curr_obs)
            if batch_size > 1:
                prev_obs[1:] = curr_obs[:-1].clone()
            prev_obs = self.prev_obs_norm(prev_obs)
        
        # Process previous action (one-hot encoding)
        emb_a = one_hot(prev_action, depth=self._num_actions)
        
        # Ensure reward and done flags have correct shape
        prev_reward = prev_reward.view(batch_size, 1)
        prev_done = prev_done.float().view(batch_size, 1)
        
        if self.use_state_history:
            # Concatenate all components including previous observation
            vec = tc.cat([
                curr_obs,           # Current observation
                prev_obs,           # Previous observation
                emb_a,              # One-hot encoded previous action
                prev_reward,        # Previous reward
                prev_done,          # Previous done flag
            ], dim=1)
        else:
            # Original behavior: concatenate without previous observation
            vec = tc.cat([
                curr_obs,
                emb_a,
                prev_reward,
                prev_done
            ], dim=1)
        
        return vec.float()
