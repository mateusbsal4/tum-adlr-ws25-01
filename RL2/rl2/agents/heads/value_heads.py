"""
Value heads for RL^2 agents.
"""

import torch as tc


class LinearValueHead(tc.nn.Module):
    """
    Value head for a reinforcement learning agent.
    """
    def __init__(self, num_features):
        super().__init__()
        self.net = tc.nn.Sequential(
            tc.nn.Linear(num_features, num_features // 2),
            tc.nn.LayerNorm(num_features // 2),
            tc.nn.ReLU(),
            tc.nn.Linear(num_features // 2, num_features // 4),
            tc.nn.LayerNorm(num_features // 4),
            tc.nn.ReLU(),
            tc.nn.Linear(num_features // 4, 1)
        )
        
        # Initialize with smaller weights
        for layer in self.net:
            if isinstance(layer, tc.nn.Linear):
                tc.nn.init.orthogonal_(layer.weight, gain=0.01)
                tc.nn.init.zeros_(layer.bias)

    def forward(self, features: tc.FloatTensor) -> tc.FloatTensor:
        """
        Computes a value estimate from features and returns it.

        Args:
            features: tc.FloatTensor of features with shape [B, ..., F].

        Returns:
            tc.FloatTensor of value estimates with shape [B, ...].
        """
        return self.net(features).squeeze(-1)
