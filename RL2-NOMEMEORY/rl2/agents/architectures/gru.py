"""
Implements MLP for RL^2.
"""

from typing import Tuple

import torch as tc

from rl2.agents.architectures.common.normalization import LayerNorm


class GRU(tc.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        use_ln=True
    ):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = hidden_dim  # Set output_dim to hidden_dim
        self._use_ln = use_ln

        self._fc1 = tc.nn.Linear(
            in_features=self._input_dim,
            out_features=self._hidden_dim,
            bias=(not self._use_ln))
        tc.nn.init.xavier_normal_(self._fc1.weight)
        if not self._use_ln:
            tc.nn.init.zeros_(self._fc1.bias)

        self._fc2 = tc.nn.Linear(
            in_features=self._hidden_dim,
            out_features=self._output_dim,
            bias=(not self._use_ln))
        tc.nn.init.xavier_normal_(self._fc2.weight)
        if not self._use_ln:
            tc.nn.init.zeros_(self._fc2.bias)

        if self._use_ln:
            self._fc1_ln = LayerNorm(units=self._hidden_dim)
            self._fc2_ln = LayerNorm(units=self._output_dim)
    
    @property
    def output_dim(self):
        return self._output_dim

    def forward(
        self,
        inputs: tc.FloatTensor
    ) -> tc.FloatTensor:
        """
        Run forward pass through the MLP.
        Args:
            inputs: input vec tensor with shape [B, ...]
        Returns:
            output features.
        """
        x = self._fc1(inputs)
        if self._use_ln:
            x = self._fc1_ln(x)
        x = tc.nn.ReLU()(x)

        x = self._fc2(x)
        if self._use_ln:
            x = self._fc2_ln(x)
        output = tc.nn.ReLU()(x)

        return output
