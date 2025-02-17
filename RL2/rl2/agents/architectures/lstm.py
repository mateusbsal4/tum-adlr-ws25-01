"""
LSTM architecture for RL^2 agents.
Implements a customized LSTM with layer normalization and forget bias options.
"""

import torch as tc
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
import logging

class LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        forget_bias: Union[float, tc.Tensor] = 1.0,
        use_ln: bool = True,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        normalize_output: bool = True,
        grad_clip: float = 1.0,
        use_skip_connection: bool = True  # Add skip connection option
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.normalize_output = normalize_output
        self.use_ln = use_ln
        self.grad_clip = grad_clip
        self.use_skip_connection = use_skip_connection
        
        # Input preprocessing layers
        self.input_norm = nn.LayerNorm(input_dim) if use_ln else nn.Identity()
        self.input_dropout = nn.Dropout(p=dropout)
        
        # Main LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output processing
        self.output_size = hidden_dim * self.num_directions
        
        # Add a projection layer to handle skip connections if input and hidden dims differ
        if use_skip_connection and input_dim != self.output_size:
            self.skip_projection = nn.Linear(input_dim, self.output_size)
        else:
            self.skip_projection = nn.Identity()
            
        self.output_norm = nn.LayerNorm(self.output_size) if normalize_output else nn.Identity()
        
        # Initialize parameters with forget bias
        self._init_parameters(forget_bias)

    def _init_parameters(self, forget_bias: Union[float, tc.Tensor]) -> None:
        """Initialize LSTM parameters with orthogonal initialization and forget bias."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                # Use different gains for input and recurrent weights
                gain = 1.0 if 'ih' in name else 5/3
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias' in name:
                # Initialize all biases to 0 except forget gate bias
                if 'bias_ih' in name or 'bias_hh' in name:
                    nn.init.zeros_(param)
                    # Support both scalar and tensor forget bias
                    if isinstance(forget_bias, tc.Tensor):
                        param.data[self.hidden_dim:2*self.hidden_dim] = forget_bias
                    else:
                        param.data[self.hidden_dim:2*self.hidden_dim].fill_(forget_bias)

    def _validate_inputs(self, inputs: tc.Tensor) -> None:
        """Validate input tensor dimensions and values."""
        if not isinstance(inputs, tc.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(inputs)}")
        
        if inputs.dim() not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D tensor, got {inputs.dim()}D")
            
        if inputs.size(-1) != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {inputs.size(-1)}")
            
        if tc.isnan(inputs).any():
            raise ValueError("NaN values detected in input tensor")
            
        if tc.isinf(inputs).any():
            raise ValueError("Infinite values detected in input tensor")

    def initial_state(
        self, 
        batch_size: int, 
        device: Optional[tc.device] = None,
        requires_grad: bool = True
    ) -> Tuple[tc.Tensor, tc.Tensor]:
        """
        Initialize hidden and cell states for the LSTM.
        
        Args:
            batch_size: Number of sequences in the batch
            device: Device to create the states on. If None, uses the device of the LSTM's parameters.
            requires_grad: Whether the initial states should require gradients
            
        Returns:
            Tuple of (hidden_state, cell_state) each with shape [num_layers * num_directions, batch_size, hidden_dim]
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Create states with shape [num_layers * num_directions, batch_size, hidden_dim]
        h0 = tc.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, 
                     device=device, requires_grad=requires_grad)
        c0 = tc.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                     device=device, requires_grad=requires_grad)
        
        return h0, c0
    
    @property
    def output_dim(self) -> int:
        """Returns the output dimension of the LSTM, accounting for bidirectionality."""
        return self.output_size

    def _process_sequence_lengths(
        self, 
        inputs: tc.Tensor, 
        sequence_lengths: Optional[tc.Tensor] = None
    ) -> Tuple[tc.Tensor, Optional[tc.Tensor]]:
        """Process and validate sequence lengths for packing."""
        if sequence_lengths is None:
            if isinstance(inputs, tc.nn.utils.rnn.PackedSequence):
                return inputs, None
            # Assume all sequences are full length
            sequence_lengths = tc.full((inputs.size(0),), inputs.size(1), 
                                    device=inputs.device, dtype=tc.long)
        else:
            # Validate sequence lengths
            if sequence_lengths.size(0) != inputs.size(0):
                raise ValueError("sequence_lengths batch size doesn't match inputs")
            if (sequence_lengths > inputs.size(1)).any():
                raise ValueError("sequence_lengths cannot be greater than input sequence length")
                
        return inputs, sequence_lengths

    def forward(
        self,
        inputs: Union[tc.Tensor, tc.nn.utils.rnn.PackedSequence],
        prev_state: Optional[Tuple[tc.Tensor, tc.Tensor]] = None,
        sequence_lengths: Optional[tc.Tensor] = None,
        return_states: bool = True
    ) -> Tuple[tc.FloatTensor, Optional[Tuple[tc.Tensor, tc.Tensor]]]:
        """
        Forward pass through the LSTM.
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_len, input_dim] or PackedSequence
            prev_state: Optional tuple of (hidden_state, cell_state)
            sequence_lengths: Optional tensor of sequence lengths for packing
            return_states: Whether to return the LSTM states
            
        Returns:
            Tuple of (output features, (hidden_state, cell_state) if return_states=True)
        """
        try:
            # Handle unbatched input
            if not isinstance(inputs, tc.nn.utils.rnn.PackedSequence) and inputs.dim() == 2:
                inputs = inputs.unsqueeze(0)  # Add batch dimension
                is_batched = False
            else:
                is_batched = True
                
            # Store original input for skip connection
            original_input = inputs
            if isinstance(original_input, tc.nn.utils.rnn.PackedSequence):
                original_input = original_input.data
                
            # Validate inputs
            if not isinstance(inputs, tc.nn.utils.rnn.PackedSequence):
                self._validate_inputs(inputs)
            
            # Get batch size and device
            if isinstance(inputs, tc.nn.utils.rnn.PackedSequence):
                batch_size = inputs.batch_sizes[0].item()
                device = inputs.data.device
            else:
                batch_size = inputs.size(0)
                device = inputs.device
            
            # Initialize or process states
            if prev_state is None:
                prev_state = self.initial_state(batch_size, device)
            else:
                h, c = prev_state
                # Ensure correct shape and device
                if h.dim() == 2:  # If missing batch dimension
                    h = h.unsqueeze(1)
                    c = c.unsqueeze(1)
                elif h.dim() == 4:  # If has extra dimension
                    h = h.squeeze(0)
                    c = c.squeeze(0)
                # Move to correct device if needed
                h = h.to(device)
                c = c.to(device)
                prev_state = (h, c)
            
            # Preprocess input
            if not isinstance(inputs, tc.nn.utils.rnn.PackedSequence):
                inputs = self.input_norm(inputs)
                inputs = self.input_dropout(inputs)
                
                # Handle sequence packing
                inputs, sequence_lengths = self._process_sequence_lengths(inputs, sequence_lengths)
                if sequence_lengths is not None:
                    inputs = nn.utils.rnn.pack_padded_sequence(
                        inputs, sequence_lengths.cpu(), 
                        batch_first=True, enforce_sorted=False
                    )
            
            # Run LSTM
            lstm_out, (h_n, c_n) = self.lstm(inputs, prev_state)
            
            # Unpack if needed
            if isinstance(lstm_out, tc.nn.utils.rnn.PackedSequence):
                lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                    lstm_out, batch_first=True
                )
            
            # Add skip connection if enabled
            if self.use_skip_connection:
                if isinstance(original_input, tc.nn.utils.rnn.PackedSequence):
                    original_input, _ = nn.utils.rnn.pad_packed_sequence(
                        original_input, batch_first=True
                    )
                projected_input = self.skip_projection(original_input)
                lstm_out = lstm_out + projected_input
            
            # Process output
            output = self.output_norm(lstm_out)
            
            # Clip gradients if needed
            if self.grad_clip > 0 and self.training:
                for param in self.parameters():
                    if param.grad is not None:
                        nn.utils.clip_grad_norm_(param, self.grad_clip)
            
            # Remove batch dimension if input was unbatched
            if not is_batched:
                output = output.squeeze(0)
                h_n = h_n.squeeze(1)
                c_n = c_n.squeeze(1)
            
            if return_states:
                return output, (h_n, c_n)
            return output, None
            
        except Exception as e:
            logging.error(f"Error in LSTM forward pass: {str(e)}")
            raise

    def reset_states(self) -> None:
        """Reset any stateful parts of the model."""
        # The LSTM states are handled externally, so we don't need to reset anything here
        pass
