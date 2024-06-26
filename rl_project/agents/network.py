"""Basic definition of the Policy Network"""

import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Define the policy network"""
        # We suggest you start by creating a fully connected layer that
        # has the following layer dimensions: (input_dim, 128) (128, output_dim)
        # BEGIN CODE

        # END CODE
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Implement the forward pass using ReLU activation for the hidden layer
        # and Softmax activation for the output layer.
        # BEGIN CODE

        # END CODE

# BONUS SUGGESTIONS (for those who want to go the extra mile)
# Implement a network with multiple hidden layers
# Implement a network with different activation functions
# Implement a value network in addition to the policy network