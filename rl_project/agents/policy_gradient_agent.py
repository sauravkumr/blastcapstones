"""Implementation of the policy gradient agent"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.network import PolicyNetwork

class PolicyGradientAgent:
    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 0.01):
        """Initialize the policy gradient agent"""
        # Create the policy network and optimizer
        self.policy_network = PolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.rewards = []
        self.log_probs = []

    def select_action(self, state: np.ndarray) -> int:
        """Select an action based on the current state"""
        # Convert state to tensor, get action probabilities from the policy network,
        # sample an action, and store the log probability of the action.
        # BEGIN CODE

        # END CODE

    def store_reward(self, reward: float):
        """Store the received reward"""
        self.rewards.append(reward)

    def update_policy(self):
        """Update the policy network using the collected rewards"""
        # Compute the discounted rewards and normalize them
        # Calculate the policy loss and update the network
        # BEGIN CODE

        # END CODE

        # Clear the rewards and log probabilities
        self.rewards = []
        self.log_probs = []

    def save(self, filepath: str):
        """Save the policy network to a file"""
        torch.save(self.policy_network.state_dict(), filepath)

    def load(self, filepath: str):
        """Load the policy network from a file"""
        self.policy_network.load_state_dict(torch.load(filepath))
