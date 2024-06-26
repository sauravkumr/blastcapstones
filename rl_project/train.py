"""Train the policy gradient agent"""

import gym
from agents.policy_gradient_agent import PolicyGradientAgent

def train(
        env_name: str,
        n_episodes: int = 1000,
        max_t: int = 1000,
        learning_rate: float = 0.01
    ):
    """Train the policy gradient agent"""
    env = gym.make(env_name)
    assert env.observation_space.shape is not None
    assert env.action_space.shape is not None
    agent = PolicyGradientAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        learning_rate
    )
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_rewards = 0

        for t in range(max_t):
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_reward(reward)
            episode_rewards += reward
            state = next_state
            if done:
                break
        
        agent.update_policy()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {episode_rewards}")
    
    env.close()
    agent.save('weights/policy_gradient.pth')

if __name__ == "__main__":
    train(env_name='CartPole-v1', n_episodes=1000, max_t=1000, learning_rate=0.01)
