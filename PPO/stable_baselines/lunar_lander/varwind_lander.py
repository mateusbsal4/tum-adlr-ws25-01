import gymnasium as gym
from stable_baselines3 import PPO
from lib.pol_eval import evaluate_policy
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import datetime
from gymnasium import Wrapper
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

def parse_args():
    parser = argparse.ArgumentParser(description='Train or evaluate a LunarLander agent with variable wind')
    parser.add_argument('mode', choices=['train', 'eval'], help='Mode to run the script in')
    parser.add_argument('--run-name', type=str, required=True, help='Name of the training/evaluation run')
    parser.add_argument('--timesteps', type=int, default=5_000_000, help='Total timesteps for training')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the environment during training/evaluation')
    return parser.parse_args()

def setup_logging(run_name: str):
    """Set up logging directory and configuration."""
    base_log_dir = os.path.join("logs", run_name)
    os.makedirs(base_log_dir, exist_ok=True)

    log_filename = os.path.join(base_log_dir, 'train_log.txt')
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    print('Starting session:', run_name)
    logging.info(f"Starting session: {run_name}")

class MetricsCallback(BaseCallback):
    def __init__(self, run_name, verbose=0):
        super().__init__(verbose)
        self.run_name = run_name
        self.rewards = []
        self.ep_lengths = []
        self.critic_losses = []
        self.wind_speeds = []
        self.checkpoint_count = 0
        self.best_mean_reward = -np.inf
        
    def _on_step(self):
        if self.n_calls % self.model.n_steps == 0:
            if len(self.model.ep_info_buffer) > 0:
                # Calculate metrics
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_ep_length = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                critic_loss = self.model.logger.name_to_value.get('train/value_loss', None)
                current_wind = self.training_env.get_attr('current_wind')[0]
                
                # Store metrics
                self.rewards.append(mean_reward)
                self.ep_lengths.append(mean_ep_length)
                self.critic_losses.append(critic_loss)
                self.wind_speeds.append(current_wind)
                
                # Log to file
                logging.info(f"Steps: {self.n_calls}")
                logging.info(f"Mean reward: {mean_reward:.2f}")
                logging.info(f"Mean episode length: {mean_ep_length:.2f}")
                logging.info(f"Critic loss: {critic_loss if critic_loss is not None else 'N/A'}")
                logging.info(f"Wind speed: {current_wind:.2f}")
                
                # Save checkpoint if we have a new best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.checkpoint_count += 1
                    checkpoint_dir = os.path.join("models", self.run_name, "checkpoints")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{self.checkpoint_count}")
                    self.model.save(checkpoint_path)
                    logging.info(f"Saved new best model checkpoint: {checkpoint_path}")
                    logging.info(f"New best mean reward: {self.best_mean_reward:.2f}")
                
                # Save current metrics
                self.save_metrics()
                
        return True

    def save_metrics(self):
        metrics_dir = os.path.join("logs", self.run_name, "training_metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        np.save(os.path.join(metrics_dir, "rewards.npy"), np.array(self.rewards))
        np.save(os.path.join(metrics_dir, "ep_lengths.npy"), np.array(self.ep_lengths))
        np.save(os.path.join(metrics_dir, "critic_losses.npy"), np.array(self.critic_losses))
        np.save(os.path.join(metrics_dir, "wind_speeds.npy"), np.array(self.wind_speeds))
        
        # Save evaluation rewards if they exist
        if hasattr(self.model, 'eval_rewards') and len(self.model.eval_rewards) > 0:
            np.save(os.path.join(metrics_dir, "eval_rewards.npy"), np.array(self.model.eval_rewards))
            np.save(os.path.join(metrics_dir, "eval_steps.npy"), np.array(self.model.eval_steps))

class RandomWindLander(Wrapper):
    """Wrapper for LunarLander that changes wind speed each episode."""
    def __init__(self, env, min_wind=0.0, max_wind=15.0):
        super().__init__(env)
        self.min_wind = min_wind
        self.max_wind = max_wind
        self.current_wind = None
        
    def reset(self, **kwargs):
        # Sample new wind speed for this episode
        self.current_wind = np.random.uniform(self.min_wind, self.max_wind)
        self.env.unwrapped.wind_power = self.current_wind
        
        # Log the new wind speed
        logging.info(f"New episode starting with wind speed: {self.current_wind:.2f}")
        
        return self.env.reset(**kwargs)

def create_env(render_mode=None):
    """Create the Lunar Lander environment with random wind."""
    env_kwargs = {
        "render_mode": render_mode,
        "wind_power": 7.0  # Initial wind power (will be overridden)
    }
    base_env = gym.make("LunarLander-v2", **env_kwargs)
    return RandomWindLander(base_env, min_wind=0.0, max_wind=15.0)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Action log standard deviations (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        # Returns action mean and value
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value
    
    def evaluate_actions(self, state, action):
        action_mean, value = self(state)
        action_std = torch.exp(self.log_std)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value
    
    def get_action(self, state, deterministic=False):
        action_mean, value = self(state)
        
        if deterministic:
            return action_mean, value
        
        action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        
        return action, value

class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
        batch_size=64,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            
        returns = advantages + values
        return advantages, returns
    
    def train_on_batch(self, states, actions, old_values, old_log_probs, advantages, returns):
        for _ in range(self.n_epochs):
            # Generate random indices for minibatches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get minibatch
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                old_values_batch = old_values[batch_indices]
                old_log_probs_batch = old_log_probs[batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                
                # Evaluate actions
                log_probs, entropy, values = self.policy.evaluate_actions(state_batch, action_batch)
                
                # Calculate policy loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Calculate value loss
                value_pred = values.squeeze()
                value_loss = 0.5 * ((value_pred - returns_batch) ** 2).mean()
                
                # Calculate entropy loss
                entropy_loss = -entropy.mean()
                
                # Calculate total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store losses
                self.value_losses.append(value_loss.item())
                self.policy_losses.append(policy_loss.item())
                self.entropy_losses.append(entropy_loss.item())
    
    def learn(self, env, total_timesteps, callback=None, eval_env=None, eval_freq=10, eval_episodes=5):
        """Train the model and optionally evaluate it periodically."""
        num_steps = 2000  # Steps per update
        states = torch.zeros((num_steps,) + env.observation_space.shape).to(self.device)
        actions = torch.zeros((num_steps,) + env.action_space.shape).to(self.device)
        rewards = torch.zeros(num_steps).to(self.device)
        dones = torch.zeros(num_steps).to(self.device)
        values = torch.zeros(num_steps).to(self.device)
        log_probs = torch.zeros(num_steps).to(self.device)
        
        # Track evaluation rewards
        self.eval_rewards = []
        self.eval_steps = []
        
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(self.device)
        
        for update in range(total_timesteps // num_steps):
            for step in range(num_steps):
                with torch.no_grad():
                    action, value = self.policy.get_action(state)
                    log_prob, _, _ = self.policy.evaluate_actions(state.unsqueeze(0), action.unsqueeze(0))
                
                next_state, reward, done, _, _ = env.step(action.cpu().numpy())
                
                states[step] = state
                actions[step] = action
                rewards[step] = reward
                dones[step] = done
                values[step] = value
                log_probs[step] = log_prob
                
                state = torch.FloatTensor(next_state).to(self.device)
                
                if done:
                    state, _ = env.reset()
                    state = torch.FloatTensor(state).to(self.device)
            
            # Calculate advantages and returns
            with torch.no_grad():
                next_value = self.policy.get_action(state)[1]
                advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Train on collected data
            self.train_on_batch(states, actions, values, log_probs, advantages, returns)
            
            if callback is not None:
                callback._on_step()
            
            # Evaluate if it's time and we have an evaluation environment
            if eval_env is not None and update > 0 and update % eval_freq == 0:
                mean_reward, std_reward = evaluate_model(self, eval_env, "current", num_episodes=eval_episodes)
                print(f"\nUpdate {update} evaluation - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
                
                # Store evaluation results
                self.eval_rewards.append(mean_reward)
                self.eval_steps.append(update * num_steps)
    
    def predict(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, _ = self.policy.get_action(state, deterministic)
        return action.cpu().numpy(), None
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path + ".pt")
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path + ".pt"))

def train_model(env, run_name, total_timesteps):
    """Initialize and train the PPO model."""
    # Create evaluation environment with different wind range
    eval_env = create_env()
    # eval_env = RandomWindLander(eval_env.unwrapped, min_wind=0.0, max_wind=15.0)  # Explicit wind range
    
    model = PPO(
        env.observation_space.shape[0],
        env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
        batch_size=64,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    metrics_callback = MetricsCallback(run_name)
    
    try:
        print("Training the model...")
        model.learn(
            env, 
            total_timesteps=total_timesteps, 
            callback=metrics_callback,
            eval_env=eval_env,
            eval_freq=10,  # Evaluate every 10 updates
            eval_episodes=5  # Use 5 episodes for each evaluation
        )
        print("Training completed.")
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        metrics_callback.save_metrics()
        eval_env.close()
        
        # Save final model
        model_path = os.path.join("models", run_name, "final_model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    return model

def evaluate_model(model, env, run_name, num_episodes=10):
    """Evaluate the trained model and log results."""
    episode_rewards = []
    episode_winds = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        # Get initial wind speed for this episode
        current_wind = env.current_wind
        episode_winds.append(current_wind)
        print(f"Episode {episode + 1} - Wind Speed: {current_wind:.2f}")
        logging.info(f"Episode {episode + 1} - Wind Speed: {current_wind:.2f}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1} - Total Reward: {episode_reward:.2f}")
        logging.info(f"Episode {episode + 1} - Total Reward: {episode_reward:.2f}")
        logging.info(f"Episode {episode + 1} - Wind/Reward: {current_wind:.2f}/{episode_reward:.2f}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_wind = np.mean(episode_winds)
    std_wind = np.std(episode_winds)
    
    # Log summary results
    logging.info("\nEvaluation Summary:")
    logging.info(f"Mean evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logging.info(f"Mean wind speed: {mean_wind:.2f} ± {std_wind:.2f}")
    print(f"\nEvaluation Summary:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean wind speed: {mean_wind:.2f} ± {std_wind:.2f}")
    
    # Save evaluation results
    eval_dir = os.path.join("logs", run_name, "evaluation_metrics")
    os.makedirs(eval_dir, exist_ok=True)
    np.save(
        os.path.join(eval_dir, "eval_results.npy"),
        {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "episode_rewards": episode_rewards,
            "episode_winds": episode_winds,
            "mean_wind": mean_wind,
            "std_wind": std_wind,
            "wind_reward_pairs": list(zip(episode_winds, episode_rewards))
        }
    )
    
    return mean_reward, std_reward

def render_agent(model, env, run_name, num_episodes=10):
    """Render and record agent's performance."""
    eval_log_dir = os.path.join("logs", run_name, "eval_plots")
    os.makedirs(eval_log_dir, exist_ok=True)
    
    for episode in range(num_episodes):
        rewards = []
        obs, _ = env.reset()
        done = False
        
        # Get and log wind speed for this episode
        current_wind = env.current_wind
        print(f"\nRendering Episode {episode + 1} - Wind Speed: {current_wind:.2f}")
        logging.info(f"Rendering Episode {episode + 1} - Wind Speed: {current_wind:.2f}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            env.render()
            
        # Plot episode rewards
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, marker="o", linestyle="-", color="b")
        plt.title(f"Episode {episode + 1} Rewards (Wind: {current_wind:.2f})")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(eval_log_dir, f"episode_{episode + 1}.png"), dpi=300)
        plt.close()
        
        # Log episode statistics
        total_reward = sum(rewards)
        print(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}")
        logging.info(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}")
        logging.info(f"Episode length: {len(rewards)}")

def main():
    args = parse_args()
    setup_logging(args.run_name)
    
    if args.render:
        render_mode = "human"
    else:
        render_mode = "rgb_array"
            
    if args.mode == 'train':
        # Create training environment
        env = create_env()
        
        # Train the model
        model = train_model(env, args.run_name, total_timesteps=args.timesteps)
        
        
        # Create evaluation environment
        eval_env = create_env(render_mode=render_mode)
        
        # Evaluate the model
        evaluate_model(model, eval_env, args.run_name, num_episodes=args.eval_episodes)
        
        # Render agent's performance
        render_agent(model, eval_env, args.run_name, num_episodes=args.eval_episodes)
        
        # Clean up
        env.close()
        eval_env.close()

    else:  # eval mode
        # Load the model
        model_path = os.path.join("models", args.run_name, "final_model")
        if not os.path.exists(model_path + ".pt"):
            raise FileNotFoundError(f"No model found at {model_path}")
        
        # Create evaluation environment
        eval_env = create_env(render_mode=render_mode)
        
        # Load and evaluate the model
        model = PPO(
            eval_env.observation_space.shape[0],
            eval_env.action_space.n,
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=10,
            batch_size=64,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        model.load(model_path)
        logging.info(f"Loaded model from {model_path}")
        
        # Evaluate the model
        evaluate_model(model, eval_env, args.run_name, num_episodes=args.eval_episodes)
        
        # Render agent's performance
        render_agent(model, eval_env, args.run_name, num_episodes=args.eval_episodes)
        
        # Clean up
        eval_env.close()

if __name__ == "__main__":
    main()

