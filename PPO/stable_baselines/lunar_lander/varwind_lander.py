import gymnasium as gym
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Wrapper
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from custom_lunar_lander import CustomLunarLander  # Import the custom environment

def parse_args():
    parser = argparse.ArgumentParser(description='Train or evaluate a LunarLander agent with variable wind')
    parser.add_argument('mode', choices=['train', 'eval'], help='Mode to run the script in')
    parser.add_argument('--run-name', type=str, required=True, help='Name of the training/evaluation run')
    parser.add_argument('--timesteps', type=int, default=1_000_000_000, help='Total timesteps for training')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the environment during training/evaluation')
    parser.add_argument('--random-target', action='store_true',
                        help='Enable random landing target position')
    return parser.parse_args()

def setup_logging(run_name: str):
    """Set up logging directory and configuration."""
    base_log_dir = os.path.join("logs", run_name)
    os.makedirs(base_log_dir, exist_ok=True)

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    fh = logging.FileHandler(os.path.join(base_log_dir, 'train_log.txt'), mode='w')
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logging.info(f"Starting session: {run_name}")

def plot_training_progress(rewards, ep_lengths, wind_speeds, plots_dir):
    """Plot training metrics."""
    plt.figure(figsize=(12, 6))
    
    # Plot training rewards
    steps = np.arange(len(rewards)) * 2000  # num_steps = 2000
    plt.plot(steps, rewards, 'b-', alpha=0.3, label='Training (per update)')
    
    # Calculate and plot moving average of training rewards
    window = 50
    if len(rewards) >= window:
        ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ma_steps = steps[window-1:len(ma_rewards)+window-1]
        plt.plot(ma_steps, ma_rewards, 'b-', linewidth=2, 
                label=f'Training ({window}-update moving average)')
    
    plt.title('Training Progress')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(plots_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics(metrics_dir, rewards, ep_lengths, wind_speeds):
    """Save training metrics to files."""
    np.save(os.path.join(metrics_dir, "rewards.npy"), np.array(rewards))
    np.save(os.path.join(metrics_dir, "ep_lengths.npy"), np.array(ep_lengths))
    np.save(os.path.join(metrics_dir, "wind_speeds.npy"), np.array(wind_speeds))


def create_env(render_mode=None):
    """Create the custom Lunar Lander environment."""
    return CustomLunarLander(render_mode=render_mode)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Actor network (policy) - outputs logits for discrete actions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        # Returns action logits and value
        action_logits = self.actor(state)
        value = self.critic(state)
        return action_logits, value
    
    def evaluate_actions(self, state, action):
        action_logits, value = self(state)
        
        # Use categorical distribution for discrete actions
        dist = torch.distributions.Categorical(logits=action_logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy, value
    
    def get_action(self, state, deterministic=False):
        action_logits, value = self(state)
        
        if deterministic:
            # Choose action with highest probability
            action = torch.argmax(action_logits, dim=-1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(logits=action_logits)
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
        
        # Initialize callback
        if callback is not None:
            callback.init_callback(model=self, env=env)
        
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
                
                # Call callback with current locals and globals
                if callback is not None:
                    locals_ = {
                        'step': step,
                        'update': update,
                        'rewards': rewards,
                        'dones': dones
                    }
                    callback._on_step(locals_, globals())
                
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
            
            # Evaluate if it's time and we have an evaluation environment
            if eval_env is not None and update > 0 and update % eval_freq == 0:
                mean_reward, std_reward = evaluate_model(self, eval_env, "current", num_episodes=eval_episodes)
                print(f"\nUpdate {update} evaluation - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
                
                # Store evaluation results
                self.eval_rewards.append(mean_reward)
                self.eval_steps.append(update * num_steps)
        
        return self
    
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
    # Create evaluation environment
    eval_env = create_env()
    
    # Setup metrics tracking
    training_rewards = []  # Renamed to avoid confusion
    ep_lengths = []
    wind_speeds = []
    best_mean_reward = -np.inf
    checkpoint_count = 0
    
    # Setup directories
    metrics_dir = os.path.join("logs", run_name, "training_metrics")
    plots_dir = os.path.join("logs", run_name, "plots")
    checkpoint_dir = os.path.join("models", run_name, "checkpoints")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
    
    try:
        print("Training the model...")
        
        # Training loop
        num_steps = 2000  # Steps per update
        total_updates = total_timesteps // num_steps
        
        for update in range(total_updates):
            # Collect episode data
            episode_rewards = []
            episode_lengths = []
            current_wind = env.get_wind_power()
            current_target = env.target_x
            
            # Log current settings
            print(f"\nUpdate {update} - Wind: {current_wind:.2f}, Target: {current_target:.2f}")
            logging.info(f"\nUpdate {update} - Wind: {current_wind:.2f}, Target: {current_target:.2f}")
            
            # These are for collecting the current batch of transitions
            states = []
            actions = []
            rewards_buffer = []
            values = []
            log_probs = []
            dones = []
            
            state, _ = env.reset()
            episode_length = 0
            episode_reward = 0
            
            # Collect steps
            for step in range(num_steps):
                state_tensor = torch.FloatTensor(state).to(model.device)
                with torch.no_grad():
                    action, value = model.policy.get_action(state_tensor)
                    log_prob, _, _ = model.policy.evaluate_actions(state_tensor.unsqueeze(0), action.unsqueeze(0))
                
                next_state, reward, done, _, _ = env.step(action.cpu().numpy())
                
                states.append(state)
                actions.append(action)
                rewards_buffer.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                dones.append(done)
                
                episode_length += 1
                episode_reward += reward
                
                if done:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    state, _ = env.reset()
                    episode_length = 0
                    episode_reward = 0
                else:
                    state = next_state
            
            # Convert to tensors for training
            states_tensor = torch.FloatTensor(np.array(states)).to(model.device)
            actions_tensor = torch.stack(actions)
            rewards_tensor = torch.FloatTensor(rewards_buffer).to(model.device)
            values_tensor = torch.stack(values)
            log_probs_tensor = torch.stack(log_probs)
            dones_tensor = torch.FloatTensor(dones).to(model.device)
            
            # Update policy
            model.train_on_batch(states_tensor, actions_tensor, values_tensor, 
                               log_probs_tensor, rewards_tensor, dones_tensor)
            
            # Log metrics
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                mean_length = np.mean(episode_lengths)
                
                training_rewards.append(mean_reward)  # Using the list for metrics
                ep_lengths.append(mean_length)
                wind_speeds.append(current_wind)
                
                logging.info(f"Update: {update}")
                logging.info(f"Mean reward: {mean_reward:.2f}")
                logging.info(f"Mean episode length: {mean_length:.2f}")
                logging.info(f"Number of episodes: {len(episode_rewards)}")
                logging.info(f"Wind speed: {current_wind:.2f}")
                logging.info(f"Target position: {current_target:.2f}")
                
                # Save best model
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    checkpoint_count += 1
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_count}")
                    model.save(checkpoint_path)
                    logging.info(f"Saved new best model checkpoint: {checkpoint_path}")
                    logging.info(f"New best mean reward: {best_mean_reward:.2f}")
            
            # Evaluate periodically
            if update % 10 == 0:
                eval_reward, eval_std = evaluate_model(model, eval_env, run_name, num_episodes=5)
                logging.info(f"Evaluation - Mean reward: {eval_reward:.2f} ± {eval_std:.2f}")
            
            # Save and plot metrics
            plot_training_progress(training_rewards, ep_lengths, wind_speeds, plots_dir)
            save_metrics(metrics_dir, training_rewards, ep_lengths, wind_speeds)
            
        print("Training completed.")
        
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        # Save final model and metrics
        model_path = os.path.join("models", run_name, "final_model")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        eval_env.close()
    
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
        current_wind = env.get_wind_power()
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
        current_wind = env.get_wind_power()
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
    
    render_mode = "human" if args.render else None
            
    if args.mode == 'train':
        # Create base environment
        env = create_env(render_mode=render_mode)
        
        # Train the model
        model = train_model(
            env, 
            args.run_name, 
            total_timesteps=args.timesteps
        )
        
        # Create evaluation environment with rendering
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
        
        # Create evaluation environment with rendering
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

