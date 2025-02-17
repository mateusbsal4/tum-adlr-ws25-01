import gymnasium as gym
from stable_baselines3 import PPO
from lib.pol_eval import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import datetime
from gymnasium import Wrapper
import argparse

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

def train_model(env, run_name, total_timesteps):
    """Initialize and train the PPO model."""
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2000,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        vf_coef=0.5,
        clip_range=0.2,
        seed=42
    )
    
    metrics_callback = MetricsCallback(run_name)
    
    try:
        print("Training the model...")
        model.learn(total_timesteps=total_timesteps, callback=metrics_callback)
        print("Training completed.")
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        metrics_callback.save_metrics()
        
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
            "std_wind": std_wind
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
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"No model found at {model_path}")
        
        # Create evaluation environment
        eval_env = create_env(render_mode=render_mode)
        
        # Load and evaluate the model
        model = PPO.load(model_path, env=eval_env)
        logging.info(f"Loaded model from {model_path}")
        
        # Evaluate the model
        evaluate_model(model, eval_env, args.run_name, num_episodes=args.eval_episodes)
        
        # Render agent's performance
        render_agent(model, eval_env, args.run_name, num_episodes=args.eval_episodes)
        
        # Clean up
        eval_env.close()

if __name__ == "__main__":
    main()

