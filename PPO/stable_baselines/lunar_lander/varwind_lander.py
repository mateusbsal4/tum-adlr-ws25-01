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

# Get run name and set up directories
RUN_NAME = input("Enter the run name: ")
base_log_dir = os.path.join("logs", RUN_NAME)
os.makedirs(base_log_dir, exist_ok=True)

# Set up logging
log_filename = os.path.join(base_log_dir, 'train_log.txt')
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
print('Starting training session:', RUN_NAME)
logging.info(f"Starting training session: {RUN_NAME}")

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
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=num_episodes, 
        return_episode_rewards=True
    )
    
    # Log evaluation results
    logging.info("Final Evaluation Results:")
    logging.info(f"Mean evaluation reward: {mean_reward:.2f}")
    logging.info(f"Std evaluation reward: {std_reward:.2f}")
    
    # Save evaluation results
    eval_dir = os.path.join("logs", run_name, "evaluation_metrics")
    os.makedirs(eval_dir, exist_ok=True)
    np.save(
        os.path.join(eval_dir, "eval_results.npy"),
        {"mean_reward": mean_reward, "std_reward": std_reward}
    )
    
    print(f"Final evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

def render_agent(model, env, run_name, num_episodes=10):
    """Render and record agent's performance."""
    eval_log_dir = os.path.join("logs", run_name, "eval_plots")
    os.makedirs(eval_log_dir, exist_ok=True)
    
    for episode in range(num_episodes):
        rewards = []
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            env.render()
            
        # Plot episode rewards
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, marker="o", linestyle="-", color="b")
        plt.title(f"Episode {episode + 1} Rewards")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(eval_log_dir, f"episode_{episode + 1}.png"), dpi=300)
        plt.close()
        
        # Log episode statistics
        logging.info(f"Evaluation episode {episode + 1}:")
        logging.info(f"Total reward: {sum(rewards):.2f}")
        logging.info(f"Episode length: {len(rewards)}")

def main():
    # Create training environment
    env = create_env()
    
    # Train the model
    model = train_model(env, RUN_NAME, total_timesteps=5_000_000_000)
    
    # Create evaluation environment with rendering
    eval_env = create_env(render_mode="rbg_array")
    
    # Evaluate the model
    evaluate_model(model, eval_env, RUN_NAME)
    
    # Render agent's performance
    render_agent(model, eval_env, RUN_NAME)
    
    # Clean up
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()

