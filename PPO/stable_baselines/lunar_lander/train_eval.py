import argparse
import os
from time import sleep
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from lib.VarPosLander import LunarLanderTargetPos
from lib.reward_plotter import SaveOnBestTrainingRewardCallback

try:
    from render_browser import render_browser
except ImportError:
    render_browser = None





# Register custom environment
register(
    id="LunarLanderTargetPos",
    entry_point=LunarLanderTargetPos,
)

def train(env_id, total_timesteps, log_rewards, model_dir, log_dir):
    """Train the PPO agent on the specified environment."""
    # create environment
    env = gym.make(env_id, render_mode=None, target_x=5, target_y=5)
    
    # reward logging
    timepoint = datetime.now().strftime("%Y%m%d_%H%M%S")     # Append date and time to the model save path and reward log
    model_name = f"model_{timepoint}"
    
    if log_rewards:
        log_dir = os.path.join(log_dir, "train", model_name)
        os.makedirs(log_dir, exist_ok=True)
        # if you want the reward plot, add --log_rewards as argument when you call file
        env = Monitor(env, log_dir)
        # Create the callback: check every 1000 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        
    # create rl model
    model = PPO("MlpPolicy", env, verbose=1)
    if log_rewards:
        model.learn(total_timesteps=total_timesteps, progress_bar = True, callback=callback)
    else: 
        model.learn(total_timesteps=total_timesteps, progress_bar = True)
        
    # save trained model
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, model_name)
    model.save(model_save_path)
    print(f"Trained Model successfully saved to {model_save_path}")
    
    #reward plot
    if log_rewards:
        plot_results(log_dir, model_name)
    env.close()


def evaluate(env_id, model_path, render_mode):
    """Evaluate a trained PPO agent in the specified environment."""
    if render_mode == "browser" and render_browser is None:
        raise ImportError("render_browser is not installed or available.")

    env_kwargs = {"target_x": 5, "target_y": 5}
    if render_mode == "human":
        env_kwargs["render_mode"] = "human"
    elif render_mode == "rgb_array":
        env_kwargs["render_mode"] = "rgb_array"

    # Ensure the latest model is loaded if model_path is default
    if model_path == "models":
        if not os.path.exists(model_path) or not os.listdir(model_path):
            raise FileNotFoundError("No models found in the 'models' directory.")
        model_path = max(
            [os.path.join(model_path, f) for f in os.listdir(model_path)],
            key=os.path.getctime
        )
        print(f"Using the latest model: {model_path}")
    
    env = gym.make(env_id, **env_kwargs)
    model = PPO.load(model_path)

    obs, info = env.reset()
    episode_over = False
    rewards = []
    total_reward = 0

    # @render_browser
    def run_episode():
        nonlocal obs, episode_over, total_reward
        while not episode_over:
            if render_mode == "human":
                env.render()
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            rewards.append(total_reward)
            episode_over = terminated or truncated
            sleep(0.02)

        env.close()
        print(f"Evaluation completed. Total reward: {total_reward}")




def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, model_name):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    title = f"{model_name} Learning Curve"
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plot_path = os.path.join(log_folder, f"curve_{model_name}.png")
    plt.savefig(plot_path)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a PPO agent on the Lunar Lander environment.")
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="Mode to run: train or eval.")
    parser.add_argument("--render_mode", choices=["human", "rgb_array", "browser"], default="human", help="Rendering mode: human, rgb_array, or browser.")
    parser.add_argument("--env_id", type=str, default="LunarLanderTargetPos", help="Gym environment ID.")
    parser.add_argument("--model_path", type=str, default="models", help="Path to save or load the model.")
    parser.add_argument("--log_path", type=str, default="reward_log", help="Path to save and plot the reward values.")
    parser.add_argument("--total_timesteps", type=int, default=100_000, help="Total timesteps for training.")
    parser.add_argument("--log_rewards", action="store_true", help="Log and plot rewards during evaluation.")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.env_id, args.total_timesteps, args.log_rewards, args.model_path, args.log_path)
    elif args.mode == "eval":
        evaluate(args.env_id, args.model_path, args.render_mode, args.log_rewards)

if __name__ == "__main__":
    main()
