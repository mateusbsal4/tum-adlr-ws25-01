import os
import time
import imageio
from render_browser import render_browser

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
#from stable_baselines3.common.env_util import make_vec_env


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plot_path = os.path.join(log_folder, "learning_curve.png")  # Save path
    plt.savefig(plot_path)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory


@render_browser
def test_policy():
    # Create log dir
    #log_dir = "monitor_logs/"
    #os.makedirs(log_dir, exist_ok=True)
    
    # Create and wrap the environment
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    #env = Monitor(env, log_dir)
    
    # Train the agent
    #model = PPO("MlpPolicy", env, verbose=1)
    #time_steps = 5000000
    #model.learn(total_timesteps=time_steps, progress_bar=True)
    #model.save("ppo_lunar")
    
    # Reload the trained model
    model = PPO.load("bestmodel")
    
    # Reset the environment and generate a GIF
    obs, info = env.reset()
    images = []  # To store frames for the GIF
    
    episode_over = False
    while not episode_over:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        # Append the rendered frame to the list
        img = env.render()
        images.append(img)
        time.sleep(0.02)
    
    env.close()
    
    # Save the GIF
    gif_path = "lander_policy.gif"
    imageio.mimsave(
        gif_path,
        [np.array(img) for i, img in enumerate(images) if i % 2 == 0],  # Skip every other frame for efficiency
        fps=25,
    )
    print(f"GIF saved at {gif_path}")

# Run the policy and serve it in a browser
test_policy()


"""  Eval Callback version    """
"""
	import os
	import time


	import gymnasium as gym
	import numpy as np
	import matplotlib.pyplot as plt
	from render_browser import render_browser

	from stable_baselines3 import PPO
	from stable_baselines3.common.callbacks import ProgressBarCallback, EvalCallback
	from stable_baselines3.common.monitor import Monitor
	from stable_baselines3.common.results_plotter import load_results, ts2xy
	#from stable_baselines3.common.env_util import make_vec_env

	def moving_average(values, window):

	    Smooth values by doing a moving average
	    :param values: (numpy array)
	    :param window: (int)
	    :return: (numpy array)

	    weights = np.repeat(1.0, window) / window
	    return np.convolve(values, weights, "valid")


	def plot_results(log_folder, title="Learning Curve"):

	    plot the results

	    :param log_folder: (str) the save location of the results to plot
	    :param title: (str) the title of the task to plot

	    x, y = ts2xy(load_results(log_folder), "timesteps")
	    y = moving_average(y, window=50)
	    # Truncate x
	    x = x[len(x) - len(y) :]

	    fig = plt.figure(title)
	    plt.plot(x, y)
	    plt.xlabel("Number of Timesteps")
	    plt.ylabel("Rewards")
	    plt.title(title + " Smoothed")
	    plot_path = os.path.join(log_folder, "learning_curve.png")  # Save path
	    plt.savefig(plot_path)  # Save the plot to a file
	    plt.close()  # Close the plot to free up memory


	@render_browser
	def test_policy():
	    log_dir = "monitor_logs/"
	    os.makedirs(log_dir, exist_ok=True)
	    #env = gym.make("LunarLander-v3", render_mode = "rgb_array", enable_wind = True)
	    env = gym.make("LunarLander-v3", render_mode = "rgb_array")
	    env = Monitor(env, filename = "monitor_logs/")
	    eval_callback = EvalCallback(env, best_model_save_path=log_dir,
		                         log_path=log_dir, eval_freq=100, n_eval_episodes =5,
		                         deterministic=True, render=False)
	    # Load or train the model
	    model = PPO("MlpPolicy", env, verbose=1)
	    time_steps = 5000
	    model.learn(total_timesteps=time_steps, progress_bar = True, callback = eval_callback)   
	    model.save("ppo_lunar")

	    # Reload the trained model

	    model = PPO.load("ppo_lunar")
	    #results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO Lunar Lander")
	    plot_results(log_dir)
	    #plot_path = os.path.join(log_dir, "training_plot.png")  # Save path
	    #plt.savefig(plot_path)  # Save the plot to a file
	    #plt.close()  # Close the plot to free up memory
	    #load_results(log_dir)
	    # Reset the environment and run the policy
	    obs, info = env.reset() 
	    episode_over = False
	    while not episode_over:
		action, _states = model.predict(obs)        #ppo policy
		obs, reward, terminated, truncated, info = env.step(action)
		episode_over = terminated or truncated
		#print("X coord: ", obs[0])
		# Yield the rendered frame as a NumPy array
		yield env.render()
		time.sleep(0.02)
	    env.close()

	# Run the policy and serve it in a browser
	test_policy()

"""