import gymnasium as gym
from stable_baselines3 import PPO
from lib.pol_eval import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
import os
import matplotlib.pyplot as plt

log_directory = os.path.join("logs")
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, 'eval_log.txt')
logging.basicConfig(
    filename=log_filename,  # Output file where logs will be saved
    level=logging.INFO,           # Log level (INFO, DEBUG, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    filemode='w')                 # 'w' for writing (overwrites existing file), 'a' for appending
print('start logging')
logging.info("start logging")


# Define the Lunar Lander environment
def create_env():
    return gym.make("LunarLander-v3")

# Create the vectorized environment
# env = DummyVecEnv([create_env])
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# # Initialize the PPO model
# model = PPO(
#     "MlpPolicy",  # Multi-Layer Perceptron Policy
#     env,           # The environment
#     verbose=1,     # Verbosity mode
#     learning_rate=3e-4,  # Learning rate for the optimizer
#     n_steps=2048,  # Number of steps per update
#     batch_size=64, # Batch size for updates
#     n_epochs=10,   # Number of epochs per update
#     gamma=0.99,    # Discount factor
#     gae_lambda=0.95, # GAE lambda for advantage estimation
#     ent_coef=0.0,    # Entropy coefficient
#     vf_coef=0.5,     # Value function coefficient
#     clip_range=0.2,  # Clipping parameter for PPO
#     seed=42         # Seed for reproducibility
# )

# # Train the model
# try:
#     print("Training the model...")
#     model.learn(total_timesteps=5_000_000)
#     print("Training completed.")
# except KeyboardInterrupt:
#     print("training interrupted.")

# # Save the model
model_path = "models/latestmodel_5mil"
# model.save(model_path)
# print(f"Model saved to {model_path}.")

# Load the model (optional, for demonstration purposes)
model = PPO.load(model_path, env=env)

# Evaluate the trained policy
def evaluate(model, env, num_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    return mean_reward, std_reward

print("Evaluating the model...")
evaluate(model, env)

# Visualize the agent's performance
def render_agent(model, env, num_episodes=2):
    rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            env.render()
        env.close()
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, marker="o", linestyle="-", color="b")
        plt.title("Reward vs. Timestep")
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.grid()

        # Save the figure to a PNG file
        plt.savefig(f"logs/plot{episode}.png", dpi=300)  # dpi=300 for high-resolution
        print(f"Plot saved as logs/plot{episode}.png")

        # Optionally show the plot (can be omitted if not needed)
        # plt.show()
    

print("Rendering the trained agent...")
render_agent(model, env)

