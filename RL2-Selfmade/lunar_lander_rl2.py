import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque
import os
from datetime import datetime
import logging
import argparse
import torch.nn.utils as nn_utils
from mpi4py import MPI
from RL2PPO import RL2PPO
from RL2PPO import mpi_avg
import re

def plot_rewards(training_rewards, eval_rewards, eval_interval, num_episodes_per_update, save_dir):
    if len(training_rewards) == 0 or len(eval_rewards) == 0:
        return None  # Don't create plot if no data
        
    plt.figure(figsize=(12, 6))
    
    # Plot training rewards
    training_episodes = [i * num_episodes_per_update for i in range(len(training_rewards))]
    plt.plot(training_episodes, training_rewards, 
             label='Training Rewards', alpha=0.6)
    
    # Calculate evaluation episodes correctly
    eval_episodes = []
    current_episode = 0
    for _ in range(len(eval_rewards)):
        eval_episodes.append(current_episode)
        current_episode += eval_interval
    
    # Ensure eval_episodes and eval_rewards have the same length
    eval_episodes = eval_episodes[:len(eval_rewards)]
    eval_rewards = eval_rewards[:len(eval_episodes)]
    
    plt.plot(eval_episodes, eval_rewards, 
             label='Evaluation Rewards', color='red', marker='o')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training and Evaluation Rewards')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(save_dir, 'rewards_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def setup_logger(log_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger('RL2PPO')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def find_latest_model(run_dir):
    """Find the latest model file by step number in the run directory."""
    model_files = []
    
    # Look in all relevant directories
    for subdir in ['', 'checkpoints', 'best_models']:
        dir_path = os.path.join(run_dir, subdir)
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith('.pth'):
                    full_path = os.path.join(dir_path, file)
                    # Extract step number from filename
                    step_match = re.search(r'step(\d+)', file)
                    if step_match:
                        step_num = int(step_match.group(1))
                        model_files.append((step_num, full_path))
    
    if not model_files:
        return None
        
    # Return the file with the highest step number
    return max(model_files, key=lambda x: x[0])[1]

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate RL2-PPO on LunarLander')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train',
                        help='Whether to train a new model or evaluate a saved one')
    parser.add_argument('--run-name', type=str, required=True,
                       help='Name for the training run directory')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the environment during training/evaluation')
    
    args = parser.parse_args()
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check if run directory exists and has models
    run_dir = os.path.join('models', args.run_name)
    
    if args.mode == 'evaluate':
        if os.path.exists(run_dir):
            latest_model = find_latest_model(run_dir)
            if latest_model:
                if rank == 0:
                    print(f"Evaluating the latest model: {latest_model}")
                    evaluate_saved_model(latest_model, render=args.render)
                return
            else:
                if rank == 0:
                    print("No model found in the specified run directory.")
                return
        else:
            if rank == 0:
                print("Specified run directory does not exist.")
            return
    
    latest_model = None
    if os.path.exists(run_dir):
        latest_model = find_latest_model(run_dir)
        if latest_model and rank == 0:
            print(f"Found existing run directory with models.")
            print(f"Latest model: {latest_model}")
            response = input("Resume training from this model? [Y/n]: ").lower()
            if response in ['', 'y', 'yes']:
                if rank == 0:
                    print(f"Resuming from {latest_model}")
            else:
                if rank == 0:
                    print("Starting fresh training run")
                latest_model = None
    
    env_name = "CustomLunarLander-v2"
    num_episodes = 10000
    eval_interval = 50
    num_episodes_per_update = 10
    num_checkpoints_to_keep = 5
    
    # Only render on rank 0
    render_mode = "human" if args.render and rank == 0 else None
    
    if latest_model:
        agent = RL2PPO.load_model(latest_model, render_mode=render_mode)
        if rank == 0:
            logger = setup_logger(agent.log_dir)
            logger.info(f"Resuming training from {latest_model}")
            # Get existing rewards history from loaded agent
            training_rewards = agent.training_rewards
            eval_rewards = agent.eval_rewards
            best_eval_reward = agent.best_eval_reward
    else:
        agent = RL2PPO(env_name, render_mode=render_mode, run_name=args.run_name)
        if rank == 0:
            logger = setup_logger(agent.log_dir)
            # Initialize empty rewards history for new run
            training_rewards = []
            eval_rewards = []
            best_eval_reward = float('-inf')
    
    reward_window = deque(maxlen=100 // num_episodes_per_update)
    
    if rank == 0:
        logger.info(f"Starting training with {size} workers...")
        logger.info("Initialization parameters:")
        logger.info(f"  Environment: {env_name}")
        logger.info(f"  Hidden dimension: {agent.hidden_dim}")
        logger.info(f"  Learning rate: {agent.optimizer.param_groups[0]['lr']}")
        logger.info(f"  Gamma (discount): {agent.gamma}")
        logger.info(f"  Epsilon (clip): {agent.epsilon}")
        logger.info(f"  PPO epochs: {agent.epochs}")
        logger.info(f"  Batch size: {agent.batch_size}")
        logger.info(f"  Run directory: {agent.run_dir}")
        logger.info("\nTraining parameters:")
        logger.info(f"  Total episodes: {num_episodes}")
        logger.info(f"  Starting from episode: {agent.total_episodes}")
        logger.info(f"  Episodes remaining: {num_episodes - agent.total_episodes}")
        logger.info(f"  Evaluation interval: {eval_interval}")
        logger.info(f"  Episodes per update: {num_episodes_per_update}")
        logger.info(f"  Checkpoints to keep: {num_checkpoints_to_keep}")
        logger.info(f"  Rendering: {args.render}")
        logger.info("-" * 50)
    
    if not latest_model:
        eval_reward = agent.evaluate()
        if rank == 0:  # Only rank 0 should append and log
            logger.info(f"Initial evaluation reward: {eval_reward:.2f}")
            logger.info("-" * 50)
    
    global_episode = 0
    try:
        while global_episode < num_episodes:  # Each worker runs its share
            avg_reward, avg_length, actor_loss, critic_loss = agent.train_episode(num_episodes_per_update)
            
            # Synchronize rewards, metrics, and episode count across workers
            avg_reward = float(mpi_avg(avg_reward))
            avg_length = float(mpi_avg(avg_length))
            actor_loss = float(mpi_avg(actor_loss))
            critic_loss = float(mpi_avg(critic_loss))
            
            # Sum up episodes across all workers
            global_episode = comm.allreduce(agent.total_episodes, op=MPI.SUM)
            
            if rank == 0:  # Only rank 0 should log
                training_rewards.append(avg_reward)
                reward_window.append(avg_reward)
            
                agent.training_rewards = training_rewards
                agent.eval_rewards = eval_rewards
                agent.best_eval_reward = best_eval_reward
            
                logger.info(f"Global Episode {global_episode}:")
                logger.info(f"  Average episode length: {avg_length:.1f}")
                logger.info(f"  Average training reward: {avg_reward:.2f}")
                logger.info(f"  Actor loss: {actor_loss:.4f}")
                logger.info(f"  Critic loss: {critic_loss:.4f}")
            
            # Use actual global episode count for evaluation and checkpoints
            if global_episode % eval_interval == 0 and rank == 0:
                eval_reward = agent.evaluate()
                if rank == 0:  # Only rank 0 should append and log
                    eval_rewards.append(eval_reward)
                    logger.info(f"  Evaluation reward: {eval_reward:.2f}")
                    logger.info("-" * 50)
                
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        model_path = agent.save_model("_best")
                        logger.info(f"  New best model saved: {model_path}")
                
                    plot_path = plot_rewards(training_rewards, eval_rewards, eval_interval, 
                                          num_episodes_per_update, agent.run_dir)
                    logger.info(f"  Updated rewards plot saved as '{plot_path}'")
            
            if global_episode % 200 == 0 and rank == 0:
                model_path = agent.save_model("_checkpoint")
                logger.info(f"  Checkpoint saved: {model_path}")
            
            agent.total_episodes += num_episodes_per_update
            
    except KeyboardInterrupt:
        if rank == 0:  # Only rank 0 should handle interruption
            logger.info("\nTraining interrupted by user!")
            model_path = agent.save_model("_interrupted")
            logger.info(f"Interrupted model saved: {model_path}")
    
    finally:
        if rank == 0:  # Only rank 0 should handle cleanup
            model_path = agent.save_model("_final")
            logger.info(f"Final model saved: {model_path}")
            
            plot_path = plot_rewards(training_rewards, eval_rewards, eval_interval, 
                                   num_episodes_per_update, agent.run_dir)
            logger.info(f"Training completed! Results plot saved as '{plot_path}'")

def evaluate_saved_model(model_path, num_episodes=10, render=True, max_steps=1000):
    print(f"Loading model from {model_path}")
    agent = RL2PPO.load_model(model_path)
    return agent.evaluate(num_episodes=num_episodes, render=render, max_steps=max_steps)

if __name__ == "__main__":
    main()