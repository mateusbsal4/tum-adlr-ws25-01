import argparse
import os
import logging
import numpy as np
from datetime import datetime
from mpi4py import MPI
import matplotlib.pyplot as plt

from context_aware_ppo import ContextAwarePPO

def setup_logger(log_dir, rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger()
    return None

def plot_rewards(training_rewards, eval_rewards, eval_interval, episodes_per_update, run_dir):
    if not training_rewards:
        return None
        
    plt.figure(figsize=(12, 6))
    
    # Plot training rewards
    # Each training reward represents episodes_per_update episodes
    train_episodes = np.arange(episodes_per_update, len(training_rewards) * episodes_per_update + 1, episodes_per_update)
    plt.plot(train_episodes, training_rewards, label='Training Rewards', alpha=0.6)
    
    # Plot evaluation rewards
    # Evaluations happen every eval_interval training updates, each update is episodes_per_update episodes
    if eval_rewards:
        eval_episodes = np.arange(eval_interval, 
                                len(eval_rewards) * eval_interval + 1, 
                                eval_interval)
        plt.plot(eval_episodes, eval_rewards, label='Evaluation Rewards', linewidth=2)
    
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Training and Evaluation Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(run_dir, 'rewards_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def find_latest_model(run_dir):
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        return None
        
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not model_files:
        return None
        
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, latest_model)

def evaluate_saved_model(model_path, num_episodes=10, render=True):
    print(f"Loading model from {model_path}")
    agent = ContextAwarePPO.load_model(model_path, render_mode="human" if render else None)
    mean_reward = agent.evaluate(num_episodes=num_episodes)
    print(f"Average reward over {num_episodes} episodes: {mean_reward:.2f}")
    return mean_reward

def main():
    parser = argparse.ArgumentParser(description='Train Context-Aware PPO on LunarLander')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train',
                        help='Whether to train a new model or evaluate a saved one')
    parser.add_argument('--run-name', type=str, required=True,
                        help='Name for the training run directory')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the environment during training/evaluation')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension for networks')
    parser.add_argument('--context-dim', type=int, default=8,
                        help='Dimension of the context vector')
    parser.add_argument('--context-kl-weight', type=float, default=0.1,
                        help='Weight for the KL divergence loss of the context encoder')
    
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
    
    # Create agent
    if latest_model:
        agent = ContextAwarePPO.load_model(
            latest_model,
            render_mode="human" if args.render else None
        )
    else:
        agent = ContextAwarePPO(
            hidden_dim=args.hidden_dim,
            context_dim=args.context_dim,
            context_kl_weight=args.context_kl_weight,
            render_mode="human" if args.render else None,
            run_name=args.run_name
        )
    
    # Training parameters
    num_episodes = 1_000_000
    eval_interval = 50
    num_episodes_per_update = 10
    
    # Setup logging
    logger = setup_logger(agent.log_dir, rank) if rank == 0 else None
    
    # Training loop
    training_rewards = []
    eval_rewards = []
    best_eval_reward = float('-inf')
    
    try:
        for global_episode in range(0, num_episodes, num_episodes_per_update):
            # Training update
            avg_reward, avg_length, actor_loss, critic_loss, context_loss = agent.train_episode(
                num_episodes_per_update=num_episodes_per_update
            )
            
            if rank == 0:  # Only rank 0 should log and save
                training_rewards.append(avg_reward)
                
                logger.info(f"Global Episode {global_episode}:")
                logger.info(f"  Average episode length: {avg_length:.1f}")
                logger.info(f"  Average training reward: {avg_reward:.2f}")
                logger.info(f"  Actor loss: {actor_loss:.4f}")
                logger.info(f"  Critic loss: {critic_loss:.4f}")
                logger.info(f"  Context loss: {context_loss:.4f}")
            
            # Evaluation
            if global_episode % eval_interval == 0 and rank == 0:
                eval_reward = agent.evaluate()
                if rank == 0:
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
        if rank == 0:
            logger.info("\nTraining interrupted by user!")
            model_path = agent.save_model("_interrupted")
            logger.info(f"Interrupted model saved: {model_path}")
    
    finally:
        if rank == 0:
            model_path = agent.save_model("_final")
            logger.info(f"Final model saved: {model_path}")
            
            plot_path = plot_rewards(training_rewards, eval_rewards, eval_interval, 
                                   num_episodes_per_update, agent.run_dir)
            logger.info(f"Training completed! Results plot saved as '{plot_path}'")

if __name__ == "__main__":
    main() 