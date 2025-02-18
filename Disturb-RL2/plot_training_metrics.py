import re
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def list_log_files(run_name):
    """List all log files in the run's log directory."""
    log_dir = os.path.join('models', run_name, 'logs')
    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory not found: {log_dir}")
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('training') and f.endswith('.log')]
    if not log_files:
        raise ValueError(f"No log files found in {log_dir}")
    
    return log_dir, sorted(log_files)

def select_log_file(log_files):
    """Let user select a log file from the list."""
    if len(log_files) == 0:
        raise ValueError("No log files found")
    elif len(log_files) == 1:
        return log_files[0]
    else: 
        print("\nAvailable log files:")
        for i, file in enumerate(log_files):
            print(f"[{i}] {file}")
    
        while True:
            try:
                choice = int(input("\nSelect a file number to plot: "))
                if 0 <= choice < len(log_files):
                    return log_files[choice]
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

def extract_metrics_from_log(log_file):
    episodes = []
    lengths = []
    rewards = []
    actor_losses = []
    critic_losses = []
    context_losses = []  # Added for context PPO
    eval_rewards = []    # Added to track evaluation rewards
    
    # Regular expressions to match the metrics
    episode_pattern = r"Global Episode (\d+):"
    length_pattern = r"Average episode length: ([-\d.]+)"
    reward_pattern = r"Average training reward: ([-\d.]+)"
    actor_loss_pattern = r"Actor loss: ([-\d.]+)"
    critic_loss_pattern = r"Critic loss: ([-\d.]+)"
    context_loss_pattern = r"Context loss: ([-\d.]+)"
    eval_reward_pattern = r"Evaluation reward: ([-\d.]+)"
    
    current_episode = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match episode number
            episode_match = re.search(episode_pattern, line)
            if episode_match:
                current_episode = int(episode_match.group(1))
                continue
            
            # Match episode length
            length_match = re.search(length_pattern, line)
            if length_match and current_episode is not None:
                lengths.append(float(length_match.group(1)))
                episodes.append(current_episode)
                continue
            
            # Match reward
            reward_match = re.search(reward_pattern, line)
            if reward_match and current_episode is not None:
                rewards.append(float(reward_match.group(1)))
                continue
            
            # Match actor loss
            actor_loss_match = re.search(actor_loss_pattern, line)
            if actor_loss_match and current_episode is not None:
                actor_losses.append(float(actor_loss_match.group(1)))
                continue
            
            # Match critic loss
            critic_loss_match = re.search(critic_loss_pattern, line)
            if critic_loss_match and current_episode is not None:
                critic_losses.append(float(critic_loss_match.group(1)))
                continue
            
            # Match context loss
            context_loss_match = re.search(context_loss_pattern, line)
            if context_loss_match and current_episode is not None:
                context_losses.append(float(context_loss_match.group(1)))
                continue
            
            # Match evaluation reward
            eval_reward_match = re.search(eval_reward_pattern, line)
            if eval_reward_match:
                eval_rewards.append(float(eval_reward_match.group(1)))
    
    return episodes, lengths, rewards, actor_losses, critic_losses, context_losses, eval_rewards

def plot_metrics(episodes, lengths, rewards, actor_losses, critic_losses, context_losses, eval_rewards, save_dir):
    mode = 'same'

    # Create figure with subplots for training metrics
    fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
    
    def calculate_moving_average(data, window):
        """Calculate moving average showing all points from start"""
        if len(data) < window:
            window = len(data)
            
        # Calculate cumulative sum
        cumsum = np.cumsum(np.insert(data, 0, 0))
        
        # Calculate moving average for all points
        ma = np.zeros(len(data))
        
        # For each point, use as many previous points as available up to window size
        for i in range(len(data)):
            start_idx = max(0, i + 1 - window)
            ma[i] = (cumsum[i + 1] - cumsum[start_idx]) / (i + 1 - start_idx)
            
        return ma
    
    window = 50  # Window size

    # Plot evaluation rewards
    if eval_rewards:
        # Calculate eval episodes (every 50 episodes)
        eval_episodes = np.arange(50, 50 * (len(eval_rewards) + 1), 50)
        ax1.plot(eval_episodes, eval_rewards, 'r-', label='Per Evaluation', alpha=0.3)
        
        # For evaluation rewards, use a smaller window if needed
        eval_window = min(window, len(eval_rewards))
        if len(eval_rewards) > 1:  # Only calculate moving average if we have more than one point
            ma = calculate_moving_average(eval_rewards, eval_window)
            ax1.plot(eval_episodes, ma, 'r-', linewidth=2, label=f'{eval_window}-episode moving average')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Evaluation Reward')
        ax1.set_title('Evaluation Rewards')
        ax1.grid(True)
        ax1.legend()
        
    
    # Plot training rewards
    ax2.plot(episodes, rewards, 'g-', alpha=0.3, label='Per Episode')
    ma = calculate_moving_average(rewards, window)
    print(max(ma))
    ax2.plot(episodes, ma, 'g-', linewidth=2, label=f'{window}-episode moving average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Training Reward')
    ax2.set_title('Training Rewards')
    ax2.grid(True)
    ax2.legend()
    
    # Plot actor losses
    ax3.plot(episodes, actor_losses, 'm-', alpha=0.3)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Actor Loss')
    ax3.set_title('Actor Loss')
    ax3.grid(True)
    
    # Plot moving average of actor losses
    ma = calculate_moving_average(actor_losses, window)
    ax3.plot(episodes, ma, 'm-', label=f'{window}-episode moving average')
    ax3.legend()
    
    # Plot critic losses
    ax4.plot(episodes, critic_losses, 'c-', alpha=0.3)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Critic Loss')
    ax4.set_title('Critic Loss')
    ax4.grid(True)
    
    # Plot moving average of critic losses
    ma = calculate_moving_average(critic_losses, window)
    ax4.plot(episodes, ma, 'c-', label=f'{window}-episode moving average')
    ax4.legend()
    
    # Plot context losses
    if context_losses:
        ax5.plot(episodes, context_losses, 'y-', alpha=0.3)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Context Loss')
        ax5.set_title('Context Loss')
        ax5.grid(True)
        
        # Plot moving average of context losses
        ma = calculate_moving_average(context_losses, window)
        ax5.plot(episodes, ma, 'y-', label=f'{window}-episode moving average')
        ax5.legend()
    
    # Plot episode lengths
    ax6.plot(episodes, lengths, 'b-', alpha=0.3)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Average Episode Length')
    ax6.set_title('Training Episode Lengths')
    ax6.grid(True)
    
    # Plot moving average of episode lengths
    ma = calculate_moving_average(lengths, window)
    ax6.plot(episodes, ma, 'b-', label=f'{window}-episode moving average')
    ax6.legend()
    
    
    plt.tight_layout()
    
    # Save training metrics plot
    metrics_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(metrics_path)
    plt.close()
    
    # Create and save reward distributions plot
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training rewards distribution
    ax1.hist(rewards, bins=50, alpha=0.7, color='g')
    ax1.axvline(np.mean(rewards), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
    ax1.axvline(np.median(rewards), color='b', linestyle='dashed', linewidth=2, label=f'Median: {np.median(rewards):.1f}')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Count')
    ax1.set_title('Training Reward Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Evaluation rewards distribution
    if eval_rewards:
        ax2.hist(eval_rewards, bins=20, alpha=0.7, color='r')
        ax2.axvline(np.mean(eval_rewards), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(eval_rewards):.1f}')
        ax2.axvline(np.median(eval_rewards), color='b', linestyle='dashed', linewidth=2, label=f'Median: {np.median(eval_rewards):.1f}')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Count')
        ax2.set_title('Evaluation Reward Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save reward distributions plot
    dist_path = os.path.join(save_dir, 'reward_distributions.png')
    plt.savefig(dist_path)
    plt.close()
    
    print(f"Training metrics plot saved as {metrics_path}")
    print(f"Reward distributions plot saved as {dist_path}")
    
    # Also save metrics to file
    metrics_path = os.path.join(save_dir, 'training_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Training Metrics Summary:\n")
        f.write("-----------------------\n")
        f.write(f"Total episodes: {len(episodes)}\n")
        f.write(f"Final episode length: {lengths[-1]:.2f}\n")
        f.write(f"Final training reward: {rewards[-1]:.2f}\n")
        f.write(f"Best training reward: {max(rewards):.2f}\n")
        f.write(f"Average training reward: {np.mean(rewards):.2f}\n")
        f.write(f"Median training reward: {np.median(rewards):.2f}\n")
        f.write(f"Training reward std: {np.std(rewards):.2f}\n")
        if eval_rewards:
            f.write(f"Best evaluation reward: {max(eval_rewards):.2f}\n")
            f.write(f"Final evaluation reward: {eval_rewards[-1]:.2f}\n")
            f.write(f"Average evaluation reward: {np.mean(eval_rewards):.2f}\n")
            f.write(f"Median evaluation reward: {np.median(eval_rewards):.2f}\n")
            f.write(f"Evaluation reward std: {np.std(eval_rewards):.2f}\n")
        f.write(f"Average episode length: {np.mean(lengths):.2f}\n")
        f.write(f"Final actor loss: {actor_losses[-1]:.4f}\n")
        f.write(f"Final critic loss: {critic_losses[-1]:.4f}\n")
        if context_losses:
            f.write(f"Final context loss: {context_losses[-1]:.4f}\n")
        f.write(f"Average actor loss: {np.mean(actor_losses):.4f}\n")
        f.write(f"Average critic loss: {np.mean(critic_losses):.4f}\n")
        if context_losses:
            f.write(f"Average context loss: {np.mean(context_losses):.4f}\n")
    print(f"Metrics summary saved as {metrics_path}")
    
    return metrics_path

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from log file')
    parser.add_argument('run_name', type=str, help='Name of the run directory in models/')
    parser.add_argument('--save-dir', type=str, default=None, 
                       help='Directory to save plots (defaults to run directory)')
    
    args = parser.parse_args()
    
    try:
        # Get list of log files
        log_dir, log_files = list_log_files(args.run_name)
        
        # Let user select a file
        selected_file = select_log_file(log_files)
        log_file_path = os.path.join(log_dir, selected_file)
        
        # Default save directory to run directory
        if args.save_dir is None:
            args.save_dir = os.path.join('models', args.run_name)
        
        # Extract and plot metrics
        episodes, lengths, rewards, actor_losses, critic_losses, context_losses, eval_rewards = extract_metrics_from_log(log_file_path)
        
        if not episodes:
            print("No metrics found in log file!")
            return
        
        plot_metrics(episodes, lengths, rewards, actor_losses, critic_losses, context_losses, eval_rewards, args.save_dir)
        
    except ValueError as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main() 