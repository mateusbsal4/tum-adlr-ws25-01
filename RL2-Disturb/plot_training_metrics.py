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
    
    log_files = [f for f in os.listdir(log_dir) if f.startswith('training_') and f.endswith('.log')]
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
    
    # Regular expressions to match the metrics
    episode_pattern = r"Global Episode (\d+):"
    length_pattern = r"Average episode length: ([-\d.]+)"
    reward_pattern = r"Average training reward: ([-\d.]+)"
    actor_loss_pattern = r"Actor loss: ([-\d.]+)"
    critic_loss_pattern = r"Critic loss: ([-\d.]+)"
    
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
                current_episode = None
    
    return episodes, lengths, rewards, actor_losses, critic_losses

def plot_metrics(episodes, lengths, rewards, actor_losses, critic_losses, save_dir):
    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot episode lengths
    ax1.plot(episodes, lengths, 'b-', alpha=0.6)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Episode Length')
    ax1.set_title('Training Episode Lengths')
    ax1.grid(True)
    
    # Plot moving average of episode lengths
    window = 50
    if len(lengths) > window:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], moving_avg, 'r-', label=f'{window}-episode moving average')
        ax1.legend()
    
    # Plot rewards
    ax2.plot(episodes, rewards, 'g-', alpha=0.6)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Training Reward')
    ax2.set_title('Training Rewards')
    ax2.grid(True)
    
    # Plot moving average of rewards
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax2.plot(episodes[window-1:], moving_avg, 'r-', label=f'{window}-episode moving average')
        ax2.legend()
    
    # Plot actor losses
    ax3.plot(episodes, actor_losses, 'm-', alpha=0.6)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Actor Loss')
    ax3.set_title('Actor Loss')
    ax3.grid(True)
    
    # Plot moving average of actor losses
    if len(actor_losses) > window:
        moving_avg = np.convolve(actor_losses, np.ones(window)/window, mode='valid')
        ax3.plot(episodes[window-1:], moving_avg, 'r-', label=f'{window}-episode moving average')
        ax3.legend()
    
    # Plot critic losses
    ax4.plot(episodes, critic_losses, 'c-', alpha=0.6)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Critic Loss')
    ax4.set_title('Critic Loss')
    ax4.grid(True)
    
    # Plot moving average of critic losses
    if len(critic_losses) > window:
        moving_avg = np.convolve(critic_losses, np.ones(window)/window, mode='valid')
        ax4.plot(episodes[window-1:], moving_avg, 'r-', label=f'{window}-episode moving average')
        ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")
    
    # Also save metrics to file
    metrics_path = os.path.join(save_dir, 'training_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Training Metrics Summary:\n")
        f.write("-----------------------\n")
        f.write(f"Total episodes: {len(episodes)}\n")
        f.write(f"Final episode length: {lengths[-1]:.2f}\n")
        f.write(f"Final reward: {rewards[-1]:.2f}\n")
        f.write(f"Best reward: {max(rewards):.2f}\n")
        f.write(f"Average reward: {np.mean(rewards):.2f}\n")
        f.write(f"Average episode length: {np.mean(lengths):.2f}\n")
        f.write(f"Final actor loss: {actor_losses[-1]:.4f}\n")
        f.write(f"Final critic loss: {critic_losses[-1]:.4f}\n")
        f.write(f"Average actor loss: {np.mean(actor_losses):.4f}\n")
        f.write(f"Average critic loss: {np.mean(critic_losses):.4f}\n")
    print(f"Metrics summary saved as {metrics_path}")

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
        episodes, lengths, rewards, actor_losses, critic_losses = extract_metrics_from_log(log_file_path)
        
        if not episodes:
            print("No metrics found in log file!")
            return
        
        plot_metrics(episodes, lengths, rewards, actor_losses, critic_losses, args.save_dir)
        
    except ValueError as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main() 