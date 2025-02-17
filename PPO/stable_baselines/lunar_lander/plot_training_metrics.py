import re
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Dict, Tuple, List

def load_training_metrics(run_dir: str) -> Dict:
    """Load training metrics from separate numpy files."""
    metrics_dir = os.path.join(run_dir, "training_metrics")
    if not os.path.exists(metrics_dir):
        raise FileNotFoundError(f"No training metrics directory found at {metrics_dir}")
    
    metrics = {}
    metric_files = {
        'rewards': 'rewards.npy',
        'ep_lengths': 'ep_lengths.npy',
        'critic_losses': 'critic_losses.npy',
        'wind_speeds': 'wind_speeds.npy'
    }
    
    for key, filename in metric_files.items():
        file_path = os.path.join(metrics_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing metric file: {file_path}")
        metrics[key] = np.load(file_path, allow_pickle=True)
        
        # Convert None values to np.nan
        if metrics[key].dtype == object:
            metrics[key] = np.array([x if x is not None else np.nan for x in metrics[key]], dtype=float)
    
    # Generate steps based on the length of rewards array and n_steps
    n_steps = 2000  # This should match the n_steps in your training
    metrics['steps'] = np.arange(len(metrics['rewards'])) * n_steps
    
    # Calculate best mean reward
    metrics['best_mean_reward'] = np.max(metrics['rewards'])
    
    return metrics

def load_eval_metrics(run_dir: str) -> Dict:
    """Load evaluation metrics from the numpy file."""
    eval_path = os.path.join(run_dir, "evaluation_metrics", "eval_results.npy")
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"No evaluation metrics found at {eval_path}")
    
    return np.load(eval_path, allow_pickle=True).item()

def calculate_moving_average(data: List[float], window: int) -> np.ndarray:
    """Calculate moving average of the data."""
    if len(data) == 0:
        return np.array([])
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_training_curves(metrics: Dict, save_dir: str, window: int = 50):
    """Plot training metrics over time."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    steps = metrics['steps']
    
    def safe_plot(ax, data, color, title, ylabel):
        """Safely plot data with moving average."""
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data available', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
        else:
            ax.plot(steps, data, f'{color}-', alpha=0.3, label='Per Update')
            ma_data = calculate_moving_average(data, min(window, len(data)))
            if len(ma_data) > 0:
                ma_steps = steps[min(window-1, len(steps)-1):len(ma_data)+min(window-1, len(steps)-1)]
                ax.plot(ma_steps, ma_data, f'{color}-', linewidth=2, 
                       label=f'{window}-update moving average')
        
        ax.set_title(title)
        ax.set_xlabel('Steps')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if len(data) > 0:
            ax.legend()
    
    # Plot each metric
    safe_plot(ax1, metrics['rewards'], 'g', 'Training Rewards', 'Reward')
    safe_plot(ax2, metrics['ep_lengths'], 'b', 'Episode Lengths', 'Length')
    safe_plot(ax3, metrics['critic_losses'], 'c', 'Critic Loss', 'Loss')
    safe_plot(ax4, metrics['wind_speeds'], 'm', 'Wind Speeds', 'Wind Speed')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()

def plot_reward_distributions(metrics: Dict, eval_metrics: Dict, save_dir: str):
    """Plot training and evaluation reward distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training rewards distribution
    if len(metrics['rewards']) > 0:
        ax1.hist(metrics['rewards'], bins=min(50, len(metrics['rewards'])), alpha=0.7, color='g')
        ax1.axvline(np.mean(metrics['rewards']), color='r', linestyle='dashed', linewidth=2,
                    label=f'Mean: {np.mean(metrics["rewards"]):.1f}')
        ax1.axvline(np.median(metrics['rewards']), color='b', linestyle='dashed', linewidth=2,
                    label=f'Median: {np.median(metrics["rewards"]):.1f}')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No training data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax1.transAxes)
    
    ax1.set_title('Training Reward Distribution')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)
    
    # Evaluation rewards
    if eval_metrics and 'mean_reward' in eval_metrics:
        ax2.bar(['Evaluation'], [eval_metrics['mean_reward']], yerr=[eval_metrics['std_reward']],
                capsize=5, color='r', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, 'No evaluation data available', 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transAxes)
    
    ax2.set_title('Evaluation Reward')
    ax2.set_ylabel('Reward')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_distributions.png'), dpi=300)
    plt.close()

def save_metrics_summary(metrics: Dict, eval_metrics: Dict, save_dir: str):
    """Save a text summary of the training and evaluation metrics."""
    summary_path = os.path.join(save_dir, 'metrics_summary.txt')
    
    with open(summary_path, 'w') as f:
        # Training metrics
        f.write("Training Metrics Summary:\n")
        f.write("-----------------------\n")
        f.write(f"Total training steps: {metrics['steps'][-1]:,}\n")
        f.write(f"Total updates: {len(metrics['rewards']):,}\n")
        f.write(f"Best mean reward: {metrics['best_mean_reward']:.2f}\n")
        f.write(f"Final training reward: {metrics['rewards'][-1]:.2f}\n")
        f.write(f"Average training reward: {np.mean(metrics['rewards']):.2f}\n")
        f.write(f"Training reward std: {np.std(metrics['rewards']):.2f}\n")
        f.write(f"Average episode length: {np.mean(metrics['ep_lengths']):.2f}\n")
        f.write(f"Final critic loss: {metrics['critic_losses'][-1]:.5f}\n")
        
        # Evaluation metrics
        if eval_metrics:
            f.write("\nEvaluation Metrics:\n")
            f.write("------------------\n")
            f.write(f"Mean evaluation reward: {eval_metrics['mean_reward']:.2f}\n")
            f.write(f"Evaluation reward std: {eval_metrics['std_reward']:.2f}\n")

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from logs')
    parser.add_argument('run_name', type=str, help='Name of the training run')
    args = parser.parse_args()
    
    # Set up directories
    run_dir = os.path.join('logs', args.run_name)
    if not os.path.exists(run_dir):
        raise ValueError(f"Run directory not found: {run_dir}")
    
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Load metrics
        print("Loading metrics...")
        training_metrics = load_training_metrics(run_dir)
        try:
            eval_metrics = load_eval_metrics(run_dir)
        except FileNotFoundError:
            print("No evaluation metrics found.")
            eval_metrics = {}
        
        # Generate plots and summary
        print("Generating plots...")
        plot_training_curves(training_metrics, plots_dir)
        plot_reward_distributions(training_metrics, eval_metrics, plots_dir)
        save_metrics_summary(training_metrics, eval_metrics, plots_dir)
        
        print(f"Plots and summary saved in {plots_dir}")
        
    except KeyboardInterrupt:
        print("\nPlotting interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 