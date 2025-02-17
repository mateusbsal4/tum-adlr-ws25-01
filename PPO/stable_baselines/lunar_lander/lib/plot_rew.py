import re
import matplotlib.pyplot as plt

def extract_ep_rew_mean(log_file_path):
    """
    Reads the text in log_file_path, finds lines with ep_rew_mean,
    and returns a list of numeric values.
    """
    ep_rew_means = []
    
    # Regex to capture the number after '|    ep_rew_mean    |'
    pattern = re.compile(r"\|\s*ep_rew_mean\s*\|\s*([-\d.]+)\s*\|")
    
    with open(log_file_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                value = float(match.group(1))
                ep_rew_means.append(value)
    
    return ep_rew_means

def plot_and_save_ep_rew_mean(values, output_file="rew_plot.png"):
    """
    Plots the ep_rew_mean values and saves the figure as output_file.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(values, marker='o', linestyle='-', color='blue')
    plt.title("ep_rew_mean Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("ep_rew_mean")
    plt.grid(True)
    
    # Save plot to PNG instead of showing it interactively
    plt.savefig(output_file, dpi=300)
    plt.close()  # Close the figure to release resources

if __name__ == "__main__":
    log_file = "./logs/train_log.txt"  # Replace with your actual log file path
    ep_rew_values = extract_ep_rew_mean(log_file)

    # Plot and save the figure
    plot_and_save_ep_rew_mean(ep_rew_values, output_file="./logs/rew_plot.png")
    print("Plot saved to ./logs/rew_plot.png")
