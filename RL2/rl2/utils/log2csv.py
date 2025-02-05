import re
import os
import csv
import matplotlib.pyplot as plt

def log2csv(checkpoint_dir: str):
    # Path to your log file
    file_path = os.path.join(checkpoint_dir, "logs")
    log_file_path = os.path.join(file_path, "training_log.txt")
    csv_file_path = os.path.join(file_path, "train_reward.csv")

    # Regex patterns to extract values
    mean_return_pattern = re.compile(r"mean meta-episode return: (-?\d+\.?\d*)")
    policy_update_pattern = re.compile(r"pol update (\d+), opt_epoch:")
    mean_episode_length_pattern = re.compile(r"mean episode length: (-?\d+\.?\d*)")

    # Data storage
    data = []

    # Parse the log file
    with open(log_file_path, "r") as log_file:
        lines = log_file.readlines()

    mean_return = None
    current_update = None
    mean_episode_length = None
    
    for line in lines:
        # Check for mean meta-episode return
        mean_match = mean_return_pattern.search(line)
        if mean_match:
            mean_return = float(mean_match.group(1))
        
        # Check for policy update
        policy_match = policy_update_pattern.search(line)
        if policy_match:
            current_update = int(policy_match.group(1))
            if mean_return is not None:
                # Save data
                # data.append((current_update, mean_return))
                mean_return = None  # Reset for the next match
                
        # Check for mean episode length
        length_match = mean_episode_length_pattern.search(line)
        if length_match:
            mean_episode_length = float(length_match.group(1))
            if mean_return is not None:
                # Save data
                data.append((current_update, mean_return, mean_episode_length))
                mean_return = None  # Reset for the next match

    # Save data to CSV
    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Policy Update", "Mean Meta-Episode Return", "Mean Episode Length"])
        writer.writerows(data)

    print(f"Data saved to {csv_file_path}")

    # Path to the CSV file
    # file_path = "checkpoints/logs/"
    # csv_file_path = os.path.join(file_path, "train_reward.csv")
    # Path to save the figure
    figure_file_path = os.path.join(file_path, "train_reward.png")

    # Read data from CSV
    policy_updates = []
    mean_returns = []
    mean_episode_length = []

    with open(csv_file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip the header
        for row in reader:
            policy_updates.append(int(row[0]))
            mean_returns.append(float(row[1]))
            mean_episode_length.append(float(row[2]))
    # Plot the data with dual y-axis: mean returns on left and mean episode length on right.
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()  # Primary axis
    ax1.plot(policy_updates, mean_returns, marker="o", linestyle="-", color="b", label="Mean Meta-Episode Return")
    ax1.set_xlabel("Policy Update")
    ax1.set_ylabel("Mean Meta-Episode Return", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    
    # Create a twin y-axis sharing the same x-axis.
    ax2 = ax1.twinx()
    ax2.plot(policy_updates, mean_episode_length, marker="s", linestyle="--", color="r", label="Mean Episode Length")
    ax2.set_ylabel("Mean Episode Length", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    
    plt.title("Mean Meta-Episode Return and Episode Length vs. Policy Update")
    plt.grid(True)
    plt.tight_layout()

    # Save the figure to a PNG file
    plt.savefig(figure_file_path, dpi=300)  # dpi=300 for high-resolution
    print(f"Plot saved as {figure_file_path}")

    # Optionally show the plot (can be omitted if not needed)
    plt.show()
    
if __name__ == "__main__":
    file_path = "checkpoints/logs/"
    log2csv(file_path)

