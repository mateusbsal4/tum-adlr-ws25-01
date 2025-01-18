import csv
import re
import matplotlib.pyplot as plt

def eval_plotter():
    # File paths
    log_file_path = "checkpoints/logs/eval_log.txt"
    plot_file_path = "checkpoints/logs/eval_analysis.png"

    # Regex pattern to extract rewards
    pattern = re.compile(
        r"x_reward: ([\-\d.]+), y_reward: ([\-\d.]+),\s+calculated reward: ([\-\d.]+), actual reward: ([\-\d.]+)"
    )

    # Lists to store extracted data
    x_rewards = []
    y_rewards = []
    calculated_rewards = []
    actual_rewards = []

    # Read the log file and extract data
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                x_rewards.append(float(match.group(1)))
                y_rewards.append(float(match.group(2)))
                calculated_rewards.append(float(match.group(3)))
                actual_rewards.append(float(match.group(4)))

    # # Save the extracted data to a CSV file
    # with open(csv_file_path, mode="w", newline="") as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["Index", "x_reward", "y_reward", "calculated_reward", "actual_reward"])
    #     for i, (x, y, calc, actual) in enumerate(zip(x_rewards, y_rewards, calculated_rewards, actual_rewards)):
    #         writer.writerow([i, x, y, calc, actual])

    # print(f"Data saved to {csv_file_path}")

    # Plot the data
    plt.figure(figsize=(12, 8))  # Set figure size
    plt.plot(x_rewards, label="x_reward")
    plt.plot(y_rewards, label="y_reward")
    plt.plot(calculated_rewards, label="calculated_reward")
    plt.plot(actual_rewards, label="actual_reward")
    plt.xlabel("Index")
    plt.ylabel("Reward Values")
    plt.title("Rewards Over Time")
    plt.legend()
    plt.grid()

    # Save the plot to a PNG file
    plt.savefig(plot_file_path, dpi=300)
    print(f"Plot saved to {plot_file_path}")

    # Show the plot (optional)
    # plt.show()

