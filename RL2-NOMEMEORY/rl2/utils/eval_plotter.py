import csv
import re
import matplotlib.pyplot as plt

def eval_plotter():
    # File paths
    log_dir = "./checkpoints_fixed/logs/"
    log_file_path = log_dir + "training_log.txt"
    plot_file_path = log_dir + "train_analysis.png"

    # Regex pattern to extract rewards
    pattern = re.compile(
        # r"x_reward: ([\-\d.]+), y_reward: ([\-\d.]+),\s+calculated reward: ([\-\d.]+), actual reward: ([\-\d.]+)"
        r"pos_reward: (?P<pos_reward>-?\d+\.?\d*), vel_reward: (?P<vel_reward>-?\d+\.?\d*),\s+"
        r"angle_reward: (?P<angle_reward>-?\d+\.?\d*), leg1_reward: (?P<leg1_reward>-?\d+\.?\d*), "
        r"leg2_reward: (?P<leg2_reward>-?\d+\.?\d*)\s+calculated reward: (?P<calculated_reward>-?\d+\.?\d*), "
        r"actual reward: (?P<actual_reward>-?\d+\.?\d*)"
    )

    # Lists to store extracted data
    pos_rewards = []
    vel_rewards = []
    angle_rewards = []
    leg1_rewards = []
    leg2_rewards = []
    calculated_rewards = []
    actual_rewards = []

    # Read the log file and extract data
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                pos_rewards.append(float(match.group(1)))
                vel_rewards.append(float(match.group(2)))
                angle_rewards.append(float(match.group(3)))
                leg1_rewards.append(float(match.group(4)))
                leg2_rewards.append(float(match.group(5)))
                calculated_rewards.append(float(match.group(6)))
                actual_rewards.append(float(match.group(7)))

    print("here:", angle_rewards)
    # # Save the extracted data to a CSV file
    # with open(csv_file_path, mode="w", newline="") as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["Index", "x_reward", "y_reward", "calculated_reward", "actual_reward"])
    #     for i, (x, y, calc, actual) in enumerate(zip(x_rewards, y_rewards, calculated_rewards, actual_rewards)):
    #         writer.writerow([i, x, y, calc, actual])

    # print(f"Data saved to {csv_file_path}")

    # Plot the data
    plt.figure(figsize=(12, 8))  # Set figure size
    plt.plot(pos_rewards, label="pos_reward")
    plt.plot(vel_rewards, label="vel_reward")
    plt.plot(angle_rewards, label="angle_reward")
    plt.plot(leg1_rewards, label="leg1_reward")
    plt.plot(leg2_rewards, label="leg2_reward")
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

if __name__ == "__main__":
    eval_plotter()