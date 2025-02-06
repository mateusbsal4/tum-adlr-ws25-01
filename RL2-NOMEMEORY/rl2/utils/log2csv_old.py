import re
import os
import csv
import matplotlib.pyplot as plt


# Path to your log file
file_path = "checkpoints_fixed/logs/"
log_file_path = os.path.join(file_path, "training_log.txt")
csv_file_path = os.path.join(file_path, "train_reward.csv")

# Regex patterns to extract values
mean_return_pattern = re.compile(r"mean meta-episode return: (-?\d+\.?\d*)")
policy_update_pattern = re.compile(r"pol update (\d+), opt_epoch:")

# Data storage
data = []

# Parse the log file
with open(log_file_path, "r") as log_file:
    lines = log_file.readlines()

mean_return = None
current_update = None

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
            data.append((current_update, mean_return))
            mean_return = None  # Reset for the next match

# Save data to CSV
with open(csv_file_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Policy Update", "Mean Meta-Episode Return"])
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

with open(csv_file_path, "r") as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # Skip the header
    for row in reader:
        policy_updates.append(int(row[0]))
        mean_returns.append(float(row[1]))

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(policy_updates, mean_returns, marker="o", linestyle="-", color="b")
plt.title("Mean Meta-Episode Return vs. Policy Update")
plt.xlabel("Policy Update")
plt.ylabel("Mean Meta-Episode Return")
plt.grid()

# Save the figure to a PNG file
plt.savefig(figure_file_path, dpi=300)  # dpi=300 for high-resolution
print(f"Plot saved as {figure_file_path}")

# Optionally show the plot (can be omitted if not needed)
plt.show()

