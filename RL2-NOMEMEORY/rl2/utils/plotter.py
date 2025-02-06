import os
import csv
import matplotlib.pyplot as plt

# Path to the CSV file
file_path = "checkpoints/logs/"
csv_file_path = os.path.join(file_path, "train_reward.csv")
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
