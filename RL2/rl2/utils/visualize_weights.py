import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for efficient file saving
import matplotlib.pyplot as plt
import torch


def visualize_network_weights(net, net_name, update_step, out_dir="weights_plots", bins=50):
    """
    Visualize the weight distributions of a given neural network by plotting histograms
    for each parameter tensor that requires gradients.

    Args:
        net (torch.nn.Module): The neural network to visualize.
        net_name (str): A name identifier for the network.
        update_step (int): The update step index, to include in the filename/title.
        out_dir (str): The directory to save the plots.
        bins (int): Number of bins for the histogram.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Filter parameters that require gradients
    params = [(name, param) for name, param in net.named_parameters() if param.requires_grad]
    num_params = len(params)
    if num_params == 0:
        print(f"No parameters to visualize for {net_name}.")
        return

    # Set up grid layout: up to 3 columns
    cols = min(3, num_params)
    rows = (num_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    # Ensure axes is iterable
    if num_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (name, param) in enumerate(params):
        ax = axes[i]
        # Flatten parameter values
        data = param.data.cpu().numpy().flatten()
        ax.hist(data, bins=bins, color='blue', alpha=0.7)
        ax.set_title(name, fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Turn off any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"{net_name} Weights at Update {update_step}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = os.path.join(out_dir, f"{net_name}_weights_update_{update_step}.png")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved weight visualization for {net_name} at update {update_step} to {filename}")


def visualize_weights(policy_net, value_net, update_step, out_dir="weights_plots", bins=50):
    """
    Visualize and save the weights of both the policy and value networks.

    Args:
        policy_net (torch.nn.Module): The policy network to visualize.
        value_net (torch.nn.Module): The value network to visualize.
        update_step (int): The update step index.
        out_dir (str): The directory to save the plots.
        bins (int): Number of bins for the histogram.
    """
    visualize_network_weights(policy_net, "policy_net", update_step, out_dir, bins)
    visualize_network_weights(value_net, "value_net", update_step, out_dir, bins)


if __name__ == "__main__":
    # Example usage (requires valid policy_net and value_net instances):
    # from your_rl_module import PolicyNet, ValueNet
    # policy_net = PolicyNet()
    # value_net  = ValueNet()
    # visualize_weights(policy_net, value_net, update_step=0)
    print("This module provides functions to visualize network weights.") 