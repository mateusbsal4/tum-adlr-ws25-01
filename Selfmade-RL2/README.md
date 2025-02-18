# Variable Target Lunar Lander with RL2-PPO

This project implements a modified version of OpenAI Gym's LunarLander environment where the landing target position can be varied. The agent is trained using PPO (Proximal Policy Optimization) with a GRU-based policy network to handle the variable target positions.

## Project Structure

- `var_lander_gym.py`: Base environment with variable target position functionality
- `custom_lunar_lander.py`: Custom wrapper for the variable target lander
- `lunar_lander_rl2.py`: Main training script with RL2-PPO implementation
- `plot_training_metrics.py`: Utility script for plotting training metrics

## Requirements

```bash
pip install gymnasium[box2d]
pip install torch
pip install mpi4py
pip install matplotlib
pip install numpy
```

## Training

To train the agent with multiple workers using MPI:

```bash
mpirun -np 8 python lunar_lander_rl2.py
```

Options:
- `--run-name NAME`: Specify custom name for the training run
- `--render`: Enable rendering during training
- `--resume-from DIR`: Resume training from latest model in specified directory

## Resuming Training

To resume training from a previous run:

```bash
mpirun -np 8 python lunar_lander_rl2.py --resume-from run_name
```

## Plotting Training Metrics

To visualize training results:

```bash
python plot_training_metrics.py run_name
```

This will:
1. List available log files in the run directory
2. Let you select which log file to analyze
3. Generate plots for:
   - Episode lengths
   - Training rewards
   - Actor losses
   - Critic losses

## Project Features

- Distributed training with MPI
- Variable landing target positions
- GRU-based policy network for temporal dependencies
- Automatic checkpoint saving
- Training metrics logging
- Visualization tools
- Resume training capability

## Directory Structure

```
.
├── models/
│   └── run_name/
│       ├── checkpoints/
│       │   └── model_checkpoints.pth
│       ├── logs/
│       │   └── training_logs.log
│       └── best_model.pth
├── var_lander_gym.py
├── custom_lunar_lander.py
├── lunar_lander_rl2.py
└── plot_training_metrics.py
```

## Training Output

The training script will create a directory under `models/` with:
- Checkpoint models saved periodically
- Best performing models
- Training logs
- Reward plots
- Training metrics summary

## Monitoring Training

During training, the script outputs:
- Global episode number
- Average episode length
- Average training reward
- Actor and critic losses
- Target positions and landing coordinates
- Evaluation results at regular intervals

## Evaluation

The agent is evaluated periodically during training using a fixed target position (center) for consistent comparison. 