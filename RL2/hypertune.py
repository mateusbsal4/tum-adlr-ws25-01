import wandb
import subprocess
import multiprocessing
import os
import json

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': 
        {
            'name': 'value_loss',
            'goal': 'minimize'
        },
    'parameters': {
        'num_features': {
            'min': 32,
            'max': 256
        },
        'ppo_clip_param': {
            'min': 0.1,
            'max': 0.3
        },
        'ppo_ent_coef': {
            'min': 0.01,
            'max': 0.1
        },
        'discount_gamma': {
            'min': 0.95,
            'max': 0.99
        },
        'gae_lambda': {
            'min': 0.9,
            'max': 0.99
        },
        'adam_lr': {
            'min': 1e-5,
            'max': 1e-3
        },
        'value_lr': {
            'min': 1e-5,
            'max': 1e-3
        },
        'timesteps_per_learner_batch': {
            'min': 32,
            'max': 256
        },
        'timesteps_per_pol_update': {
            'min': 1024,
            'max': 8192
        },
        'ppo_opt_epochs': {
            'min': 1,
            'max': 20
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        'eta': 2,
        'max_iter': 27
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="rl2_project")

# Define the training function
def train():
    # Initialize wandb
    with wandb.init(reinit=True):

        # Get the hyperparameters
        config = wandb.config

        # Determine test ID
        log_directory = "checkpoints/"
        os.makedirs(log_directory, exist_ok=True)
        
        registry_file = os.path.join(log_directory, 'test_registry.json')
        
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            test_id = len(registry) + 1
        else:
            test_id = 1
            registry = []

        # Add the test information to the registry
        test_info = {
            "test_id": test_id,
            "parameters": {
                "num_features": config.num_features,
                "ppo_clip_param": config.ppo_clip_param,
                "ppo_ent_coef": config.ppo_ent_coef,
                "discount_gamma": config.discount_gamma,
                "gae_lambda": config.gae_lambda,
                "adam_lr": config.adam_lr,
                "value_lr": config.value_lr,
                "timesteps_per_learner_batch": config.timesteps_per_learner_batch,
                "timesteps_per_pol_update": config.timesteps_per_pol_update,
                "ppo_opt_epochs": config.ppo_opt_epochs,
            }
        }
        registry.append(test_info)
        
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=4)
            
        print(f"Running test {test_id}")

        # Run the training script with the hyperparameters
        subprocess.run([
            'python', 'train.py',
            '--num_features', str(config.num_features),
            '--ppo_clip_param', str(config.ppo_clip_param),
            '--ppo_ent_coef', str(config.ppo_ent_coef),
            '--discount_gamma', str(config.discount_gamma),
            '--gae_lambda', str(config.gae_lambda),
            '--adam_lr', str(config.adam_lr),
            '--value_lr', str(config.value_lr),
            '--timesteps_per_learner_batch', str(config.timesteps_per_learner_batch),
            '--timesteps_per_pol_update', str(config.timesteps_per_pol_update),
            '--ppo_opt_epochs', str(config.ppo_opt_epochs),
            '--checkpoint_dir', f'checkpoints/checkpoint_{test_id}',
            '--wandb_project', 'rl2_project',
            '--wandb_entity', wandb.run.entity
        ])

# Function to run the wandb agent
def run_agent():
    wandb.agent(sweep_id, function=train, count=20)  # Remove fixed count for continuous runs

# Number of parallel workers
num_workers = 1

# Create and start multiple processes
processes = []
for _ in range(num_workers):
    p = multiprocessing.Process(target=run_agent)
    p.start()
    processes.append(p)

# Gracefully handle keyboard interrupt during join
try:
    for p in processes:
        p.join()
except KeyboardInterrupt:
    print("KeyboardInterrupt caught. Terminating processes...")
    for p in processes:
        p.terminate()
    for p in processes:
        p.join()
