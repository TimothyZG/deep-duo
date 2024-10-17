# scripts/hypertune.py

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import f1_score

# Add the project directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import create_model
from data.dataloader import get_dataloaders
from utils.config import load_config

def train_model(config, num_classes, root_dir, dataset_name, device):
    # Unpack hyperparameters
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_workers = config['num_workers']

    # Create model
    model = create_model(model_name=config['model_name'], num_classes=num_classes)
    model = model.to(device)

    # Get data transforms
    from data.transforms import get_transforms
    transforms = get_transforms(dataset_name)

    # Prepare data loaders
    data_loaders = get_dataloaders(
        dataset_name=dataset_name,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transforms=transforms
    )

    # Set up criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    best_validation_f1 = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for batch in data_loaders['train']:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in data_loaders['val']:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute macro F1 score
        validation_f1 = f1_score(all_labels, all_preds, average='macro')

        # Report metrics to Ray Tune
        tune.report(validation_f1=validation_f1)

        # Save checkpoint if this is the best model so far
        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)

        # Update scheduler
        scheduler.step()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning script')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--config_dir', type=str, default='deep-duo/configs', help='Directory of the configuration files')
    args = parser.parse_args()

    # Load configurations
    model_config_path = os.path.join(args.config_dir, 'model_config.yaml')
    dataset_config_path = os.path.join(args.config_dir, 'dataset_config.yaml')
    training_config_path = os.path.join(args.config_dir, f'training_config_{args.dataset_name}.yaml')

    model_config = load_config(model_config_path)
    dataset_config = load_config(dataset_config_path)
    training_config = load_config(training_config_path)

    # Check if model is in model_config
    model_name = args.model_name
    if model_name.lower() not in [m.lower() for m in model_config['models']]:
        raise ValueError(f"Model '{model_name}' not found in model_config.yaml")

    # Get dataset-specific configurations
    dataset_name = args.dataset_name
    dataset_params = next(
        (item for item in dataset_config['datasets'] if item['name'].lower() == dataset_name.lower()), None)
    if not dataset_params:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset_config.yaml")

    num_classes = dataset_params['num_classes']
    root_dir = os.path.expandvars(dataset_params['root_dir'])

    # Access training-specific configurations
    training_params = training_config.get('training', {})
    batch_size = training_params.get('batch_size', 64)
    num_epochs = training_params.get('num_epochs', 1)
    num_workers = training_params.get('num_workers', 4)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the hyperparameter search space
    config = {
        'model_name': model_name,
        'learning_rate': tune.loguniform(1e-5, 1e-3),
        'weight_decay': tune.loguniform(1e-6, 1e-4),
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'num_workers': num_workers,
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=['learning_rate', 'weight_decay'],
        metric_columns=['validation_f1', 'training_iteration']
    )

    result = tune.run(
        tune.with_parameters(
            train_model,
            num_classes=num_classes,
            root_dir=root_dir,
            dataset_name=dataset_name,
            device=device
        ),
        resources_per_trial={'cpu': num_workers, 'gpu': 1 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=12,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir='ray_results',
        checkpoint_at_end=True
    )

    best_trial = result.get_best_trial('validation_f1', 'max', 'last')
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['validation_accuracy']}")

    # Load the best model checkpoint
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint.pt"))['model_state_dict']

    # Recreate the model and load state
    best_model = create_model(model_name=best_trial.config['model_name'], num_classes=num_classes)
    best_model.load_state_dict(model_state)
    best_model.to(device)

    # Save the best model
    os.makedirs('checkpoints', exist_ok=True)
    model_save_path = f'checkpoints/best_model_{model_name}_{dataset_name}.pth'
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Best model saved to '{model_save_path}'")

if __name__ == '__main__':
    main()
