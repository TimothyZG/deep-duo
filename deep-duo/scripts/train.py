# scripts/train.py

import argparse
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mf.model_factory import create_model
from data.dataloader import get_dataloaders
from trainers.trainer import Trainer
from trainers.logger import Logger
from trainers.saver import Saver
from utils.config import load_config

def main(config, checkpoint_dir=None):
    # Extract hyperparameters from config
    model_name = config['model_name']
    dataset_name = config['dataset_name']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    batch_size = config.get('batch_size', 64)
    num_epochs = config.get('num_epochs', 1)
    num_workers = config.get('num_workers', 4)
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--config_dir', type=str, default='deep-duo/configs', help='Directory of the configuration files')
    args = parser.parse_args()

    # Load configurations
    dataset_config_path = os.path.join(args.config_dir, 'dataset_config.yaml')
    training_config_path = os.path.join(args.config_dir, f'training_config_{args.dataset_name}.yaml')

    print(f"{os.getcwd()=}")
    dataset_config = load_config(dataset_config_path)
    training_config = load_config(training_config_path)
    print(f"{dataset_config=}")
    print(f"{training_config=}")


    # Get dataset-specific configurations
    dataset_name = args.dataset_name
    dataset_params = next((item for item in dataset_config['datasets'] if item['name'].lower() == dataset_name.lower()), None)
    if not dataset_params:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset_config.yaml")

    num_classes = dataset_params['num_classes']
    root_dir = os.path.expandvars(dataset_params['root_dir'])
    print(f"{root_dir=}")
    # Access training-specific configurations
    training_params = training_config.get('training', {})
    print(f"{training_params}")
    # Get batch size, learning rate, weight decay, and number of epochs from the nested structure
    batch_size = training_params.get('batch_size', 64)
    # Get optimizer-related parameters
    optimizer_params = training_params.get('optimizer', {})
    learning_rate = optimizer_params.get('lr', 1e-4)
    weight_decay = optimizer_params.get('weight_decay', 1e-5)

    # Other training parameters
    num_epochs = training_params.get('num_epochs', 1)
    num_workers = training_params.get('num_workers', 4)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_model(model_name=model_name, num_classes=num_classes)
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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize logger and saver
    logger = Logger(project_name='deep-duo', config={
        'model_name': model_name,
        'dataset_name': dataset_name,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
    })
    checkpoint_dir=f'checkpoints/{dataset_name}/{model_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    saver = Saver(checkpoint_dir)

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        dataloaders=data_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        saver=saver,
        num_epochs=num_epochs,
        dataset_name=dataset_name
    )

    # Train the model
    trained_model, best_metric = trainer.train()
    
    # Save the trained model
    torch.save(trained_model.state_dict(), f'checkpoints/{dataset_name}/trained_model_{model_name}_{dataset_name}.pth')
    print(f"Trained model with {best_metric=} saved to 'checkpoints/{dataset_name}/trained_model_{model_name}_{dataset_name}.pth'")

    # Finish logging
    logger.finish()

if __name__ == '__main__':
    main()
