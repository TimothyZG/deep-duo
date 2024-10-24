import argparse
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mf.model_factory import create_model
from data.dataloader import get_dataloaders
from trainers.evaluator import evaluate_model
from trainers.logger import Logger
from utils.config import load_config

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config_dir', type=str, default='deep-duo/config', help='Directory of the configuration files')
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
    model_params = next((item for item in model_config['models'] if item.lower() == model_name.lower()), None)
    if not model_params:
        raise ValueError(f"Model '{model_name}' not found in model_config.yaml")
    
    # Get dataset-specific configurations
    dataset_name = args.dataset_name
    dataset_params = next((item for item in dataset_config['datasets'] if item['name'].lower() == dataset_name.lower()), None)
    if not dataset_params:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset_config.yaml")

    num_classes = dataset_params['num_classes']
    root_dir = os.path.expandvars(dataset_params['root_dir'])
    training_params = training_config.get('training', {})
    batch_size = training_params.get('batch_size', 32)
    num_workers = training_params.get('num_workers', 4)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_model(model_name=model_name, num_classes=num_classes)
    model = model.to(device)

    # Load the saved state dictionary
    checkpoint_path = args.checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

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

    # Set up criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize logger
    logger = Logger(project_name='deep-duo-eval', config={
        'model_name': model_name,
        'dataset_name': dataset_name,
        # Add other configurations
    })

    # Evaluate the model on in-distribution test set
    print("Evaluating on In-Distribution Test Set:")
    test_loss_ind, test_metric_ind = evaluate_model(
        model=model,
        dataloader=data_loaders['test'],
        criterion=criterion,
        device=device,
        dataset_name=dataset_name,
        model_name=model_name,
        eval_type='ind'
    )

    # Evaluate the model on out-of-distribution test set
    print("Evaluating on Out-of-Distribution Test Set:")
    test_loss_ood, test_metric_ood = evaluate_model(
        model=model,
        dataloader=data_loaders['ood_test'],
        criterion=criterion,
        device=device,
        dataset_name=dataset_name,
        model_name=model_name,
        eval_type='ood'
    )

    # Log results
    logger.log({
        'test_loss_ind': test_loss_ind,
        'test_metric_ind': test_metric_ind,
        'test_loss_ood': test_loss_ood,
        'test_metric_ood': test_metric_ood,
    })

    # Print results
    print(f"InD Test Loss: {test_loss_ind:.4f}, InD Test F1: {test_metric_ind:.4f}")
    print(f"OOD Test Loss: {test_loss_ood:.4f}, OOD Test F1: {test_metric_ood:.4f}")

    # Finish logging
    logger.finish()

if __name__ == '__main__':
    main()
