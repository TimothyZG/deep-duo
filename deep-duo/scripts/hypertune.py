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
from sklearn.metrics import f1_score, accuracy_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from mf.model_factory import create_model
import pandas as pd

def save_results_csv(model_name, phase, learning_rate, weight_decay, batch_size, epoch, best_validation_metric, filename):
    print(f"checking if saving hyperparams result to {filename}...")
    file_exists = os.path.exists(filename)
    
    # Load existing CSV or create an empty DataFrame
    if file_exists:
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["model_name", "phase", "learning_rate", "weight_decay", "batch_size", 
                                   "num_epochs", "validation_metric"])

    # Filter existing records for the same model & phase
    mask = (df["model_name"] == model_name) & (df["phase"] == phase)

    if mask.any():  # Check if this model-phase exists in CSV
        prev_best_metric = df.loc[mask, "validation_metric"].max()
        if best_validation_metric <= prev_best_metric:
            print(f"Skipping saving for {model_name} ({phase}) as current metric is not better.")
            return  # Do not save if new metric is not better

        # Remove the old record since new one is better
        df = df[~mask]

    # Append new best result
    new_entry = pd.DataFrame([{
        "model_name": model_name,
        "phase": phase,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_epochs": epoch,
        "validation_metric": best_validation_metric
    }])

    df = pd.concat([df, new_entry], ignore_index=True)
    print("hyperparams csv saved as")
    print(df)
    # Save back to CSV
    df.to_csv(filename, index=False)
    print(f"Updated {filename} with new best results.")
                
def train_model(config, num_classes, root_dir, dataset_name, device, model_state=None, finetune=False):
    import sys
    import os
    import ray
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mf.model_factory import create_model
    from data.transforms import get_transforms
    from data.dataloader import get_dataloaders
    from ray.air import session, Checkpoint
    print(f"Starting training with model: {config['model_name']} on dataset: {dataset_name}")
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_workers = config['num_workers']

    # Create model
    model = create_model(model_name=config['model_name'], num_classes=num_classes)
    
    if model_state:
        print("Using the model_state pass in as starting point")
        model.load_state_dict(model_state)
        
    if finetune:
        print("Fully Finetune Phase:")
        for param in model.parameters():
            param.requires_grad = True
        print("All grad activated!")
    else:
        print("Linear Probing Phase:")
    phase = "fully_finetuned" if finetune else "linear_probing"
        
    model = model.to(device)

    # Get data transforms
    resize = 384 if config['model_name']== "vit_b_16" else 512 if config['model_name'] == "vit_l_16" else 224
    transforms = get_transforms(dataset_name,resize)

    # Prepare data loaders
    data_loaders = get_dataloaders(
        dataset_name=dataset_name,
        root_dir=root_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        transforms=transforms,
        resize = resize
    )

    # Set up criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    trial_id = session.get_trial_id()
    checkpoint_dir = os.path.join(os.getcwd(), f"checkpoint_{trial_id}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Training loop
    best_validation_metric = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for batch in data_loaders['train']:
            if dataset_name=="iwildcam":
                inputs, labels, metadata = batch
            else:
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
                if dataset_name=="iwildcam":
                    inputs, labels, metadata = batch
                else:
                    inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute macro F1 score
        validation_f1 = f1_score(all_labels, all_preds, average='macro')
        validation_accuracy = accuracy_score(all_labels, all_preds)
        metrics = {
            "validation_f1": validation_f1,
            "validation_accuracy": validation_accuracy
        }
        validation_metric = validation_f1 if dataset_name=="iwildcam" else validation_accuracy
        # Save checkpoint if this is the best model so far
        if validation_metric > best_validation_metric:
            best_validation_metric = validation_metric
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            session.report(metrics, checkpoint=checkpoint)
            
        else:
            session.report(metrics)


def main():
    num_finetune_samples = 16
    num_lp_sample = 8
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning script')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to train')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--config_dir', type=str, default='deep-duo/configs', help='Directory of the configuration files')
    parser.add_argument('--skip_lp', type=bool, default=False, help='Directory of the configuration files')
    args = parser.parse_args()

    # Load configurations
    dataset_config_path = os.path.join(args.config_dir, 'dataset_config.yaml')
    training_config_path = os.path.join(args.config_dir, f'training_config_{args.dataset_name}.yaml')

    dataset_config = load_config(dataset_config_path)
    training_config = load_config(training_config_path)

    model_name = args.model_name

    dataset_name = args.dataset_name
    dataset_params = next(
        (item for item in dataset_config['datasets'] if item['name'].lower() == dataset_name.lower()), None)
    if not dataset_params:
        raise ValueError(f"Dataset '{dataset_name}' not found in dataset_config.yaml")

    validation_metric = "validation_f1" if dataset_name=="iwildcam" else "validation_accuracy"
    num_classes = dataset_params['num_classes']
    root_dir = os.path.expandvars(dataset_params['root_dir'])

    # Access training-specific configurations
    training_params = training_config.get('training', {})
    batch_size = training_params.get('batch_size', 64)
    num_epochs_lp = training_params.get('num_epochs_lp', 1)
    num_workers = training_params.get('num_workers', 4)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (not args.skip_lp):
        ray.init(
            runtime_env={"working_dir": os.path.dirname(os.path.dirname(os.path.abspath(__file__)))},
            log_to_driver=False
        )
        
        config_lp = {
            'model_name': model_name,
            'learning_rate': tune.loguniform(1e-4, 1e-2),
            'weight_decay': tune.loguniform(1e-7, 1e-4),
            'batch_size': batch_size,
            'num_epochs': num_epochs_lp,
            'num_workers': num_workers,
        }

        scheduler = ASHAScheduler(
            max_t=num_epochs_lp,
            grace_period=2,
            reduction_factor=3,
            metric=validation_metric,
            mode='max'
        )

        reporter = CLIReporter(
            parameter_columns=['learning_rate', 'weight_decay'],
            metric_columns=[validation_metric, 'training_iteration'],
            max_report_frequency=900
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
            config=config_lp,
            num_samples=num_lp_sample,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir='ray_results',
            keep_checkpoints_num=1,
            fail_fast=False
        )

        best_trial = result.get_best_trial(validation_metric, 'max', 'last')
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation metric: {best_trial.last_result[validation_metric]}")
        

        best_checkpoint = result.get_best_checkpoint(
            trial=best_trial,
            metric=validation_metric,
            mode='max'
        )
        
        # Access the checkpoint directory
        with best_checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            checkpoint_data = torch.load(checkpoint_path)
            model_state = checkpoint_data['model_state_dict']
        
        best_config_saved = best_trial.config
        hyper_results_dir = f"hyperparams/{dataset_name}"
        os.makedirs(hyper_results_dir, exist_ok=True)
        hyper_csv_path = os.path.join(hyper_results_dir, "tuning_results.csv")
        
        save_results_csv(
            model_name=best_config_saved["model_name"],
            phase="linear_probing",
            learning_rate=best_config_saved["learning_rate"],
            weight_decay=best_config_saved["weight_decay"],
            batch_size=best_config_saved["batch_size"],
            epoch=best_config_saved["num_epochs"],
            best_validation_metric=best_trial.last_result[validation_metric],
            filename=hyper_csv_path
        )

        best_model = create_model(model_name=best_trial.config['model_name'], num_classes=num_classes)
        best_model.load_state_dict(model_state)
        best_model.to(device)

        # Save the best model
        os.makedirs('checkpoints', exist_ok=True)
        model_save_path = f'checkpoints/best_lp_model_{model_name}_{dataset_name}.pth'
        torch.save(best_model.state_dict(), model_save_path)
        print(f"Best lp model saved to '{model_save_path}'")
    
    # Fully Finetune Phase
    ###########===============##YumiYumi#XiguaXigua#############
    
    finetune_config = {
        'model_name': model_name,
        'learning_rate': tune.loguniform(1e-6, 1e-4),
        'weight_decay': tune.loguniform(1e-8, 1e-5),
        'batch_size': batch_size,
        'num_epochs': training_params.get('num_epochs_ff', 1),
        'num_workers': num_workers,
    }
    finetune_scheduler = ASHAScheduler(
        max_t=finetune_config['num_epochs'],
        grace_period=1,
        reduction_factor=2,
        metric=validation_metric,
        mode='max'
    )
    finetune_result = tune.run(
        tune.with_parameters(
            train_model,
            num_classes=num_classes,
            root_dir=root_dir,
            dataset_name=dataset_name,
            device=device,
            model_state=model_state,
            finetune=True
        ),
        resources_per_trial={'cpu': num_workers, 'gpu': 1 if torch.cuda.is_available() else 0},
        config=finetune_config,
        num_samples=num_finetune_samples,
        scheduler=finetune_scheduler,
        progress_reporter=reporter,
        local_dir='ray_results',
        keep_checkpoints_num=1,
        fail_fast=False
    )
    best_finetune_trial = finetune_result.get_best_trial(validation_metric, 'max', 'last')
    print(f"Best finetune trial config: {best_finetune_trial.config}")
    print(f"Best finetune trial final validation metric: {best_finetune_trial.last_result[validation_metric]}")

    best_finetune_checkpoint = finetune_result.get_best_checkpoint(
        trial=best_finetune_trial,
        metric=validation_metric,
        mode='max'
    )
    
    best_config_saved = best_finetune_trial.config
    save_results_csv(
        model_name=best_config_saved["model_name"],
        phase="fully_finetuned",
        learning_rate=best_config_saved["learning_rate"],
        weight_decay=best_config_saved["weight_decay"],
        batch_size=best_config_saved["batch_size"],
        epoch=best_config_saved["num_epochs"],
        best_validation_metric=best_finetune_trial.last_result[validation_metric],
        filename=hyper_csv_path
    )


    
    with best_finetune_checkpoint.as_directory() as checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        checkpoint_data = torch.load(checkpoint_path)
        model_state = checkpoint_data['model_state_dict']

    best_finetune_model = create_model(model_name=best_finetune_trial.config['model_name'], num_classes=num_classes)
    best_finetune_model.load_state_dict(model_state)
    best_finetune_model.to(device)

    # Save the best finetuned model
    finetune_model_save_path = f'checkpoints/best_ff_model_{model_name}_{dataset_name}.pth'
    torch.save(best_finetune_model.state_dict(), finetune_model_save_path)
    print(f"Best finetuned model saved to '{finetune_model_save_path}'")


if __name__ == '__main__':
    main()
