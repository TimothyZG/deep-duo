import os
import torch
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

def evaluate_model(model, dataloader, criterion, device, dataset_name, model_name, eval_type,temp_scaled=False):
    """
    Evaluate the model on a given dataloader and save results.

    Args:
    - model: The PyTorch model being evaluated.
    - dataloader: The dataloader for the dataset being evaluated.
    - criterion: Loss function.
    - device: Device to run the evaluation on (CPU/GPU).
    - dataset_name: Name of the dataset.
    - model_name: Name of the model being evaluated.
    - eval_type: Type of evaluation ('ind' for in-distribution, 'ood' for out-of-distribution).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []

    # Create directory for saving results if it doesn't exist
    if dataset_name=="iwildcam":
        pred_folder_path = f"./{dataset_name}-{eval_type}"
    else:
        pred_folder_path = f"./{dataset_name}"
    if temp_scaled:
        pred_folder_path = f"{pred_folder_path}_tp"
    target_folder_path = pred_folder_path
    os.makedirs(pred_folder_path, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, mininterval=6.0):
            if dataset_name=="iwildcam":
                data, targets, metadata = batch
            else:
                data, targets = batch
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * data.size(0)

            # Save logits (as an n x k matrix, where n = number of samples, k = number of classes)
            all_logits.extend(outputs.cpu().numpy())

            # Save predictions and labels
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)
    if dataset_name.lower() == 'iwildcam':
        total_metric = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        total_metric = accuracy_score(all_labels, all_preds)

    print(f'{eval_type.upper()} Test Loss: {total_loss:.4f}, Test Metric: {total_metric:.4f}')

    # Convert logits (n samples x k classes) and targets (n samples) to DataFrames
    logits_df = pd.DataFrame(all_logits)  # Each row is a sample, each column is a class logit

    # Save logits and targets to separate CSV files with 'ind' or 'ood' in the file name
    logits_csv_file_path = os.path.join(pred_folder_path, f"{model_name}_logits.csv")
    targets_csv_file_path = os.path.join(target_folder_path, f"{eval_type}_targets.csv")

    logits_df.to_csv(logits_csv_file_path, index=False)
    print(f"Prediction saved to {logits_csv_file_path}")
    
    if os.path.exists(targets_csv_file_path):
        print("Target Files already exists.")
    else:
        targets_df = pd.DataFrame(all_labels, columns=['targets'])  # Targets in a separate DataFrame
        targets_df.to_csv(targets_csv_file_path, index=False)

    return total_loss, total_metric
