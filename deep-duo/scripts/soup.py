import argparse
import os
import torch
import sys
import glob
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mf.model_factory import create_model
from data.dataloader import get_dataloaders
from trainers.evaluator import evaluate_model
from trainers.logger import Logger
from utils.config import load_config

def average_state_dicts(state_dicts):
    avg_state = {}
    for key in state_dicts[0]:
        if state_dicts[0][key].dtype in (torch.float32, torch.float64, torch.float16):
            avg_state[key] = sum(sd[key].float() for sd in state_dicts) / len(state_dicts)
        else:
            # Just copy the value from the first model (e.g., integers like num_batches_tracked)
            avg_state[key] = state_dicts[0][key].clone()
    return avg_state

parser = argparse.ArgumentParser(description='Evaluation script')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to evaluate')
parser.add_argument('--config_dir', type=str, default='deep-duo/config', help='Directory of the configuration files')
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to use')
args = parser.parse_args()

dataset_config_path = os.path.join(args.config_dir, 'dataset_config.yaml')
training_config_path = os.path.join(args.config_dir, f'training_config_{args.dataset_name}.yaml')

dataset_config = load_config(dataset_config_path)
training_config = load_config(training_config_path)
model_name = args.model_name
dataset_name = args.dataset_name
dataset_params = next((item for item in dataset_config['datasets'] if item['name'].lower() == dataset_name.lower()), None)
if not dataset_params:
    raise ValueError(f"Dataset '{dataset_name}' not found in dataset_config.yaml")

num_classes = dataset_params['num_classes']
root_dir = os.path.expandvars(dataset_params['root_dir'])
training_params = training_config.get('training', {})
batch_size = training_params.get('batch_size', 32)
num_workers = training_params.get('num_workers', 4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_dir = f"checkpoints/{dataset_name}"
ingredients_ls = sorted(glob.glob(f"{checkpoint_dir}/{model_name}_trial*.pth"))
print(f"{ingredients_ls=}")

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

criterion = torch.nn.CrossEntropyLoss()

# ---------- UNIFORM SOUP ----------
print("Building Uniform Soup...")
uniform_models = []
for ckpt in ingredients_ls:
    model = create_model(model_name=args.model_name, num_classes=num_classes).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    uniform_models.append(copy.deepcopy(model.state_dict()))

# Average the weights
uniform_soup_state = average_state_dicts(uniform_models)

# Load into a model
uniform_soup = create_model(args.model_name, num_classes=num_classes).to(device)
uniform_soup.load_state_dict(uniform_soup_state)
uniform_soup.eval()

# Evaluate Uniform Soup
print("Evaluating Uniform Soup on In-Distribution Test Set:")
test_loss_ind, test_metric_ind = evaluate_model(
    model=uniform_soup,
    dataloader=data_loaders['test'],
    criterion=criterion,
    device=device,
    dataset_name=dataset_name,
    model_name=model_name+"uniform_soup",
    eval_type='ind'
)

print(f"[Uniform Soup] InD Test Loss: {test_loss_ind:.4f}, F1: {test_metric_ind:.4f}")

# ---------- GREEDY SOUP ----------
print("Building Greedy Soup...")
greedy_ckpts = []
greedy_state_dicts = []
best_val_metric = -1

for ckpt in ingredients_ls:
    temp_model = create_model(model_name=args.model_name, num_classes=num_classes).to(device)
    state = torch.load(ckpt, map_location=device)
    temp_model.load_state_dict(state)
    temp_model.eval()

    # If this is the first one, add it
    if not greedy_ckpts:
        greedy_ckpts.append(ckpt)
        greedy_state_dicts.append(state)
        best_val_metric = evaluate_model(
            model=temp_model,
            dataloader=data_loaders['val'],
            criterion=criterion,
            device=device,
            dataset_name=dataset_name,
            model_name=model_name+"greedy_soup",
            eval_type='val'
        )[1]
        continue

    # Try adding this one to the greedy soup
    candidate_state_dicts = greedy_state_dicts + [state]
    avg_state = average_state_dicts(candidate_state_dicts)

    candidate_model = create_model(model_name=args.model_name, num_classes=num_classes).to(device)
    candidate_model.load_state_dict(avg_state)
    candidate_model.eval()

    _, candidate_val_metric = evaluate_model(
        model=candidate_model,
        dataloader=data_loaders['val'],
        criterion=criterion,
        device=device,
        dataset_name=dataset_name,
        model_name=model_name+"candidate_greedy",
        eval_type='val'
    )

    if candidate_val_metric > best_val_metric:
        print(f"Adding {ckpt} to greedy soup (val metric improved to {candidate_val_metric:.4f})")
        greedy_ckpts.append(ckpt)
        greedy_state_dicts = candidate_state_dicts
        best_val_metric = candidate_val_metric
    else:
        print(f"Skipping {ckpt} (val metric would drop to {candidate_val_metric:.4f})")

# Final greedy soup
greedy_soup_state = average_state_dicts(greedy_state_dicts)
greedy_soup = create_model(args.model_name, num_classes=num_classes).to(device)
greedy_soup.load_state_dict(greedy_soup_state)
greedy_soup.eval()

# Evaluate Greedy Soup
print("Evaluating Greedy Soup on In-Distribution Test Set:")
test_loss_gs, test_metric_gs = evaluate_model(
    model=greedy_soup,
    dataloader=data_loaders['test'],
    criterion=criterion,
    device=device,
    dataset_name=dataset_name,
    model_name=model_name+"greedy_soup",
    eval_type='ind'
)

print(f"[Greedy Soup] InD Test Loss: {test_loss_gs:.4f}, F1: {test_metric_gs:.4f}")

# ---------- Logging ----------
logger = Logger(project_name='deep-duo-eval', config={
    'model_name': model_name,
    'dataset_name': dataset_name,
})

logger.log({
    'uniform_soup_f1': test_metric_ind,
    'greedy_soup_f1': test_metric_gs,
})
logger.finish()
