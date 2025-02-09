import argparse
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mf.model_factory import create_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate ImageNet logits for different models')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Path to the ImageNet validation set (organized for ImageFolder)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
args = parser.parse_args()

# make "./ImgNet" folder if not exist yet
output_folder = "./ImgNet"
os.makedirs(output_folder, exist_ok=True)

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageNet(args.data_dir, split="val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

target_df = pd.DataFrame(val_dataset.samples, columns=['filepath', 'label'])
target_csv_path = os.path.join(output_folder, "target.csv")
target_df.to_csv(target_csv_path, index=False)
print(f"Saved targets to {target_csv_path}")

model_ls = ["mobilenet_v3_large",
            "mnasnet1_3",
            "shufflenet_v2_x2_0",
            "efficientnet_b1",
            "efficientnet_b2",
            "convnext_tiny",
            "swin_t",
            "maxvit_t",
            "convnext_small",
            "swin_v2_s",
            "convnext_base",
            ]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model in model_ls:
    print(f"Processing model: {model}")
    try:
        # Create and move model to device; set it to eval mode
        curr_model = create_model(model,use_imagenet=True)
        curr_model.to(device)
        curr_model.eval()

        logits_list = []
        with torch.no_grad():
            for images, _ in val_loader:  # We don't need labels here
                images = images.to(device)
                outputs = curr_model(images)  # outputs are logits (no softmax applied)
                logits_list.append(outputs.cpu())

        # Concatenate all batch outputs (shape: [num_samples, num_classes])
        logits_tensor = torch.cat(logits_list, dim=0)
        df_logits = pd.DataFrame(logits_tensor.numpy())

        logits_csv_path = os.path.join(output_folder, f"{model}_logits.csv")
        df_logits.to_csv(logits_csv_path, index=False)
        print(f"Saved logits for {model} to {logits_csv_path}")
    except Exception as e:
        print(f"Error processing model {model}: {e}")