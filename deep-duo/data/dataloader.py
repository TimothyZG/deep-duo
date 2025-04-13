from .datasets import IWildCamDataset, RxRx1Dataset
from .transforms import get_transforms
from torchvision.datasets import Caltech256
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torch

def get_dataloaders(dataset_name, root_dir, batch_size, num_workers, transforms, resize=224):
    
    # Initialize dataset and get data loaders
    if dataset_name.lower() == 'iwildcam':
        dataset = IWildCamDataset(root_dir=root_dir)
    elif dataset_name.lower() == 'rxrx1':
        dataset = RxRx1Dataset(root_dir=root_dir)
    elif dataset_name.lower() == 'caltech256':
        return get_caltech256_dataloaders(root_dir, batch_size, num_workers, resize)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    data_loaders = dataset.get_splits(
        transforms=transforms,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return data_loaders


def get_caltech256_dataloaders(root_dir, batch_size, num_workers, resize=224):
    import os
    print(f"{root_dir=}")
    print("Files in root_dir:")
    print(os.listdir(root_dir))

    print("\nFiles in parent_dir:")
    print(os.listdir(os.path.dirname(root_dir)))
    transform_pipeline = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),  # Ensure image is in RGB mode
        T.Resize((resize, resize)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean and std
    ])
    
    # Load the entire Caltech256 dataset
    dataset = Caltech256(root=root_dir, transform=transform_pipeline, download=False)
    
    # Define dataset sizes for train, val, and test (e.g., 70% train, 15% val, 15% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Split the dataset
    seed = 42
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    # Create DataLoaders for each split
    data_loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    }
    
    return data_loaders

def collate_fn(batch):
    # batch is a list of tuples: (image, label)
    pixel_values = torch.stack([x[0] for x in batch])  # x[0] for image
    labels = torch.tensor([x[1] for x in batch])  # x[1] for label
    return {
        'pixel_values': pixel_values,
        'labels': labels
    }
    
    