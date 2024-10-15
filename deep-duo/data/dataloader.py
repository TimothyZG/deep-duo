from .datasets import IWildCamDataset, RxRx1Dataset
from .transforms import get_transforms

def get_dataloaders(dataset_name, root_dir, batch_size, num_workers, transforms, resize=224):
    
    # Initialize dataset and get data loaders
    if dataset_name.lower() == 'iwildcam':
        dataset = IWildCamDataset(root_dir=root_dir)
    elif dataset_name.lower() == 'rxrx1':
        dataset = RxRx1Dataset(root_dir=root_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    data_loaders = dataset.get_splits(
        transforms=transforms,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return data_loaders
