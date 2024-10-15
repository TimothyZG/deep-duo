import os
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, dataset_name, root_dir, download=True):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.download = download
        self.dataset = get_dataset(dataset=self.dataset_name, root_dir=self.root_dir, download=self.download)
        
    def get_splits(self):
        raise NotImplementedError("Subclasses should implement this method.")

class IWildCamDataset(BaseDataset):
    def __init__(self, root_dir, download=True):
        super().__init__('iwildcam', root_dir, download)

    def get_splits(self, transforms, batch_size, num_workers):
        # Get the data subsets
        train_data = self.dataset.get_subset('train', transform=transforms['train'])
        val_data = self.dataset.get_subset('id_val', transform=transforms['val'])
        test_data = self.dataset.get_subset('id_test', transform=transforms['test'])
        ood_test_data = self.dataset.get_subset('test', transform=transforms['test'])

        # Create data loaders
        train_loader = get_train_loader('standard', train_data, batch_size=batch_size, num_workers=num_workers)
        val_loader = get_eval_loader('standard', val_data, batch_size=batch_size, num_workers=num_workers)
        test_loader = get_eval_loader('standard', test_data, batch_size=batch_size, num_workers=num_workers)
        ood_test_loader = get_eval_loader('standard', ood_test_data, batch_size=batch_size, num_workers=num_workers)

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'ood_test': ood_test_loader
        }

class RxRx1Dataset(BaseDataset):
    def __init__(self, root_dir, download=True):
        super().__init__('rxrx1', root_dir, download)

    def get_splits(self, transforms, batch_size, num_workers):
        # Get the data subsets
        train_data = self.dataset.get_subset('train', transform=transforms['train'])
        val_data = self.dataset.get_subset('val', transform=transforms['val'])
        test_data = self.dataset.get_subset('id_test', transform=transforms['test'])
        ood_test_data = self.dataset.get_subset('test', transform=transforms['test'])

        # Create data loaders
        train_loader = get_train_loader('standard', train_data, batch_size=batch_size, num_workers=num_workers)
        val_loader = get_eval_loader('standard', val_data, batch_size=batch_size, num_workers=num_workers)
        test_loader = get_eval_loader('standard', test_data, batch_size=batch_size, num_workers=num_workers)
        ood_test_loader = get_eval_loader('standard', ood_test_data, batch_size=batch_size, num_workers=num_workers)

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'ood_test': ood_test_loader
        }
