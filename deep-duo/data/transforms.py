from torchvision import transforms
import torchvision.transforms.functional as F
import random

def get_transforms(dataset_name, resize=224):
    if dataset_name.lower() == 'rxrx1':
        # Specific transforms for RxRx1
        def random_rotation(image):
            angles = [0, 90, 180, 270]
            angle = random.choice(angles)
            return F.rotate(image, angle)
        
        train_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.Lambda(random_rotation),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif dataset_name.lower() == 'iwildcam':
        # Training transforms for iWildCam with RandAugment
        train_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        # Default training transforms (if needed)
        train_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    # Validation and test transforms
    val_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform  # Usually same as validation transforms
    }
