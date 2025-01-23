import torch
import torch.nn as nn
from torchvision import models

def create_model(model_name, num_classes, freeze_layers=True):
    model_name = model_name.lower()
    
    if model_name == 'densenet_121':
        from torchvision.models import DenseNet121_Weights
        weights = DenseNet121_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.densenet121(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    elif model_name == 'efficientnetv2_s':
        from torchvision.models import EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.efficientnet_v2_s(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    elif model_name == 'efficientnetv2_m':
        from torchvision.models import EfficientNet_V2_M_Weights
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.efficientnet_v2_m(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    elif model_name == 'efficientnetv2_l':
        from torchvision.models import EfficientNet_V2_L_Weights
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.efficientnet_v2_l(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    elif model_name == 'maxvit_t':
        from torchvision.models import MaxVit_T_Weights
        weights = MaxVit_T_Weights.DEFAULT  # Only default weights available
        model = models.maxvit_t(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.block_channels[-1]  # Last block channels
        # Replace the classification head
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Pooling
            nn.Flatten(),            # Flatten into a vector
            nn.LayerNorm(num_features),  # LayerNorm for stability
            nn.Linear(num_features, num_features),  # Fully Connected Layer
            nn.Tanh(),               # Non-linearity
            nn.Linear(num_features, num_classes),  # Output Layer
        )

    elif model_name == 'mnasnet1_3':
        from torchvision.models import MNASNet1_3_Weights
        weights = MNASNet1_3_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.mnasnet1_3(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.classifier.in_features # Error
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),  # Keep the dropout layer
            nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes)
        )

    elif model_name == 'resnext50_32x4d':
        from torchvision.models import ResNeXt50_32X4D_Weights
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2  # V2 weights available
        model = models.resnext50_32x4d(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == 'resnext101_32x8d': # Cuda Out Of Memory Error
        from torchvision.models import ResNeXt101_32X8D_Weights
        weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V2  # V2 weights available
        model = models.resnext101_32x8d(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == 'shufflenetv2x2_0':
        from torchvision.models import ShuffleNet_V2_X2_0_Weights
        weights = ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.shufflenet_v2_x2_0(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif model_name == 'swinv2_b': # Cuda Ran Out of Memory Error
        from torchvision.models import Swin_V2_B_Weights
        weights = Swin_V2_B_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.swin_v2_b(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)

    elif model_name == 'swinv2_s':
        from torchvision.models import Swin_V2_S_Weights
        weights = Swin_V2_S_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.swin_v2_s(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)

    elif model_name == 'swinv2_t': # Image Error in DataLoader
        from torchvision.models import Swin_V2_T_Weights
        weights = Swin_V2_T_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.swin_v2_t(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, num_classes)

    elif model_name == 'vit_b_32':
        from torchvision.models import ViT_B_32_Weights
        weights = ViT_B_32_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.vit_b_32(weights=weights)
        num_features = model.heads.head.in_features
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        model.heads.head = nn.Linear(num_features, num_classes)

    elif model_name == 'vit_l_32':
        from torchvision.models import ViT_L_32_Weights
        weights = ViT_L_32_Weights.IMAGENET1K_V1  # Only V1 available
        model = models.vit_l_32(weights=weights)
        num_features = model.heads.head.in_features
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        model.heads.head = nn.Linear(num_features, num_classes)

    elif model_name == 'wideresnet101':
        from torchvision.models import Wide_ResNet101_2_Weights
        weights = Wide_ResNet101_2_Weights.IMAGENET1K_V2  # V2 weights available
        model = models.wide_resnet101_2(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    # Added ResNet50
    elif model_name == 'resnet50':
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    # Added ResNet18
    elif model_name == 'resnet18':
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    # Added ViT_L_16
    elif model_name == 'vit_l_16':
        from torchvision.models import ViT_L_16_Weights
        weights = ViT_L_16_Weights.IMAGENET1K_V1
        model = models.vit_l_16(weights=weights)
        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
        num_features = model.heads.head.in_features
        model.heads.head = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    return model
