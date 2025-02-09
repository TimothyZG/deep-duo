import torch
import torch.nn as nn
from torchvision import models
import open_clip

def freeze_model_layers(model):
    """Helper function to freeze all parameters of a model."""
    for param in model.parameters():
        param.requires_grad = False

def create_model(model_name, num_classes=0, freeze_layers=True, use_imagenet=False):
    """
    Creates a model from torchvision with specified pretrained weights,
    optionally freezes all layers, and replaces the classifier/head
    with a new Linear layer that outputs `num_classes`.
    """
    model_name = model_name.lower()

    # ---------------------------
    # MobileNet V3 Large
    # ---------------------------
    if model_name == 'mobilenet_v3_large':
        from torchvision.models import MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # MNASNet 1.3
    # ---------------------------
    elif model_name == 'mnasnet1_3':
        from torchvision.models import MNASNet1_3_Weights
        weights = MNASNet1_3_Weights.IMAGENET1K_V1
        model = models.mnasnet1_3(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # ShuffleNet V2 x2.0
    # ---------------------------
    elif model_name == 'shufflenet_v2_x2_0':
        from torchvision.models import ShuffleNet_V2_X2_0_Weights
        weights = ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1
        model = models.shufflenet_v2_x2_0(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

    # ---------------------------
    # EfficientNet B1
    # ---------------------------
    elif model_name == 'efficientnet_b1':
        from torchvision.models import EfficientNet_B1_Weights
        weights = EfficientNet_B1_Weights.IMAGENET1K_V1
        model = models.efficientnet_b1(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # EfficientNet B2
    # ---------------------------
    elif model_name == 'efficientnet_b2':
        from torchvision.models import EfficientNet_B2_Weights
        weights = EfficientNet_B2_Weights.IMAGENET1K_V1
        model = models.efficientnet_b2(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            
    # ---------------------------
    # EfficientNet B3
    # ---------------------------
    elif model_name == 'efficientnet_b3':
        from torchvision.models import EfficientNet_B3_Weights
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
            
    # ---------------------------
    # EfficientNet B4
    # ---------------------------
    elif model_name == 'efficientnet_b4':
        from torchvision.models import EfficientNet_B4_Weights
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # ConvNeXt Tiny
    # ---------------------------
    elif model_name == 'convnext_tiny':
        from torchvision.models import ConvNeXt_Tiny_Weights
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # Swin Transformer (Tiny)
    # ---------------------------
    elif model_name == 'swin_t':
        from torchvision.models import Swin_T_Weights
        weights = Swin_T_Weights.IMAGENET1K_V1
        model = models.swin_t(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)

    # ---------------------------
    # MaxViT Tiny
    # ---------------------------
    elif model_name == 'maxvit_t':
        from torchvision.models import MaxVit_T_Weights
        weights = MaxVit_T_Weights.IMAGENET1K_V1
        model = models.maxvit_t(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[5].in_features
            model.classifier[5] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # Swin V2 T
    # ---------------------------
    elif model_name == 'swin_v2_t':
        from torchvision.models import Swin_V2_T_Weights
        weights = Swin_V2_T_Weights.IMAGENET1K_V1
        model = models.swin_v2_t(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
            
    # ---------------------------
    # ConvNeXt Small
    # ---------------------------
    elif model_name == 'convnext_small':
        from torchvision.models import ConvNeXt_Small_Weights
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = models.convnext_small(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # Swin V2 Small
    # ---------------------------
    elif model_name == 'swin_v2_s':
        from torchvision.models import Swin_V2_S_Weights
        weights = Swin_V2_S_Weights.IMAGENET1K_V1
        model = models.swin_v2_s(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        
    # ---------------------------
    # Swin V2 Base
    # ---------------------------
    elif model_name == 'swin_v2_b':
        from torchvision.models import Swin_V2_B_Weights
        weights = Swin_V2_B_Weights.IMAGENET1K_V1
        model = models.swin_v2_b(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)

    # ---------------------------
    # EfficientNet V2 M
    # ---------------------------
    elif model_name == 'efficientnet_v2_m':
        from torchvision.models import EfficientNet_V2_M_Weights
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_m(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # ConvNeXt Base
    # ---------------------------
    elif model_name == 'convnext_base':
        from torchvision.models import ConvNeXt_Base_Weights
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        model = models.convnext_base(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)
        
    # ---------------------------
    # ConvNeXt Large
    # ---------------------------
    elif model_name == 'convnext_large':
        from torchvision.models import ConvNeXt_Large_Weights
        weights = ConvNeXt_Large_Weights.IMAGENET1K_V1
        model = models.convnext_large(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # ViT Base 16
    # ---------------------------
    elif model_name == 'vit_b_16':
        from torchvision.models import ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        model = models.vit_b_16(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        # ViT has a "heads" attribute: model.heads.head
        if not use_imagenet:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)

    # ---------------------------
    # EfficientNet V2 L
    # ---------------------------
    elif model_name == 'efficientnet_v2_l':
        from torchvision.models import EfficientNet_V2_L_Weights
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_l(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

    # ---------------------------
    # ViT Large 16
    # ---------------------------
    elif model_name == 'vit_l_16':
        from torchvision.models import ViT_L_16_Weights
        weights = ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
        model = models.vit_l_16(weights=weights)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)

    elif model_name == 'vit_base_clip':
        import timm
        model = timm.create_model(
            "vit_base_patch16_clip_224.laion2b", 
            pretrained=True,  # Download and load pretrained weights
            num_classes=num_classes  # Immediately sets up the final layer for our task
        )
        
        # Optionally freeze all layers except the final classifier
        if freeze_layers:
            for name, param in model.named_parameters():
                if "head" not in name:  # 'head' is typically timm's classification layer
                    param.requires_grad = False


    else:
        raise ValueError(f"Model '{model_name}' is not supported or not implemented.")

    return model
