import torch
import torch.nn as nn
from torchvision import models
import open_clip

def freeze_model_layers(model):
    """Helper function to freeze all parameters of a model."""
    for param in model.parameters():
        param.requires_grad = False

def create_model(model_name, num_classes=0, freeze_layers=True, use_imagenet=False, get_transform=False, weight=None):
    """
    Creates a model from torchvision with specified pretrained weights,
    optionally freezes all layers, and replaces the classifier/head
    with a new Linear layer that outputs `num_classes`.
    """
    model_name = model_name.lower()
    weight_name = weight if weight!=None else "IMAGENET1K_V1"

    # ---------------------------
    # MobileNet V3 Large
    # ---------------------------
    if model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
        transform = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()


    # ---------------------------
    # MNASNet 1.3
    # ---------------------------
    elif model_name == 'mnasnet1_3':
        model = models.mnasnet1_3(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.MNASNet1_3_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # ShuffleNet V2 x2.0
    # ---------------------------
    elif model_name == 'shufflenet_v2_x2_0':
        model = models.shufflenet_v2_x2_0(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        transform = models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1.transforms()
        
    # ---------------------------
    # EfficientNet B0
    # ---------------------------
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    
    # ---------------------------
    # EfficientNet B1
    # ---------------------------
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.EfficientNet_B1_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # EfficientNet B2
    # ---------------------------
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # EfficientNet B3
    # ---------------------------
    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.EfficientNet_B3_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # EfficientNet B4
    # ---------------------------
    elif model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.EfficientNet_B4_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # EfficientNet B5
    # ---------------------------
    elif model_name == 'efficientnet_b5':
        model = models.efficientnet_b5(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.EfficientNet_B5_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # ConvNeXt Tiny
    # ---------------------------
    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)
        transform = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # Swin Transformer (Tiny)
    # ---------------------------
    elif model_name == 'swin_t':
        model = models.swin_t(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        transform = models.Swin_T_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # MaxViT Tiny
    # ---------------------------
    elif model_name == 'maxvit_t':
        model = models.maxvit_t(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[5].in_features
            model.classifier[5] = nn.Linear(in_features, num_classes)
        transform = models.MaxVit_T_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # Swin V2 T
    # ---------------------------
    elif model_name == 'swin_v2_t':
        model = models.swin_v2_t(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        transform = models.Swin_V2_T_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # ConvNeXt Small
    # ---------------------------
    elif model_name == 'convnext_small':
        model = models.convnext_small(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)
        transform = models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # Swin V2 Small
    # ---------------------------
    elif model_name == 'swin_v2_s':
        model = models.swin_v2_s(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        transform = models.Swin_V2_S_Weights.IMAGENET1K_V1.transforms()
    # ---------------------------
    # Swin V2 Base
    # ---------------------------
    elif model_name == 'swin_v2_b':
        model = models.swin_v2_b(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        transform = models.Swin_V2_B_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # EfficientNet V2 S
    # ---------------------------
    elif model_name == 'efficientnet_v2_s':
        model = models.efficientnet_v2_s(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()

    
    # ---------------------------
    # EfficientNet V2 M
    # ---------------------------
    elif model_name == 'efficientnet_v2_m':
        model = models.efficientnet_v2_m(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # ConvNeXt Base
    # ---------------------------
    elif model_name == 'convnext_base':
        model = models.convnext_base(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)
        transform = models.ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # ConvNeXt Large
    # ---------------------------
    elif model_name == 'convnext_large':
        model = models.convnext_large(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)
        transform = models.ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms()

    # ---------------------------
    # ViT Base 16
    # ---------------------------
    elif model_name == 'vit_b_16':
        if weight==None:
            weight_name = "IMAGENET1K_SWAG_E2E_V1"
            print("WARNING: weight not specified for ViT, default to E2E, GFLOPS might be incorrect.")
        model = models.vit_b_16(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        # ViT has a "heads" attribute: model.heads.head
        if not use_imagenet:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        if weight=="IMAGENET1K_SWAG_E2E_V1":
            transform = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        elif weight=="IMAGENET1K_SWAG_LINEAR_V1":
            transform = models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()

    # ---------------------------
    # EfficientNet V2 L
    # ---------------------------
    elif model_name == 'efficientnet_v2_l':
        model = models.efficientnet_v2_l(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        transform = models.EfficientNet_V2_L_Weights.IMAGENET1K_V1.transforms()


    # ---------------------------
    # ViT Large 16
    # ---------------------------
    elif model_name == 'vit_l_16':
        if weight==None:
            weight_name = "IMAGENET1K_SWAG_E2E_V1"
            print("WARNING: weight not specified for ViT, default to E2E, GFLOPS might be incorrect.")
        model = models.vit_l_16(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        if weight=="IMAGENET1K_SWAG_E2E_V1":
            transform = models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        elif weight=="IMAGENET1K_SWAG_LINEAR_V1":
            transform = models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()
    # ---------------------------
    # ViT Huge 14
    # ---------------------------
    elif model_name == 'vit_h_14':
        if weight==None:
            weight_name = "IMAGENET1K_SWAG_LINEAR_V1"
            print("WARNING: weight not specified for ViT, default to LP, GFLOPS might be incorrect.")
        model = models.vit_h_14(weights=weight_name)
        if freeze_layers:
            freeze_model_layers(model)
        if not use_imagenet:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        if weight=="IMAGENET1K_SWAG_E2E_V1":
            transform = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        elif weight=="IMAGENET1K_SWAG_LINEAR_V1":
            transform = models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.transforms()

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

    if get_transform:
        return model, transform
    return model
