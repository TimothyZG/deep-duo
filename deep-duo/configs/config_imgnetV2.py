dataset = "ImgNetV2"
num_class = 1000

logit_pred_dir = f"{dataset}"
sv_pred_dir = f"pred/softvote-{dataset}"
cd_pred_dir = f"pred/confident-{dataset}"
dd_pred_dir = f"pred/dictatorial-{dataset}"
unc_dir = f"pred/uncertainty-{dataset}"
backbone_csv_path = f"backbones_selected.csv"

target_dir = f"{dataset}"

large_model_ls = ["ViT_H_14-IMAGENET1K_SWAG_LINEAR_V1",
                  "ConvNeXt_Large-IMAGENET1K_V1",
                  "EfficientNet_V2_L-IMAGENET1K_V1",
                  "Swin_V2_B-IMAGENET1K_V1"
                ]
