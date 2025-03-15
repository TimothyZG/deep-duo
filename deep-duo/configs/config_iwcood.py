dataset = "iwildcam-ood"
num_class = 182

logit_pred_dir = f"{dataset}"
sv_pred_dir = f"pred/softvote-{dataset}"
cd_pred_dir = f"pred/confident-{dataset}"
dd_pred_dir = f"pred/dictatorial-{dataset}"
unc_dir = f"pred/uncertainty-{dataset}"
backbone_csv_path = f"backbones_selected.csv"

target_dir = f"{dataset}"


large_model_ls = ["convnext_base-IMAGENET1K_V1",
                #   "convnext_large-IMAGENET1K_V1",
                #   "convnext_small-IMAGENET1K_V1",
                  "swin_v2_s-IMAGENET1K_V1",
                #   "swin_v2_b-IMAGENET1K_V1",
                  ]