test_ls = [""]

dataset = "ImgNet"
num_class = 1000

logit_pred_dir = f"{dataset}"
sv_pred_dir = f"pred/softvote-{dataset}"
cd_pred_dir = f"pred/confident-{dataset}"
dd_pred_dir = f"pred/dictatorial-{dataset}"
unc_dir = f"pred/uncertainty-{dataset}"
backbone_csv_path = f"backbones-iwildcam.csv"

target_dir = f"{dataset}"