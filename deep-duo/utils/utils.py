import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score
import seaborn as sns

def intersection(lst1, lst2):
    return [item for item in lst2 if item in set(lst1)]

def calc_entr_torch(P, device):
    P_tensor = torch.tensor(P.values, dtype=torch.float32, device=device)
    P_log_softmax = F.log_softmax(P_tensor, dim=1)
    P_softmax = torch.exp(P_log_softmax)  # More stable than softmax directly
    elem_wise_entr = -torch.sum(P_softmax * P_log_softmax, dim=1)
    return elem_wise_entr.cpu().numpy()  # Ensure CPU conversion

def calc_cross_entr_torch(P, Q, device):
    P_tensor = torch.tensor(P.values, dtype=torch.float32, device=device)
    Q_tensor = torch.tensor(Q.values, dtype=torch.float32, device=device)
    P_log_softmax = F.log_softmax(P_tensor, dim=1)
    P_softmax = torch.exp(P_log_softmax)
    Q_log_softmax = F.log_softmax(Q_tensor, dim=1)
    elem_wise_cross_entr = -(P_softmax*Q_log_softmax).sum(dim=1)
    return elem_wise_cross_entr.cpu().numpy()


def calc_kl_torch(P, Q, device):
    return calc_cross_entr_torch(P,Q,device) - calc_entr_torch(P,device)

def softvote(pred1, pred2):
    res = (pred1+pred2)/2
    return res

def softmax(P):
    P_tensor = torch.tensor(P.values, dtype=torch.float32)
    P_softmax = F.softmax(P_tensor, dim=1)
    return P_softmax.numpy()

def softmax_response_unc(P):
    P_softmax = softmax(P)
    P_softmax_response = np.max(P_softmax,axis=1)
    return 1-P_softmax_response
    
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def brier_score(y, p):
  return np.mean(np.power(p - y, 2))

def calibration(y, p_mean, num_bins=10):
  class_pred = np.argmax(p_mean, axis=1)
  conf = np.max(p_mean, axis=1)
  y = np.argmax(y, axis=1)
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
  esce = np.average(
      np.square(mean_conf - acc_tab),
      weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
  mce = np.max(np.absolute(mean_conf - acc_tab))
  return ece, esce, mce

def get_rank(unc_pred):
    return unc_pred.rank(method='first')

# Function to evaluate models and store metrics
def evaluate_models(predictions, label, prefix, testtype, metrics_dict, category, num_classes):
    for method in predictions:
        pred_col_name = f"pred_{method}"
        pred = pd.read_csv(f"{prefix}{testtype}_{method}.csv")
        label = pd.concat([label,pd.DataFrame({pred_col_name: pred.idxmax(axis=1)})],axis=1)
        label[pred_col_name] = label[pred_col_name].str.extract('(\d+)').astype(int)
        
        curr_acc = np.mean(label[pred_col_name]==label["target"])
        curr_f1 = f1_score(label["target"], label[pred_col_name], average='macro')
        curr_brier = brier_score(one_hot(np.array(label["target"]), num_classes), softmax(pred))
        curr_ece, curr_mce = calibration(one_hot(np.array(label["target"]), num_classes), softmax(pred))
        
        # Store metrics
        metrics_dict["Model"].append(method)
        metrics_dict["Test Set"].append(testtype)
        metrics_dict["Acc"].append(curr_acc)
        metrics_dict["F1"].append(curr_f1)
        metrics_dict["Brier"].append(curr_brier)
        metrics_dict["ECE"].append(curr_ece)
        metrics_dict["MCE"].append(curr_mce)
        metrics_dict["Predictor Category"].append(category)
    return metrics_dict

# Function to calculate AUROC
def calculate_auroc(pred_df, unc_df, pred_vec, target_vec, unc_vec):
    is_correct = pred_df[pred_vec] != pred_df[target_vec]
    fpr, tpr, _ = roc_curve(is_correct, unc_df)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def calculate_f1_acc_cov_auc_and_sac(pred_df, unc, pred_vec, target_vec, unc_vec, cov_range, sac_targets=[0.7,0.8,0.9,0.95,0.98,0.99],sfc_targets=[0.4,0.5,0.6,0.7,0.8,0.9,0.95]):
    """
    Calculate Coverage AUC for F1-Score, Accuracy, and SAC for specified accuracy levels.
    
    Parameters:
    - pred_df: DataFrame containing predictions and true targets.
    - unc_df: DataFrame containing uncertainty values.
    - pred_vec: Column name for predictions in pred_df.
    - target_vec: Column name for target labels in pred_df.
    - unc_vec: Column name for uncertainty in unc_df.
    - cov_range: List of coverage percentages (e.g., [10, 20, 30, ...]).
    - sac_targets: List of accuracy levels for SAC calculation (default: [0.95, 0.98]).
    
    Returns:
    - A dictionary with F1-Cov AUC, Acc-Cov AUC, and SAC results for each target.
    """
    # Get rank based on uncertainty
    rank = get_rank(unc)
    
    # Calculate coverage threshold values
    coverage_ls = cov_range / 100 * pred_df.shape[0]
    
    # Initialize lists for metrics
    f1_metrics = []
    acc_metrics = []
    sac_results = {f"SAC_{int(target * 100)}": None for target in sac_targets}  # Initialize SAC results with keys like "SAC_95"
    sfc_results = {f"SFC_{target}": None for target in sfc_targets}  # Initialize SAC results with keys like "SAC_95"
    pred_df = pred_df[[target_vec,pred_vec]]
    # Loop through coverage thresholds
    for cov in coverage_ls:
        # Filter predictions within the coverage threshold
        cov_pred = pred_df.loc[rank < cov]
        if cov_pred.empty:
            print(f"Empty DataFrame for coverage threshold: {cov}")
            print(rank.head())
            print(rank.min(), rank.max())
            continue
        
        # Calculate F1 and Accuracy
        f1=f1_score(cov_pred[target_vec], cov_pred[pred_vec], average='macro')
        f1_metrics.append(f1)
        accuracy = accuracy_score(cov_pred[target_vec], cov_pred[pred_vec])
        acc_metrics.append(accuracy)
        
        # Check SAC for each accuracy target
        for sac_target in sac_targets:
            sac_key = f"SAC_{int(sac_target * 100)}"
            if ((sac_results[sac_key] is None) or  (accuracy >= sac_target)):
                sac_results[sac_key] = cov/pred_df.shape[0]*100

         # Check SAC for each accuracy target
        for sfc_target in sfc_targets:
            sfc_key = f"SFC_{sfc_target}"
            if ((sfc_results[sfc_key] is None) or (f1 >= sfc_target)):
                sfc_results[sfc_key] = cov/pred_df.shape[0]*100
    
    # Calculate area under the curve for both metrics
    # f1_cov_auc = np.sum(f1_metrics)
    # acc_cov_auc = np.sum(acc_metrics)
    f1_cov_auc = np.trapz(f1_metrics, dx=0.5)  # Integrate using trapezoidal rule
    acc_cov_auc = np.trapz(acc_metrics, dx=0.5)
    
    # Combine results into a dictionary
    results = {
        "f1_cov_auc": f1_cov_auc,
        "acc_cov_auc": acc_cov_auc,
        **sac_results,  # Merge SAC results into the dictionary
        **sfc_results
    }
    
    return results


def add_uncertainty_measures(category, model, unc_ls, unc_des_ls, main_model, sec_model):
    if category=="Single Model":
        entropy_unc = f"Entr[{model}]"
        softmax_unc = f"SR[{model}]"
        unc_ls.extend([entropy_unc, softmax_unc])
        unc_des_ls.extend(["entropy", "softmax response"])
    elif category in ["Softvote Duo", "Confident Duo"]:
        entropy_unc = f"Entr[{model}]"
        softmax_unc = f"SR[{model}]"
        unc_ls.extend([entropy_unc, softmax_unc])
        unc_des_ls.extend(["entropy", "softmax response"])
        kl_unc_ls = f"KL[{main_model}||{sec_model}]"
        unc_ls.extend([kl_unc_ls])
        unc_des_ls.extend(["kl_ls"])
        unc_ls.append(f"Entr[{model}]+{kl_unc_ls}")
        unc_des_ls.append("entropy+kl(ls)")
        # kl_unc_sl = f"KL[{sec_model}||{main_model}]"
        # mi_unc = f"MI[{sec_model}||{main_model}]"
        # unc_ls.extend([kl_unc_sl])
        # unc_des_ls.extend(["kl_sl"])
        # unc_ls.extend([mi_unc])
        # unc_des_ls.extend(["mi"])
        # unc_ls.append(f"Entr[{model}]+{kl_unc_sl}")
        # unc_des_ls.append("entropy+kl(sl)")
        # unc_ls.append(f"Entr[{model}]+{mi_unc}")
        # unc_des_ls.append("entropy+mi")
    elif category == "Dictatorial Duo":
        entropy_unc = f"Entr[{main_model}]"
        softmax_unc = f"SR[{main_model}]"
        unc_ls.extend([entropy_unc, softmax_unc])
        kl_unc_ls = f"KL[{main_model}||{sec_model}]"
        unc_ls.extend([kl_unc_ls])
        unc_des_ls.extend(["kl_ls"])
        unc_des_ls.extend(["entropy", "softmax response"])
        unc_ls.append(f"Entr[{main_model}]+{kl_unc_ls}")
        unc_des_ls.append("entropy+kl(ls)")
    
    return unc_ls, unc_des_ls

def evaluate_metrics(pred, unc, model, target_vec, unc_quantifier, cov_range):
    auroc = calculate_auroc(pred, unc, f"pred_{model}", target_vec, unc_quantifier)
    cov_auc_and_sac = calculate_f1_acc_cov_auc_and_sac(
        pred, unc, f"pred_{model}", target_vec, unc_quantifier, cov_range
    )
    metrics = {
        "auroc": auroc,
        **cov_auc_and_sac  # Include F1-Cov AUC, Acc-Cov AUC, and SAC results
    }
    return metrics


def get_member_models(model):
    members = model.replace("confident(", "").replace("softvote(", "").replace("dictatorial(", "").replace(")", "")
    return members.split(",")

def get_main_model(model):
    return get_member_models(model)[1]

def get_secondary_model(model):
    return get_member_models(model)[0]

def process_category_subprogress_bar(task, shared_data, target_col="targets"):
    """
    Each subprocess calls this function to handle one category's models.
    It loads the shared data (pred and unc), updates progress in progress_dict,
    and returns all result rows for that category.
    """
    category, models, testset, cov_range = task
    
    # Access DataFrames from shared memory (no additional reading from disk)
    pred = shared_data['pred']
    unc = shared_data['unc']
    
    category_results = []
    for idx, model in enumerate(models):
        # Uncertainty measure definitions
        main_model = get_main_model(model) if category != "Single Model" else model
        sec_model = get_secondary_model(model) if category != "Single Model" else model
        unc_ls, unc_des_ls = add_uncertainty_measures(
            category, model, [], [], main_model, sec_model
        )

        for unc_quantifier, unc_des in zip(unc_ls, unc_des_ls):
            # Evaluate metrics for (model, unc_quantifier)
            unc_series = unc[unc_quantifier]
            if unc_quantifier not in unc.columns:
                print(f"{unc_quantifier} is not in unc.")
                print(f"Available columns are {unc.columns}")
            metrics = evaluate_metrics(pred, unc_series, model, target_col, unc_quantifier, cov_range)

            # Create a row dict for the final DataFrame
            row = {
                "Predictor": model,
                "Test Set": testset,
                "Uncertainty Measure": unc_quantifier,
                "Uncertainty Type": unc_des,
                "F1-Cov AUC": metrics["f1_cov_auc"],
                "Acc-Cov AUC": metrics["acc_cov_auc"],
                "Correctness Prediction AUROC": metrics["auroc"],
                "Predictor Category": category,
                "Main Model": main_model,
                "Secondary Model": sec_model,
            }
            # Add any SAC_* keys
            for sac_key in metrics:
                if sac_key.startswith("SAC_"):
                    row[sac_key] = metrics[sac_key]
            for sfc_key in metrics:
                if sfc_key.startswith("SFC_"):
                    row[sfc_key] = metrics[sfc_key]

            category_results.append(row)

    return category_results

def confident_predictor(pred_1, pred_2, unc1, unc2):
    mask = (np.array(unc1) < np.array(unc2))
    res = pred_2.copy()
    res[mask] = pred_1[mask]
    return res
