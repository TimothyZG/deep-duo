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

def calc_mi_torch(P, Q, device):
    return (calc_kl_torch(P, Q, device)+calc_kl_torch(Q, P, device))/2

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


# Function to calculate AUROC
def calculate_auroc(pred_vec, unc_vec, target_vec):
    is_correct = pred_vec != target_vec
    fpr, tpr, _ = roc_curve(is_correct, unc_vec)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def area_under_risk_coverage_curve(uncertainties: np.ndarray, correctnesses: np.ndarray, risk="error") -> float:
    uncertainties = uncertainties.astype(np.float64)
    correctnesses = correctnesses.astype(np.float64)

    sorted_indices = np.argsort(uncertainties)
    correctnesses = correctnesses[sorted_indices]
    total_samples = len(uncertainties)

    cumulative_incorrect = np.cumsum(1 - correctnesses)
    indices = np.arange(1, total_samples + 1, dtype=np.float64)

    if risk == "error":
        aurc = np.sum(cumulative_incorrect / indices) / total_samples
    elif risk == "f1":
        cumulative_f1 = np.array([
            f1_score(correctnesses[:i], np.ones(i)) if i > 0 else 0
            for i in range(1, total_samples + 1)
        ])
        aurc = np.sum(1-cumulative_f1) / total_samples
    else:
        raise ValueError("Invalid risk type. Choose either 'error' or 'f1'.")
    return float(aurc)

def area_under_risk_coverage_vec(uncertainties: np.ndarray, correctnesses: np.ndarray, risk="error") -> float:
    uncertainties = uncertainties.astype(np.float64)
    correctnesses = correctnesses.astype(np.float64)

    sorted_indices = np.argsort(uncertainties)
    correctnesses = correctnesses[sorted_indices]
    total_samples = len(uncertainties)

    cumulative_incorrect = np.cumsum(1 - correctnesses)
    indices = np.arange(1, total_samples + 1, dtype=np.float64)
    
    if risk == "error":
        return cumulative_incorrect / indices
    elif risk == "f1":
        cumulative_f1 = np.array([
            f1_score(correctnesses[:i], np.ones(i)) if i > 0 else 0
            for i in range(1, total_samples + 1)
        ])
        return 1 - cumulative_f1
    else:
        raise ValueError("Invalid risk type. Choose either 'error' or 'f1'.")


def calculate_f1_acc_cov_auc_and_sac(pred_vec, unc_vec, target_vec, sac_targets=[0.9,0.95,0.98,0.99],sfc_targets=[0.8,0.9,0.95,0.98,0.99]):
    start_index = 200 # to reduce noise in beginning of unc, according to https://github.com/bmucsanyi/untangle/blob/main/untangle/utils/metric.py
    correctnesses = (pred_vec == target_vec).astype(np.float64).values
    uncertainties = unc_vec.astype(np.float64)
    cumulative_incorrect = area_under_risk_coverage_vec(uncertainties, correctnesses, risk="error")
    #cumulative_f1_risk = area_under_risk_coverage_vec(uncertainties, correctnesses, risk="f1")
    aurc_acc = np.mean(cumulative_incorrect)
    #aurc_f1_risk = np.mean(cumulative_f1_risk)
    aurc_f1_risk = 0
    # Initialize SAC and SFC results
    sac_results = {f"SAC_{int(target * 100)}": None for target in sac_targets}
    sfc_results = {f"SFC_{sfc_target}": None for sfc_target in sfc_targets}
    for acc_tar in sac_targets:
        over_threshold = cumulative_incorrect > 1 - acc_tar
        if np.all(over_threshold):
            sac_results[f"SAC_{int(acc_tar * 100)}"] = 1
        else:
            coverage_for_accuracy_strict = np.argmax(over_threshold)/len(correctnesses)
            coverage_for_accuracy_nonstrict = (
                np.argmax((cumulative_incorrect[start_index:] >1-acc_tar))+ start_index)/len(correctnesses)
            sac_results[f"SAC_{int(acc_tar * 100)}"] = coverage_for_accuracy_nonstrict if coverage_for_accuracy_nonstrict>start_index else coverage_for_accuracy_strict
    for f1_tar in sfc_targets:
        # WARNING!! THIS IS TEMPORARY TO MAKE COMPUTATION FASTER
        # TODO: figure out a faster way to compute this
        sfc_results[f"SFC_{f1_tar}"]=0
        continue
        # WARNING END
        over_threshold = cumulative_f1_risk > 1 - f1_tar
        if np.all(over_threshold):
            sfc_results[f"SFC_{f1_tar}"] = 1
        else:
            coverage_for_f1_strict = np.argmax(over_threshold)/len(correctnesses)
            coverage_for_f1_nonstrict = (
                np.argmax((cumulative_f1_risk[start_index:] >1-f1_tar))+ start_index)/len(correctnesses)
            sfc_results[f"SFC_{f1_tar}"] = coverage_for_f1_nonstrict if coverage_for_f1_nonstrict>start_index else coverage_for_f1_strict
    results = {
        "f1_cov_auc": aurc_f1_risk,
        "acc_cov_auc": aurc_acc,
        **sac_results,
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
        sr_main_unc = f"SR[{main_model}]"
        unc_ls.append(sr_main_unc)
        unc_des_ls.append("SR_main")
        entropy_main_unc = f"Entr[{main_model}]"
        unc_ls.append(entropy_main_unc)
        unc_des_ls.append("Entr_main")
    elif category == "Dictatorial Duo":
        entropy_unc = f"Entr[{main_model}]"
        softmax_unc = f"SR[{main_model}]"
        unc_ls.extend([entropy_unc, softmax_unc])
        unc_des_ls.extend(["entropy", "softmax response"])
        kl_unc_ls = f"KL[{main_model}||{sec_model}]"
        unc_ls.extend([kl_unc_ls])
        unc_des_ls.extend(["kl_ls"])
        unc_ls.append(f"Entr[{main_model}]+{kl_unc_ls}")
        unc_des_ls.append("entropy+kl(ls)")
        sv_duo_model = model.replace("dictatorial","softvote")
        sr_duo_unc = f"SR[{sv_duo_model}]"
        unc_ls.append(sr_duo_unc)
        unc_des_ls.append("SR_svduo")
        entropy_duo_unc = f"Entr[{sv_duo_model}]"
        unc_ls.append(entropy_duo_unc)
        unc_des_ls.append("Entr_svduo")
        
        # kl_unc_sl = f"KL[{sec_model}||{main_model}]"
        # mi_unc = f"MI[{sec_model}||{main_model}]"
        # unc_ls.extend([kl_unc_sl])
        # unc_des_ls.extend(["kl_sl"])
        # unc_ls.extend([mi_unc])
        # unc_des_ls.extend(["mi"])
        # unc_ls.append(f"Entr[{main_model}]+{kl_unc_sl}")
        # unc_des_ls.append("entropy+kl(sl)")
        # unc_ls.append(f"Entr[{main_model}]+{mi_unc}")
        # unc_des_ls.append("entropy+mi")
    
    return unc_ls, unc_des_ls

def evaluate_metrics(pred_vec, unc_vec, target_vec):
    auroc = calculate_auroc(pred_vec, unc_vec, target_vec)
    cov_auc_and_sac = calculate_f1_acc_cov_auc_and_sac(
        pred_vec, unc_vec, target_vec
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
    true_target = pred[target_col]
    for idx, model in enumerate(models):
        # Uncertainty measure definitions
        main_model = get_main_model(model) if category != "Single Model" else model
        sec_model = get_secondary_model(model) if category != "Single Model" else model
        unc_ls, unc_des_ls = add_uncertainty_measures(
            category, model, [], [], main_model, sec_model
        )
        pred_col_name = f"pred_{model}"
        pred_vec = pred[pred_col_name]
        for unc_quantifier, unc_des in zip(unc_ls, unc_des_ls):
            # Evaluate metrics for (model, unc_quantifier)
            if unc_quantifier not in unc.columns:
                print(f"{unc_quantifier} is not in unc.")
                print(f"Available columns are {unc.columns}")
            unc_series = unc[unc_quantifier]
            metrics = evaluate_metrics(pred_vec, unc_series, true_target)

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
