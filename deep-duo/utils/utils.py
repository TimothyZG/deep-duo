import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.metrics import f1_score, roc_curve, auc, accuracy_score
import seaborn as sns

def intersection(lst1, lst2):
    return [item for item in lst2 if item in set(lst1)]

# Get sum or imbalance of properties of a pair of backbones, like GFlops and IMGNet_T1_Acc
def get_duo_property(backbone_df_path, arch1: str, arch2: str, property: str, type: str="sum"):
    backbone_df = pd.read_csv(backbone_df_path)
    prop1 = backbone_df.loc[backbone_df["Architecture"].str.lower() == arch1.lower()][property].iloc[0]
    prop2 = backbone_df.loc[backbone_df["Architecture"].str.lower() == arch2.lower()][property].iloc[0]
    prop_ls = np.array([prop1,prop2])
    res = np.sum(prop_ls) if type=="sum" else np.min(prop_ls)/np.max(prop_ls)
    return res

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

def calc_kl_ignoretop_torch(P, Q):
    # Convert DataFrames to NumPy arrays
    P_vals = P.values
    Q_vals = Q.values

    # Get the indices of the maximum values along each row
    max_indices = np.argmax(P_vals, axis=1)

    # Create masks to exclude the maximum indices
    mask = np.ones_like(P_vals, dtype=bool)
    mask[np.arange(len(P_vals)), max_indices] = False

    # Apply the mask to exclude the maximum values
    P_sub = P_vals[mask].reshape(len(P_vals), -1)
    Q_sub = Q_vals[mask].reshape(len(Q_vals), -1)

    # Convert back to DataFrames for compatibility with calc_kl_torch
    P_sub_df = pd.DataFrame(P_sub)
    Q_sub_df = pd.DataFrame(Q_sub)

    # Compute KL divergence
    return calc_kl_torch(P_sub_df, Q_sub_df)

def softvote(pred1, pred2):
    res = (pred1+pred2)/2
    return res

def eval_pred(pred):
    results = []
    for col in pred.columns.drop("target"):
        if isinstance(pred[col].iloc[0], str):
            acc = pred.apply(lambda row: str(row["target"]) in row[col], axis=1).mean()
        else:
            acc = (pred[col] == pred["target"]).mean()
        results.append({"Method": col, "Accuracy": acc})
    return pd.DataFrame(results)

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
  mce = np.max(np.absolute(mean_conf - acc_tab))
  return ece, mce

def auroc(pred_df, unc_df, pred_vec, target_vec, unc_vec, ax):
    label = f"{unc_vec} on {pred_vec}"
    if isinstance(pred_df[pred_vec].iloc[0],str):
        is_correct = pred_df.apply(lambda row: str(row[target_vec]) not in row[pred_vec], axis=1)
    else:
        is_correct = pred_df[pred_vec]!=pred_df[target_vec]
    fpr, tpr, threshold = metrics.roc_curve(is_correct, unc_df[unc_vec])
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, label = f'{label}: AUC = %0.3f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

def get_rank(unc_pred):
    # rank = pd.DataFrame()
    # for curr_metric in unc_pred.columns:
    #     rank[curr_metric] = unc_pred[curr_metric].rank()
    # return rank
    return unc_pred.rank(method='first')

def acc_cov_tradeoff(pred_df, pred_vec, target_vec, unc_vec, rank, ax, cov_range, criteria="f1"):
    temp=pd.DataFrame()
    temp["coverage"] = cov_range
    coverage_ls = (cov_range+1)/100*pred_df.shape[0]
    for i, cov in enumerate(coverage_ls):
        cov_pred = pred_df[rank[unc_vec]<cov]
        temp.loc[i,"acc"] = np.mean(cov_pred[pred_vec]==cov_pred[target_vec])
        temp.loc[i,"f1"] = metrics.f1_score(cov_pred[target_vec], cov_pred[pred_vec], average='macro')
    area=np.sum(temp[criteria])
    label = f"(AUC: {area:.3f}) - {unc_vec} on {pred_vec}"
    sns.lineplot(temp,x="coverage",y=f"{criteria}",label=label,ax = ax)
    ax.set_ylabel(f'{criteria}')
    ax.set_xlabel('Coverage')
    ax.legend(loc = 'upper right')
    
def auroc_ood(pred_df, unc_df, pred_vec, unc_vec, ax):
    label = f"{unc_vec}"
    is_ood = np.where(pred_df["test_type"].isin([ "ood and cor", "ood and inc"]),True,False)
    fpr, tpr, threshold = metrics.roc_curve(is_ood, unc_df[unc_vec])
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, label = f'{label}: AUC = %0.3f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    
# Function to load predictions
# Takes in a list of models and return a list of predictions
def load_predictions(prefix, testtype, models):
    return [pd.read_csv(f"{prefix}{testtype}_{model}.csv") for model in models]

# Function to save ensemble predictions
def save_ensemble_predictions(prefix, testtype, name, predictions):
    predictions.to_csv(f"{prefix}{testtype}_{name}.csv", index=False)

# Function to generate ensemble name
def generate_ensemble_name(base_name, models):
    return f"{base_name}_{'_'.join(models)}"

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
    if isinstance(pred_df[pred_vec].iloc[0], str):
        is_correct = pred_df.apply(lambda row: str(row[target_vec]) not in row[pred_vec], axis=1)
    else:
        is_correct = pred_df[pred_vec] != pred_df[target_vec]
    fpr, tpr, _ = roc_curve(is_correct, unc_df)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def calculate_aufcc(pred_df, unc_df, pred_vec, target_vec, unc_vec):
    if isinstance(pred_df[pred_vec].iloc[0], str):
        is_correct = pred_df.apply(lambda row: str(row[target_vec]) not in row[pred_vec], axis=1)
    else:
        is_correct = pred_df[pred_vec] != pred_df[target_vec]
    fpr, tpr, _ = roc_curve(is_correct, unc_df[unc_vec])
    roc_auc = auc(fpr, tpr)
    return roc_auc

# Function to update AUROC results
def update_auroc_results(auroc_results, model, testtype, category, unc_measure, auroc_value):
    auroc_results["Model"].append(model)
    auroc_results["Test Set"].append(testtype)
    auroc_results["Uncertainty Measure"].append(unc_measure)
    auroc_results["Predictor Category"].append(category)
    auroc_results["AUROC"].append(auroc_value)
    return auroc_results

# Function to calculate F1-Coverage AUC
def calculate_f1_cov_auc(pred_df, unc_df, pred_vec, target_vec, unc_vec, cov_range):
    print("WARNING: DEPRECATED, use calculate_cov_auc instead!")
    rank = get_rank(unc_df)
    temp = pd.DataFrame()
    temp["coverage"] = cov_range
    coverage_ls = (cov_range + 1) / 100 * pred_df.shape[0]
    for i, cov in enumerate(coverage_ls):
        cov_pred = pred_df.loc[rank[unc_vec] < cov]
        temp.loc[i, "f1"] = f1_score(cov_pred[target_vec], cov_pred[pred_vec], average='macro')
    area = np.sum(temp["f1"])
    return area

# Function to calculate Acc-Coverage AUC
def calculate_acc_cov_auc(pred_df, unc_df, pred_vec, target_vec, unc_vec, cov_range):
    print("WARNING: DEPRECATED, use calculate_cov_auc instead!")
    rank = get_rank(unc_df)
    temp = pd.DataFrame()
    temp["coverage"] = cov_range
    coverage_ls = (cov_range + 1) / 100 * pred_df.shape[0]
    for i, cov in enumerate(coverage_ls):
        cov_pred = pred_df.loc[rank[unc_vec] < cov]
        temp.loc[i, "acc"] = accuracy_score(cov_pred[target_vec], cov_pred[pred_vec])
    area = np.sum(temp["acc"])
    return area

def calculate_cov_auc(pred_df, unc_df, pred_vec, target_vec, unc_vec, cov_range, metric_func):
    """
    Calculate Coverage AUC for a given metric function.
    
    Parameters:
    - pred_df: DataFrame containing predictions and true targets.
    - unc_df: DataFrame containing uncertainty values.
    - pred_vec: Column name for predictions in pred_df.
    - target_vec: Column name for target labels in pred_df.
    - unc_vec: Column name for uncertainty in unc_df.
    - cov_range: List of coverage percentages (e.g., [10, 20, 30, ...]).
    - metric_func: Function to calculate the metric (e.g., f1_score, accuracy_score).
    
    Returns:
    - AUC value based on the selected metric.
    """
    # Get rank based on uncertainty
    rank = get_rank(unc_df)
    
    # Calculate coverage threshold values
    coverage_ls = (cov_range + 1) / 100 * pred_df.shape[0]
    
    # Compute metric for each coverage threshold
    metrics = [
        metric_func(
            pred_df.loc[rank[unc_vec] < cov, target_vec], 
            pred_df.loc[rank[unc_vec] < cov, pred_vec]
        )
        for cov in coverage_ls
    ]
    
    # Calculate area under the curve
    return np.sum(metrics)

def calculate_f1_acc_cov_auc(pred_df, unc_df, pred_vec, target_vec, unc_vec, cov_range):
    """
    Calculate Coverage AUC for both F1-Score and Accuracy.
    
    Parameters:
    - pred_df: DataFrame containing predictions and true targets.
    - unc_df: DataFrame containing uncertainty values.
    - pred_vec: Column name for predictions in pred_df.
    - target_vec: Column name for target labels in pred_df.
    - unc_vec: Column name for uncertainty in unc_df.
    - cov_range: List of coverage percentages (e.g., [10, 20, 30, ...]).
    
    Returns:
    - A tuple (f1_cov_auc, acc_cov_auc), representing the AUC for F1-Score and Accuracy.
    """
    # Get rank based on uncertainty
    rank = get_rank(unc_df)
    
    # Calculate coverage threshold values
    coverage_ls = (cov_range + 1) / 100 * pred_df.shape[0]
    
    # Initialize lists for metrics
    f1_metrics = []
    acc_metrics = []
    
    # Compute metrics for each coverage threshold
    for cov in coverage_ls:
        # Filter predictions within the coverage threshold
        cov_pred = pred_df.loc[rank[unc_vec] < cov]
        
        # Calculate F1 and Accuracy
        f1_metrics.append(f1_score(cov_pred[target_vec], cov_pred[pred_vec], average='macro'))
        acc_metrics.append(accuracy_score(cov_pred[target_vec], cov_pred[pred_vec]))
    
    # Calculate area under the curve for both metrics
    f1_cov_auc = np.sum(f1_metrics)
    acc_cov_auc = np.sum(acc_metrics)
    
    return f1_cov_auc, acc_cov_auc

def calculate_f1_acc_cov_auc_and_sac(pred_df, unc, pred_vec, target_vec, unc_vec, cov_range, sac_targets=[0.7,0.8,0.9,0.95,0.98],sfc_targets=[0.4,0.5,0.6,0.7,0.8,0.9,0.95]):
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
            if ((sac_results[sac_key] is None) or (cov>sac_results[sac_key])) and accuracy >= sac_target:
                sac_results[sac_key] = cov/pred_df.shape[0]*100
         # Check SAC for each accuracy target
        for sfc_target in sfc_targets:
            sfc_key = f"SFC_{sfc_target}"
            if ((sfc_results[sfc_key] is None) or (cov>sfc_results[sfc_key])) and f1 >= sfc_target:
                sfc_results[sfc_key] = cov/pred_df.shape[0]*100
    
    # Calculate area under the curve for both metrics
    f1_cov_auc = np.sum(f1_metrics)
    acc_cov_auc = np.sum(acc_metrics)
    
    # Combine results into a dictionary
    results = {
        "f1_cov_auc": f1_cov_auc,
        "acc_cov_auc": acc_cov_auc,
        **sac_results,  # Merge SAC results into the dictionary
        **sfc_results
    }
    
    return results


# Function to update F1-Coverage results
def update_f1_cov_results(f1_cov_results, model, testtype, category, unc_measure, f1_cov_auc_value):
    f1_cov_results["Model"].append(model)
    f1_cov_results["Test Set"].append(testtype)
    f1_cov_results["Uncertainty Measure"].append(unc_measure)
    f1_cov_results["Predictor Category"].append(category)
    f1_cov_results["F1-Cov AUC"].append(f1_cov_auc_value)
    return f1_cov_results

# Function to calculate AUROC for OOD detection
def calculate_auroc_ood(pred_df, unc_df, pred_vec, unc_vec):
    is_ood = np.where(pred_df["test_type"].isin(["ood"]), True, False)
    fpr, tpr, _ = roc_curve(is_ood, unc_df[unc_vec])
    roc_auc = auc(fpr, tpr)
    return roc_auc


# Function to update AUROC OOD results
def update_auroc_ood_results(auroc_ood_results, model, category, unc_measure, auroc_ood_value):
    auroc_ood_results["Model"].append(model)
    auroc_ood_results["Test Set"].append("combined")
    auroc_ood_results["Uncertainty Measure"].append(unc_measure)
    auroc_ood_results["Predictor Category"].append(category)
    auroc_ood_results["AUROC OOD"].append(auroc_ood_value)
    return auroc_ood_results

def generate_duos(pred_prefix, base_name, testtype, smaller_model_ls, larger_model_ls, predictor_categories):
    for smaller_model in smaller_model_ls:
        for larger_model in larger_model_ls:
            curr_small_logits = pd.read_csv(f"{pred_prefix}{testtype}_{smaller_model}.csv")
            curr_large_logits = pd.read_csv(f"{pred_prefix}{testtype}_{larger_model}.csv")
            # Softvote Ensemble
            curr_duo_pred_softvote = softvote([curr_small_logits,curr_large_logits])
            curr_duo_name_softvote = generate_ensemble_name(base_name+"(Softvote)", [smaller_model, larger_model])
            save_ensemble_predictions(pred_prefix, testtype, curr_duo_name_softvote, curr_duo_pred_softvote)
            if not (curr_duo_name_softvote in predictor_categories[base_name]):
                predictor_categories[base_name].append(curr_duo_name_softvote)
    return predictor_categories


def generate_scaled_duos(pred_df_combined, pred_prefix, smaller_model_ls, larger_model_ls,scale):
    auroc_ood_ls = []
    for smaller_model in smaller_model_ls:
        for larger_model in larger_model_ls:
            pred_smaller_ood = pd.read_csv(f"{pred_prefix}ood_{smaller_model}.csv") * scale
            pred_larger_ood = pd.read_csv(f"{pred_prefix}ood_{larger_model}.csv") * scale
            ce_larger_smaller_ood = calc_cross_entr_torch(pred_larger_ood,pred_smaller_ood)
            
            pred_smaller_ind = pd.read_csv(f"{pred_prefix}ind_{smaller_model}.csv") * scale
            pred_larger_ind = pd.read_csv(f"{pred_prefix}ind_{larger_model}.csv") * scale
            ce_larger_smaller_ind = calc_cross_entr_torch(pred_larger_ind,pred_smaller_ind)
            
            ce_combined = np.concatenate([ce_larger_smaller_ood, ce_larger_smaller_ind])
            
            is_ood = np.where(pred_df_combined["test_type"].isin(["ood"]), True, False)
            fpr, tpr, _ = roc_curve(is_ood, ce_combined)
            roc_auc = auc(fpr, tpr)
            auroc_ood_ls.append(roc_auc)
    return np.mean(auroc_ood_ls)


def add_uncertainty_measures(category, model, unc_ls, unc_des_ls, main_model, sec_model):
    entropy_unc = f"Entr[{model}]"
    softmax_unc = f"SR[{model}]"
    unc_ls.extend([entropy_unc, softmax_unc])
    unc_des_ls.extend(["entropy", "softmax response"])
    
    if category in ["Softvote Duo", "Confident Duo", "Dictatorial Duo"]:
        kl_unc_ls = f"KL[{main_model}||{sec_model}]"
        # kl_unc_sl = f"KL[{sec_model}||{main_model}]"
        # mi_unc = f"MI[{sec_model}||{main_model}]"
        unc_ls.extend([kl_unc_ls])
        unc_des_ls.extend(["kl_ls"])
        # unc_ls.extend([kl_unc_sl])
        # unc_des_ls.extend(["kl_sl"])
        # unc_ls.extend([mi_unc])
        # unc_des_ls.extend(["mi"])
        unc_ls.append(f"Entr[{model}]+{kl_unc_ls}")
        unc_des_ls.append("entropy+kl(ls)")
        # unc_ls.append(f"Entr[{model}]+{kl_unc_sl}")
        # unc_des_ls.append("entropy+kl(sl)")
        # unc_ls.append(f"Entr[{model}]+{mi_unc}")
        # unc_des_ls.append("entropy+mi")
    
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

def process_uncertainty_measure(
    testset, model, unc_quantifier, unc_des, category, main_model, sec_model, pred, unc, target_vec, cov_range
):
    # Evaluate metrics
    metrics = evaluate_metrics(pred, unc, model, target_vec, unc_quantifier, cov_range)
    
    # Create the row dictionary
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
    for sac_key in metrics:
        if sac_key.startswith("SAC_"):
            row[sac_key] = metrics[sac_key]
    
    return row


def get_member_models(model):
    members = model.replace("confident(", "").replace("softvote(", "").replace("dictatorial(", "").replace(")", "")
    return members.split(",")

def get_main_model(model):
    return get_member_models(model)[1]

def get_secondary_model(model):
    return get_member_models(model)[0]

def process_category_subprogress_bar(task, progress_dict, shared_data, target_col="targets"):
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
        # Update progress so the main process can reflect it in the inner progress bar
        progress_dict[category] = idx + 1

        # Uncertainty measure definitions
        main_model = get_main_model(model) if category != "Single Model" else model
        sec_model = get_secondary_model(model) if category != "Single Model" else model
        unc_ls, unc_des_ls = add_uncertainty_measures(
            category, model, [], [], main_model, sec_model
        )

        for unc_quantifier, unc_des in zip(unc_ls, unc_des_ls):
            # Evaluate metrics for (model, unc_quantifier)
            unc_series = unc[unc_quantifier]
            metrics = evaluate_metrics(pred, unc_series, model,target_col, unc_quantifier, cov_range)

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