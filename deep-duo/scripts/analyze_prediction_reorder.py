import sys
import pandas as pd
import numpy as np
import itertools
import json
import os
from sklearn.metrics import f1_score
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.utils as utils
import argparse
import time
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Run model evaluation.")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (ImageNet, Caltech256, IwildCamIND, IwildCamOOD)")
args = parser.parse_args()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

dataset = args.dataset

arch_col_name = "Weight"
gflops_col_name = "GFLOPS"
cov_range = np.arange(1, 100.1,0.5)

if dataset == "ImageNet":
    import configs.config_imgnet as config
elif dataset == "ImageNetV2":
    import configs.config_imgnetV2 as config
elif dataset == "Caltech256":
    import configs.config_caltech256 as config
elif dataset == "IwildCamIND":
    import configs.config_iwcind as config
elif dataset == "IwildCamOOD":
    import configs.config_iwcood as config

num_class = config.num_class

def get_pred_dir(category):
    return config.logit_pred_dir if category=="Single Model" else config.sv_pred_dir if category=="Softvote Duo" else config.cd_pred_dir if category == "Confident Duo" else config.logit_pred_dir

predictor_categories = {
    "Single Model":[],
    "Softvote Duo":[],
    "Confident Duo":[],
    "Dictatorial Duo":[],
}

backbones = pd.read_csv(config.backbone_csv_path).sort_values(gflops_col_name,ignore_index=True)
backbones_ls = backbones[arch_col_name]
raw_pred_ls = os.listdir(config.logit_pred_dir)
print(f"{raw_pred_ls=}")
available_pred_ls_unsorted = [raw_pred[:-11] for raw_pred in raw_pred_ls]
print(f"{available_pred_ls_unsorted=}")
available_pred_map = {x.lower(): x for x in available_pred_ls_unsorted}
# Keep the order of backbones_ls but use original strings from available_pred_ls_unsorted
available_pred_ls_resorted = [
    available_pred_map[x.lower()] for x in backbones_ls if x.lower() in available_pred_map
]
available_pred_ls = available_pred_ls_resorted
avail_model_dict = {"available models":available_pred_ls}
print(f"{available_pred_ls=}")
print(avail_model_dict)
# Deduplicate:
predictor_categories["Single Model"] = list(set(available_pred_ls))

########### Create Directories If Not Exist ###########
softvote_save_dir = config.sv_pred_dir
if not os.path.exists(config.sv_pred_dir):
    os.makedirs(config.sv_pred_dir)
conf_dir = config.cd_pred_dir
if not os.path.exists(conf_dir):
    os.makedirs(conf_dir)
dict_dir = config.dd_pred_dir
if not os.path.exists(dict_dir):
    os.makedirs(dict_dir)
    

############# Enumerate Duos of Interest ###########
large_model_ls = config.large_model_ls
duo_combination = []
for comb in itertools.combinations(available_pred_ls, 2):
    member1, member2 = comb
    if member2 in large_model_ls:
        duo_combination.append((member1,member2))
print(f"{duo_combination=}")
print(f"{len(duo_combination)}")

############# Make Single Model Uncertainties ############
unc_dir = config.unc_dir
if not os.path.exists(unc_dir):
    os.makedirs(unc_dir)

unc = pd.DataFrame()
print(f"{predictor_categories=}")
for model in predictor_categories["Single Model"]:
    pred = pd.read_csv(f"{config.logit_pred_dir}/{model}_logits.csv")
    unc[f"Entr[{model}]"]=utils.calc_entr_torch(pred,device)
    unc[f"SR[{model}]"]=utils.softmax_response_unc(pred)
        
############# Make Duo Uncertainties ############
for (memS,memL) in duo_combination:
    pred_memS = pd.read_csv(f"{config.logit_pred_dir}/{memS}_logits.csv")
    pred_memL = pd.read_csv(f"{config.logit_pred_dir}/{memL}_logits.csv")
    unc[f"KL[{memL}||{memS}]"]=utils.calc_kl_torch(pred_memL,pred_memS,device)
    unc[f"Entr[{memL}]+KL[{memL}||{memS}]"]=unc[f"KL[{memL}||{memS}]"]+unc[f"Entr[{memL}]"]
    unc[f"KL[{memS}||{memL}]"]=utils.calc_kl_torch(pred_memS,pred_memL,device)
    unc[f"Entr[{memL}]+KL[{memS}||{memL}]"]=unc[f"KL[{memS}||{memL}]"]+unc[f"Entr[{memL}]"]
    unc[f"MI[{memS}||{memL}]"]=utils.calc_mi_torch(pred_memL,pred_memS,device)
    unc[f"Entr[{memL}]+MI[{memS}||{memL}]"]=unc[f"MI[{memS}||{memL}]"]+unc[f"Entr[{memL}]"]
############## Make Softvote & Confident & Dictatorial Duos ############
for (member1, member2) in duo_combination:
    softvote_predictor_name = f"softvote({member1},{member2})"
    if not (softvote_predictor_name in predictor_categories["Softvote Duo"]):
        predictor_categories["Softvote Duo"].append(softvote_predictor_name)
    save_loc=f"{config.sv_pred_dir}/{softvote_predictor_name}_logits.csv"
    pred_1 = pd.read_csv(f"{config.logit_pred_dir}/{member1}_logits.csv")
    pred_2 = pd.read_csv(f"{config.logit_pred_dir}/{member2}_logits.csv")
    softvote_prediction = utils.softvote(pred_1,pred_2)
    softvote_prediction.to_csv(save_loc, index=False)
    unc[f"Entr[{softvote_predictor_name}]"]=utils.calc_entr_torch(softvote_prediction,device)
    unc[f"SR[{softvote_predictor_name}]"]=utils.softmax_response_unc(softvote_prediction)
    unc[f"Entr[{softvote_predictor_name}]+KL[{member2}||{member1}]"]=unc[f"Entr[{softvote_predictor_name}]"]+unc[f"KL[{member2}||{member1}]"]
    unc[f"Entr[{softvote_predictor_name}]+KL[{member1}||{member2}]"]=unc[f"Entr[{softvote_predictor_name}]"]+unc[f"KL[{member1}||{member2}]"]
    unc[f"Entr[{softvote_predictor_name}]+MI[{member1}||{member2}]"]=unc[f"MI[{member1}||{member2}]"]+unc[f"Entr[{softvote_predictor_name}]"]
    confident_predictor_name = f"confident({member1},{member2})"
    if not (confident_predictor_name in predictor_categories["Confident Duo"]):
        predictor_categories["Confident Duo"].append(confident_predictor_name)
    save_loc=f"{conf_dir}/{confident_predictor_name}_logits.csv"
    confident_prediction = utils.confident_predictor(pred_1,pred_2,unc[f"SR[{member1}]"],unc[f"SR[{member2}]"])
    confident_prediction.to_csv(save_loc, index=False)
    unc[f"Entr[{confident_predictor_name}]"]=utils.calc_entr_torch(confident_prediction,device)
    unc[f"SR[{confident_predictor_name}]"]=utils.softmax_response_unc(confident_prediction)
    unc[f"Entr[{confident_predictor_name}]+KL[{member2}||{member1}]"]=unc[f"Entr[{confident_predictor_name}]"]+unc[f"KL[{member2}||{member1}]"]
    unc[f"Entr[{confident_predictor_name}]+KL[{member1}||{member2}]"]=unc[f"Entr[{confident_predictor_name}]"]+unc[f"KL[{member1}||{member2}]"]
    unc[f"Entr[{confident_predictor_name}]+MI[{member1}||{member2}]"]=unc[f"MI[{member1}||{member2}]"]+unc[f"Entr[{confident_predictor_name}]"]
    dictatorial_predictor_name = f"dictatorial({member1},{member2})"
    if not (dictatorial_predictor_name in predictor_categories["Dictatorial Duo"]):
        predictor_categories["Dictatorial Duo"].append(dictatorial_predictor_name)
############# Save unc df and predictor json ############
unc.to_csv(f"{unc_dir}/unc.csv", index=False)
with open(f"{config.dataset}_predictor_categories.json", "w") as file:
    json.dump(predictor_categories, file)

    
## Notebook 2

with open(f"{config.dataset}_predictor_categories.json", "r") as file:
    predictor_categories = json.load(file) 
target_col="label"
target = pd.read_csv(f"{config.target_dir}/targets.csv")
point_target = target[target_col]
metrics_dict = {"Predictor": [], "Acc": [], "F1": [], "Brier": [], "CP_Brier":[], "ECE": [],"ESCE":[], "MCE": [], "Predictor Category": [], "Main Model": [], "Secondary Model": []}

############ Evaluate duo predictors ##################
for category, models in predictor_categories.items():
    category_start_time = time.time()
    print(f"Evaluating {category} predictors...")
    pred_dir_curr = get_pred_dir(category)
    
    for model in models:
        main_model = utils.get_main_model(model) if category != "Single Model" else model
        sec_model = utils.get_secondary_model(model) if category != "Single Model" else model
        if category == "Dictatorial Duo":
            pred = pd.read_csv(f"{pred_dir_curr}/{main_model}_logits.csv")
        else:
            pred = pd.read_csv(f"{pred_dir_curr}/{model}_logits.csv")
        curr_point_pred = np.argmax(pred, 1)
        curr_sm_pred = utils.softmax(pred)
        curr_correctness = (point_target == curr_point_pred)
        accuracy = np.mean(curr_correctness)
        f1 = f1_score(point_target, curr_point_pred, average='macro')
        curr_brier = utils.brier_score(utils.one_hot(np.array(point_target), num_class), curr_sm_pred)
        curr_cp_brier = utils.brier_score(curr_correctness, np.max(curr_sm_pred,axis=1))
        curr_ece, curr_esce, curr_mce = utils.calibration(utils.one_hot(np.array(point_target), num_class), curr_sm_pred)
        
        metrics_dict["Predictor"].append(model)
        metrics_dict["Acc"].append(accuracy)
        metrics_dict["F1"].append(f1)
        metrics_dict["Brier"].append(curr_brier)
        metrics_dict["CP_Brier"].append(curr_cp_brier)
        metrics_dict["ECE"].append(curr_ece)
        metrics_dict["ESCE"].append(curr_esce)
        metrics_dict["MCE"].append(curr_mce)
        metrics_dict["Predictor Category"].append(category)
        metrics_dict["Main Model"].append(main_model)
        metrics_dict["Secondary Model"].append(sec_model)
        target[f"pred_{model}"] = curr_point_pred
    category_time = time.time() - category_start_time
    print(f"Completed {category} category in {category_time:.0f} seconds.")
target.to_csv(f"evaluation/{config.dataset}_predictions.csv")
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(f"evaluation/{config.dataset}_predictors.csv")

# Define task parameters
max_processes = 4
tasks = [(category, models, "val", cov_range) for category, models in predictor_categories.items()]

# Load predictions and uncertainty data
pred_df = pd.read_csv(f"evaluation/{config.dataset}_predictions.csv")
unc_df = pd.read_csv(f"{config.unc_dir}/unc.csv")

# compute unc quant metrics in parallel
############# evaliate duo uncertainty quantifiers ############
def process_tasks(tasks, max_processes):
    results = []
    manager = Manager()
    shared_data = manager.dict()
    shared_data['pred'] = pred_df
    shared_data['unc'] = unc_df
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {
            executor.submit(
                utils.process_category_subprogress_bar, 
                task,
                shared_data,
                target_col
            ): task[0]
            for task in tasks
        }
        print("Processing tasks...")
        for future in as_completed(futures):
            category = futures[future]
            try:
                result = future.result()
                results.extend(result)
                print(f"Completed: {category}")
            except Exception as e:
                print(f"Error in {category}: {e}")
    print("All tasks completed.")
    return results
results = process_tasks(tasks, max_processes)

# Convert all results to a DataFrame
auroc_results = pd.DataFrame(results)
print("Final AUROC Results:")
print(auroc_results.head())

auroc_results.to_csv(f"evaluation/{config.dataset}_uncertainty_usefulness.csv",index=False)

