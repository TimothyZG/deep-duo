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
import configs.config_imgnet as config
import utils.utils as utils


## notebook 1 gen-data imagenet

predictor_categories = {
    "Single Model":[],
    "Softvote Duo":[],
    "Confident Duo":[],
    "Dictatorial Duo":[],
}

backbones = pd.read_csv(config.backbone_csv_path).sort_values("GFlops",ignore_index=True)
backbones_ls = backbones["Architecture"].str.lower()
raw_pred_ls = os.listdir(config.logit_pred_dir)

ff_available_ls_unsorted = [raw_pred[:-11] for raw_pred in raw_pred_ls]
ff_available_ls_resorted = utils.intersection(ff_available_ls_unsorted, backbones_ls)
ff_available_ls = [bck for bck in ff_available_ls_resorted]

# Deduplicate:
avail_model_dict = {"ff":ff_available_ls}
print(f"{len(ff_available_ls)=}")
print(avail_model_dict)
"""
for ff_available in ff_available_ls: predictor_categories["Single Model"].append(ff_available)
# Deduplicate:
predictor_categories["Single Model"] = list(set(predictor_categories["Single Model"]))

softvote_save_dir = config.sv_pred_dir
if not os.path.exists(config.sv_pred_dir):
    os.makedirs(config.sv_pred_dir)

for (model_type, model_ls) in avail_model_dict.items():
    for comb in itertools.combinations(model_ls, 2):
        member1, member2 = comb
        softvote_predictor_name = f"softvote({member1},{member2})"
        save_loc=f"{config.sv_pred_dir}/{softvote_predictor_name}_logits.csv"
        pred_1 = pd.read_csv(f"{config.logit_pred_dir}/{member1}_logits.csv")
        pred_2 = pd.read_csv(f"{config.logit_pred_dir}/{member2}_logits.csv")
        softvote_prediction = utils.softvote(pred_1,pred_2)
        softvote_prediction.to_csv(save_loc, index=False)
        if not (softvote_predictor_name in predictor_categories["Softvote Duo"]):
            predictor_categories["Softvote Duo"].append(softvote_predictor_name)
            
unc_dir = config.unc_dir
if not os.path.exists(unc_dir):
    os.makedirs(unc_dir)
warnings.filterwarnings('ignore')
unc = pd.DataFrame()
for category, models in predictor_categories.items():
    print(f"Evaluating {category} predictors...")
    for predictor in models:
        pred_dir = config.logit_pred_dir if category=="Single Model" else config.sv_pred_dir
        pred = pd.read_csv(f"{pred_dir}/{predictor}_logits.csv")
        unc[f"Entr[{predictor}]"]=utils.calc_entr_torch(pred)
        unc[f"SR[{predictor}]"]=utils.softmax_response_unc(pred)
        if(category == "Softvote Duo"):
            models = predictor.replace("softvote(", "").replace(")", "")
            memS, memL = models.split(",")
            pred_memS = pd.read_csv(f"{config.logit_pred_dir}/{memS}_logits.csv")
            pred_memL = pd.read_csv(f"{config.logit_pred_dir}/{memL}_logits.csv")
            unc[f"KL[{memL}||{memS}]"]=utils.calc_kl_torch(pred_memL,pred_memS)
            # unc[f"KL[{memS}||{memL}]"]=utils.calc_kl_torch(pred_memS,pred_memL)
            # unc[f"MI[{memS}||{memL}]"]=(unc[f"KL[{memL}||{memS}]"]+unc[f"KL[{memS}||{memL}]"])/2
            # unc[f"Entr[{predictor}]+KL_noise[{memL}||{memS}]"]=unc[f"Entr[{predictor}]"]+unc[f"KL_noise[{memL}||{memS}]"]
            unc[f"Entr[{predictor}]+KL[{memL}||{memS}]"]=unc[f"Entr[{predictor}]"]+unc[f"KL[{memL}||{memS}]"]
            # unc[f"Entr[{predictor}]+KL[{memS}||{memL}]"]=unc[f"Entr[{predictor}]"]+unc[f"KL[{memS}||{memL}]"]
            # unc[f"Entr[{predictor}]+MI[{memS}||{memL}]"]=unc[f"Entr[{predictor}]"]+unc[f"MI[{memS}||{memL}]"]
            unc[f"Entr[{predictor}]*KL[{memL}||{memS}]"]=unc[f"Entr[{predictor}]"]*unc[f"KL[{memL}||{memS}]"]
unc.to_csv(f"{unc_dir}/unc.csv", index=False)

backbone_df = pd.read_csv(config.backbone_csv_path)

def confident_predictor(pred_1, pred_2, unc1, unc2, gflops_balance = 0.5):
    print(f"{gflops_balance=}")
    mask = (np.array(unc1)/np.array(unc2) < np.sqrt(gflops_balance))
    res = pred_2.copy()
    res[mask] = pred_1[mask]
    perc_small = np.mean(mask)
    print(f"{perc_small} of predictions made by small member")
    return res

conf_dir = config.cd_pred_dir
if not os.path.exists(conf_dir):
    os.makedirs(conf_dir)

for (model_type, model_ls) in avail_model_dict.items():
    print(f"Making confident csv for {model_type}")
    print(f"Current {model_ls=}")
    for comb in itertools.combinations(model_ls, 2):
        member1, member2 = comb
        confident_predictor_name = f"confident({member1},{member2})"
        save_loc=f"{conf_dir}/{confident_predictor_name}_logits.csv"
        unc = pd.read_csv(f"{config.unc_dir}/unc.csv")
        pred_1 = pd.read_csv(f"{config.logit_pred_dir}/{member1}_logits.csv")
        pred_2 = pd.read_csv(f"{config.logit_pred_dir}/{member2}_logits.csv")
        print(f"{member1=}, {member2=}")
        gflops1=backbone_df.loc[backbone_df["Architecture"].str.lower() == member1]["GFlops"].iloc[0]
        gflops2=backbone_df.loc[backbone_df["Architecture"].str.lower() == member2]["GFlops"].iloc[0]
        gfbal = gflops1/gflops2
        confident_prediction = confident_predictor(pred_1,pred_2,unc[f"SR[{member1}]"],unc[f"SR[{member2}]"],gfbal)
        confident_prediction.to_csv(save_loc, index=False)
        if not (confident_predictor_name in predictor_categories["Confident Duo"]):
                predictor_categories["Confident Duo"].append(confident_predictor_name)
                
                
unc = pd.read_csv(f"{config.unc_dir}/unc.csv")
for predictor in predictor_categories["Confident Duo"]:
    pred_dir = config.cd_pred_dir
    pred = pd.read_csv(f"{pred_dir}/{predictor}_logits.csv")
    unc[f"Entr[{predictor}]"]=utils.calc_entr_torch(pred)
    unc[f"SR[{predictor}]"]=utils.softmax_response_unc(pred)
    models = predictor.replace("confident(", "").replace(")", "")
    memS, memL = models.split(",")
    pred_memS = pd.read_csv(f"{config.logit_pred_dir}/{memS}_logits.csv")
    pred_memL = pd.read_csv(f"{config.logit_pred_dir}/{memL}_logits.csv")
    unc[f"Entr[{predictor}]+KL[{memL}||{memS}]"]=unc[f"Entr[{predictor}]"]+unc[f"KL[{memL}||{memS}]"]
    # unc[f"Entr[{predictor}]+KL[{memS}||{memL}]"]=unc[f"Entr[{predictor}]"]+unc[f"KL[{memS}||{memL}]"]
    # unc[f"Entr[{predictor}]+MI[{memS}||{memL}]"]=unc[f"Entr[{predictor}]"]+unc[f"MI[{memS}||{memL}]"]
    unc[f"Entr[{predictor}]*KL[{memL}||{memS}]"]=unc[f"Entr[{predictor}]"]*unc[f"KL[{memL}||{memS}]"]
unc.to_csv(f"{config.unc_dir}/unc.csv", index=False)

"""
"""
# TODO: temporary
with open(f"{config.dataset}_predictor_categories.json", "r") as file:
    predictor_categories = json.load(file)
# END: temporary
    
dict_dir = config.dd_pred_dir
if not os.path.exists(dict_dir):
    os.makedirs(dict_dir)
    
unc = pd.read_csv(f"{config.unc_dir}/unc.csv")
for (model_type, model_ls) in avail_model_dict.items():
    print(f"Making Dictatorial csv for {model_type}")
    print(f"Current {model_ls=}")
    for comb in itertools.combinations(model_ls, 2):
        member1, member2 = comb
        pred_2 = pd.read_csv(f"{config.logit_pred_dir}/{member2}_logits.csv")
        dictatorial_predictor_name = f"dictatorial({member1},{member2})"
        dictatorial_prediction = pred_2
        unc[f"Entr[{dictatorial_predictor_name}]"]=unc[f"Entr[{member2}]"]
        unc[f"SR[{dictatorial_predictor_name}]"]=unc[f"SR[{member2}]"]
        save_loc=f"{dict_dir}/{dictatorial_predictor_name}_logits.csv"
        dictatorial_prediction.to_csv(save_loc, index=False)
        if not (dictatorial_predictor_name in predictor_categories["Dictatorial Duo"]):
                predictor_categories["Dictatorial Duo"].append(dictatorial_predictor_name)
unc.to_csv(f"{config.unc_dir}/unc.csv", index=False)
                    

unc = pd.read_csv(f"{config.unc_dir}/unc.csv")
print(f"Calculating Uncertainty Measures on test set")
for predictor in predictor_categories["Dictatorial Duo"]:
    pred_dir = config.dd_pred_dir
    pred = pd.read_csv(f"{pred_dir}/{predictor}_logits.csv")
    models = predictor.replace("dictatorial(", "").replace(")", "")
    memS, memL = models.split(",")
    unc[f"Entr[{predictor}]+KL[{memL}||{memS}]"]=unc[f"Entr[{predictor}]"]+unc[f"KL[{memL}||{memS}]"]
    # unc[f"Entr[{predictor}]+KL[{memS}||{memL}]"]=unc[f"Entr[{predictor}]"]+unc[f"KL[{memS}||{memL}]"]
    # unc[f"Entr[{predictor}]+MI[{memS}||{memL}]"]=unc[f"Entr[{predictor}]"]+unc[f"MI[{memS}||{memL}]"]
    unc[f"Entr[{predictor}]*KL[{memL}||{memS}]"]=unc[f"Entr[{predictor}]"]*unc[f"KL[{memL}||{memS}]"]
unc.to_csv(f"{config.unc_dir}/unc.csv", index=False)



with open(f"{config.dataset}_predictor_categories.json", "w") as file:
    json.dump(predictor_categories, file)
"""
    
## Notebook 2
num_class = config.num_class
pred_dir = config.logit_pred_dir

cov_range = np.arange(1, 100)


with open(f"{config.dataset}_predictor_categories.json", "r") as file:
    predictor_categories = json.load(file)
    
with open("forward_pass.json", "r") as json_file:
    gflops_data = json.load(json_file)
    
target_col="label"

metrics_dict = {
    "Predictor": [],
    "Acc": [],
    "F1": [],
    "Brier": [],
    "ECE": [],
    "MCE": [],
    "Predictor Category": [],
    "Main Model":[],
    "Secondary Model":[],
}

def get_member_models(model):
    members = model.replace("confident(", "").replace("softvote(", "").replace("dictatorial(", "").replace(")", "")
    return members.split(",")

def get_main_model(model):
    return get_member_models(model)[1]

def get_secondary_model(model):
    return get_member_models(model)[0]
    
def get_pred_dir(category):
    return config.logit_pred_dir if category=="Single Model" else config.sv_pred_dir if category=="Softvote Duo" else config.cd_pred_dir if category == "Confident Duo" else config.dd_pred_dir

def process_tasks(tasks, max_processes):
    results = []
    manager = Manager()
    shared_data = manager.dict()
    # We'll also store progress in a manager dict so all processes can update it
    progress_dict = manager.dict()
    progress_dict.clear()
    for category in predictor_categories.keys():
        progress_dict[category] = 0
    shared_data['pred'] = pred_df
    shared_data['unc'] = unc_df
    with ProcessPoolExecutor(max_workers=max_processes) as executor:
        futures = {
        executor.submit(
            utils.process_category_subprogress_bar, 
            task, 
            progress_dict, 
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

# Load target data
target = pd.read_csv(f"{config.target_dir}/target.csv")
metrics_dict = {"Predictor": [], "Acc": [], "F1": [], "Brier": [], "ECE": [], "MCE": [], "Predictor Category": [], "Main Model": [], "Secondary Model": []}

for category, models in predictor_categories.items():
    print(f"Evaluating {category} predictors...")
    pred_dir_curr = get_pred_dir(category)
    
    for model in models:
        main_model = get_main_model(model) if category != "Single Model" else model
        sec_model = get_secondary_model(model) if category != "Single Model" else model
        pred = pd.read_csv(f"{pred_dir_curr}/{model}_logits.csv")
        
        accuracy = np.mean(target[target_col] == np.argmax(pred, 1))
        f1 = f1_score(target[target_col], np.argmax(pred, 1), average='macro')
        curr_brier = utils.brier_score(utils.one_hot(np.array(target[target_col]), num_class), utils.softmax(pred))
        curr_ece, curr_mce = utils.calibration(utils.one_hot(np.array(target[target_col]), num_class), utils.softmax(pred))
        
        metrics_dict["Predictor"].append(model)
        metrics_dict["Acc"].append(accuracy)
        metrics_dict["F1"].append(f1)
        metrics_dict["Brier"].append(curr_brier)
        metrics_dict["ECE"].append(curr_ece)
        metrics_dict["MCE"].append(curr_mce)
        metrics_dict["Predictor Category"].append(category)
        metrics_dict["Main Model"].append(main_model)
        metrics_dict["Secondary Model"].append(sec_model)
        
        target[f"pred_{model}"] = np.argmax(pred, 1)

target.to_csv(f"evaluation/{config.dataset}_predictions.csv")
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(f"evaluation/{config.dataset}_predictors.csv")

# Load predictions and uncertainty data
pred_df = pd.read_csv(f"evaluation/{config.dataset}_predictions.csv")
unc_df = pd.read_csv(f"{config.unc_dir}/unc.csv")

# Define task parameters
max_processes = 4
cov_range = np.arange(1, 100)
tasks = [(category, models, "val", cov_range) for category, models in predictor_categories.items()]

# Process tasks in parallel
results = process_tasks(tasks, max_processes)

# Convert all results to a DataFrame
auroc_results = pd.DataFrame(results)
print("Final AUROC Results:")
print(auroc_results.head())

auroc_results.to_csv(f"evaluation/{config.dataset}_uncertainty_usefulness.csv")

## Notebook 3
