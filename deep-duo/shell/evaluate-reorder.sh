#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1       
#SBATCH --time=4:00:00       
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/evaluate_duo/evaluate-duo-%j.out

module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

#ImageNet,ImageNetV2, Caltech256, IwildCamIND, IwildCamOOD
DATASET="IwildCamOOD"
echo "DATASET = $DATASET"
python deep-duo/scripts/analyze_prediction_reorder.py --dataset $DATASET
