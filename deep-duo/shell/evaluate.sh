#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1       
#SBATCH --time=1:00:00       
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/evaluate-duo-%j.out

module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

#ImageNet, Caltech256, IwildCamIND, IwildCamOOD

DATASET="Caltech256"
echo "DATASET = $DATASET"
python deep-duo/scripts/analyze_prediction.py --dataset $DATASET
