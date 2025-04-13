#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/make_soup/%j.out
module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate
# Prepare data

export DATA_DIR=$SLURM_TMPDIR/data/caltech256
mkdir -p $DATA_DIR
tar xf Data/256_ObjectCategories.tar -C $DATA_DIR

# swin_v2_s, convnext_base
MODEL_NAME=swin_v2_s
DATASET_NAME=caltech256

echo "MODEL_NAME = $MODEL_NAME"
echo "DATASET_NAME = $DATASET_NAME"

python deep-duo/scripts/soup.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --config_dir deep-duo/configs \
