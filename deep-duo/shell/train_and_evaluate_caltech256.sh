#!/bin/bash
#SBATCH --mem=64000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/train_and_evaluate-%j.out

module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

# Prepare data

export DATA_DIR=$SLURM_TMPDIR/data/caltech256
mkdir -p $DATA_DIR
tar xf Data/256_ObjectCategories.tar -C $DATA_DIR

MODEL_NAME=resnext101_32x8d
DATASET_NAME=caltech256

# Train the model
python deep-duo/scripts/hypertune.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model
python deep-duo/scripts/evaluate.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/best_lp_model_${MODEL_NAME}_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs \
    --mode lp

# Evaluate the fully-finetuned model
python deep-duo/scripts/evaluate.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/best_ff_model_${MODEL_NAME}_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs \
    --mode ff
