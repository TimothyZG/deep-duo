#!/bin/bash
#SBATCH --mem=64000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:4               # Request 1 GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16         # Number of CPU cores per task
#SBATCH --time=24:00:00            # 24 hours
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/train_and_evaluate-%j.out

module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

# Prepare data
export DATA_DIR=$SLURM_TMPDIR/data/iwildcam_v2.0
mkdir -p $DATA_DIR
tar xf Data/iwildcam_v2.0.tar -C $DATA_DIR --strip-components=1

MODEL_NAME=resnet18

# Train the model
python deep-duo/scripts/hypertune.py \
    --model_name $MODEL_NAME \
    --dataset_name iwildcam \
    --config_dir deep-duo/configs

# Evaluate the model
python deep-duo/scripts/evaluate.py \
    --model_name $MODEL_NAME \
    --dataset_name iwildcam \
    --checkpoint checkpoints/best_model_${MODEL_NAME}_iwildcam.pth \
    --config_dir deep-duo/configs
