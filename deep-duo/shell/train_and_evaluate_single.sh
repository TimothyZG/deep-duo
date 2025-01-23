#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:2         
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         
#SBATCH --time=16:00:00            
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

MODEL_NAME=maxvit_t
DATASET_NAME=iwildcam

echo "MODEL_NAME = $MODEL_NAME"
echo "DATASET_NAME = $DATASET_NAME"

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
    --config_dir deep-duo/configs

# Evaluate the fully-finetuned model
python deep-duo/scripts/evaluate.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/best_ff_model_${MODEL_NAME}_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs
