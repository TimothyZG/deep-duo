#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/cal256_train_and_evaluate/cal256_train_and_evaluate-%j.out
module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate
# Prepare data

export DATA_DIR=$SLURM_TMPDIR/data/caltech256
mkdir -p $DATA_DIR
tar xf Data/256_ObjectCategories.tar -C $DATA_DIR

MODEL_NAME=swin_v2_s
DATASET_NAME=caltech256

echo "MODEL_NAME = $MODEL_NAME"
echo "DATASET_NAME = $DATASET_NAME"

# Train the model
python deep-duo/scripts/hypertune.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --config_dir deep-duo/configs \
    --skip_lp True\
    --soup True\
    --init_pth checkpoints/${DATASET_NAME}/best_lp_model_${MODEL_NAME}_${DATASET_NAME}.pth


echo "Start Evaluation"
# Evaluate the fully-finetuned model
python deep-duo/scripts/evaluate.py \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_${MODEL_NAME}_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs \
    --mode ff
