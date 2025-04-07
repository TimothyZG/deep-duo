#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:1      
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         
#SBATCH --time=10:00:00            
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/temp_scale_and_evaluate-%j.out

module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

# Prepare data
export DATA_DIR=$SLURM_TMPDIR/data/iwildcam_v2.0
mkdir -p $DATA_DIR
tar xf Data/iwildcam_v2.0.tar -C $DATA_DIR --strip-components=1

DATASET_NAME=iwildcam

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name mobilenet_v3_large \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/${DATASET_NAME}/best_ff_model_mobilenet_v3_large_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name mnasnet1_3 \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_mnasnet1_3_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name shufflenet_v2_x2_0 \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_shufflenet_v2_x2_0_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name efficientnet_b1 \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_efficientnet_b1_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name efficientnet_b2 \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_efficientnet_b2_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name convnext_tiny \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_convnext_tiny_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name swin_t \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_swin_t_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name maxvit_t \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_maxvit_t_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name convnext_small \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_convnext_small_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name swin_v2_s \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_swin_v2_s_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs

# Evaluate the linear-probed model (prediction csv has suffix lp as indicator)
python deep-duo/scripts/temp_scale_and_evaluate.py \
    --model_name convnext_base \
    --dataset_name $DATASET_NAME \
    --checkpoint checkpoints/${DATASET_NAME}/best_ff_model_convnext_base_${DATASET_NAME}.pth \
    --config_dir deep-duo/configs
