#!/bin/bash
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/throughput/%j.out
module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate


MODEL1_NAME=swin_v2_b
MODEL2_NAME=convnext_small
M_HEAD=3
BS=64

# Single Model
python deep-duo/scripts/throughput.py \
    --model1 $MODEL1_NAME \
    --m_head $M_HEAD \
    --batch_size $BS

# Duo
python deep-duo/scripts/throughput.py \
    --model1 $MODEL1_NAME \
    --model2 $MODEL2_NAME \
    --m_head $M_HEAD \
    --batch_size $BS
