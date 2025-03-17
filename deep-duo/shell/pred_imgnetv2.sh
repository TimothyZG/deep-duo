#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:1       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         
#SBATCH --time=00:30:00            
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/misc/pred_on_imgnetv2-%j.out

module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

# Prepare data
export DATA_DIR=$SLURM_TMPDIR/data
mkdir -p $DATA_DIR
cp Data/imagenetv2-matched-frequency.tar $DATA_DIR
tar -xf $DATA_DIR/imagenetv2-matched-frequency.tar -C $DATA_DIR

python deep-duo/scripts/pred_on_imagenet.py \
    --data_dir $DATA_DIR \
    --version "v2"
