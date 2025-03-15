#!/bin/bash
#SBATCH --mem=4G
#SBATCH --nodes=1       
#SBATCH --time=1:00:00       
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/take200-%j.out

module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

python deep-duo/scripts/take_200.py\
    --origin "caltech256full"\
    --target "caltech256"

python deep-duo/scripts/take_200.py\
    --origin "ImgNetfull"\
    --target "ImgNet"

python deep-duo/scripts/take_200.py\
    --origin "iwildcam-indfull"\
    --target "iwildcam-ind"

python deep-duo/scripts/take_200.py\
    --origin "iwildcam-oodfull"\
    --target "iwildcam-ood"