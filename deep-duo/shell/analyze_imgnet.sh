#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1       
#SBATCH --time=04:00:00          
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/analyze-imgnet-%j.out

module load python/3.10
module load scipy-stack/2023b
source ~/mlenv/bin/activate

python deep-duo/scripts/analyze_imgnet.py
