#!/bin/bash
#SBATCH --mem=32000M
#SBATCH --nodes=1      
#SBATCH --ntasks-per-node=1    
#SBATCH --time=00:20:00            
#SBATCH --mail-user=<tiange.zhou@outlook.com>
#SBATCH --mail-type=ALL
#SBATCH --output=logs/train_and_evaluate-%j.out


wget -c https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar