#!/bin/bash

#SBATCH --job-name=train_pix
#SBATCH --gres=gpu:1
#SBATCH -o ./f_check_same.out
#SBATCH --time=24:00:00

#cp -r /data/dataset/GTA5  /local_datasets/GTA5
#cp -r /data/dataset/SYNTHIA  /local_datasets/SYNTHIA
#cp -r /data/dataset/IDD  /local_datasets/IDD
#cp -r /data/dataset/MapillaryVistas  /local_datasets/MapillaryVistas
source /data/seunan/init.sh
conda activate torch38gpu

HYDRA_FULL_ERROR=1 python main.py --config-name=gta5 lam_aug=0.10 name=f_check_same


#rm -rf /local_datasets/GTA5
#rm -rf /local_datasets/SYNTHIA
#rm -rf /local_datasets/CityScapes