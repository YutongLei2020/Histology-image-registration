#!/bin/bash
#SBATCH --job-name=test_run_testing
#SBATCH --time=3-
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem 128G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leiy28@uci.edu
#SBATCH --partition=zhanglab.p
#SBATCH --out=test_run_testing.txt
#SBATCH -w laniakea

read_dir='/home/leiy28/Histology-image-registration/'
# out_dir='/home/leiy28/Histology-image-registration/'

moving_image=${read_dir}/small_dataset_testing/2_ER_val.tif # Placeholder
fixed_image=${read_dir}/small_dataset_testing/2_HE_val.tif # Placeholder
global_model=${read_dir}/pretrained_models/checkpoint_global_deformation.pth
local_model=${read_dir}/pretrained_models/checkpoint_local_deformation.pth
save_dir=${read_dir}/example_output/ # Placeholder

python registration.py \
    --moving_path ${moving_image} \
    --fixed_path ${fixed_image} \
    --global_model_path ${global_model} \
    --local_model_path ${local_model} \
    --save_dir ${save_dir}