#!/bin/bash
#SBATCH --job-name=test_run_training
#SBATCH --time=3-
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem 64G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=leiy28@uci.edu
#SBATCH --partition=zhanglab.p
#SBATCH --out=test_run_training.txt
#SBATCH -w laniakea

read_dir='/extra/zhanglab0/CommonData/image_registration/acrobat/data/train'
out_dir='/home/leiy28/Histology-image-registration/small_dataset/'
mkdir -p ${out_dir}

for i in {2..2}; do
    image1=${read_dir}/${i}_KI67_train.tif
    image2=${read_dir}/${i}_PGR_train.tif
    if [ -f ${image1} ]; then
        echo ${image1}
    else
        echo "File does not exist."
        continue
    fi

    if [ -f ${image2} ]; then
        echo ${image2}
    else
        echo "File does not exist."
        continue
    fi
    
    python preprocessing.py \
      --data1_path ${image1} \
      --data2_path ${image2} \
      --save_dir ${out_dir}/KI67_PGR_${i}/
done;


save_path='/home/leiy28/Histology-image-registration/example_output/checkpoint_global.pth'

# Ensure the directory for save_path exists
mkdir -p $(dirname "${save_path}")

python Global_deformation_train.py \
      --input_path ${out_dir}/ \
      --save_path ${save_path}


read_dir=${out_dir}
model_path=${save_path}

# Run global deformation inference
python run_global_deformation.py \
    --input_dir ${read_dir}/ \
    --model_path ${model_path}

# Divide into patches
for i in {2..2}; do
    image1=${read_dir}/KI67_PGR_${i}/preprocess_out/cropped_fixed.tif
    image2=${read_dir}/KI67_PGR_${i}/preprocess_out/global_registered.tif
    outpath=${read_dir}/KI67_PGR_${i}
    
    if [ -f ${image2} ]; then
        echo ${image2}
    else
        echo "${image2} File does not exist."
        continue
    fi

    mkdir -p ${outpath}
    rm -rf ${read_dir}/KI67_PGR_${i}/tissue_seg
    
    python divide_patch.py \
      --input_fixed ${image1} \
      --input_moving ${image2} \
      --save_dir ${outpath}
done;


save_path='/home/leiy28/Histology-image-registration/example_output/checkpoint_local.pth'

# Ensure the directory for save_path exists
mkdir -p $(dirname "${save_path}")

python Local_deformation_train.py \
    --input_path ${read_dir}/ \
    --save_path ${save_path}
