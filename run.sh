#!/bin/bash

# Activate the environment if needed
# source activate general_python3

INPUT_DIR="/extra/zhanglab0/INDV/leiy28/image_registration/acrobat_train_KI67_PGR"
MODEL_PATH="/extra/zhanglab0/INDV/leiy28/image_registration/global_deform/test1/checkpoint_real_sample_v3.pth"

python run_global_deformation.py \
    --input_dir "$INPUT_DIR" \
    --model_path "$MODEL_PATH"
