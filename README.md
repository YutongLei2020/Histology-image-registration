# [Project Name]

[Brief description of the project]

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Download Training Data](#download-training-data)
- [Usage](#usage)
- [License](#license)

## Prerequisites

Before running the project, ensure you have the following installed:

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Installation

Follow these steps to set up the development environment:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/YutongLei2020/Histology-image-registration.git
    cd Histology-image-registration
    ```

2.  **Set up the environment**
    
    Create the Conda environment using the provided `environment.yml` file. This will install all necessary dependencies, including Python 3.9 and required libraries.
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment**
    ```bash
    conda activate general_python3
    ```


## Download Training Data

1.  Go to [ACROBAT Grand Challenge](https://acrobat.grand-challenge.org/).
2.  Register and download the data from their website.

## Usage

Clear instructions on how to run the project.

1.  **Preprocessing Data**

    Create paired data from all images. The following example shows how to create pairs of KI67 and PGR stained images:

    ```bash
    read_dir='/extra/zhanglab0/CommonData/image_registration/acrobat/data/train'
    out_dir='/extra/zhanglab0/INDV/leiy28/image_registration/acrobat_train_KI67_PGR'
    mkdir -p ${out_dir}

    for i in {120..120}; do
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
          --data_path ${image1} \
          --data2_path ${image2} \
          --save_dir ${out_dir}/KI67_PGR_${i}/
    done;
    ```

2.  **Training Global Deformation Model**

    Train the global deformation model using the preprocessed data.

    ```bash
    out_dir='/extra/zhanglab0/INDV/leiy28/image_registration/acrobat_train_KI67_PGR'
    save_path='/extra/zhanglab0/INDV/leiy28/image_registration/global_deform/test1/checkpoint_global.pth'
    
    # Ensure the directory for save_path exists
    mkdir -p $(dirname "${save_path}")

    python Global_deformation_train.py \
          --input_path ${out_dir}/ \
          --save_path ${save_path}
    ```

3.  **Run Global Deformation & Divide Patches**

    Run the trained global deformation model on the training dataset and divide the results into patches.

    ```bash
    read_dir="/extra/zhanglab0/INDV/leiy28/image_registration/acrobat_train_KI67_PGR"
    model_path="/extra/zhanglab0/INDV/leiy28/image_registration/global_deform/test1/checkpoint_global.pth"

    # Run global deformation inference
    python run_global_deformation.py \
        --input_dir ${read_dir}/ \
        --model_path ${model_path}

    # Divide into patches
    for i in {0..120}; do
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
        
        python divide_patch.py \
          --input_fixed ${image1} \
          --input_moving ${image2} \
          --save_dir ${read_dir}
    done;
    ```

4.  **Train Local Deformation Model**

    Train the local deformation model using the patches generated in the previous step.

    ```bash
    read_dir="/extra/zhanglab0/INDV/leiy28/image_registration/acrobat_train_KI67_PGR"
    save_path="/extra/zhanglab0/INDV/leiy28/image_registration/local_deform/test1/checkpoint_local.pth"

    # Ensure the directory for save_path exists
    mkdir -p $(dirname "${save_path}")

    python Local_deformation_train.py \
        --input_path ${read_dir}/ \
        --save_path ${save_path}
    ```

5.  **Run Pipeline on Single Image Pair**

    Run the full registration pipeline on a single pair of images using the trained models.

    ```bash
    moving_image="/path/to/moving_image.tif" # Placeholder
    fixed_image="/path/to/fixed_image.tif" # Placeholder
    global_model="/extra/zhanglab0/INDV/leiy28/image_registration/global_deform/test1/checkpoint_global.pth"
    local_model="/extra/zhanglab0/INDV/leiy28/image_registration/local_deform/test1/checkpoint_local.pth"
    save_dir="/path/to/output_directory" # Placeholder

    python registration.py \
        --moving_path ${moving_image} \
        --fixed_path ${fixed_image} \
        --global_model_path ${global_model} \
        --local_model_path ${local_model} \
        --save_dir ${save_dir}
    ```

