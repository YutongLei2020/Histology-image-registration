# Histology image registration

The problem we aim to solve is the spatial misalignment between high-resolution histology images obtained from adjacent tissue sections. Each tissue sample is cut into thin slices, stained, and scanned separately, which causes distortions, tears, and nonlinear differences across slides. The input to our system will be a pair of whole-slide images (WSIs)—a fixed image and a moving image—and the goal is to produce a deformation field that warps the moving image so that it aligns accurately with the fixed one.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Download Training Data](#download-training-data)
- [Usage](#usage)

## Prerequisites

Before running the project, ensure you have the following installed:

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

> **Note:** Training and inference processes may require a significant amount of system memory (RAM) and a GPU with sufficient VRAM.

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

> **Note:** If you just want to test if the method runs, you can use the small dataset provided in this repository (`small_dataset_training` and `small_dataset_testing`) without downloading the full dataset.

## Usage

Clear instructions on how to run the project.


1.  **Quick Start / Testing**

    For a simplified run to test the pipeline on a small dataset, you can use the provided shell scripts. **If you run these scripts, you do NOT need to manually execute Steps 2 through 6.**
    
    *   **Training:** Run `test_run_training.sh` to train the models on a small subset.
    *   **Testing:** Run `test_run_testing.sh` to perform inference on a small subset.

    > **Note:** You may need to adjust the paths in these scripts to match your local environment.

2.  **Preprocessing Data**

    Create paired data from all images. The following example shows how to create pairs of KI67 and PGR stained images:

    ```bash
    read_dir='/extra/zhanglab0/CommonData/image_registration/acrobat/data/train'
    out_dir='/extra/zhanglab0/INDV/leiy28/image_registration/acrobat_train_KI67_PGR'
    mkdir -p ${out_dir}

    for i in {0..120}; do
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
    ```

3.  **Training Global Deformation Model**

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

4.  **Run Global Deformation & Divide Patches**

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

5.  **Train Local Deformation Model**

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

6.  **Run Pipeline on Single Image Pair**

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



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

