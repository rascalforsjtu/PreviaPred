#!/bin/bash
set -e  # Terminate script if any command fails

##############################################################################
# 1. Configuration parameters (modify according to your needs!)
##############################################################################
RAW_IMG_DIR="backend/src/data/Raw_image_MRI"          # Raw MRI images directory
LABEL_DIR="backend/src/data/Label_mask_MRI"           # Label mask directory
OUTPUT_BASE_DIR="backend/src/data"                    # Root directory for results
DATASET_ID=1314                                       # Custom nnUNet dataset ID (e.g., 123)
DATASET_NAME="Plancenta_Segmentation"                 # Dataset name
MAX_LABEL=1                                           # Maximum allowed label value (set 1 for binary classification)
TRAINING_CONFIGS="3d_fullres"                         # Training configuration (2d/3d_lowres/3d_cascade_fullres)
FOLDS="0"                                             # 5-fold cross-validation
TRAINER="nnUNetTrainer"                               # Trainer (default: nnUNetTrainer, use nnUNetTrainer_5epochs for quick training)
NUM_PROCESSES=8                                       # Number of processes for preprocessing/training (adjust based on CPU/GPU)
LABEL_MAPPING='{"background":0, "target":1}'          # Label mapping (modify for multi-class)

# Auto-generate subsequent paths
NNUNET_RAW_DIR="${OUTPUT_BASE_DIR}/nnUNet_raw"        # nnUNet raw data directory
NNUNET_PREPROC_DIR="${OUTPUT_BASE_DIR}/nnUNet_preprocessed"  # Preprocessed data directory
NNUNET_RESULTS_DIR="${OUTPUT_BASE_DIR}/nnUNet_results"        # Training results (models/logs) directory
DATASET_FULL_NAME="Dataset$(printf "%03d" ${DATASET_ID})_${DATASET_NAME}"  # Full nnUNet dataset name


##############################################################################
# 2. Data validation
##############################################################################
echo -e "\n========================================"
echo "Step 1: Data validation (check image-label consistency)"
echo "========================================"
python backend/src/Script/validate_mri_data.py \
    --raw-img-dir "${RAW_IMG_DIR}" \
    --label-dir "${LABEL_DIR}" \
    --max-label "${MAX_LABEL}"

if [ $? -ne 0 ]; then
    echo "❌ Data validation failed! Please fix issues and retry."
    exit 1
fi


##############################################################################
# 3. Convert to nnUNet format
##############################################################################
echo -e "\n========================================"
echo "Step 2: Convert to nnUNet standard format"
echo "========================================"
python backend/src/Script/convert_to_nnunet_format.py \
    --raw-img-dir "${RAW_IMG_DIR}" \
    --label-dir "${LABEL_DIR}" \
    --output-base-dir "${NNUNET_RAW_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-name "${DATASET_NAME}" \
    --label-mapping "${LABEL_MAPPING}"

if [ $? -ne 0 ]; then
    echo "❌ nnUNet format conversion failed!"
    exit 1
fi


##############################################################################
# 4. nnUNet planning and preprocessing
##############################################################################
echo -e "\n========================================"
echo "Step 3: Experiment planning and data preprocessing"
echo "========================================"
# Set nnUNet environment variables (specify data/results paths)
export nnUNet_raw="${NNUNET_RAW_DIR}"
export nnUNet_preprocessed="${NNUNET_PREPROC_DIR}"
export nnUNet_results="${NNUNET_RESULTS_DIR}"

# Execute planning and preprocessing (--np specifies number of processes)
nnUNetv2_plan_and_preprocess \
    -d "${DATASET_ID}" \
    -c ${TRAINING_CONFIGS} \
    -np "${NUM_PROCESSES}"

if [ $? -ne 0 ]; then
    echo "❌ Experiment planning and preprocessing failed!"
    exit 1
fi


##############################################################################
# 5. Model training (5-fold cross-validation)
##############################################################################
echo -e "\n========================================"
echo "Step 4: Model training (5-fold cross-validation)"
echo "========================================"
for config in ${TRAINING_CONFIGS}; do
    echo -e "\n📌 Starting training for config: ${config}"
    for fold in ${FOLDS}; do
        echo -e "🔄 Training fold ${fold}..."
        nnUNetv2_train \
            "${DATASET_ID}" \
            "${config}" \
            "${fold}" \
            -tr "${TRAINER}" \
            --npz  # Save prediction probability maps (for subsequent integration)
    done
done

if [ $? -ne 0 ]; then
    echo "❌ Model training failed!"
    exit 1
fi
