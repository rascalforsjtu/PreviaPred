#!/bin/bash
set -e  # Exit script if any command fails

##############################################################################
# 1. Configure inference parameters (must match training script! Adjust as needed)
##############################################################################
# Base path configuration (match training script)
OUTPUT_BASE_DIR="backend/src/data"                    # Root directory for results (same as training)
DATASET_ID=1314                                       # Custom dataset ID (same as training)
DATASET_NAME="Plancenta_Segmentation"                 # Dataset name (same as training)
TRAINING_CONFIGS="3d_fullres"                         # Training configuration (same as training: 2d/3d_lowres/3d_cascade_fullres)
FOLDS="0"                                             # Single fold inference (specify 1 fold: 0/1/2/3/4)
TRAINER="nnUNetTrainer"                               # Trainer (same as training: e.g., nnUNetTrainer/nnUNetTrainer_5epochs)
CHECKPOINT_TYPE="best"                                # Inference weight type: best (optimal on validation)/final (end of training)

# Inference-specific configuration
INPUT_IMG_DIR="backend/src/data/Total_Cases"          # Directory with input images (user-specified)
OUTPUT_MASK_DIR="backend/src/data/Predicted_Masks"    # Output directory for masks + probability maps
TMP_INPUT_DIR="${OUTPUT_MASK_DIR}/tmp_input_suffix"   # Temporary dir (for files with _0000 suffix, auto-deleted after inference)


# Auto-generated dependent paths (no modification needed)
NNUNET_RAW_DIR="${OUTPUT_BASE_DIR}/nnUNet_raw"                # nnUNet raw data dir (same as training)
NNUNET_PREPROC_DIR="${OUTPUT_BASE_DIR}/nnUNet_preprocessed"   # nnUNet preprocessing dir (same as training)
NNUNET_RESULTS_DIR="${OUTPUT_BASE_DIR}/nnUNet_results"         # nnUNet training results dir (same as training)
DATASET_FULL_NAME="Dataset$(printf "%03d" ${DATASET_ID})_${DATASET_NAME}"  # Full dataset name (same as training)


##############################################################################
# 2. Environment preparation and input validation
##############################################################################
echo -e "\n========================================"
echo "Step 1: Environment preparation and input validation"
echo "========================================"

# Check nnU-Net environment variables
export nnUNet_raw="${NNUNET_RAW_DIR}"
export nnUNet_preprocessed="${NNUNET_PREPROC_DIR}"
export nnUNet_results="${NNUNET_RESULTS_DIR}"

if [ -z "${nnUNet_raw}" ] || [ -z "${nnUNet_preprocessed}" ] || [ -z "${nnUNet_results}" ]; then
    echo "❌ nnU-Net environment variables configuration failed! Check OUTPUT_BASE_DIR path"
    exit 1
fi
echo "✅ Environment variables configured:"
echo "  - nnUNet_raw: ${nnUNet_raw}"
echo "  - nnUNet_preprocessed: ${nnUNet_preprocessed}"
echo "  - nnUNet_results: ${nnUNet_results}"

# Check input image directory exists and is not empty
if [ ! -d "${INPUT_IMG_DIR}" ]; then
    echo "❌ Input image directory does not exist: ${INPUT_IMG_DIR}"
    exit 1
fi

INPUT_IMGS=$(find "${INPUT_IMG_DIR}" -maxdepth 1 -type f \( -name "*.nii.gz" -o -name "*.mha" \) | wc -l)
if [ "${INPUT_IMGS}" -eq 0 ]; then
    echo "❌ No valid image files in input directory (only .nii.gz/.mha supported): ${INPUT_IMG_DIR}"
    exit 1
fi
echo "✅ Input directory validation passed: ${INPUT_IMGS} valid image files found"

# Create output mask directory if it doesn't exist
mkdir -p "${OUTPUT_MASK_DIR}"
if [ ! -d "${OUTPUT_MASK_DIR}" ]; then
    echo "❌ Failed to create output mask directory: ${OUTPUT_MASK_DIR}"
    exit 1
fi
echo "✅ Output directory prepared: ${OUTPUT_MASK_DIR}"
echo "⚠️ Note: Probability map saving enabled, ensure sufficient disk space (probability maps ~10-20x larger than masks)"

# Check single fold training weights exist
for config in ${TRAINING_CONFIGS}; do
    for fold in ${FOLDS}; do
        WEIGHT_PATH="${NNUNET_RESULTS_DIR}/${DATASET_FULL_NAME}/${TRAINER}__nnUNetPlans__${config}/fold_${fold}/checkpoint_${CHECKPOINT_TYPE}.pth"
        if [ ! -f "${WEIGHT_PATH}" ]; then
            echo "❌ Single fold training weights not found: ${WEIGHT_PATH}"
            echo "   Verify: 1. Fold ${fold} completed training 2. Dataset ID/config/trainer match training"
            exit 1
        fi
    done
done
echo "✅ Single fold training weights validated (fold=${FOLDS}, weight type=${CHECKPOINT_TYPE})"

# Prepare temporary directory: copy files and add _0000 suffix (original files unmodified)
echo -e "\n📁 [Suffix processing] Preparing temporary input directory (adding _0000 suffix)..."
# Clean old temporary directory
if [ -d "${TMP_INPUT_DIR}" ]; then
    rm -rf "${TMP_INPUT_DIR}"
fi
mkdir -p "${TMP_INPUT_DIR}"
if [ ! -d "${TMP_INPUT_DIR}" ]; then
    echo "❌ Failed to create temporary input directory: ${TMP_INPUT_DIR}"
    exit 1
fi

# Add _0000 suffix to .nii.gz files (e.g., image.nii.gz → image_0000.nii.gz)
for img in $(find "${INPUT_IMG_DIR}" -maxdepth 1 -type f -name "*.nii.gz" | grep -v "_0000\.nii\.gz"); do
    img_basename=$(basename "${img}")
    new_basename="${img_basename%.nii.gz}_0000.nii.gz"
    cp -a "${img}" "${TMP_INPUT_DIR}/${new_basename}"
    echo "  - Processed .nii.gz: ${img_basename} → ${new_basename}"
done

# Add _0000 suffix to .mha files (e.g., image.mha → image_0000.mha)
for img in $(find "${INPUT_IMG_DIR}" -maxdepth 1 -type f -name "*.mha" | grep -v "_0000\.mha"); do
    img_basename=$(basename "${img}")
    new_basename="${img_basename%.mha}_0000.mha"
    cp -a "${img}" "${TMP_INPUT_DIR}/${new_basename}"
    echo "  - Processed .mha: ${img_basename} → ${new_basename}"
done

# Check temporary directory has valid files
TMP_VALID_IMGS=$(find "${TMP_INPUT_DIR}" -maxdepth 1 -type f \( -name "*.nii.gz" -o -name "*.mha" \) | wc -l)
if [ "${TMP_VALID_IMGS}" -eq 0 ]; then
    echo "❌ No valid processed images in temporary directory (all files may already have _0000 suffix)"
    rm -rf "${TMP_INPUT_DIR}"
    exit 1
fi
echo "✅ Temporary directory prepared: ${TMP_INPUT_DIR} (contains ${TMP_VALID_IMGS} files with _0000 suffix)"


##############################################################################
# 3. Execute nnU-Net inference
##############################################################################
echo -e "\n========================================"
echo "Step 2: Execute single fold inference (saving probability maps)"
echo "========================================"

# Run inference for configured settings (using temporary directory as input)
for config in ${TRAINING_CONFIGS}; do
    echo -e "\n📌 Inference config: ${config} | Single fold: ${FOLDS} | Weight type: ${CHECKPOINT_TYPE}"
    echo "🔄 Running inference (with probability map saving)..."
    
    nnUNetv2_predict \
        -d "${DATASET_ID}" \
        -i "${TMP_INPUT_DIR}" \
        -o "${OUTPUT_MASK_DIR}" \
        -c "${config}" \
        -f "${FOLDS}" \
        -tr "${TRAINER}" \
        -chk "checkpoint_${CHECKPOINT_TYPE}.pth" \
        --save_probabilitie \
        --verbose \

    # Verify inference results
    PRED_MASKS=$(find "${OUTPUT_MASK_DIR}" -maxdepth 1 -type f -name "*.nii.gz" | wc -l)
    PRED_PROBS=$(find "${OUTPUT_MASK_DIR}" -maxdepth 1 -type f -name "*.npz" | wc -l)
    if [ "${PRED_MASKS}" -eq 0 ] || [ "${PRED_PROBS}" -eq 0 ]; then
        echo "❌ Inference failed: no masks or probability maps generated"
        echo "  - Mask count: ${PRED_MASKS} | Probability map count: ${PRED_PROBS}"
        rm -rf "${TMP_INPUT_DIR}"
        exit 1
    fi
    echo "✅ Single fold inference completed:"
    echo "  - Generated prediction masks (.nii.gz): ${PRED_MASKS}"
    echo "  - Generated probability maps (.npz): ${PRED_PROBS}"
done


##############################################################################
# 4. Clean temporary files + inference summary
##############################################################################
echo -e "\n========================================"
echo "Step 3: Clean temporary files + inference summary"
echo "========================================"

# Clean temporary directory (original files unmodified)
echo "🧹 Deleting temporary input directory (original files unchanged)..."
rm -rf "${TMP_INPUT_DIR}"
if [ ! -d "${TMP_INPUT_DIR}" ]; then
    echo "✅ Temporary directory deleted: ${TMP_INPUT_DIR}"
else
    echo "⚠️ Failed to delete temporary directory, please clean manually: ${TMP_INPUT_DIR}"
fi

# Inference completion summary
echo -e "\n🎉 Single fold inference task (with probability maps + suffix processing) completed!"
echo "📊 Summary:"
echo "  - Original input directory (unmodified): ${INPUT_IMG_DIR}"
echo "  - Output directory (masks + probability maps): ${OUTPUT_MASK_DIR}"
echo "  - Dataset config: ${DATASET_FULL_NAME} (ID: ${DATASET_ID})"
echo "  - Inference parameters: ${TRAINER} | ${TRAINING_CONFIGS} | fold=${FOLDS} | weight=${CHECKPOINT_TYPE}"
echo "  - Generated results:"
echo "    · Prediction masks (.nii.gz): $(find "${OUTPUT_MASK_DIR}" -maxdepth 1 -type f -name "*.nii.gz" | wc -l)"
echo "    · Probability maps (.npz): $(find "${OUTPUT_MASK_DIR}" -maxdepth 1 -type f -name "*.npz" | wc -l)"
echo -e "\n💡 Tips:"
echo "  1. Masks can be viewed with ITK-SNAP/3D Slicer (.nii.gz format)"
echo "  2. Probability maps (.npz) can be used for subsequent multi-model ensembling (e.g., nnUNetv2_ensemble) or quantitative analysis"
echo "  3. Probability maps are large; delete if not needed to free disk space"
