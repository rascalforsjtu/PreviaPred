#!/bin/bash
set -e  # Exit script if any command fails


# Configure path parameters (must match prediction script! Modify as needed)
OUTPUT_BASE_DIR="backend/src/data"
INPUT_IMG_DIR="backend/src/data/Total_Cases"
MASK_DIR="backend/src/data/Predicted_Masks"
RADIOMICS_OUTPUT_DIR="${OUTPUT_BASE_DIR}/Radiomics_Results"
PYTHON_SCRIPT="backend/src/Script/extract_placenta_radiomics.py"


# Create result directory if not exists
mkdir -p "${RADIOMICS_OUTPUT_DIR}"
echo "✅ Radiomics results directory created: ${RADIOMICS_OUTPUT_DIR}"


echo -e "\n========================================"
echo "Step 1: Environment Dependency Check"
echo "========================================"


echo -e "\n========================================"
echo "Step 2: Input File Validation"
echo "========================================"


# Validate raw image directory
if [ ! -d "${INPUT_IMG_DIR}" ]; then
    echo "❌ Raw image directory does not exist: ${INPUT_IMG_DIR}"
    exit 1
fi
RAW_IMGS=$(find "${INPUT_IMG_DIR}" -maxdepth 1 -type f -name "*.nii.gz" | wc -l)
if [ "${RAW_IMGS}" -eq 0 ]; then
    echo "❌ No .nii.gz files found in raw image directory: ${INPUT_IMG_DIR}"
    exit 1
fi
echo "✅ Raw image validation completed: ${RAW_IMGS} .nii.gz files found"


# Validate mask directory (match raw image count)
if [ ! -d "${MASK_DIR}" ]; then
    echo "❌ Predicted mask directory does not exist: ${MASK_DIR}"
    exit 1
fi
MASK_IMGS=$(find "${MASK_DIR}" -maxdepth 1 -type f -name "*.nii.gz" ! -name "*_0000.nii.gz" | wc -l)
if [ "${MASK_IMGS}" -ne "${RAW_IMGS}" ]; then
    echo "⚠️  Mismatch: ${RAW_IMGS} raw images vs ${MASK_IMGS} masks - may affect results"
else
    echo "✅ Mask validation completed: ${MASK_IMGS} matching mask files found"
fi


echo -e "\n========================================"
echo "Step 3: Execute Radiomics Feature Extraction"
echo "========================================"


# Run Python extraction script with parameters
python3 "${PYTHON_SCRIPT}" \
    --image-folder "${INPUT_IMG_DIR}" \
    --mask-folder "${MASK_DIR}" \
    --output-folder "${RADIOMICS_OUTPUT_DIR}" \
    --label 1 \
    --bin-width 10 \
    --resampled-spacing 1,1,1 \
    --sigma-values 1.0,2.0,3.0 \
    --wavelet-type coif1 \
    --wavelet-level 2


# Verify extraction results
FINAL_CSV="${RADIOMICS_OUTPUT_DIR}/Placenta_Features_Final.csv"
REPORT_TXT="${RADIOMICS_OUTPUT_DIR}/extraction_report.txt"
if [ -f "${FINAL_CSV}" ] && [ -f "${REPORT_TXT}" ]; then
    echo -e "\n🎉 Radiomics extraction completed successfully!"
    echo "📊 Result files:"
    echo "  - Feature table: ${FINAL_CSV}"
    echo "  - Extraction report: ${REPORT_TXT}"
    echo -e "\n💡 Tip: Use Excel or Pandas to analyze the .csv file"
else
    echo -e "\n❌ Radiomics extraction failed: Final result files not generated"
    exit 1
fi