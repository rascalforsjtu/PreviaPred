import os
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm

def get_file_basename(file_path: str) -> str:
    """Extract filename without extension (e.g., D1293632.nii.gz -> D1293632)"""
    return os.path.basename(file_path).replace(".nii.gz", "")

def validate_file_count(raw_img_dir: str, label_dir: str) -> bool:
    """Check if raw images and label files count match"""
    raw_files = [f for f in os.listdir(raw_img_dir) if f.endswith(".nii.gz")]
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".nii.gz")]
    
    if len(raw_files) != len(label_files):
        print(f"❌ Mismatched file counts: {len(raw_files)} raw images, {len(label_files)} labels")
        return False
    
    raw_basenames = set(get_file_basename(f) for f in raw_files)
    label_basenames = set(get_file_basename(f) for f in label_files)
    missing_in_raw = label_basenames - raw_basenames
    missing_in_label = raw_basenames - label_basenames
    
    if missing_in_raw:
        print(f"❌ Labels exist but raw images missing: {missing_in_raw}")
        return False
    if missing_in_label:
        print(f"❌ Raw images exist but labels missing: {missing_in_label}")
        return False
    
    print(f"✅ File counts match: {len(raw_files)} samples total")
    return True

def validate_image_properties(raw_path: str, label_path: str) -> bool:
    """Check if image properties (size, spacing, origin) match for a single sample"""
    raw_img = sitk.ReadImage(raw_path)
    label_img = sitk.ReadImage(label_path)
    
    if raw_img.GetSize() != label_img.GetSize():
        print(f"❌ Size mismatch: {get_file_basename(raw_path)} "
              f"raw {raw_img.GetSize()} vs label {label_img.GetSize()}")
        return False
    
    raw_spacing = np.round(raw_img.GetSpacing(), 3)
    label_spacing = np.round(label_img.GetSpacing(), 3)
    if not np.allclose(raw_spacing, label_spacing):
        print(f"❌ Spacing mismatch: {get_file_basename(raw_path)} "
              f"raw {raw_spacing} vs label {label_spacing}")
        return False
    
    raw_origin = np.round(raw_img.GetOrigin(), 3)
    label_origin = np.round(label_img.GetOrigin(), 3)
    if not np.allclose(raw_origin, label_origin):
        print(f"❌ Origin mismatch: {get_file_basename(raw_path)} "
              f"raw {raw_origin} vs label {label_origin}")
        return False
    
    return True

def validate_data_validity(raw_path: str) -> bool:
    """Check raw image data for anomalies (NaN/infinity)"""
    raw_img = sitk.ReadImage(raw_path)
    raw_np = sitk.GetArrayFromImage(raw_img)
    
    if np.isnan(raw_np).any():
        print(f"❌ Raw image contains NaN: {get_file_basename(raw_path)}")
        return False
    if np.isinf(raw_np).any():
        print(f"❌ Raw image contains infinity: {get_file_basename(raw_path)}")
        return False
    
    return True

def validate_label_validity(label_path: str, max_label: int = 5) -> bool:
    """Check if labels are integers and within valid range (0~max_label by default)"""
    label_img = sitk.ReadImage(label_path)
    label_np = sitk.GetArrayFromImage(label_img)
    
    if not np.issubdtype(label_np.dtype, np.integer):
        print(f"❌ Label not integer type: {get_file_basename(label_path)} (type: {label_np.dtype})")
        return False
    
    label_unique = np.unique(label_np)
    if np.min(label_unique) < 0 or np.max(label_unique) > max_label:
        print(f"❌ Label out of valid range (0~{max_label}): {get_file_basename(label_path)} "
              f"unique values: {label_unique}")
        return False
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MRI data validation script: check image-label consistency")
    parser.add_argument("--raw-img-dir", required=True, help="Directory with raw MRI images")
    parser.add_argument("--label-dir", required=True, help="Directory with label masks")
    parser.add_argument("--max-label", type=int, default=5, help="Maximum allowed label value (adjust for task)")
    args = parser.parse_args()

    if not validate_file_count(args.raw_img_dir, args.label_dir):
        exit(1)
    
    raw_files = [f for f in os.listdir(args.raw_img_dir) if f.endswith(".nii.gz")]
    all_valid = True
    
    for raw_file in tqdm(raw_files, desc="Validating samples"):
        raw_path = os.path.join(args.raw_img_dir, raw_file)
        label_basename = get_file_basename(raw_file)
        label_path = os.path.join(args.label_dir, f"{label_basename}.nii.gz")
        
        if not validate_image_properties(raw_path, label_path):
            all_valid = False
        if not validate_data_validity(raw_path):
            all_valid = False
        if not validate_label_validity(label_path, args.max_label):
            all_valid = False
    
    if all_valid:
        print("\n✅ All data validated successfully! Ready for further processing")
        exit(0)
    else:
        print("\n❌ Some data failed validation. Please fix issues and retry")
        exit(1)

if __name__ == "__main__":
    main()