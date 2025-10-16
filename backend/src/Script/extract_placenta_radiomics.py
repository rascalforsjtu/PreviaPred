import os
import time
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import psutil
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description="Placental radiomics feature extraction (based on nnU-Net predicted masks)")
    parser.add_argument("--image-folder", required=True, help="Original image directory (.nii.gz)")
    parser.add_argument("--mask-folder", required=True, help="Predicted mask directory (.nii.gz)")
    parser.add_argument("--output-folder", required=True, help="Radiomics results output directory")
    parser.add_argument("--label", type=int, default=1, help="Target label value in mask (default: 1)")
    parser.add_argument("--bin-width", type=int, default=10, help="Gray level histogram bin width (default: 10)")
    parser.add_argument("--resampled-spacing", type=str, default="1,1,1", help="Resampling spacing (default: 1,1,1)")
    parser.add_argument("--sigma-values", type=str, default="1.0,2.0,3.0", help="LoG filter sigma values (default: 1.0,2.0,3.0)")
    parser.add_argument("--wavelet-type", type=str, default="coif1", help="Wavelet transform type (default: coif1)")
    parser.add_argument("--wavelet-level", type=int, default=2, help="Wavelet transform level (default: 2)")
    return parser.parse_args()

def create_extractor(args):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    
    extractor.disableAllFeatures()
    
    # Enable shape features
    extractor.enableFeatureClassByName('shape', enabled=True)
    shape_features = [
        'VoxelVolume', 'SurfaceArea', 'Sphericity', 'Maximum3DDiameter',
        'Elongation', 'Flatness', 'SurfaceVolumeRatio', 'MeshVolume', 'LeastAxisLength'
    ]
    extractor.enableFeaturesByName(shape=shape_features)
    
    # Enable first-order statistics
    extractor.enableFeatureClassByName('firstorder', enabled=True)
    firstorder_features = [
        'Mean', 'Median', 'Minimum', 'Maximum', 'Range', 'Variance',
        'Skewness', 'Kurtosis', 'Energy', 'Entropy', 'RootMeanSquared',
        'RobustMeanAbsoluteDeviation', 'InterquartileRange',
        'MeanAbsoluteDeviation', '10Percentile', '90Percentile'
    ]
    extractor.enableFeaturesByName(firstorder=firstorder_features)
    
    # Enable texture features
    # GLCM features
    extractor.enableFeatureClassByName('glcm', enabled=True)
    glcm_features = [
        'Autocorrelation', 'ClusterProminence', 'ClusterShade', 'ClusterTendency',
        'Contrast', 'Correlation', 'DifferenceEntropy', 'JointEnergy', 'JointEntropy',
        'Imc1', 'Imc2'
    ]
    extractor.enableFeaturesByName(glcm=glcm_features)
    
    # GLRLM features (all)
    extractor.enableFeatureClassByName('glrlm', enabled=True)
    # GLSZM features
    extractor.enableFeatureClassByName('glszm', enabled=True)
    glszm_features = [
        'LargeAreaEmphasis', 'LargeAreaLowGrayLevelEmphasis',
        'SizeZoneNonUniformity', 'ZonePercentage'
    ]
    extractor.enableFeaturesByName(glszm=glszm_features)
    
    # GLDM features
    extractor.enableFeatureClassByName('gldm', enabled=True)
    gldm_features = ['DependenceEntropy', 'DependenceVariance', 'GrayLevelVariance']
    extractor.enableFeaturesByName(gldm=gldm_features)
    
    # NGTDM features
    extractor.enableFeatureClassByName('ngtdm', enabled=True)
    ngtdm_features = ['Coarseness', 'Contrast']
    extractor.enableFeaturesByName(ngtdm=ngtdm_features)
    
    # Configure extraction parameters
    resampled_spacing = [float(x) for x in args.resampled_spacing.split(',')]
    sigma_values = [float(x) for x in args.sigma_values.split(',')]
    
    settings = {
        'label': args.label,
        'binWidth': args.bin_width,
        'resampledPixelSpacing': resampled_spacing,
        'interpolator': sitk.sitkBSpline,
        'interpolatorMask': sitk.sitkNearestNeighbor,
        'normalize': True,
        'removeOutliers': 3,
        'padDistance': 5,
        'force2D': False
    }
    extractor.settings.update(settings)
    
    # Enable image types
    extractor.enableImageTypeByName('Original')
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': sigma_values})
    extractor.enableImageTypeByName(
        'Wavelet', 
        customArgs={'wavelet': args.wavelet_type, 'level': args.wavelet_level}
    )
    extractor.enableImageTypeByName('Gradient', customArgs={'gradientUseSpacing': True})
    
    return extractor

def process_single_case(img_path, mask_path, args):
    img_filename = os.path.basename(img_path)
    
    try:
        start_time = time.time()
        proc = psutil.Process()
        mem_before = proc.memory_info().rss / (1024 ** 2)
        
        # Read image and mask
        image = sitk.ReadImage(img_path, sitk.sitkFloat32)
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        
        # Validate mask validity
        mask_arr = sitk.GetArrayFromImage(mask)
        target_voxels = np.sum(mask_arr == args.label)
        if target_voxels < 10:
            print(f"[Skipped][{img_filename}] Insufficient target voxels ({target_voxels} < 10)")
            return None
        
        # Validate image-mask size consistency
        if image.GetSize() != mask.GetSize():
            print(f"[Skipped][{img_filename}] Image-mask size mismatch (image: {image.GetSize()}, mask: {mask.GetSize()})")
            return None
        
        # Extract features
        extractor = create_extractor(args)
        feat_result = extractor.execute(image, mask)
        
        # Organize features
        feat_dict = {
            "filename": img_filename,
            "TargetVoxelCount": target_voxels
        }
        
        # Keep non-diagnostic and non-original features
        for key, value in feat_result.items():
            if key.startswith("diagnostics") or key.startswith("original"):
                continue
            
            if isinstance(value, (int, float)):
                feat_dict[key] = round(value, 6)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                feat_dict[key] = round(value[0], 6) if isinstance(value[0], (int, float)) else np.nan
            elif isinstance(value, np.ndarray) and value.size > 0:
                feat_dict[key] = round(value.item(), 6) if value.ndim == 0 else round(value.ravel()[0], 6)
            else:
                feat_dict[key] = np.nan
        
        # Calculate memory change
        mem_after = proc.memory_info().rss / (1024 ** 2)
        mem_delta = round(mem_after - mem_before, 2)
        process_time = round(time.time() - start_time, 2)
        print(f"[Completed][{img_filename}] Features: {len(feat_dict)-1} | Time: {process_time}s | Memory: +{mem_delta}MB")
        
        return feat_dict

    except Exception as e:
        print(f"[Error][{img_filename}] Processing failed: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        # Force memory cleanup
        try:
            del image, mask, mask_arr, extractor
        except:
            pass
        gc.collect()

def main():
    args = parse_args()
    
    # Create output directory if not exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Collect image and mask files
    img_files = [f for f in os.listdir(args.image_folder) if f.endswith(".nii.gz")]
    if not img_files:
        print(f"❌ No .nii.gz image files found in {args.image_folder}")
        return
    
    # Build image-mask path mapping
    case_path_map = {}
    for img_file in img_files:
        case_id = img_file.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        mask_file = f"{case_id}.nii.gz"
        mask_path = os.path.join(args.mask_folder, mask_file)
        
        if os.path.exists(mask_path):
            case_path_map[case_id] = {
                "img": os.path.join(args.image_folder, img_file),
                "mask": mask_path
            }
        else:
            print(f"[Warning][{case_id}] Corresponding mask not found: {mask_path}")
    
    if not case_path_map:
        print(f"❌ No matching image-mask pairs found, cannot proceed with extraction")
        return
    print(f"\n[Info] Found {len(case_path_map)} valid image-mask pairs")
    
    # Auto-allocate batch size based on memory
    mem_available = psutil.virtual_memory().available / (1024 ** 3)
    batch_size = max(1, min(int(mem_available // 2), 8))
    case_list = list(case_path_map.values())
    batches = [case_list[i:i+batch_size] for i in range(0, len(case_list), batch_size)]
    print(f"[Info] Auto-batched: {len(batches)} batches, {batch_size} cases per batch")
    
    # Process all batches in parallel
    all_features = []
    total_start = time.time()
    
    for batch_idx, batch in enumerate(batches, 1):
        batch_start = time.time()
        print(f"\n[Batch {batch_idx}/{len(batches)}] Processing {len(batch)} cases")
        
        with ProcessPoolExecutor(max_workers=min(len(batch), os.cpu_count())) as executor:
            futures = {}
            for case in batch:
                future = executor.submit(
                    process_single_case,
                    case["img"],
                    case["mask"],
                    args
                )
                futures[future] = case["img"]
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_features.append(result)
    
    # Save final results
    if all_features:
        # Convert to DataFrame and adjust column order
        df = pd.DataFrame(all_features)
        cols = ["filename", "TargetVoxelCount"] + [
            col for col in df.columns if col not in ["filename", "TargetVoxelCount"]
        ]
        df = df[cols]
        
        # Save feature table
        final_csv = os.path.join(args.output_folder, "Placenta_Features_Final.csv")
        df.to_csv(final_csv, index=False, encoding="utf-8")
        
        # Generate extraction report
        total_time = round(time.time() - total_start, 2)
        success_count = len(all_features)
        total_count = len(case_path_map)
        success_rate = round((success_count / total_count) * 100, 1)
        feature_dim = len(df.columns) - 2
        
        report = f"""
=======================================
Placental Radiomics Feature Extraction Report
=======================================
Basic Information:
  Total cases: {total_count}
  Successfully processed: {success_count} ({success_rate}%)
  Failed/skipped: {total_count - success_count}
  Total time: {total_time}s ({round(total_time/60, 2)}min)
  Average time per case: {round(total_time/success_count, 2)}s

Extraction Parameters:
  Target label: {args.label}
  Gray level bin width: {args.bin_width}
  Resampling spacing: {args.resampled_spacing}
  LoG sigma values: {args.sigma_values}
  Wavelet type/level: {args.wavelet_type}/{args.wavelet_level}

Result Dimensions:
  Number of cases: {success_count}
  Number of features: {feature_dim} (including shape, first-order, texture; original features filtered out)

Output Files:
  Feature table: {final_csv}
  Report file: {os.path.join(args.output_folder, "extraction_report.txt")}
=======================================
"""
        # Save report
        report_path = os.path.join(args.output_folder, "extraction_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(report)
        print(f"✅ All results saved to: {args.output_folder}")
    else:
        print(f"❌ No valid features extracted, please check input files and parameters")

if __name__ == "__main__":
    # Limit multi-thread usage to avoid CPU overload
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    main()