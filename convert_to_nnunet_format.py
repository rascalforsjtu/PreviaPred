import os
import json
import random
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import shutil

def get_file_basename(file_path: str) -> str:
    return os.path.basename(file_path).replace(".nii.gz", "")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert MRI data to nnUNet standard format")
    parser.add_argument("--raw-img-dir", required=True, help="Directory containing raw MRI images")
    parser.add_argument("--label-dir", required=True, help="Directory containing label masks")
    parser.add_argument("--output-base-dir", required=True, help="nnUNet raw data root directory (e.g., backend/src/data/nnUNet_raw)")
    parser.add_argument("--dataset-id", type=int, required=True, help="nnUNet dataset ID (e.g., 123)")
    parser.add_argument("--dataset-name", default="MRI_Segmentation", help="nnUNet dataset name")
    parser.add_argument("--train-val-split", type=float, default=0.7, help="Training set proportion (default 70%)")
    parser.add_argument("--label-mapping", default='{"background":0, "target":1}', help="Label mapping JSON (adjust for multi-class)")
    args = parser.parse_args()

    # Define nnUNet dataset path (format: DatasetXXX_Name)
    dataset_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    nnunet_raw_dir = join(args.output_base_dir, dataset_name)
    images_tr_dir = join(nnunet_raw_dir, "imagesTr")
    labels_tr_dir = join(nnunet_raw_dir, "labelsTr")
    maybe_mkdir_p(images_tr_dir)
    maybe_mkdir_p(labels_tr_dir)

    # Copy and format data (nnUNet single-modal naming: xxx_0000.nii.gz, label: xxx.nii.gz)
    raw_files = [f for f in os.listdir(args.raw_img_dir) if f.endswith(".nii.gz")]
    for raw_file in raw_files:
        basename = get_file_basename(raw_file)
        src_img = join(args.raw_img_dir, raw_file)
        dst_img = join(images_tr_dir, f"{basename}_0000.nii.gz")
        shutil.copy(src_img, dst_img)
        
        src_label = join(args.label_dir, f"{basename}.nii.gz")
        dst_label = join(labels_tr_dir, f"{basename}.nii.gz")
        shutil.copy(src_label, dst_label)

    # Generate splits_final.json (train/validation split)
    case_names = [get_file_basename(f) for f in raw_files]
    random.seed(42)
    random.shuffle(case_names)
    split_idx = int(len(case_names) * args.train_val_split)
    train_cases = case_names[:split_idx]
    val_cases = case_names[split_idx:]
    
    splits = [{"train": train_cases, "val": val_cases}]
    with open(join(nnunet_raw_dir, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent=4)

    # Generate dataset.json (required by nnUNet)
    label_dict = json.loads(args.label_mapping)
    generate_dataset_json(
        output_folder=nnunet_raw_dir,
        channel_names={0: "MRI"},
        labels=label_dict,
        num_training_cases=len(case_names),
        file_ending=".nii.gz",
        dataset_name=dataset_name,
        reference="Custom MRI Segmentation Dataset",
        license="Custom",
        overwrite_image_reader_writer="NibabelIOWithReorient"
    )

    print(f"✅ nnUNet format conversion completed! Dataset path: {nnunet_raw_dir}")
    print(f"📊 Data split: {len(train_cases)} training cases, {len(val_cases)} validation cases")

if __name__ == "__main__":
    main()