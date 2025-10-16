import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def mri_nnunet_inference(
    model_dir: str,          # Path to the trained model folder
    input_img_dir: str,      # Input path for raw MRI images
    output_mask_dir: str,    # Output path for predicted masks
    use_folds: tuple = (0,), # Folds to use (e.g., (0,1,2,3,4) for multi-fold training)
    checkpoint_name: str = "checkpoint_final.pth",  # Weight file name
    device: str = "cuda"     # Inference device ("cuda" for GPU, "cpu" for CPU)
):
    # Initialize nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,        # Sliding window step size (0.5 balances speed and accuracy)
        use_gaussian=True,         # Use Gaussian weighting for window fusion
        use_mirroring=True,        # Use mirroring for test-time augmentation
        perform_everything_on_device=(device == "cuda"),  # Run all operations on GPU if available
        device=torch.device(device),  # Device configuration
        verbose=False,             # Disable detailed logging
        verbose_preprocessing=False,  # Disable preprocessing logs
        allow_tqdm=True            # Show progress bar
    )

    # Load weights and configuration from trained model folder
    print(f"Loading model: {model_dir}, folds: {use_folds}")
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir,
        use_folds=use_folds,
        checkpoint_name=checkpoint_name
    )

    # Perform inference: read images from input folder and save predictions to output folder
    print(f"Starting inference: input={input_img_dir}, output={output_mask_dir}")
    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_img_dir,
        output_folder_or_list_of_truncated_output_files=output_mask_dir,
        save_probabilities=False,  # Do not save probability maps
        overwrite=True,            # Overwrite existing files
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4,
        folder_with_segs_from_prev_stage=None,  # No previous stage
        num_parts=1,  # Single machine inference
        part_id=0
    )
    print(f"Inference completed! Predicted masks saved to: {output_mask_dir}")


if __name__ == "__main__":
    # Modify parameters as needed
    MODEL_DIR = "backend/src/data/nnUNet_results/Dataset1314_Plancenta_Segmentation/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres"
    INPUT_IMG_DIR = "backend/src/data/Raw_image_MRI"
    OUTPUT_MASK_DIR = "backend/src/data/Predicted_Masks"
    USE_FOLDS = (0,1,2,3,4)  # Use model ensemble for multi-fold training
    DEVICE = "cuda"  # Use "cpu" if no GPU available

    # Run inference
    mri_nnunet_inference(
        model_dir=MODEL_DIR,
        input_img_dir=INPUT_IMG_DIR,
        output_mask_dir=OUTPUT_MASK_DIR,
        use_folds=USE_FOLDS,
        device=DEVICE
    )
