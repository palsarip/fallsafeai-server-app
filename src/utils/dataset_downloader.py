"""
Utility to download the Le2i dataset from Kaggle
"""

import os
import kaggle
import zipfile
import shutil

def download_le2i_dataset(output_dir="data/le2i"):
    """
    Download and extract the Le2i fall detection dataset from Kaggle
    
    Args:
        output_dir: Directory to save the dataset
    
    Returns:
        Path to the extracted dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading Le2i dataset to {output_dir}...")
    
    try:
        # Download the dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "tuyenldvn/falldataset-imvia", 
            path=output_dir,
            unzip=True
        )
        
        print(f"Dataset downloaded and extracted to {output_dir}")
        
        # If download creates a subdirectory, move files up
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if subdirs and "fallvideos" in subdirs:
            subdir_path = os.path.join(output_dir, "fallvideos")
            # Move all files from subfolder to main folder
            for item in os.listdir(subdir_path):
                src = os.path.join(subdir_path, item)
                dst = os.path.join(output_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            
            # Remove the subfolder
            shutil.rmtree(subdir_path)
        
        return output_dir
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative manual approach:")
        print("1. Go to https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia")
        print("2. Click 'Download' button")
        print(f"3. Extract the downloaded zip file to {output_dir}")
        return None

def count_videos(dataset_dir):
    """
    Count and print information about the dataset
    """
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        return
    
    fall_count = 0
    nonfall_count = 0
    
    # Count video files by category
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".avi", ".mp4")):
                if "fall" in file.lower():
                    fall_count += 1
                else:
                    nonfall_count += 1
    
    print(f"\nDataset statistics:")
    print(f"Total videos: {fall_count + nonfall_count}")
    print(f"Fall videos: {fall_count}")
    print(f"Non-fall videos: {nonfall_count}")

if __name__ == "__main__":
    # Download the dataset
    dataset_path = download_le2i_dataset()
    
    # Print statistics
    if dataset_path:
        count_videos(dataset_path)