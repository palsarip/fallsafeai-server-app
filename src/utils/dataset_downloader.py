import os
import kaggle
import zipfile
import shutil
from pathlib import Path

def check_dataset_exists(output_dir, marker_files=None):
    """
    Check if dataset already exists in the output directory
    
    Args:
        output_dir: Directory to check
        marker_files: List of files that should exist if dataset is downloaded
                     If None, just checks if directory exists and is not empty
    
    Returns:
        bool: True if dataset appears to be already downloaded
    """
    if not os.path.exists(output_dir):
        return False
    
    # If specific marker files are provided, check for their existence
    if marker_files:
        return all(os.path.exists(os.path.join(output_dir, f)) for f in marker_files)
    
    # Otherwise check if directory is not empty
    return len(os.listdir(output_dir)) > 0

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
    
    # Check if dataset already exists
    if check_dataset_exists(output_dir):
        print(f"Le2i dataset already exists in {output_dir}, skipping download")
        return output_dir
    
    print(f"Downloading Le2i dataset to {output_dir}...")
    
    try:
        # Download the dataset with progress bar
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "tuyenldvn/falldataset-imvia", 
            path=output_dir,
            unzip=True,
            quiet=False  # Show progress bar
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
        print(f"Error downloading Le2i dataset: {e}")
        print("\nAlternative manual approach:")
        print("1. Go to https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia")
        print("2. Click 'Download' button")
        print(f"3. Extract the downloaded zip file to {output_dir}")
        return None

def download_workout_images_dataset(output_dir="data/workout_images"):
    """
    Download the workout exercises images dataset using kaggle.api
    
    Args:
        output_dir: Directory to save the dataset
    
    Returns:
        Path to the downloaded dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset already exists
    if check_dataset_exists(output_dir):
        print(f"Workout images dataset already exists in {output_dir}, skipping download")
        return output_dir
    
    print(f"Downloading workout exercises images dataset to {output_dir}...")
    
    try:
        # Download the dataset with progress bar
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "hasyimabdillah/workoutexercises-images", 
            path=output_dir,
            unzip=True,
            quiet=False  # Show progress bar
        )
        
        print(f"Workout images dataset downloaded and extracted to {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"Error downloading workout images dataset: {e}")
        print("\nAlternative manual approach:")
        print("1. Go to https://www.kaggle.com/datasets/hasyimabdillah/workoutexercises-images")
        print("2. Click 'Download' button")
        print(f"3. Extract the downloaded zip file to {output_dir}")
        return None

def download_workout_videos_dataset(output_dir="data/workout_videos"):
    """
    Download the workout fitness video dataset using kaggle.api
    
    Args:
        output_dir: Directory to save the dataset
    
    Returns:
        Path to the downloaded dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset already exists
    if check_dataset_exists(output_dir):
        print(f"Workout videos dataset already exists in {output_dir}, skipping download")
        return output_dir
    
    print(f"Downloading workout fitness video dataset to {output_dir}...")
    
    try:
        # Download the dataset with progress bar
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "hasyimabdillah/workoutfitness-video", 
            path=output_dir,
            unzip=True,
            quiet=False  # Show progress bar
        )
        
        print(f"Workout videos dataset downloaded and extracted to {output_dir}")
        return output_dir
        
    except Exception as e:
        print(f"Error downloading workout videos dataset: {e}")
        print("\nAlternative manual approach:")
        print("1. Go to https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video")
        print("2. Click 'Download' button")
        print(f"3. Extract the downloaded zip file to {output_dir}")
        return None
    
def download_human_action_recognition_HAR_dataset(output_dir="data/human_activity"):
    """
    Download the human action recognition dataset using kaggle.api
    This includes both videos and images related to human activity recognition
    
    Args:
        output_dir: Directory to save the dataset
    
    Returns:
        Path to the downloaded dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if dataset already exists
    if check_dataset_exists(output_dir):
        print(f"Human activity dataset already exists in {output_dir}, skipping download")
        return output_dir
    
    print(f"Downloading human action recognition dataset to {output_dir}...")
    
    try:
        # Download the dataset with progress bar
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "meetnagadia/human-action-recognition-har-dataset", 
            path=output_dir,
            unzip=True,
            quiet=False  # Show progress bar
        )
        
        print(f"Human activity videos dataset downloaded and extracted to {output_dir}")
        
        # Also download the images dataset
        image_subdir = os.path.join(output_dir, "images")
        os.makedirs(image_subdir, exist_ok=True)
        
        kaggle.api.dataset_download_files(
            "niharika41298/human-activity-recognition", 
            path=image_subdir,
            unzip=True,
            quiet=False  # Show progress bar
        )
        
        print(f"Human activity images dataset downloaded and extracted to {image_subdir}")
        
        return output_dir
        
    except Exception as e:
        print(f"Error downloading human activity dataset: {e}")
        print("\nAlternative manual approach:")
        print("1. Go to https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset")
        print("2. Click 'Download' button")
        print(f"3. Extract the downloaded zip file to {output_dir}")
        print("\nAnd for the images dataset:")
        print("1. Go to https://www.kaggle.com/datasets/niharika41298/human-activity-recognition")
        print("2. Click 'Download' button")
        print(f"3. Extract the downloaded zip file to {output_dir}/images")
        return None

def count_videos(dataset_dir):
    """
    Count and print information about video datasets
    """
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        return
    
    fall_count = 0
    nonfall_count = 0
    workout_count = 0
    human_activity_count = 0
    
    # Count video files by category
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".avi", ".mp4")):
                if "fall" in file.lower():
                    fall_count += 1
                elif "workout" in root.lower() or "fitness" in root.lower():
                    workout_count += 1
                elif "human_activity" in root.lower() or "har" in root.lower():
                    human_activity_count += 1
                else:
                    nonfall_count += 1
    
    print(f"\nVideo dataset statistics for {dataset_dir}:")
    print(f"Total videos: {fall_count + nonfall_count + workout_count + human_activity_count}")
    if fall_count > 0:
        print(f"Fall videos: {fall_count}")
    if nonfall_count > 0:
        print(f"Non-fall videos: {nonfall_count}")
    if workout_count > 0:
        print(f"Workout videos: {workout_count}")
    if human_activity_count > 0:
        print(f"Human activity videos: {human_activity_count}")

def count_images(dataset_dir):
    """
    Count and print information about image datasets
    """
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        return
    
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")
    image_count = 0
    category_counts = {}
    human_activity_count = 0
    
    # Count image files and categorize them
    for root, dirs, files in os.walk(dataset_dir):
        relative_path = os.path.relpath(root, dataset_dir)
        category = relative_path.split(os.path.sep)[0] if relative_path != "." else "uncategorized"
        
        is_human_activity = "human_activity" in root.lower() or "har" in root.lower()
        
        for file in files:
            if file.lower().endswith(image_extensions):
                image_count += 1
                if is_human_activity:
                    human_activity_count += 1
                category_counts[category] = category_counts.get(category, 0) + 1
    
    # Remove the 'uncategorized' category if it's empty
    if "uncategorized" in category_counts and category_counts["uncategorized"] == 0:
        del category_counts["uncategorized"]
    
    # Print statistics
    print(f"\nImage dataset statistics for {dataset_dir}:")
    print(f"Total images: {image_count}")
    print(f"Categories/classes: {len(category_counts)}")
    
    if human_activity_count > 0:
        print(f"Human activity images: {human_activity_count}")
    
    if category_counts:
        print("Images per category:")
        for category, count in sorted(category_counts.items()):
            if category != "uncategorized":
                print(f"  - {category}: {count} images")

def verify_kaggle_credentials():
    """
    Verify that Kaggle credentials are correctly set up
    """
    try:
        # Check if KAGGLE_USERNAME and KAGGLE_KEY environment variables are set
        username = os.environ.get('KAGGLE_USERNAME')
        key = os.environ.get('KAGGLE_KEY')
        
        if not username or not key:
            # Check if kaggle.json exists
            kaggle_json = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')
            if not os.path.exists(kaggle_json):
                print("Kaggle credentials not found!")
                print("Please set up your Kaggle credentials by:")
                print("1. Go to https://www.kaggle.com/account")
                print("2. Click on 'Create API Token' to download kaggle.json")
                print("3. Place the downloaded file in ~/.kaggle/kaggle.json")
                print("   OR set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
                return False
        
        # Test authentication
        kaggle.api.authenticate()
        print("Kaggle credentials verified successfully!")
        return True
        
    except Exception as e:
        print(f"Error verifying Kaggle credentials: {e}")
        return False

if __name__ == "__main__":
    # Print header
    print("=" * 60)
    print("Dataset Download Utility")
    print("=" * 60)
    
    # Verify Kaggle credentials before attempting downloads
    if not verify_kaggle_credentials():
        print("Aborting downloads due to credential issues.")
        exit(1)
    
    # Download the datasets
    print("\nStarting dataset downloads...\n")
    
    le2i_path = download_le2i_dataset()
    workout_images_path = download_workout_images_dataset()
    workout_videos_path = download_workout_videos_dataset()
    human_activity_path = download_human_action_recognition_HAR_dataset()
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    # Print statistics for each dataset
    if le2i_path:
        count_videos(le2i_path)
    
    if workout_videos_path:
        count_videos(workout_videos_path)
    
    if workout_images_path:
        count_images(workout_images_path)

    if human_activity_path:
        count_videos(human_activity_path)
        count_images(human_activity_path)
    
    print("\nProcessing complete!")
    print("=" * 60)