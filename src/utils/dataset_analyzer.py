"""
Utility to analyze the Le2i dataset and find optimal fall detection parameters
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm  # For progress bars

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.system import FallDetectionSystem
from src.utils.dataset_downloader import download_le2i_dataset, count_videos

def analyze_le2i_dataset(dataset_dir="data/le2i", sample_rate=5, max_videos=None):
    """
    Analyze the Le2i dataset to collect statistics on fall vs. non-fall cases
    
    Args:
        dataset_dir: Directory containing the dataset
        sample_rate: Process every Nth frame to speed up analysis
        max_videos: Maximum number of videos to analyze (None for all)
    
    Returns:
        DataFrame with measurements
    """
    # Check if dataset exists, download if needed
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print("Dataset not found. Downloading...")
        dataset_dir = download_le2i_dataset(dataset_dir)
        if not dataset_dir:
            return None
    
    # Count videos in the dataset
    count_videos(dataset_dir)
    
    # Initialize our fall detection system
    system = FallDetectionSystem(
        detector_model='yolo11n.pt',
        pose_model='yolo11n-pose.pt'
    )
    
    # Get all video files
    video_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".avi", ".mp4")):
                video_files.append(os.path.join(root, file))
    
    # Limit the number of videos if specified
    if max_videos and max_videos < len(video_files):
        video_files = video_files[:max_videos]
    
    print(f"Analyzing {len(video_files)} videos...")
    
    # Lists to store measurements
    all_measurements = []
    
    # Process each video with progress bar
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Determine if this is a fall video based on filename or path
        is_fall = "fall" in os.path.basename(video_path).lower()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        
        frame_count = 0
        frame_measurements = []
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Skip frames to speed up processing
            if frame_count % sample_rate != 0:
                continue
            
            # Process frame
            detections = system.detector.detect(frame)
            poses = system.pose_estimator.estimate_pose(frame)
            
            # Skip if no poses detected
            if not poses:
                continue
            
            # For each pose, collect measurements
            for pose in poses:
                pose_id = pose['id']
                
                # Update fall detector
                system.fall_detector.update(pose_id, pose['keypoints'], pose['confidence'])
                
                # Get measurements from fall detector
                if pose_id in system.fall_detector.pose_history and len(system.fall_detector.pose_history[pose_id]) >= 2:
                    history = system.fall_detector.pose_history[pose_id]
                    frame_height, _ = frame.shape[:2]
                    result = system.fall_detector._analyze_fall(history, frame_height)
                    
                    # Skip invalid results
                    if not result or not isinstance(result, dict):
                        continue
                    
                    # Save measurements with safe access
                    measurements = {
                        'video': os.path.basename(video_path),
                        'frame': frame_count,
                        'is_fall_video': is_fall,
                        'detected_as_fall': bool(result.get('is_fall', False)),
                        'torso_angle': float(result.get('torso_angle', 0)),
                        'width_height_ratio': float(result.get('width_height_ratio', 0)),
                        'vertical_speed': float(result.get('vertical_speed', 0)),
                        'is_low': bool(result.get('is_low', False)),
                        'fast_movement': bool(result.get('fast_movement', False))
                    }
                    frame_measurements.append(measurements)
        
        # Release video
        cap.release()
        
        # Add frame measurements to all measurements
        all_measurements.extend(frame_measurements)
    
    # Check if we have measurements
    if not all_measurements:
        print("No measurements collected. Check your dataset or detection system.")
        return None
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_measurements)
    
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Save measurements
    df.to_csv("outputs/le2i_measurements.csv", index=False)
    print(f"Measurements saved to outputs/le2i_measurements.csv")
    
    # Analyze measurements
    analyze_measurements(df)
    
    return df

def analyze_measurements(df):
    """
    Analyze measurements to find optimal thresholds
    
    Args:
        df: DataFrame with measurements
    """
    # Check if DataFrame is empty
    if df.empty:
        print("No data to analyze.")
        return
    
    # Separate fall and non-fall videos
    fall_df = df[df['is_fall_video'] == True]
    nonfall_df = df[df['is_fall_video'] == False]
    
    # Check if we have both fall and non-fall data
    if fall_df.empty:
        print("No fall videos detected in dataset.")
        return
    if nonfall_df.empty:
        print("No non-fall videos detected in dataset.")
        return
    
    # Calculate statistics
    print("\nStatistics for fall videos:")
    print(fall_df[['torso_angle', 'width_height_ratio', 'vertical_speed']].describe())
    
    print("\nStatistics for non-fall videos:")
    print(nonfall_df[['torso_angle', 'width_height_ratio', 'vertical_speed']].describe())
    
    # Plot distributions
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Torso angle
    axs[0].hist(fall_df['torso_angle'], bins=30, alpha=0.5, label='Fall')
    axs[0].hist(nonfall_df['torso_angle'], bins=30, alpha=0.5, label='Non-Fall')
    axs[0].set_title('Torso Angle Distribution')
    axs[0].set_xlabel('Angle (degrees)')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Width-height ratio
    axs[1].hist(fall_df['width_height_ratio'], bins=30, alpha=0.5, label='Fall')
    axs[1].hist(nonfall_df['width_height_ratio'], bins=30, alpha=0.5, label='Non-Fall')
    axs[1].set_title('Width-Height Ratio Distribution')
    axs[1].set_xlabel('Ratio')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Vertical speed
    axs[2].hist(fall_df['vertical_speed'].abs(), bins=30, alpha=0.5, label='Fall')
    axs[2].hist(nonfall_df['vertical_speed'].abs(), bins=30, alpha=0.5, label='Non-Fall')
    axs[2].set_title('Vertical Speed Distribution (Absolute)')
    axs[2].set_xlabel('Speed (pixels/s)')
    axs[2].set_ylabel('Frequency')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("outputs/le2i_analysis.png")
    print("Analysis plots saved to outputs/le2i_analysis.png")
    
    # Find optimal thresholds using ROC curve analysis
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # Calculate optimal thresholds
    thresholds = {}
    
    for feature in ['torso_angle', 'width_height_ratio', 'vertical_speed']:
        # Use absolute values for vertical speed
        values = df[feature].abs() if feature == 'vertical_speed' else df[feature]
        
        # Calculate ROC curve
        fpr, tpr, threshold = roc_curve(df['is_fall_video'], values)
        
        # Find the optimal threshold (maximize TPR while minimizing FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = threshold[optimal_idx]
        
        # For vertical speed, sometimes the optimal threshold is too low
        # Use 75th percentile of fall videos as a fallback
        if feature == 'vertical_speed' and optimal_threshold < 50:
            optimal_threshold = fall_df['vertical_speed'].abs().quantile(0.75)
        
        thresholds[feature] = optimal_threshold
    
    print("\nRecommended thresholds based on ROC analysis:")
    print(f"Angle threshold: {thresholds['torso_angle']:.1f} degrees")
    print(f"Width-height ratio threshold: {thresholds['width_height_ratio']:.2f}")
    print(f"Vertical speed threshold: {thresholds['vertical_speed']:.1f} pixels/s")
    
    # Calculate potential accuracy with these thresholds
    predictions = (
        (df['torso_angle'] > thresholds['torso_angle']) & 
        (df['width_height_ratio'] > thresholds['width_height_ratio']) & 
        ((df['vertical_speed'].abs() > thresholds['vertical_speed']) | df['is_low'])
    )
    
    accuracy = (predictions == df['is_fall_video']).mean()
    print(f"Potential accuracy with these thresholds: {accuracy:.2%}")
    
    # Save thresholds to a file
    with open("outputs/le2i_thresholds.txt", "w") as f:
        f.write(f"angle_threshold = {thresholds['torso_angle']:.1f}\n")
        f.write(f"width_height_ratio_threshold = {thresholds['width_height_ratio']:.2f}\n")
        f.write(f"vertical_speed_threshold = {thresholds['vertical_speed']:.1f}\n")
    
    print("Thresholds saved to outputs/le2i_thresholds.txt")

if __name__ == "__main__":
    # Set sample_rate higher to speed up analysis (process fewer frames)
    analyze_le2i_dataset(sample_rate=10, max_videos=20)