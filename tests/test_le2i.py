"""
Test the fall detection system on the Le2i dataset.
"""

import os
import sys
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system import FallDetectionSystem
from src.utils.dataset_downloader import download_le2i_dataset, count_videos

def load_le2i_thresholds():
    """Load optimized thresholds from file"""
    thresholds = {
        'angle_threshold': 50,
        'width_height_ratio_threshold': 1.0,
        'vertical_speed_threshold': 150
    }
    
    # Try to load from file if it exists
    threshold_file = "outputs/le2i_thresholds.txt"
    if os.path.exists(threshold_file):
        with open(threshold_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=")
                    key = key.strip()
                    value = float(value.strip())
                    if key == "angle_threshold":
                        thresholds['angle_threshold'] = value
                    elif key == "width_height_ratio_threshold":
                        thresholds['width_height_ratio_threshold'] = value
                    elif key == "vertical_speed_threshold":
                        thresholds['vertical_speed_threshold'] = value
    
    return thresholds

def test_on_le2i(dataset_dir="data/le2i", output_dir="outputs/le2i_results", 
                max_videos=None, use_optimized=True, sample_rate=5):
    """
    Test fall detection system on Le2i dataset videos
    
    Args:
        dataset_dir: Directory containing the dataset
        output_dir: Directory to save results
        max_videos: Maximum number of videos to test (None for all)
        use_optimized: Whether to use optimized thresholds
        sample_rate: Process every Nth frame to speed up testing
    
    Returns:
        DataFrame with results
    """
    # Check if dataset exists, download if needed
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print("Dataset not found. Downloading...")
        dataset_dir = download_le2i_dataset(dataset_dir)
        if not dataset_dir:
            return None
    
    # Count videos in the dataset
    count_videos(dataset_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize system
    system = FallDetectionSystem(
        detector_model='yolo11n.pt',
        pose_model='yolo11n-pose.pt'
    )
    
    # Use optimized thresholds if requested
    if use_optimized:
        thresholds = load_le2i_thresholds()
        system.fall_detector.angle_threshold = thresholds['angle_threshold']
        system.fall_detector.width_height_ratio_threshold = thresholds['width_height_ratio_threshold']
        system.fall_detector.vertical_speed_threshold = thresholds['vertical_speed_threshold']
        print(f"Using optimized thresholds:")
        print(f"  Angle threshold: {system.fall_detector.angle_threshold:.1f}°")
        print(f"  Width/height ratio threshold: {system.fall_detector.width_height_ratio_threshold:.2f}")
        print(f"  Vertical speed threshold: {system.fall_detector.vertical_speed_threshold:.1f} px/s")
    
    # Get all video files
    video_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".avi", ".mp4")):
                video_files.append(os.path.join(root, file))
    
    # Limit the number of videos if specified
    if max_videos and max_videos < len(video_files):
        video_files = video_files[:max_videos]
    
    print(f"Testing on {len(video_files)} videos...")
    
    # Results tracking
    results = []
    
    # Process each video with progress bar
    for video_path in tqdm(video_files, desc="Testing videos"):
        video_name = os.path.basename(video_path)
        
        # Determine if this is a fall video based on filename
        is_fall = "fall" in video_name.lower()
        
        # Output path
        output_path = os.path.join(output_dir, f"result_{video_name}")
        
        # Reset frame counter
        system.frame_count = 0
        
        # Testing variables
        fall_detected = False
        fall_frames = []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip frames to speed up processing
            if frame_idx % sample_rate != 0:
                continue
            
            # Process frame
            _, frame_fall_detected = system.process_frame(frame)
            
            # Track falls
            if frame_fall_detected:
                fall_detected = True
                fall_frames.append(frame_idx)
        
        # Release video
        cap.release()
        
        # Add to results
        results.append({
            'video': video_name,
            'is_fall_video': is_fall,
            'fall_detected': fall_detected,
            'fall_frames': fall_frames if fall_frames else [],
            'correct_detection': fall_detected == is_fall
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    y_true = results_df['is_fall_video']
    y_pred = results_df['fall_detected']
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Fall    No Fall")
    print(f"Actual Fall     {cm[1, 1]}      {cm[1, 0]}")
    print(f"       No Fall  {cm[0, 1]}      {cm[0, 0]}")
    
    # Calculate metrics
    true_positives = cm[1, 1]
    false_positives = cm[0, 1]
    true_negatives = cm[0, 0]
    false_negatives = cm[1, 0]
    
    # Calculate performance metrics
    total = true_positives + false_positives + true_negatives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # Print metrics
    print("\nPerformance metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Fall', 'Fall']))
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, "le2i_results.csv"), index=False)
    print(f"Results saved to {os.path.join(output_dir, 'le2i_results.csv')}")
    
    # Save metrics to file
    with open(os.path.join(output_dir, "le2i_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n")
    
    return results_df

if __name__ == "__main__":
    # Set max_videos to limit the number of videos to test
    # Set sample_rate higher to speed up testing
    test_on_le2i(max_videos=20, sample_rate=10)