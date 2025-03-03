"""
Script to test different threshold combinations for fall detection.
"""

import os
import sys
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import debug_video function directly
# Instead of importing from tests.debug_le2i, use a direct import approach
from debug_le2i import debug_video  # This assumes both files are in the same directory

def test_threshold_combinations(dataset_dir="data/le2i", max_videos=10):
    """
    Test different combinations of thresholds for fall detection
    
    Args:
        dataset_dir: Directory containing the dataset
        max_videos: Maximum number of videos to test
    """
    # Get fall videos
    fall_videos = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".avi", ".mp4")) and "fall" in file.lower():
                fall_videos.append(os.path.join(root, file))
    
    # Check if videos were found
    if not fall_videos:
        print(f"No fall videos found in directory: {dataset_dir}")
        print("Please check that the dataset is correctly installed.")
        return None
    
    # Limit number of videos if specified
    if max_videos and max_videos < len(fall_videos):
        fall_videos = fall_videos[:max_videos]
    
    print(f"Testing threshold combinations on {len(fall_videos)} videos")
    
    # Define threshold combinations to test
    angle_thresholds = [30, 45, 60]
    ratio_thresholds = [0.8, 1.0, 1.2]
    speed_thresholds = [50, 100, 150]
    
    # Test each combination
    results = []
    
    for angle in angle_thresholds:
        for ratio in ratio_thresholds:
            for speed in speed_thresholds:
                print(f"\nTesting thresholds: Angle={angle}°, Ratio={ratio}, Speed={speed} px/s")
                
                # Modify system thresholds
                success_count = 0
                
                # Test on each video
                for video_path in tqdm(fall_videos, desc="Testing videos"):
                    # Create a new instance for each test to avoid state issues
                    from src.system import FallDetectionSystem
                    system = FallDetectionSystem(
                        detector_model='yolo11n.pt',
                        pose_model='yolo11n-pose.pt'
                    )
                    
                    # Set thresholds
                    system.fall_detector.angle_threshold = angle
                    system.fall_detector.width_height_ratio_threshold = ratio
                    system.fall_detector.vertical_speed_threshold = speed
                    
                    # Simplified test function
                    fall_detected = test_video_with_system(video_path, system)
                    
                    if fall_detected:
                        success_count += 1
                
                # Calculate success rate
                success_rate = success_count / len(fall_videos) * 100
                
                # Save result
                results.append({
                    'angle_threshold': angle,
                    'ratio_threshold': ratio,
                    'speed_threshold': speed,
                    'success_count': success_count,
                    'total_videos': len(fall_videos),
                    'success_rate': success_rate
                })
                
                print(f"Success rate: {success_rate:.1f}%")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by success rate
    results_df = results_df.sort_values('success_rate', ascending=False)
    
    # Save results
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/threshold_testing_results.csv", index=False)
    
    # Print best combination
    best = results_df.iloc[0]
    print("\nBest threshold combination:")
    print(f"Angle threshold: {best['angle_threshold']}°")
    print(f"Width/height ratio threshold: {best['ratio_threshold']}")
    print(f"Vertical speed threshold: {best['speed_threshold']} px/s")
    print(f"Success rate: {best['success_rate']:.1f}%")
    
    # Save best thresholds
    with open("outputs/best_thresholds.txt", "w") as f:
        f.write(f"angle_threshold = {best['angle_threshold']}\n")
        f.write(f"width_height_ratio_threshold = {best['ratio_threshold']}\n")
        f.write(f"vertical_speed_threshold = {best['speed_threshold']}\n")
    
    return results_df

def test_video_with_system(video_path, system):
    """Simple test function that returns True if a fall is detected"""
    # Open video
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
        
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fall_detected = False
    frame_count = 0
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue
            
        # Process frame
        detections = system.detector.detect(frame)
        poses = system.pose_estimator.estimate_pose(frame)
        
        # Check for falls
        for pose in poses:
            pose_id = pose['id']
            
            # Update fall detector
            system.fall_detector.update(pose_id, pose['keypoints'], pose['confidence'])
            
            # Analyze for falls
            if pose_id in system.fall_detector.pose_history and len(system.fall_detector.pose_history[pose_id]) >= 2:
                result = system.fall_detector.analyze(pose_id, frame_height)
                
                if result.get('is_fall', False):
                    fall_detected = True
                    break
        
        if fall_detected:
            break
    
    # Release video
    cap.release()
    
    return fall_detected

if __name__ == "__main__":
    test_threshold_combinations(max_videos=10)