"""
Debug script for analyzing falls in the Le2i dataset.
"""

import cv2
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system import FallDetectionSystem

def debug_video(video_path, output_dir="outputs/debug_frames", save_frames=False):
    """
    Debug a single video to see detailed metrics and visualize pose detection
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save debug frames
        save_frames: Whether to save frames with pose visualization
    """
    # Create output directory if saving frames
    if save_frames:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize system with more sensitive thresholds
    system = FallDetectionSystem(
        detector_model='yolo11n.pt',
        pose_model='yolo11n-pose.pt'
    )
    
    # Make fall detection more sensitive
    system.fall_detector.angle_threshold = 45  # Lower from 60
    system.fall_detector.width_height_ratio_threshold = 1.0  # Lower from 1.2
    system.fall_detector.vertical_speed_threshold = 100  # Lower from 200
    
    print(f"Using modified thresholds:")
    print(f"  Angle threshold: {system.fall_detector.angle_threshold}°")
    print(f"  Width/height ratio threshold: {system.fall_detector.width_height_ratio_threshold}")
    print(f"  Vertical speed threshold: {system.fall_detector.vertical_speed_threshold} px/s")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
        
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Statistics
    fall_frames = []
    value_history = []
    
    # Create a window for visualization
    window_name = f"Fall Analysis: {os.path.basename(video_path)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame to speed up
            continue
            
        # Detect people and poses
        detections = system.detector.detect(frame)
        poses = system.pose_estimator.estimate_pose(frame)
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        # Draw detections
        if detections:
            vis_frame = system.detector.visualize(vis_frame, detections)
        
        # Draw poses
        if poses:
            vis_frame = system.pose_estimator.visualize(vis_frame, poses)
        
        # Analyze poses
        is_fall_detected = False
        for pose in poses:
            pose_id = pose['id']
            
            # Update fall detector
            system.fall_detector.update(pose_id, pose['keypoints'], pose['confidence'])
            
            # Get analysis result
            if pose_id in system.fall_detector.pose_history and len(system.fall_detector.pose_history[pose_id]) >= 2:
                history = system.fall_detector.pose_history[pose_id]
                result = system.fall_detector._analyze_fall(history, frame_height)
                
                # Store values for tracking
                value_history.append({
                    'frame': frame_count,
                    'torso_angle': result.get('torso_angle', 0),
                    'width_height_ratio': result.get('width_height_ratio', 0),
                    'vertical_speed': result.get('vertical_speed', 0),
                    'is_fall': result.get('is_fall', False)
                })
                
                # Add metrics to visualization
                y_pos = 30
                cv2.putText(vis_frame, f"Frame: {frame_count}/{total_frames}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
                
                cv2.putText(vis_frame, f"Torso angle: {result.get('torso_angle', 0):.1f}°", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
                
                cv2.putText(vis_frame, f"Width/height: {result.get('width_height_ratio', 0):.2f}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
                
                cv2.putText(vis_frame, f"Vertical speed: {result.get('vertical_speed', 0):.1f} px/s", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
                
                # Check for fall and add to visualization
                if result.get('is_fall', False):
                    is_fall_detected = True
                    cv2.putText(vis_frame, "FALL DETECTED!", (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Print detailed metrics
                    print(f"Frame {frame_count}:")
                    print(f"  Torso angle: {result.get('torso_angle', 0):.1f}° (threshold: {system.fall_detector.angle_threshold}°)")
                    print(f"  Width/height ratio: {result.get('width_height_ratio', 0):.2f} (threshold: {system.fall_detector.width_height_ratio_threshold})")
                    print(f"  Vertical speed: {result.get('vertical_speed', 0):.1f} px/s (threshold: {system.fall_detector.vertical_speed_threshold})")
                    print(f"  Is low in frame: {result.get('is_low', False)}")
                    print(f"  Fast movement: {result.get('fast_movement', False)}")
                    print(f"  Is fall: {result.get('is_fall', False)}")
                    print()
                
        # If a fall is detected, add to list
        if is_fall_detected:
            fall_frames.append(frame_count)
        
        # Display the frame
        cv2.imshow(window_name, vis_frame)
        
        # Save frame if requested
        if save_frames and (poses or is_fall_detected):
            frame_filename = f"{output_dir}/frame_{frame_count:04d}_{'fall' if is_fall_detected else 'normal'}.jpg"
            cv2.imwrite(frame_filename, vis_frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # Pause on 'p'
            cv2.waitKey(0)
    
    # Release video and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\nSummary for {os.path.basename(video_path)}:")
    print(f"Total frames processed: {len(value_history)}")
    print(f"Fall frames detected: {len(fall_frames)}")
    
    # Calculate statistics
    if value_history:
        angles = [v['torso_angle'] for v in value_history]
        ratios = [v['width_height_ratio'] for v in value_history]
        speeds = [abs(v['vertical_speed']) for v in value_history]
        
        print(f"\nStatistics:")
        print(f"Torso angle - Min: {min(angles):.1f}°, Max: {max(angles):.1f}°, Avg: {sum(angles)/len(angles):.1f}°")
        print(f"Width/height ratio - Min: {min(ratios):.2f}, Max: {max(ratios):.2f}, Avg: {sum(ratios)/len(ratios):.2f}")
        print(f"Vertical speed - Min: {min(speeds):.1f} px/s, Max: {max(speeds):.1f} px/s, Avg: {sum(speeds)/len(speeds):.1f} px/s")
    
    # Return if falls were detected
    return len(fall_frames) > 0

def process_all_fall_videos(dataset_dir="data/le2i", max_videos=10):
    """Process all fall videos in the dataset"""
    # Get all video files
    fall_videos = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".avi", ".mp4")) and "fall" in file.lower():
                fall_videos.append(os.path.join(root, file))
    
    print(f"Found {len(fall_videos)} fall videos")
    
    # If no videos were found, print a helpful message and exit
    if not fall_videos:
        print("No fall videos found. Please check:")
        print(f"1. That the dataset directory '{dataset_dir}' exists")
        print("2. The directory structure contains video files with 'fall' in their filenames")
        print("3. Video files have .avi or .mp4 extensions")
        print("\nYou can also specify a specific video path directly:")
        print("python tests/debug_le2i.py /path/to/your/video.avi")
        return
    
    # Limit number of videos if specified
    if max_videos and max_videos < len(fall_videos):
        fall_videos = fall_videos[:max_videos]
        print(f"Processing first {max_videos} videos")
    
    # Process each video
    success_count = 0
    for i, video_path in enumerate(fall_videos):
        print(f"\nProcessing video {i+1}/{len(fall_videos)}: {os.path.basename(video_path)}")
        fall_detected = debug_video(video_path, save_frames=(i < 2))  # Save frames for first two videos
        
        if fall_detected:
            success_count += 1
    
    # Print overall success rate (only if videos were found)
    print("\nOverall results:")
    print(f"Successfully detected falls in {success_count}/{len(fall_videos)} videos")
    print(f"Success rate: {(success_count/len(fall_videos))*100:.1f}%")

if __name__ == "__main__":
    # Check if a specific video path is provided
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"Processing specific video: {video_path}")
        debug_video(video_path, save_frames=True)
    else:
        # Process several fall videos from the dataset
        process_all_fall_videos(max_videos=10)