"""
Test script for pose estimation using YOLO.
"""

import sys
import os
import cv2

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose.estimator import PoseEstimator

def main():
    """Test pose estimation on webcam feed"""
    # Initialize pose estimator
    pose_estimator = PoseEstimator(model_path='yolo11n-pose.pt', confidence=0.5) 
    print("Pose estimator initialized. Press 'q' to quit.")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create window
    cv2.namedWindow("Pose Estimation Test", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
            
            # Estimate poses
            poses = pose_estimator.estimate_pose(frame)
            print(f"Detected {len(poses)} poses")
            
            # Visualize poses
            vis_frame = pose_estimator.visualize(frame, poses)
            
            # Display frame
            cv2.imshow("Pose Estimation Test", vis_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Test completed")

if __name__ == "__main__":
    main()