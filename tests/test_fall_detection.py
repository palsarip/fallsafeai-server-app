"""
Test script for the complete fall detection system.
"""

import sys
import os
import cv2

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.system import FallDetectionSystem
from src.detection.detector import PersonDetector
from src.pose.estimator import PoseEstimator
from src.activity.classifier import FallDetector

def test_fall_detection():
    """Test the complete fall detection system on a sample frame"""
    # Initialize system
    system = FallDetectionSystem(
        detector_model='yolo11n.pt',
        pose_model='yolo11n-pose.pt',
        telegram_token=None,  # Set your token if you want Telegram alerts
        telegram_chat_id=None,  # Set your chat ID if you want Telegram alerts
        detection_confidence=0.5,
        pose_confidence=0.5
    )
    
    # Load a test image
    test_image_path = os.path.join(os.path.dirname(__file__), 'test_image.jpg')
    
    # Check if test image exists
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    # Read the test image
    frame = cv2.imread(test_image_path)
    
    if frame is None:
        print(f"Could not read the image: {test_image_path}")
        return
    
    # Detect people
    detector = PersonDetector(model_path='yolo11n.pt', confidence=0.5)
    detections = detector.detect(frame)
    
    # Estimate poses
    pose_estimator = PoseEstimator(model_path='yolo11n-pose.pt', confidence=0.5)
    poses = pose_estimator.estimate_pose(frame)
    
    # Create fall detector
    fall_detector = FallDetector(history_size=15)
    
    # Frame height for fall detection
    frame_height = frame.shape[0]
    
    # Analyze each detected pose
    for pose in poses:
        pose_id = pose['id']
        
        # Update fall detector with new pose data
        fall_detector.update(pose_id, pose['keypoints'], pose['confidence'])
        
        # Analyze for falls (include frame height)
        result = fall_detector.analyze(pose_id, frame_height)
        
        print(f"Pose {pose_id} Analysis:")
        print(f"  Activity: {result['activity']}")
        print(f"  Is Fall: {result['is_fall']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Torso Angle: {result['torso_angle']:.1f}°")
        if 'vertical_speed' in result:
            print(f"  Vertical Speed: {result['vertical_speed']:.1f} px/s")
    
    print("Fall detection test complete.")

def main():
    """Run the fall detection test"""
    print("Starting fall detection system test...")
    test_fall_detection()

if __name__ == "__main__":
    main()