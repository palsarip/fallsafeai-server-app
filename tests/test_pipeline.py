import sys
import os
import time
import cv2

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.detector import PersonDetector
from src.pose.estimator import PoseEstimator
from src.activity.classifier import FallDetector

# Initialize components
person_detector = PersonDetector()
pose_estimator = PoseEstimator()
fall_detector = FallDetector()

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # Get frame height for fall detection
        frame_height = frame.shape[0]
        
        # Detect people
        detections = person_detector.detect(frame)
        
        # Estimate poses
        poses = pose_estimator.estimate_pose(frame)
        
        # Track and analyze poses for fall detection
        for i, pose in enumerate(poses):
            # Use detection index as a simple tracking ID
            pose_id = i
            
            # Update fall detector with new pose data
            fall_detector.update(pose_id, pose['keypoints'], pose['confidence'])
            
            # Analyze for falls (now including frame height)
            result = fall_detector.analyze(pose_id, frame_height)
            
            # Display activity status
            if i < len(detections):
                x1, y1, x2, y2 = detections[i]['bbox']
                
                # Choose color based on activity
                if result["is_fall"]:
                    color = (0, 0, 255)  # Red for falls
                    label = f"FALL DETECTED! ({result['confidence']:.2f})"
                else:
                    color = (0, 255, 0)  # Green for normal
                    label = f"{result['activity']} ({result['confidence']:.2f})"
                
                # Draw box with activity
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Visualize poses
        vis_frame = frame.copy()
        vis_frame = pose_estimator.visualize(vis_frame, poses)
        
        # Display the frame
        cv2.imshow("FallSafeAI Detection", vis_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()