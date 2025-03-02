"""
Pose estimation module using YOLO model.
"""

import cv2
import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    """Pose estimation using YOLO pose model"""
    def __init__(self, model_path='yolo11n-pose.pt', confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"Loaded pose model: {model_path}")
        
    def estimate_pose(self, frame):
        """
        Estimate poses for people in frame
        
        Args:
            frame: Image/frame as numpy array
            
        Returns:
            poses: List of pose dictionaries with keypoints
        """
        results = self.model(frame, conf=self.confidence)
        
        poses = []
        for result in results:
            if result.keypoints is None:
                continue
                
            # Process all detected poses
            for i in range(len(result.keypoints.data)):
                kpts = result.keypoints.data[i].cpu().numpy()  # Get keypoints for person i
                
                # Extract coordinates and confidence
                keypoints = []
                confidences = []
                
                for kp in kpts:
                    x, y, conf = kp
                    keypoints.append((float(x), float(y)))
                    confidences.append(float(conf))
                
                poses.append({
                    'keypoints': keypoints,
                    'confidence': confidences,
                    'id': i  # Track ID for temporal analysis
                })
                
        return poses
        
    def visualize(self, frame, poses):
        """
        Draw pose skeleton on frame
        
        Args:
            frame: Input image
            poses: List of pose dictionaries
            
        Returns:
            vis_frame: Visualized frame with pose skeletons
        """
        vis_frame = frame.copy()
        
        # Define the pose connections (pairs of keypoint indices that form limbs)
        limbs = [
            (5, 7), (7, 9), (6, 8), (8, 10),  # arms
            (5, 6), (5, 11), (6, 12), (11, 12),  # torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # legs
        ]
        
        for pose in poses:
            keypoints = pose['keypoints']
            conf = pose['confidence']
            
            # Draw keypoints
            for i, (x, y) in enumerate(keypoints):
                if i < len(conf) and conf[i] > self.confidence:
                    cv2.circle(vis_frame, (int(x), int(y)), 5, (0, 255, 255), -1)
            
            # Draw limbs
            for p1, p2 in limbs:
                if (p1 < len(keypoints) and p2 < len(keypoints) and 
                    p1 < len(conf) and p2 < len(conf) and
                    conf[p1] > self.confidence and conf[p2] > self.confidence):
                    cv2.line(vis_frame, 
                            (int(keypoints[p1][0]), int(keypoints[p1][1])),
                            (int(keypoints[p2][0]), int(keypoints[p2][1])),
                            (0, 255, 0), 2)
                            
        return vis_frame