"""
Fall detection module based on pose analysis.
Implementation based on research paper: "Enhanced Fall Detection Using YOLOv7-W6-Pose for Real-Time Elderly Monitoring"
"""

import math
import time
from collections import deque

class FallDetector:
    """Fall detection based on pose analysis"""
    def __init__(self, history_size=15, min_alert_interval=5.0):
        self.pose_history = {}  # Dictionary to track poses by person ID
        self.history_size = history_size  # Number of frames to keep
        self.min_alert_interval = min_alert_interval  # Minimum time between alerts (seconds)
        self.last_alert_time = {}  # To avoid repeated alerts
        
        # Parameters from the research paper
        self.angle_threshold = 60  # Degrees, threshold for horizontal orientation
        self.width_height_ratio_threshold = 1.2  # Threshold for unusual body proportions
        self.vertical_speed_threshold = 200  # Pixels per second, for sudden drops
        self.height_ratio_threshold = 0.7  # Relative height in frame
        
        print("Fall detector initialized with parameters:")
        print(f"  - Angle threshold: {self.angle_threshold}°")
        print(f"  - Width/height ratio threshold: {self.width_height_ratio_threshold}")
        print(f"  - Vertical speed threshold: {self.vertical_speed_threshold} px/s")
        print(f"  - Height ratio threshold: {self.height_ratio_threshold}")
        
    def update(self, pose_id, keypoints, confidence):
        """
        Add new pose data to history
        
        Args:
            pose_id: ID of the person
            keypoints: List of (x,y) coordinates for each keypoint
            confidence: List of confidence values for each keypoint
        """
        # Initialize history for new person
        if pose_id not in self.pose_history:
            self.pose_history[pose_id] = deque(maxlen=self.history_size)
            self.last_alert_time[pose_id] = 0
            
        # Add new pose to history
        self.pose_history[pose_id].append({
            'keypoints': keypoints,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
    def analyze(self, pose_id, frame_height):
        """
        Analyze pose history to detect falls
        
        Args:
            pose_id: ID of the person to analyze
            frame_height: Height of the video frame
            
        Returns:
            Dictionary containing analysis results
        """
        if pose_id not in self.pose_history or len(self.pose_history[pose_id]) < 2:
            return {"is_fall": False, "activity": "unknown", "confidence": 0.0}
            
        history = self.pose_history[pose_id]
        
        # Fall detection analysis
        result = self._analyze_fall(history, frame_height)
        
        # Throttle alerts to avoid spamming
        current_time = time.time()
        if result["is_fall"] and (current_time - self.last_alert_time.get(pose_id, 0)) > self.min_alert_interval:
            self.last_alert_time[pose_id] = current_time
            result["should_alert"] = True
        else:
            result["should_alert"] = False
            
        return result
    
    def _analyze_fall(self, history, frame_height):
        """
        Analyze if a fall has occurred based on pose data
        Implementation based on the journal's approach
        
        Args:
            history: Queue of pose data over time
            frame_height: Height of the video frame
            
        Returns:
            Dictionary containing detailed analysis results
        """
        latest_pose = history[-1]
        prev_pose = history[-2]  # Previous frame for speed calculation
        
        keypoints = latest_pose['keypoints']
        conf = latest_pose['confidence']
        
        # Ensure we have enough keypoints for analysis
        if len(keypoints) < 17:  # YOLO pose has 17 keypoints
            return {"is_fall": False, "activity": "unknown", "confidence": 0.0}
            
        # 1. Calculate length factor (used to adjust thresholds based on body size)
        # As described in the paper, this helps adapt the algorithm to different body sizes
        left_shoulder = keypoints[5] if 5 < len(keypoints) else None
        left_hip = keypoints[11] if 11 < len(keypoints) else None
        
        if left_shoulder is None or left_hip is None or conf[5] < 0.5 or conf[11] < 0.5:
            return {"is_fall": False, "activity": "unknown", "confidence": 0.0}
            
        length_factor = math.sqrt((left_shoulder[0] - left_hip[0])**2 + 
                                 (left_shoulder[1] - left_hip[1])**2)
        
        # 2. Extract key body points with good confidence
        required_indices = [5, 6, 11, 12]  # Left & right shoulders, left & right hips
        if any(i >= len(keypoints) or i >= len(conf) or conf[i] < 0.5 for i in required_indices):
            return {"is_fall": False, "activity": "unknown", "confidence": 0.0}
        
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # 3. Calculate midpoints for shoulders and hips
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0])/2, 
                       (left_shoulder[1] + right_shoulder[1])/2)
        hip_mid = ((left_hip[0] + right_hip[0])/2, 
                  (left_hip[1] + right_hip[1])/2)
        
        # 4. Calculate torso angle with vertical axis
        # This is a key metric from the paper - analyzing body orientation
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        torso_angle = abs(math.degrees(math.atan2(dx, dy)))  # 0=vertical, 90=horizontal
        
        # 5. Calculate body dimensions 
        # The paper emphasizes analyzing the difference between width and height
        torso_height = abs(shoulder_mid[1] - hip_mid[1])
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        body_width = max(shoulder_width, hip_width)
        
        width_height_ratio = body_width / torso_height if torso_height > 0 else 0
        
        # 6. Calculate position in frame (key point from the paper)
        # This checks if the person is lower in the frame (fallen on floor)
        shoulder_height_ratio = shoulder_mid[1] / frame_height if frame_height > 0 else 0
        is_low = shoulder_height_ratio > self.height_ratio_threshold
        
        # 7. Calculate vertical movement speed between frames
        # The paper highlights this as important for distinguishing falls from lying down
        prev_keypoints = prev_pose['keypoints']
        vertical_speed = 0
        fast_movement = False
        
        if len(prev_keypoints) >= 17 and 5 < len(prev_keypoints) and 6 < len(prev_keypoints):
            prev_left_shoulder = prev_keypoints[5]
            prev_right_shoulder = prev_keypoints[6]
            prev_shoulder_mid = ((prev_left_shoulder[0] + prev_right_shoulder[0])/2, 
                                (prev_left_shoulder[1] + prev_right_shoulder[1])/2)
            
            time_diff = latest_pose['timestamp'] - prev_pose['timestamp']
            if time_diff > 0:
                vertical_speed = (shoulder_mid[1] - prev_shoulder_mid[1]) / time_diff
                fast_movement = abs(vertical_speed) > self.vertical_speed_threshold
        
        # 8. Combine factors to detect falls (as described in the journal's algorithm)
        is_horizontal = torso_angle > self.angle_threshold
        unusual_ratio = width_height_ratio > self.width_height_ratio_threshold
        
        # Core fall detection logic from the paper
        is_fall = is_horizontal and unusual_ratio and (is_low or fast_movement)
        
        # 9. Determine activity and confidence
        if is_fall:
            activity = "FALL"
            # Higher confidence for clearer falls (more horizontal, faster movement)
            confidence = min(1.0, (torso_angle / 90.0) * 
                           (width_height_ratio / self.width_height_ratio_threshold) * 
                           (1.0 if fast_movement else 0.8))
        elif is_horizontal and not fast_movement:
            activity = "LYING DOWN"
            confidence = 0.7
        elif unusual_ratio:
            activity = "BENDING"
            confidence = 0.6
        else:
            activity = "STANDING"
            confidence = 1.0 - (torso_angle / 90.0)
        
        # Return detailed analysis results (helpful for debugging and tuning)
        return {
            "is_fall": is_fall,
            "activity": activity,
            "confidence": confidence,
            "torso_angle": torso_angle,
            "width_height_ratio": width_height_ratio,
            "is_horizontal": is_horizontal,
            "is_low": is_low,
            "vertical_speed": vertical_speed,
            "fast_movement": fast_movement,
            "length_factor": length_factor
        }