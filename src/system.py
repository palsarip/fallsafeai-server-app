"""
Main system module that integrates all components of the fall detection system.
"""

import os
import cv2
import time

from src.detection.detector import PersonDetector
from src.pose.estimator import PoseEstimator
from src.activity.classifier import FallDetector
from src.utils.telegram_bot import TelegramBot

class FallDetectionSystem:
    """Complete fall detection system"""
    def __init__(self, detector_model='yolo11n.pt', pose_model='yolo11n-pose.pt', 
                 telegram_token=None, telegram_chat_id=None, 
                 detection_confidence=0.5, pose_confidence=0.5):
        """
        Initialize the fall detection system
        
        Args:
            detector_model: Path to YOLO model for person detection
            pose_model: Path to YOLO model for pose estimation
            telegram_token: Telegram bot token for sending alerts
            telegram_chat_id: Telegram chat ID to send alerts to
            detection_confidence: Confidence threshold for person detection
            pose_confidence: Confidence threshold for pose estimation
        """
        print("Initializing Fall Detection System...")
        self.detector = PersonDetector(model_path=detector_model, confidence=detection_confidence)
        self.pose_estimator = PoseEstimator(model_path=pose_model, confidence=pose_confidence)
        self.fall_detector = FallDetector(history_size=15)
        
        # Initialize telegram bot if credentials provided
        self.telegram_enabled = False
        if telegram_token and telegram_chat_id:
            self.telegram_bot = TelegramBot(token=telegram_token, chat_id=telegram_chat_id)
            self.telegram_enabled = True
            print("Telegram alerts enabled")
        else:
            print("Telegram alerts disabled - no token/chat ID provided")
        
        self.frame_count = 0
        print("Fall Detection System initialized successfully")
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input video frame
            
        Returns:
            vis_frame: Processed frame with visualizations
            fall_detected: Boolean indicating if a fall was detected
        """
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]
        
        # 1. Detect people
        detections = self.detector.detect(frame)
        
        # 2. Estimate poses
        poses = self.pose_estimator.estimate_pose(frame)
        
        # 3. Visualize detections and poses
        vis_frame = self.detector.visualize(frame.copy(), detections)
        vis_frame = self.pose_estimator.visualize(vis_frame, poses)
        
        # 4. Analyze poses for fall detection
        fall_detected = False
        for pose in poses:
            try:
                pose_id = pose.get('id', 0)
                
                # Update fall detector with new pose data
                self.fall_detector.update(pose_id, pose['keypoints'], pose['confidence'])
                
                # Analyze for falls
                result = self.fall_detector.analyze(pose_id, frame_height)
                
                # Safe value extraction with default values
                is_fall = bool(result.get('is_fall', False))
                confidence = float(result.get('confidence', 0.0))
                activity = str(result.get('activity', 'Unknown'))
                torso_angle = float(result.get('torso_angle', 0.0))
                vertical_speed = float(result.get('vertical_speed', 0.0))
                
                # Choose color and label based on activity
                if is_fall:
                    color = (0, 0, 255)  # Red for falls
                    activity_label = f"FALL DETECTED! ({confidence:.2f})"
                    fall_detected = True
                    
                    # Add additional information for debugging
                    debug_info = (
                        f"Angle: {torso_angle:.1f}°, "
                        f"Speed: {vertical_speed:.1f} px/s"
                    )
                    
                    # Send alert via Telegram if enabled
                    if self.telegram_enabled and result.get("should_alert", False):
                        alert_message = (
                            f"⚠️ FALL DETECTED! ⚠️\n"
                            f"Confidence: {confidence:.2f}\n"
                            f"Time: {time.strftime('%H:%M:%S')}\n"
                            f"Angle: {torso_angle:.1f}°\n"
                            f"Speed: {abs(vertical_speed):.1f} px/s"
                        )
                        self.telegram_bot.send_alert(alert_message)
                else:
                    color = (0, 255, 0)  # Green for normal
                    activity_label = f"{activity} ({confidence:.2f})"
                    debug_info = f"Angle: {torso_angle:.1f}°"
                
                # Display activity status below bounding box if we have detections
                if len(detections) > pose_id:
                    x1, y1, x2, y2 = detections[pose_id]['bbox']
                    # Draw status below the bounding box
                    cv2.putText(vis_frame, activity_label, (x1, y2 + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    # Add debug info
                    cv2.putText(vis_frame, debug_info, (x1, y2 + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            except Exception as e:
                print(f"Error processing pose: {e}")
                continue
        
        # 5. Add system info to frame
        cv2.putText(vis_frame, f"Frame: {self.frame_count}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add fall indicator in top right
        if fall_detected:
            cv2.putText(vis_frame, "FALL DETECTED!", (frame_width - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis_frame, fall_detected
    
    def run_camera(self, camera_id=0, output_path=None):
        """
        Run detection on camera feed
        
        Args:
            camera_id: Camera index to use
            output_path: Path to save output video file
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera initialized: {width}x{height} @ {fps}fps")
        
        # Initialize video writer if output path specified
        video_writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Recording to: {output_path}")
        
        # Create display window
        window_name = "Fall Detection System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Track falls for summary
        fall_frames = []
        total_frames = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error reading from camera")
                    break
                
                total_frames += 1
                
                # Process frame
                processed_frame, fall_detected = self.process_frame(frame)
                
                # Track falls
                if fall_detected:
                    fall_frames.append(total_frames)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Write frame if recording
                if video_writer:
                    video_writer.write(processed_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed")
                    break
                elif key == ord('p'):  # Pause on 'p'
                    print("Paused - press any key to continue")
                    cv2.waitKey(0)
                elif key == ord('s') and video_writer is None and output_path:
                    # Start recording on 's' key
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    print(f"Started recording to: {output_path}")
                elif key == ord('e') and video_writer:
                    # End recording on 'e' key
                    video_writer.release()
                    video_writer = None
                    print("Recording stopped")
                
                # Add a small delay to prevent excessive CPU usage
                time.sleep(0.01)
        
        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            # Clean up
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Print summary
            print(f"Video processing complete: {self.frame_count}/{total_frames} frames")
            print(f"Falls detected: {len(fall_frames)}")
            if fall_frames:
                print(f"Fall frames: {fall_frames}")
            
            print("Analysis complete")
            print("Camera released and windows closed")
    
    def run_video(self, video_path, output_path=None):
        """
        Run detection on video file
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video file
        """
        # Check if video file exists
        if not os.path.isfile(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video loaded: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Initialize video writer if output path specified
        video_writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Will save output to: {output_path}")
        
        # Create display window
        window_name = f"Fall Detection - {os.path.basename(video_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Track falls for summary
        fall_frames = []
        
        # Reset frame count
        self.frame_count = 0
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("End of video or error reading frame")
                    break
                
                # Process frame
                processed_frame, fall_detected = self.process_frame(frame)
                
                # Track falls
                if fall_detected:
                    fall_frames.append(self.frame_count)
                
                # Add progress indicator
                progress = f"Frame: {self.frame_count}/{total_frames} ({self.frame_count/total_frames*100:.1f}%)"
                cv2.putText(processed_frame, progress, (10, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Write frame if recording
                if video_writer:
                    video_writer.write(processed_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit key pressed")
                    break
                elif key == ord('p'):  # Pause on 'p'
                    print("Paused - press any key to continue")
                    cv2.waitKey(0)
        
        finally:
            # Clean up
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Print summary
            print(f"Video processing complete: {self.frame_count}/{total_frames} frames")
            print(f"Falls detected: {len(fall_frames)}")
            if fall_frames:
                print(f"Fall frames: {fall_frames}")
                
            print("Analysis complete")
            print("Video released and windows closed")