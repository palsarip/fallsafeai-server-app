"""
Test script for person detection using YOLO.
"""

import sys
import os
import cv2

# Add parent directory to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import PersonDetector

def main():
    """Test person detection on webcam feed"""
    # Initialize detector
    detector = PersonDetector(model_path='yolo11n.pt', confidence=0.5)
    print("Detector initialized. Press 'q' to quit.")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create window
    cv2.namedWindow("Person Detection Test", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
            
            # Detect people
            detections = detector.detect(frame)
            print(f"Detected {len(detections)} people")
            
            # Visualize detections
            vis_frame = detector.visualize(frame, detections)
            
            # Display frame
            cv2.imshow("Person Detection Test", vis_frame)
            
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