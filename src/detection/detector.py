"""
Person detection module using YOLO.
"""

import cv2
from ultralytics import YOLO

class PersonDetector:
    """Person detection using YOLO"""
    def __init__(self, model_path='yolo11n.pt', confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"Loaded detection model: {model_path}")
        
    def detect(self, frame):
        """
        Detect people in frame
        
        Args:
            frame: Image/frame as numpy array (BGR format)
               
        Returns:
            detections: List of detection dictionaries with boxes, scores, etc.
        """
        results = self.model(frame, classes=[0], conf=self.confidence)  # class 0 = person
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence
                })
                
        return detections
           
    def visualize(self, frame, detections):
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input image
            detections: List of detection dictionaries
            
        Returns:
            vis_frame: Visualized frame with bounding boxes
        """
        vis_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label above the box
            label = f"Person: {conf:.2f}"
            cv2.putText(vis_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return vis_frame