"""
Main script to run the fall detection system.
"""

import os
import argparse
from src.system import FallDetectionSystem

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fall Detection System')
    
    # Input source
    parser.add_argument('--source', type=str, default='0', 
                        help='Source for detection: camera index (e.g., 0) or video file path')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video file')
    
    # Model options
    parser.add_argument('--detector-model', type=str, default='yolo11n.pt', 
                        help='Path to yolo11 detector model')
    parser.add_argument('--pose-model', type=str, default='yolo11n-pose.pt', 
                        help='Path to yolo11 pose model')
    
    # Detection parameters
    parser.add_argument('--conf', type=float, default=0.5, 
                        help='Confidence threshold for detections')
    
    # Telegram alerts
    parser.add_argument('--telegram-token', type=str, default=None, 
                        help='Telegram bot token for alerts')
    parser.add_argument('--telegram-chat-id', type=str, default=None, 
                        help='Telegram chat ID for alerts')
    
    return parser.parse_args()

def main():
    """Main function to run the fall detection system"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory if needed
    if args.output and not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize the system
    system = FallDetectionSystem(
        detector_model=args.detector_model,
        pose_model=args.pose_model,
        telegram_token=args.telegram_token,
        telegram_chat_id=args.telegram_chat_id,
        detection_confidence=args.conf,
        pose_confidence=args.conf
    )
    
    # Determine if source is camera or video file
    source = args.source
    try:
        # If source is a number, it's a camera index
        camera_id = int(source)
        print(f"Using camera {camera_id}")
        system.run_camera(camera_id=camera_id, output_path=args.output)
    except ValueError:
        # If source is not a number, it's a video file
        if os.path.isfile(source):
            print(f"Processing video file: {source}")
            system.run_video(video_path=source, output_path=args.output)
        else:
            print(f"Error: Source '{source}' not found")

if __name__ == "__main__":
    main()