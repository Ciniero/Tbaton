#!/usr/bin/env python3
"""
Video to Step-by-Step Instructions Generator

This script analyzes silent video recordings and generates detailed step-by-step
text instructions describing the actions performed in the video.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import json
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, model_path: str = None):
        """Initialize the video analyzer with optional pre-trained model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize object detection (using a lightweight model)
        self.setup_object_detection()
        
        # Action recognition setup
        self.setup_action_recognition()
        
        # Frame analysis parameters
        self.frame_skip = 5  # Analyze every 5th frame
        self.min_confidence = 0.5
        
    def setup_object_detection(self):
        """Setup object detection model."""
        try:
            # Try to use YOLOv5 if available, otherwise use OpenCV DNN
            import ultralytics
            self.yolo_model = ultralytics.YOLO('yolov8n.pt')
            self.use_yolo = True
            logger.info("Using YOLOv8 for object detection")
        except ImportError:
            # Fallback to OpenCV DNN
            self.use_yolo = False
            self.setup_opencv_dnn()
            logger.info("Using OpenCV DNN for object detection")
    
    def setup_opencv_dnn(self):
        """Setup OpenCV DNN for object detection."""
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def setup_action_recognition(self):
        """Setup action recognition capabilities."""
        # Simple action recognition based on object movements and interactions
        self.action_templates = {
            'cooking': ['knife', 'cutting board', 'pan', 'stove', 'food'],
            'cleaning': ['cleaning supplies', 'sponge', 'broom', 'vacuum'],
            'working': ['laptop', 'keyboard', 'mouse', 'desk', 'chair'],
            'exercising': ['person', 'sports equipment', 'gym equipment'],
            'repairing': ['tools', 'screwdriver', 'hammer', 'wrench'],
            'organizing': ['box', 'container', 'shelf', 'storage']
        }
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video for analysis."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % self.frame_skip == 0:
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in a frame."""
        if self.use_yolo:
            return self.detect_objects_yolo(frame)
        else:
            return self.detect_objects_opencv(frame)
    
    def detect_objects_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO."""
        results = self.yolo_model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = box.conf.item()
                    if conf > self.min_confidence:
                        cls = int(box.cls.item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            'class': self.yolo_model.names[cls],
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2]
                        })
        
        return detections
    
    def detect_objects_opencv(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using OpenCV DNN (simplified version)."""
        # This is a simplified implementation
        # In practice, you'd load a pre-trained DNN model
        detections = []
        
        # Simple person detection using HOG descriptor
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        for i, (x, y, w, h) in enumerate(boxes):
            if weights[i] > self.min_confidence:
                detections.append({
                    'class': 'person',
                    'confidence': weights[i],
                    'bbox': [x, y, x + w, y + h]
                })
        
        return detections
    
    def analyze_frame_sequence(self, frames: List[np.ndarray]) -> List[Dict]:
        """Analyze a sequence of frames to identify actions."""
        frame_analyses = []
        
        for i, frame in enumerate(frames):
            detections = self.detect_objects(frame)
            
            # Analyze object interactions and movements
            analysis = {
                'frame_number': i * self.frame_skip,
                'objects': detections,
                'timestamp': i * self.frame_skip / 30.0,  # Assuming 30 FPS
                'actions': self.identify_actions(detections, frame_analyses)
            }
            
            frame_analyses.append(analysis)
        
        return frame_analyses
    
    def identify_actions(self, current_objects: List[Dict], previous_analyses: List[Dict]) -> List[str]:
        """Identify actions based on objects and their context."""
        actions = []
        
        # Get object classes
        object_classes = [obj['class'] for obj in current_objects]
        
        # Identify context-based actions
        for context, objects in self.action_templates.items():
            if any(obj in object_classes for obj in objects):
                actions.append(f"Engaging in {context}")
        
        # Identify specific object interactions
        if 'person' in object_classes:
            person_objects = [obj for obj in current_objects if obj['class'] == 'person']
            other_objects = [obj for obj in current_objects if obj['class'] != 'person']
            
            if other_objects:
                for obj in other_objects:
                    actions.append(f"Person interacting with {obj['class']}")
        
        return actions
    
    def generate_instructions(self, frame_analyses: List[Dict]) -> List[str]:
        """Generate step-by-step instructions from frame analyses."""
        instructions = []
        step_number = 1
        previous_actions = set()
        
        for analysis in frame_analyses:
            current_actions = set(analysis['actions'])
            new_actions = current_actions - previous_actions
            
            if new_actions:
                for action in new_actions:
                    timestamp = analysis['timestamp']
                    instruction = f"Step {step_number}: {action} (at {timestamp:.1f}s)"
                    instructions.append(instruction)
                    step_number += 1
                
                previous_actions = current_actions
        
        return instructions
    
    def process_video(self, video_path: str) -> Dict:
        """Process a video and generate instructions."""
        logger.info(f"Processing video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        if not frames:
            raise ValueError("No frames extracted from video")
        
        # Analyze frames
        frame_analyses = self.analyze_frame_sequence(frames)
        
        # Generate instructions
        instructions = self.generate_instructions(frame_analyses)
        
        # Create summary
        summary = {
            'video_path': video_path,
            'total_frames': len(frames),
            'analysis_frames': len(frame_analyses),
            'instructions': instructions,
            'frame_analyses': frame_analyses
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Generate step-by-step instructions from silent video')
    parser.add_argument('video_path', help='Path to the input video file')
    parser.add_argument('--output', '-o', help='Output file path (JSON format)')
    parser.add_argument('--model', '-m', help='Path to custom model file')
    parser.add_argument('--frame-skip', type=int, default=5, help='Frame skip interval for analysis')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return
    
    # Initialize analyzer
    analyzer = VideoAnalyzer(args.model)
    analyzer.frame_skip = args.frame_skip
    analyzer.min_confidence = args.confidence
    
    try:
        # Process video
        result = analyzer.process_video(args.video_path)
        
        # Print instructions
        print("\n" + "="*50)
        print("STEP-BY-STEP INSTRUCTIONS")
        print("="*50)
        for instruction in result['instructions']:
            print(instruction)
        
        print(f"\nTotal steps: {len(result['instructions'])}")
        print(f"Video duration: {result['analysis_frames'] * args.frame_skip / 30.0:.1f} seconds")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to: {args.output}")
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

if __name__ == "__main__":
    main()