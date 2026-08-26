"""
Advanced object detection with confidence filtering, NMS, and tracking.
Provides high-accuracy real-time object detection with reduced false positives.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict, deque
import time


class ObjectDetector:
    """
    Advanced object detection with:
    - Configurable confidence thresholds
    - Non-Maximum Suppression (NMS)
    - Per-frame tracking for stability
    - Detection filtering and deduplication
    - Performance metrics
    """
    
    def __init__(self, model_path='weights/yolo11n.pt', conf_threshold=0.5, nms_threshold=0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Model and inference
        self.model = None
        self.device = None
        
        # Tracking and stability
        self.track_history = defaultdict(lambda: deque(maxlen=30))  # 30-frame history
        self.frame_detections = []  # Detections in current frame
        self.smoothed_boxes = {}  # Stabilized bounding boxes
        
        # Performance metrics
        self.inference_time = 0
        self.total_detections = 0
        self.filtered_detections = 0
        self.frame_count = 0
        
        # Detection history for spam prevention
        self.recent_detections = defaultdict(float)
        self.detection_cooldown = 2.0  # Seconds
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model with GPU support if available"""
        if not os.path.exists(self.model_path):
            print(f"[ObjectDetector] Error: Model not found at {self.model_path}")
            return False
        
        try:
            # Auto-detect GPU
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[ObjectDetector] Using device: {self.device}")
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"[ObjectDetector] Model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"[ObjectDetector] Load error: {e}")
            return False
    
    def detect(self, frame, conf_threshold=None, nms_threshold=None, 
               apply_tracking=True, apply_filtering=True):
        """
        Detect objects in frame with optional post-processing.
        
        Args:
            frame: Input image
            conf_threshold: Override default confidence threshold
            nms_threshold: Override default NMS threshold
            apply_tracking: Apply temporal tracking for stability
            apply_filtering: Apply confidence and deduplication filtering
        
        Returns:
            List of detections: [
                {
                    'class': class_name,
                    'confidence': float,
                    'bbox': [x1, y1, x2, y2],
                    'center': [cx, cy],
                    'area': float,
                    'tracked_id': int (if tracking enabled),
                    'stable': bool
                }
            ]
        """
        if not self.model:
            return []
        
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        nms = nms_threshold if nms_threshold is not None else self.nms_threshold
        
        # Run inference
        start_time = time.time()
        results = self.model(frame, conf=conf, iou=nms, verbose=False)
        self.inference_time = time.time() - start_time
        
        detections = []
        
        for result in results:
            for box in result.boxes:
                det = self._parse_detection(box, result.names)
                
                # Apply filtering
                if apply_filtering:
                    if not self._should_keep_detection(det):
                        self.filtered_detections += 1
                        continue
                
                detections.append(det)
                self.total_detections += 1
        
        # Apply temporal tracking for stability
        if apply_tracking:
            detections = self._apply_tracking(detections)
        
        self.frame_count += 1
        self.frame_detections = detections
        
        return detections
    
    def _parse_detection(self, box, class_names):
        """Parse YOLO detection into standard format"""
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())
        class_name = class_names[class_id]
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        
        return {
            'class': class_name,
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2],
            'center': [cx, cy],
            'area': area,
            'class_id': class_id,
            'tracked_id': None,
            'stable': False
        }
    
    def _should_keep_detection(self, detection):
        """Filter detection based on various criteria"""
        # Minimum confidence
        if detection['confidence'] < self.conf_threshold:
            return False
        
        # Minimum bounding box size (avoid tiny detections)
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < 10 or height < 10:
            return False
        
        # Filter very small bounding boxes (likely false positives)
        if detection['area'] < 400:  # 20x20 minimum
            return False
        
        return True
    
    def _apply_tracking(self, detections):
        """Apply temporal tracking for smooth, stable detections"""
        for det in detections:
            # Find matching detection from previous frame
            best_match = None
            best_distance = float('inf')
            threshold = 50  # Pixel distance threshold
            
            for prev_id, history in self.track_history.items():
                if not history:
                    continue
                
                prev_center = history[-1]['center']
                curr_center = det['center']
                
                distance = np.sqrt(
                    (prev_center[0] - curr_center[0])**2 +
                    (prev_center[1] - curr_center[1])**2
                )
                
                if distance < best_distance and distance < threshold:
                    best_distance = distance
                    best_match = prev_id
            
            # Assign tracking ID
            if best_match is not None:
                det['tracked_id'] = best_match
                self.track_history[best_match].append(det)
                det['stable'] = len(self.track_history[best_match]) > 3  # Stable after 3 frames
            else:
                # New detection
                new_id = max(self.track_history.keys()) + 1 if self.track_history else 0
                det['tracked_id'] = new_id
                self.track_history[new_id].append(det)
                det['stable'] = False
        
        return detections
    
    def draw_detections(self, frame, detections, show_confidence=True, 
                       show_tracking_id=False, stable_only=False):
        """
        Draw detections on frame.
        
        Args:
            frame: Image to draw on
            detections: List of detections
            show_confidence: Show confidence percentage
            show_tracking_id: Show tracking ID
            stable_only: Only draw stable detections
        """
        for det in detections:
            if stable_only and not det['stable']:
                continue
            
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class']
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = class_name
            if show_confidence:
                label += f" {confidence:.1%}"
            if show_tracking_id and det['tracked_id'] is not None:
                label += f" #{det['tracked_id']}"
            
            # Draw label with background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-25), (x1 + text_size[0] + 5, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def set_confidence_threshold(self, threshold):
        """Update confidence threshold"""
        self.conf_threshold = max(0.0, min(1.0, threshold))
    
    def set_nms_threshold(self, threshold):
        """Update NMS threshold"""
        self.nms_threshold = max(0.0, min(1.0, threshold))
    
    def get_stats(self):
        """Return detection statistics"""
        return {
            "total_detections": self.total_detections,
            "filtered": self.filtered_detections,
            "frames_processed": self.frame_count,
            "inference_time_ms": round(self.inference_time * 1000, 2),
            "avg_fps": round(1 / max(self.inference_time, 0.001), 1),
            "confidence_threshold": self.conf_threshold,
            "nms_threshold": self.nms_threshold,
            "tracked_objects": len(self.track_history)
        }
    
    def clear_tracking(self):
        """Clear tracking history"""
        self.track_history.clear()
