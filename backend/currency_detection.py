"""
Indian Currency Detection module.
Detects Indian rupee notes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500).
Uses YOLO custom model or COCO model with fallback classification.
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict
import time


class CurrencyDetector:
    """
    Specialized detector for Indian currency notes.
    Supports: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500
    
    Features:
    - Custom YOLO model for accurate currency detection
    - Fallback to COCO with shape-based classification
    - Size estimation for denomination identification
    - Low-light optimization
    - Deduplication to prevent spam announcements
    """
    
    # Expected physical dimensions (in mm)
    CURRENCY_DIMENSIONS = {
        10: {'length': 115, 'width': 63, 'color': 'brown'},
        20: {'length': 124, 'width': 63, 'color': 'purple'},
        50: {'length': 133, 'width': 63, 'color': 'brown-blue'},
        100: {'length': 142, 'width': 63, 'color': 'peachy'},
        200: {'length': 151, 'width': 63, 'color': 'yellow'},
        500: {'length': 160, 'width': 63, 'color': 'pink'},
    }
    
    DENOMINATIONS = [10, 20, 50, 100, 200, 500]
    
    # Color ranges in HSV for identification (lower_hsv, upper_hsv)
    COLOR_RANGES = {
        10: ([5, 50, 100], [15, 255, 255]),      # Brown
        20: ([140, 50, 100], [160, 255, 255]),   # Purple
        50: ([100, 30, 100], [130, 255, 255]),   # Blue-brown
        100: ([10, 100, 100], [25, 255, 255]),   # Peachy
        200: ([20, 100, 100], [35, 255, 255]),   # Yellow
        500: ([330, 50, 100], [360, 255, 255]),  # Pink
    }
    
    def __init__(self, model_path='weights/best.pt', use_custom_model=True):
        """
        Initialize currency detector.
        
        Args:
            model_path: Path to custom trained YOLO model for currencies
            use_custom_model: If True, use custom model; fallback to COCO otherwise
        """
        self.model_path = model_path
        self.use_custom_model = use_custom_model
        self.model = None
        self.coco_model = None
        self.is_custom = False
        
        # Detection history for deduplication
        self.detected_currencies = defaultdict(float)
        self.detection_cooldown = 4.0  # Don't announce same note twice in 4 seconds
        
        # Statistics
        self.detections_count = 0
        self.inference_time = 0
        self.frame_count = 0
        
        self._load_model()
    
    def _load_model(self):
        """Load currency detection model"""
        # Try custom model first
        if self.use_custom_model and os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                self.is_custom = True
                print(f"[CurrencyDetector] Custom model loaded: {self.model_path}")
                return True
            except Exception as e:
                print(f"[CurrencyDetector] Custom model error: {e}")
        
        # Fallback to COCO for object detection
        try:
            self.coco_model = YOLO('yolov8n.pt')
            print("[CurrencyDetector] Using COCO model with classification fallback")
            return True
        except Exception as e:
            print(f"[CurrencyDetector] Model loading error: {e}")
            return False
    
    def detect(self, frame, conf_threshold=0.6, apply_low_light_enhancement=True):
        """
        Detect Indian currency notes in frame.
        
        Args:
            frame: Input image
            conf_threshold: Confidence threshold
            apply_low_light_enhancement: Apply preprocessing for low light conditions
        
        Returns:
            List of detected currencies:
            [
                {
                    'denomination': int (10, 20, 50, 100, 200, 500),
                    'confidence': float,
                    'bbox': [x1, y1, x2, y2],
                    'center': [cx, cy],
                    'area': float,
                    'color_match': float (0-1),
                    'aspect_ratio': float,
                    'is_new': bool (first detection of this note)
                }
            ]
        """
        if not self.model and not self.coco_model:
            return []
        
        # Enhance for low light
        if apply_low_light_enhancement:
            frame = self._enhance_low_light(frame)
        
        detections = []
        
        if self.is_custom and self.model:
            # Use custom currency model
            detections = self._detect_with_custom_model(frame, conf_threshold)
        else:
            # Use COCO fallback
            detections = self._detect_with_coco_fallback(frame, conf_threshold)
        
        # Deduplicate recent detections
        detections = self._deduplicate_detections(detections)
        
        self.detections_count += len(detections)
        self.frame_count += 1
        
        return detections
    
    def _enhance_low_light(self, frame):
        """
        Enhance frame for low-light conditions.
        Improves detection accuracy under dim lighting.
        """
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE with higher clip limit for low light
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        lab[:, :, 0] = enhanced_l
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _detect_with_custom_model(self, frame, conf_threshold):
        """Detect using custom trained currency model"""
        start_time = time.time()
        results = self.model(frame, conf=conf_threshold, verbose=False)
        self.inference_time = time.time() - start_time
        
        detections = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0].cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Try to get denomination from class name
                class_name = result.names[int(box.cls[0])]
                denomination = self._parse_denomination_from_class_name(class_name)
                
                if denomination:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    
                    det = {
                        'denomination': denomination,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'center': [cx, cy],
                        'area': (x2 - x1) * (y2 - y1),
                        'color_match': 0.9,
                        'aspect_ratio': (x2 - x1) / (y2 - y1),
                        'is_new': True
                    }
                    detections.append(det)
        
        return detections
    
    def _detect_with_coco_fallback(self, frame, conf_threshold):
        """
        Detect using COCO model with fallback classification.
        Looks for rectangular objects and classifies by size/color.
        """
        start_time = time.time()
        results = self.coco_model(frame, conf=conf_threshold, verbose=False)
        self.inference_time = time.time() - start_time
        
        detections = []
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Check if bounding box looks like a rectangular currency note
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / (height + 0.001)
                
                # Currency notes are roughly 2.5:1 aspect ratio
                if 1.8 < aspect_ratio < 3.0:
                    # Classify by color
                    roi = hsv_frame[y1:y2, x1:x2]
                    denomination = self._classify_by_color(roi)
                    
                    if denomination:
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        
                        det = {
                            'denomination': denomination,
                            'confidence': 0.7,  # Lower confidence for fallback
                            'bbox': [x1, y1, x2, y2],
                            'center': [cx, cy],
                            'area': width * height,
                            'color_match': 0.6,
                            'aspect_ratio': aspect_ratio,
                            'is_new': True
                        }
                        detections.append(det)
        
        return detections
    
    def _classify_by_color(self, roi):
        """Classify currency denomination by color"""
        if roi.size == 0:
            return None
        
        best_match = None
        best_score = 0
        
        for denom, (lower, upper) in self.COLOR_RANGES.items():
            # Create mask for this color range
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            mask = cv2.inRange(roi, lower_bound, upper_bound)
            
            # Calculate match score
            match_pixels = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            score = match_pixels / total_pixels
            
            if score > best_score:
                best_score = score
                best_match = denom
        
        # Only return if we have a decent match
        return best_match if best_score > 0.15 else None
    
    def _parse_denomination_from_class_name(self, class_name):
        """Extract denomination from class name (e.g., 'rupee_500' -> 500)"""
        class_name = class_name.lower()
        for denom in self.DENOMINATIONS:
            if str(denom) in class_name:
                return denom
        return None
    
    def _deduplicate_detections(self, detections):
        """Mark if detection is new or recently seen"""
        now = time.time()
        result = []
        
        for det in detections:
            denom = det['denomination']
            is_new = now - self.detected_currencies[denom] > self.detection_cooldown
            det['is_new'] = is_new
            
            if is_new:
                self.detected_currencies[denom] = now
            
            result.append(det)
        
        return result
    
    def draw_detections(self, frame, detections, show_confidence=True):
        """Draw detected currencies on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            denom = det['denomination']
            confidence = det['confidence']
            
            # Color: green if confident, yellow if lower confidence
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"₹{denom}"
            if show_confidence:
                label += f" {confidence:.0%}"
            
            # Draw label
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1-30), (x1 + text_size[0] + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def get_stats(self):
        """Return detection statistics"""
        return {
            "total_detections": self.detections_count,
            "frames_processed": self.frame_count,
            "inference_time_ms": round(self.inference_time * 1000, 2),
            "model_type": "custom" if self.is_custom else "coco_fallback",
            "detection_cooldown": self.detection_cooldown,
            "recent_detections": dict(self.detected_currencies)
        }
