"""
Frame preprocessing module for enhanced detection
Includes brightness normalization, contrast enhancement, denoising
"""
import cv2
import numpy as np


class FramePreprocessor:
    """
    Advanced frame preprocessing for improved detection quality.
    Handles lighting conditions, noise, contrast enhancement.
    """
    
    def __init__(self, enable_adaptive_histogram=True):
        self.enable_adaptive_histogram = enable_adaptive_histogram
        self.prev_brightness = 0
        self.prev_contrast = 1.0
    
    def normalize_brightness(self, frame, target_brightness=100):
        """
        Normalize frame brightness for consistent lighting.
        Helps with detection under various lighting conditions.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate current brightness
        current_brightness = np.mean(l_channel)
        
        # Smooth transitions
        alpha = 0.8  # Smoothing factor
        self.prev_brightness = alpha * self.prev_brightness + (1 - alpha) * current_brightness
        
        # Adjust brightness
        if self.prev_brightness < target_brightness - 10:
            # Image too dark, brighten it
            lab[:, :, 0] = cv2.add(lab[:, :, 0], 20)
        elif self.prev_brightness > target_brightness + 10:
            # Image too bright, darken it
            lab[:, :, 0] = cv2.subtract(lab[:, :, 0], 15)
        
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def enhance_contrast(self, frame, clip_limit=2.0):
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Better for local contrast without over-enhancing noise.
        """
        if not self.enable_adaptive_histogram:
            return frame
        
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE only to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        lab[:, :, 0] = enhanced_l
        
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def denoise(self, frame, method='bilateral', strength=10):
        """
        Denoise frame while preserving edges.
        - bilateral: Good for edge preservation
        - morphological: Good for structural noise
        - gaussian: Simple but blurs edges
        """
        if method == 'bilateral':
            return cv2.bilateralFilter(frame, 9, strength, strength)
        elif method == 'morphological':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
            return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        elif method == 'gaussian':
            return cv2.GaussianBlur(frame, (5, 5), 0)
        else:
            return frame
    
    def sharpen(self, frame, strength=1.5):
        """
        Sharpen frame to enhance edges for better detection.
        """
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) / 1.0
        
        sharpened = cv2.filter2D(frame, -1, kernel)
        
        # Blend original with sharpened (to avoid over-sharpening artifacts)
        return cv2.addWeighted(frame, 1.0 - strength/10, sharpened, strength/10, 0)
    
    def preprocess_for_detection(self, frame, config=None):
        """
        Complete preprocessing pipeline for detection.
        Args:
            frame: Input frame
            config: Dictionary with preprocessing settings
                {
                    'normalize_brightness': bool,
                    'enhance_contrast': bool,
                    'denoise': bool,
                    'sharpen': bool,
                    'denoise_strength': int (0-20),
                    'clahe_clip': float (1.0-4.0)
                }
        """
        if config is None:
            config = {
                'normalize_brightness': True,
                'enhance_contrast': True,
                'denoise': True,
                'sharpen': False,
                'denoise_strength': 8,
                'clahe_clip': 2.0
            }
        
        result = frame.copy()
        
        # Apply preprocessing in sequence
        if config.get('normalize_brightness', True):
            result = self.normalize_brightness(result)
        
        if config.get('enhance_contrast', True):
            result = self.enhance_contrast(result, clip_limit=config.get('clahe_clip', 2.0))
        
        if config.get('denoise', True):
            result = self.denoise(result, method='bilateral', 
                                 strength=config.get('denoise_strength', 8))
        
        if config.get('sharpen', False):
            result = self.sharpen(result, strength=1.0)
        
        return result
    
    def get_lighting_conditions(self, frame):
        """Analyze and return current lighting conditions"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        brightness = np.mean(l_channel)
        
        if brightness < 80:
            condition = "too_dark"
        elif brightness < 100:
            condition = "dim"
        elif brightness > 180:
            condition = "too_bright"
        elif brightness > 160:
            condition = "bright"
        else:
            condition = "optimal"
        
        return {
            "brightness": round(brightness, 1),
            "condition": condition,
            "std_dev": round(np.std(l_channel), 1)
        }
