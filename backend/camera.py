"""
Optimized camera capture with threading and GPU support
Provides low-latency frame capture for real-time processing
"""
import cv2
import threading
import queue
import time
import numpy as np
from collections import deque


class ThreadedCamera:
    """
    Threaded camera capture for low-latency video streaming.
    Runs capture in background thread to prevent blocking.
    """
    
    def __init__(self, camera_id=0, fps=30, resize_scale=0.75):
        self.camera_id = camera_id
        self.fps = fps
        self.resize_scale = resize_scale
        
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_buffer = deque(maxlen=5)  # For frame smoothing/stability
        self.latest_frame = None
        self.frame_count = 0
        self.dropped_frames = 0
        self.fps_actual = 0
        self.last_time = time.time()
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            print(f"[Camera] Initialized - FPS: {self.fps}, Scale: {self.resize_scale}")
            return True
        except Exception as e:
            print(f"[Camera] ERROR: {e}")
            return False
    
    def _capture_loop(self):
        """Background thread loop for continuous frame capture"""
        skip_counter = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("[Camera] Failed to read frame")
                    continue
                
                # Skip frames for performance (every 2nd frame = ~15fps for 30fps input)
                skip_counter += 1
                if skip_counter < 2:  # Process every 2nd frame
                    self.dropped_frames += 1
                    continue
                skip_counter = 0
                
                # Resize for faster processing
                if self.resize_scale < 1.0:
                    h, w = frame.shape[:2]
                    new_w = int(w * self.resize_scale)
                    new_h = int(h * self.resize_scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                self.frame_buffer.append(frame)
                
                # Try to put frame in queue, drop if queue is full
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    self.dropped_frames += 1
                
                self.frame_count += 1
                
                # Calculate actual FPS every 30 frames
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - self.last_time
                    self.fps_actual = 30 / elapsed if elapsed > 0 else 0
                    self.last_time = current_time
                
            except Exception as e:
                print(f"[Camera] Capture error: {e}")
                continue
    
    def start(self):
        """Start the camera capture thread"""
        if not self.running and self.cap:
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("[Camera] Capture thread started")
    
    def get_frame(self, timeout=1.0):
        """Get latest frame from queue"""
        try:
            self.latest_frame = self.frame_queue.get(timeout=timeout)
            return self.latest_frame
        except queue.Empty:
            return self.latest_frame
    
    def get_stable_frame(self, use_median=False):
        """
        Get stabilized frame from buffer (averages recent frames).
        Reduces jitter and noise in video stream.
        """
        if not self.frame_buffer:
            return self.latest_frame
        
        if use_median:
            # Use median for better outlier rejection
            frames_array = np.array(list(self.frame_buffer))
            return np.median(frames_array, axis=0).astype(np.uint8)
        else:
            # Simple averaging
            frames_array = np.array(list(self.frame_buffer))
            return np.mean(frames_array, axis=0).astype(np.uint8)
    
    def stop(self):
        """Stop camera capture and cleanup"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print("[Camera] Stopped")
    
    def get_stats(self):
        """Return camera performance stats"""
        return {
            "fps_actual": round(self.fps_actual, 1),
            "frames_captured": self.frame_count,
            "frames_dropped": self.dropped_frames,
            "queue_size": self.frame_queue.qsize()
        }
