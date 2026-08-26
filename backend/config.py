"""
Configuration management for the vision system.
Centralizes all tunable parameters and settings.
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class CameraConfig:
    """Camera capture settings"""
    camera_id: int = 0
    fps: int = 30
    resize_scale: float = 0.75
    frame_buffer_size: int = 5
    skip_frames: int = 2  # Process every Nth frame


@dataclass
class ObjectDetectionConfig:
    """Object detection settings"""
    model_path: str = "weights/yolo11n.pt"
    conf_threshold: float = 0.5
    nms_threshold: float = 0.45
    apply_tracking: bool = True
    apply_filtering: bool = True
    min_box_size: int = 10
    min_area: int = 400
    show_confidence: bool = True
    show_tracking_id: bool = False
    stable_only: bool = False


@dataclass
class CurrencyDetectionConfig:
    """Currency detection settings"""
    model_path: str = "weights/best.pt"
    use_custom_model: bool = True
    conf_threshold: float = 0.6
    apply_low_light_enhancement: bool = True
    detection_cooldown: float = 4.0
    show_confidence: bool = True


@dataclass
class PreprocessingConfig:
    """Frame preprocessing settings"""
    normalize_brightness: bool = True
    enhance_contrast: bool = True
    denoise: bool = True
    sharpen: bool = False
    denoise_strength: int = 8
    clahe_clip: float = 2.0
    target_brightness: int = 100


@dataclass
class VoiceConfig:
    """Voice/TTS settings"""
    enabled: bool = True
    rate: int = 160
    volume: float = 1.0
    global_cooldown: float = 1.5
    object_cooldown_duration: float = 3.0
    profile: str = 'normal'  # normal, slow, fast, quiet, loud


@dataclass
class SpatialConfig:
    """Position / distance thresholds for accessibility spatial cues."""
    left_max: float = 0.33
    right_min: float = 0.66
    near_area_ratio: float = 0.12
    far_area_ratio: float = 0.03
    path_band_top: float = 0.45


@dataclass
class SystemConfig:
    """Overall system configuration"""
    mode: Optional[str] = None  # 'object', 'currency', 'face', None
    use_gpu: bool = True
    debug_mode: bool = False
    log_file: str = "logs/detection_log.txt"
    max_concurrent_threads: int = 4
    
    # Sub-configs
    camera: CameraConfig = None
    object_detection: ObjectDetectionConfig = None
    currency_detection: CurrencyDetectionConfig = None
    preprocessing: PreprocessingConfig = None
    voice: VoiceConfig = None
    spatial: SpatialConfig = None
    
    def __post_init__(self):
        if self.camera is None:
            self.camera = CameraConfig()
        if self.object_detection is None:
            self.object_detection = ObjectDetectionConfig()
        if self.currency_detection is None:
            self.currency_detection = CurrencyDetectionConfig()
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.voice is None:
            self.voice = VoiceConfig()
        if self.spatial is None:
            self.spatial = SpatialConfig()

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Return preprocessing settings as a plain dict for the pipeline."""
        return asdict(self.preprocessing)


class ConfigManager:
    """Manages system configuration with JSON persistence"""
    
    CONFIG_FILE = "system_config.json"
    
    def __init__(self):
        self.config = SystemConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file if exists"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                self._apply_dict_to_config(data)
                print(f"[ConfigManager] Loaded config from {self.CONFIG_FILE}")
            except Exception as e:
                print(f"[ConfigManager] Error loading config: {e}")
        else:
            self._save_config()
    
    def _apply_dict_to_config(self, data: Dict[str, Any]):
        """Apply dictionary data to config objects"""
        # Apply top-level settings
        for key, value in data.items():
            if key in [
                'camera',
                'object_detection',
                'currency_detection',
                'preprocessing',
                'voice',
                'spatial',
            ]:
                sub_config = getattr(self.config, key)
                if isinstance(sub_config, object):
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _save_config(self):
        """Save current configuration to JSON file"""
        try:
            os.makedirs('config', exist_ok=True)
            
            config_dict = {
                'mode': self.config.mode,
                'use_gpu': self.config.use_gpu,
                'debug_mode': self.config.debug_mode,
                'log_file': self.config.log_file,
                'max_concurrent_threads': self.config.max_concurrent_threads,
                'camera': asdict(self.config.camera),
                'object_detection': asdict(self.config.object_detection),
                'currency_detection': asdict(self.config.currency_detection),
                'preprocessing': asdict(self.config.preprocessing),
                'voice': asdict(self.config.voice),
                'spatial': asdict(self.config.spatial),
            }
            
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"[ConfigManager] Saved config to {self.CONFIG_FILE}")
        except Exception as e:
            print(f"[ConfigManager] Error saving config: {e}")
    
    def get_config(self) -> SystemConfig:
        """Get current configuration"""
        return self.config
    
    def set_mode(self, mode: Optional[str]):
        """Set detection mode (object, currency, face, None)"""
        self.config.mode = mode
    
    def set_confidence_threshold(self, mode: str, threshold: float):
        """Update confidence threshold for detection mode"""
        threshold = max(0.0, min(1.0, threshold))
        
        if mode == 'object':
            self.config.object_detection.conf_threshold = threshold
        elif mode == 'currency':
            self.config.currency_detection.conf_threshold = threshold
    
    def set_camera_fps(self, fps: int):
        """Update camera FPS"""
        self.config.camera.fps = max(1, min(60, fps))
    
    def set_camera_scale(self, scale: float):
        """Update frame resize scale (0.5 - 1.0)"""
        self.config.camera.resize_scale = max(0.3, min(1.0, scale))
    
    def set_voice_cooldowns(self, global_cooldown: float, object_cooldown: float):
        """Update voice announcement cooldowns"""
        self.config.voice.global_cooldown = max(0.5, global_cooldown)
        self.config.voice.object_cooldown_duration = max(1.0, object_cooldown)
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration as dictionary"""
        return asdict(self.config.preprocessing)
    
    def export_config(self, filepath: str = None):
        """Export configuration to file"""
        filepath = filepath or self.CONFIG_FILE
        try:
            config_dict = {
                'mode': self.config.mode,
                'use_gpu': self.config.use_gpu,
                'debug_mode': self.config.debug_mode,
                'camera': asdict(self.config.camera),
                'object_detection': asdict(self.config.object_detection),
                'currency_detection': asdict(self.config.currency_detection),
                'preprocessing': asdict(self.config.preprocessing),
                'voice': asdict(self.config.voice),
                'spatial': asdict(self.config.spatial),
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"[ConfigManager] Exported to {filepath}")
            return True
        except Exception as e:
            print(f"[ConfigManager] Export error: {e}")
            return False
    
    def import_config(self, filepath: str):
        """Import configuration from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self._apply_dict_to_config(data)
            print(f"[ConfigManager] Imported from {filepath}")
            return True
        except Exception as e:
            print(f"[ConfigManager] Import error: {e}")
            return False
    
    def get_json(self) -> str:
        """Get current configuration as JSON string"""
        config_dict = {
            'mode': self.config.mode,
            'camera': asdict(self.config.camera),
            'object_detection': asdict(self.config.object_detection),
            'currency_detection': asdict(self.config.currency_detection),
            'preprocessing': asdict(self.config.preprocessing),
            'voice': asdict(self.config.voice),
            'spatial': asdict(self.config.spatial),
        }
        return json.dumps(config_dict, indent=2)
