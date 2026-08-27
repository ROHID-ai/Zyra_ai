"""
Advanced voice engine with cooldown management, queueing, and better TTS handling.
Prevents spam, manages announcement timing, and improves accessibility.
"""
import pyttsx3
import threading
import time
import queue
from collections import defaultdict


class VoiceEngine:
    """
    Advanced text-to-speech engine with:
    - Announcement queueing and prioritization
    - Per-object cooldown timers (prevents spam for same object)
    - Global cooldown between announcements
    - Threaded execution for non-blocking speech
    - Multiple voice profiles for different use cases
    """
    
    def __init__(self, rate=160, volume=1.0):
        self.rate = rate
        self.volume = volume
        
        # Voice management
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Queuing system
        self.speech_queue = queue.PriorityQueue()
        self._queue_lock = threading.Lock()
        self.queue_thread = None
        self.running = False
        
        # Cooldown management
        self.last_speech_time = 0
        self.global_cooldown = 1.5  # Minimum seconds between any announcements
        self.object_cooldowns = defaultdict(float)  # Per-object cooldown
        self.object_cooldown_duration = 3.0  # Don't repeat same object for 3s
        
        # Statistics
        self.total_announcements = 0
        self.queued_announcements = 0
        self.skipped_announcements = 0
        self.last_announcement = ""
        
        # Voice profiles for different contexts
        self.profiles = {
            'normal': {'rate': 160, 'volume': 1.0},
            'slow': {'rate': 120, 'volume': 1.0},
            'fast': {'rate': 200, 'volume': 0.9},
            'quiet': {'rate': 160, 'volume': 0.6},
            'loud': {'rate': 140, 'volume': 1.0},
        }
        
        self._start_queue_processor()
    
    def _start_queue_processor(self):
        """Start background thread for processing speech queue"""
        if not self.running:
            self.running = True
            self.queue_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.queue_thread.start()
            print("[VoiceEngine] Queue processor started")
    
    def _process_queue(self):
        """Process queued announcements"""
        while self.running:
            try:
                # Non-blocking get with small timeout
                priority, speech_data = self.speech_queue.get(timeout=0.5)
                text, profile, callback = speech_data
                
                self._speak_now(text, profile)
                
                if callback:
                    callback()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceEngine] Queue processing error: {e}")
    
    def _drain_non_critical_pending(self) -> None:
        """Drop queued non-critical speech so urgent warnings are not delayed."""
        with self._queue_lock:
            kept: list[tuple] = []
            while True:
                try:
                    item = self.speech_queue.get_nowait()
                except queue.Empty:
                    break
                priority = item[0]
                if priority <= 0:
                    kept.append(item)
            for item in kept:
                self.speech_queue.put(item)

    def _enqueue(self, priority: int, text: str, profile: str, callback=None) -> None:
        with self._queue_lock:
            self.speech_queue.put((priority, (text, profile, callback)))

    def _speak_now(self, text, profile='normal'):
        """Actually speak the text (thread-safe)"""
        local_engine = None
        try:
            local_engine = pyttsx3.init()
            
            if profile in self.profiles:
                props = self.profiles[profile]
                local_engine.setProperty('rate', props['rate'])
                local_engine.setProperty('volume', props['volume'])
            
            self.last_announcement = text
            local_engine.say(text)
            local_engine.runAndWait()
            self.total_announcements += 1

        except Exception as e:
            print(f"[VoiceEngine] Speech error: {e}")
        finally:
            if local_engine is not None:
                try:
                    local_engine.stop()
                except Exception:
                    pass
    
    def announce(self, text, object_key=None, profile='normal', 
                priority=5, wait_for_cooldown=True, callback=None):
        """
        Queue an announcement with cooldown checking.
        
        Args:
            text: Text to speak
            object_key: Unique key for this object (for per-object cooldown)
            profile: Voice profile ('normal', 'slow', 'fast', 'quiet', 'loud')
            priority: Priority in queue (0-10, lower = higher priority)
            wait_for_cooldown: If True, respects cooldown timers
            callback: Optional callback after speech completes
        
        Returns:
            bool: True if queued, False if skipped due to cooldown
        """
        now = time.time()
        
        # Check global cooldown
        if wait_for_cooldown:
            if now - self.last_speech_time < self.global_cooldown:
                self.skipped_announcements += 1
                return False
            
            # Check per-object cooldown
            if object_key and now - self.object_cooldowns[object_key] < self.object_cooldown_duration:
                self.skipped_announcements += 1
                return False
        
        # Queue the announcement
        self._enqueue(priority, text, profile, callback)
        self.queued_announcements += 1
        self.last_speech_time = now
        
        if object_key:
            self.object_cooldowns[object_key] = now
        
        return True
    
    def announce_immediate(self, text, profile='normal'):
        """Immediately announce text (bypasses queue, use sparingly)"""
        self._speak_now(text, profile)
    
    def announce_priority(self, text, object_key=None, profile='normal', callback=None):
        """Queue high-priority announcement (goes to front of queue)"""
        return self.announce(text, object_key, profile, priority=1, callback=callback)

    def announce_critical(self, text, object_key=None, profile='loud', callback=None):
        """
        Safety-critical announcement: highest queue priority, shorter object cooldown.
        Drops pending non-critical queue items so urgent warnings are not delayed.
        """
        now = time.time()
        self._drain_non_critical_pending()

        if object_key and now - self.object_cooldowns[object_key] < 1.2:
            self.skipped_announcements += 1
            return False

        self._enqueue(0, text, profile, callback)
        self.queued_announcements += 1
        self.last_speech_time = now
        if object_key:
            self.object_cooldowns[object_key] = now
        return True

    def announce_event(self, text, *, critical=False, object_key=None, profile=None):
        """Route event text through priority-aware announcement helpers."""
        if critical:
            return self.announce_critical(
                text, object_key=object_key, profile=profile or "loud"
            )
        return self.announce_priority(
            text, object_key=object_key, profile=profile or "normal"
        )
    
    def set_profile(self, profile_name):
        """Set active voice profile"""
        if profile_name in self.profiles:
            props = self.profiles[profile_name]
            self.rate = props['rate']
            self.volume = props['volume']
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
    
    def set_cooldowns(self, global_cooldown=None, object_cooldown=None):
        """Adjust cooldown timers"""
        if global_cooldown is not None:
            self.global_cooldown = global_cooldown
        if object_cooldown is not None:
            self.object_cooldown_duration = object_cooldown
    
    def clear_object_cooldown(self, object_key):
        """Manually clear cooldown for specific object"""
        if object_key in self.object_cooldowns:
            del self.object_cooldowns[object_key]
    
    def clear_all_cooldowns(self):
        """Clear all cooldown timers"""
        self.object_cooldowns.clear()
        self.last_speech_time = 0
    
    def get_stats(self):
        """Return voice engine statistics"""
        return {
            "total_announcements": self.total_announcements,
            "queued": self.queued_announcements,
            "skipped": self.skipped_announcements,
            "queue_size": self.speech_queue.qsize(),
            "last_announcement": self.last_announcement,
            "active_cooldowns": len(self.object_cooldowns),
            "global_cooldown": self.global_cooldown,
            "object_cooldown_duration": self.object_cooldown_duration
        }
    
    def shutdown(self):
        """Stop voice engine and cleanup"""
        self.running = False
        if self.queue_thread:
            self.queue_thread.join(timeout=2)
        try:
            self.engine.stop()
        except:
            pass
        print("[VoiceEngine] Shutdown complete")
