export type DetectionMode = "object" | "currency" | null;

export type CoreVisualState =
  | "idle"
  | "listening"
  | "thinking"
  | "responding"
  | "warning"
  | "vision"
  | "offline";

export interface CameraStats {
  fps_actual?: number;
  frames_dropped?: number;
  frames_captured?: number;
  queue_size?: number;
}

export interface DetectorStats {
  total_detections?: number;
  inference_time_ms?: number;
  frames_processed?: number;
  filtered?: number;
}

export interface VoiceStats {
  total_announcements?: number;
  queued?: number;
  skipped?: number;
  last_announcement?: string;
}

export interface RecentDetection {
  label: string;
  confidence: number;
  position?: string;
  distance?: string;
  motion?: string;
}

export interface LiveObject {
  name: string;
  class?: string;
  position?: string;
  distance?: string;
  motion?: string;
  confidence?: number;
  tracked_id?: number | null;
  priority?: number;
  kind?: string;
  in_path?: boolean;
  stable?: boolean;
}

export interface PathDetail {
  status?: string;
  lanes?: {
    left?: string;
    center?: string;
    right?: string;
  };
  blocking_object?: string | null;
  position?: string | null;
  suggestion?: string | null;
}

export interface HazardInfo {
  name?: string;
  reason?: string;
  priority?: number;
  position?: string;
  distance?: string;
  motion?: string;
}

export interface CurrencyItem {
  value: number;
  count: number;
}

export interface CurrencySummary {
  currency?: CurrencyItem[];
  total?: number;
  signature?: string;
  spoken?: string;
}

export interface SceneEvent {
  type?: string;
  message?: string;
  priority?: number;
  critical?: boolean;
  ts?: number;
}

export interface SystemStatus {
  mode: DetectionMode;
  logs: string[];
  camera_stats: CameraStats;
  object_detector_stats: DetectorStats;
  currency_detector_stats: DetectorStats;
  voice_engine_stats: VoiceStats;
  recent_detections?: RecentDetection[];
  objects?: LiveObject[];
  path?: string;
  path_detail?: PathDetail;
  hazard?: HazardInfo | null;
  danger?: string;
  currency?: CurrencySummary;
  recent_events?: SceneEvent[];
  last_response?: string;
  core_state?: CoreVisualState | string;
  groq?: { enabled?: boolean; model?: string | null; error?: string };
  camera?: CameraStats;
  voice?: VoiceStats;
}
