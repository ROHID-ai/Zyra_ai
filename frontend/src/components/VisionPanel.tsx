"use client";

interface VisionPanelProps {
  videoUrl: string;
  summary: string;
  live: boolean;
  fps: number;
}

export default function VisionPanel({
  videoUrl,
  summary,
  live,
  fps,
}: VisionPanelProps) {
  return (
    <aside className="veyra-panel veyra-panel--right">
      <div className="veyra-panel-header">
        <span className="veyra-panel-title">VISION</span>
        <span className={`veyra-live ${live ? "veyra-live--on" : ""}`}>
          {live ? "LIVE" : "OFF"}
        </span>
      </div>

      <div className="veyra-video-frame">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={videoUrl} alt="Live camera feed" className="veyra-video" />
        {!live && <div className="veyra-video-overlay">Camera standby</div>}
      </div>

      <p className="veyra-vision-summary">{summary}</p>
      <p className="veyra-fps">{fps > 0 ? `${fps.toFixed(1)} FPS` : ""}</p>
    </aside>
  );
}
