"use client";

import { useState } from "react";
import type { CameraFacing } from "@/hooks/useMobileCamera";
import { useMobileCamera } from "@/hooks/useMobileCamera";

interface VisionPanelProps {
  videoUrl: string;
  summary: string;
  live: boolean;
  fps: number;
  mobile?: boolean;
  backendReady?: boolean;
}

export default function VisionPanel({
  videoUrl,
  summary,
  live,
  fps,
  mobile = false,
  backendReady = false,
}: VisionPanelProps) {
  const [facing, setFacing] = useState<CameraFacing>("environment");
  const { videoRef, annotatedUrl, error, streaming, processingFps, needsHttps } =
    useMobileCamera({
      cameraEnabled: mobile && backendReady,
      detectEnabled: mobile && backendReady && live,
      facing,
    });

  const displayFps = mobile && live ? processingFps : fps;
  const showAnnotated = Boolean(annotatedUrl && live);

  return (
    <aside className="zyra-panel zyra-panel--right">
      <div className="zyra-panel-header">
        <span className="zyra-panel-title">VISION</span>
        <span className={`zyra-live ${live ? "zyra-live--on" : ""}`}>
          {live ? "LIVE" : mobile && streaming ? "PREVIEW" : "OFF"}
        </span>
      </div>

      {mobile && (
        <div className="zyra-camera-toggle" role="group" aria-label="Camera facing">
          <button
            type="button"
            className={`zyra-camera-btn ${facing === "user" ? "active" : ""}`}
            onClick={() => setFacing("user")}
            aria-pressed={facing === "user"}
          >
            Front
          </button>
          <button
            type="button"
            className={`zyra-camera-btn ${facing === "environment" ? "active" : ""}`}
            onClick={() => setFacing("environment")}
            aria-pressed={facing === "environment"}
          >
            Back
          </button>
        </div>
      )}

      <div className="zyra-video-frame">
        {mobile ? (
          <>
            <video
              ref={videoRef}
              className={`zyra-video ${showAnnotated ? "zyra-video--under" : ""}`}
              muted
              playsInline
              autoPlay
            />
            {showAnnotated && (
              /* eslint-disable-next-line @next/next/no-img-element */
              <img
                src={annotatedUrl!}
                alt="Detections overlay"
                className="zyra-video zyra-video--overlay"
              />
            )}
            {!backendReady && (
              <div className="zyra-video-overlay">Waiting for backend…</div>
            )}
            {backendReady && !streaming && !error && (
              <div className="zyra-video-overlay">Starting camera…</div>
            )}
            {backendReady && streaming && !live && !error && (
              <div className="zyra-video-banner">
                Tap <strong>ACTIVATE VISION</strong> to detect objects
              </div>
            )}
            {(error || needsHttps) && (
              <div className="zyra-video-overlay zyra-video-overlay--error">
                {error ||
                  "Use HTTPS: npm run dev:mobile → https://192.168.0.10:3000"}
              </div>
            )}
          </>
        ) : (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={videoUrl} alt="Live camera feed" className="zyra-video" />
            {!live && <div className="zyra-video-overlay">Camera standby</div>}
          </>
        )}
      </div>

      <p className="zyra-vision-summary">{summary}</p>
      <p className="zyra-fps">
        {displayFps > 0 ? `${displayFps.toFixed(1)} FPS` : ""}
        {mobile && streaming ? " · phone camera" : ""}
      </p>
    </aside>
  );
}
