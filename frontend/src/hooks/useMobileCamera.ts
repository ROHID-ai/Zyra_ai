"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { postMobileFrame } from "@/lib/api";

export type CameraFacing = "user" | "environment";

interface UseMobileCameraOptions {
  /** Start phone camera preview (mobile) */
  cameraEnabled: boolean;
  /** Send frames to backend for YOLO (vision mode on) */
  detectEnabled: boolean;
  facing: CameraFacing;
  intervalMs?: number;
}

function isSecureCameraContext(): boolean {
  if (typeof window === "undefined") return true;
  return (
    window.isSecureContext ||
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
  );
}

export function useMobileCamera({
  cameraEnabled,
  detectEnabled,
  facing,
  intervalMs = 300,
}: UseMobileCameraOptions) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [annotatedUrl, setAnnotatedUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [processingFps, setProcessingFps] = useState(0);
  const tickRef = useRef(0);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStreaming(false);
  }, []);

  const startStream = useCallback(async () => {
    stopStream();
    setAnnotatedUrl(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera not supported in this browser.");
      return;
    }

    if (!isSecureCameraContext()) {
      setError(
        "Camera blocked on HTTP. On your phone open https://192.168.0.10:3000 instead (run: npm run dev:mobile on Mac, then accept the certificate warning).",
      );
      return;
    }

    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        video.setAttribute("playsinline", "true");
        video.muted = true;
        await video.play();
        setStreaming(true);
      }
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Camera permission denied";
      setError(
        msg.toLowerCase().includes("secure") || msg.toLowerCase().includes("permission")
          ? "Allow camera access, or use https://192.168.0.10:3000 (npm run dev:mobile)."
          : `Camera error: ${msg}`,
      );
    }
  }, [facing, stopStream]);

  useEffect(() => {
    if (cameraEnabled) {
      startStream();
    } else {
      stopStream();
      setAnnotatedUrl(null);
      setError(null);
    }
    return () => stopStream();
  }, [cameraEnabled, facing, startStream, stopStream]);

  useEffect(() => {
    if (!detectEnabled || !streaming) {
      window.clearInterval(tickRef.current);
      if (!detectEnabled) setAnnotatedUrl(null);
      return;
    }

    if (!canvasRef.current) {
      canvasRef.current = document.createElement("canvas");
    }

    let cancelled = false;
    let frames = 0;
    let lastFps = performance.now();

    const tick = async () => {
      if (cancelled) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;

      const w = video.videoWidth;
      const h = video.videoHeight;
      if (w < 1 || h < 1) return;

      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, w, h);

      canvas.toBlob(
        async (blob) => {
          if (cancelled || !blob) return;
          try {
            const result = await postMobileFrame(blob);
            if (result.ok && result.frame_b64) {
              setAnnotatedUrl(`data:image/jpeg;base64,${result.frame_b64}`);
              frames += 1;
              const now = performance.now();
              if (now - lastFps >= 1000) {
                setProcessingFps(frames);
                frames = 0;
                lastFps = now;
              }
            }
          } catch {
            /* keep live video preview */
          }
        },
        "image/jpeg",
        0.78,
      );
    };

    tickRef.current = window.setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(tickRef.current);
    };
  }, [detectEnabled, streaming, intervalMs]);

  return {
    videoRef,
    annotatedUrl,
    error,
    streaming,
    processingFps,
    needsHttps: cameraEnabled && !isSecureCameraContext(),
  };
}
