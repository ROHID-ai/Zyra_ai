import type { SystemStatus } from "@/types";

// Same-origin proxy via Next.js rewrites (/api -> backend)
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    throw new Error(`API ${path} failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getApiUrl(): string {
  return API_URL;
}

export function getVideoFeedUrl(): string {
  return `${API_URL}/video_feed`;
}

export function fetchHealth(): Promise<{
  status: string;
  ready: boolean;
  error?: string;
}> {
  return request("/health");
}

export function fetchStatus(): Promise<SystemStatus> {
  return request<SystemStatus>("/status");
}

export function startObjectDetection(): Promise<{ status: string }> {
  return request("/start-object");
}

export function startCurrencyDetection(): Promise<{ status: string }> {
  return request("/start-currency");
}

export function stopDetection(): Promise<{ status: string }> {
  return request("/stop");
}

export function askZyra(
  question: string,
): Promise<{ answer: string; live_state: unknown }> {
  return request("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

export function sendCommand(
  question: string,
): Promise<{ result: string; mode: string | null }> {
  return request("/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
}

export function setConfidence(
  mode: string | null,
  threshold: number,
): Promise<{ status: string }> {
  return request("/config/set-confidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, threshold }),
  });
}

export function setCameraScale(scale: number): Promise<{ status: string }> {
  return request("/config/set-camera-scale", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scale }),
  });
}

export function postMobileFrame(
  blob: Blob,
): Promise<{ ok: boolean; frame_b64?: string; error?: string; mode?: string | null }> {
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  return fetch(`${API_URL}/mobile-frame`, {
    method: "POST",
    body: form,
  }).then(async (response) => {
    const data = (await response.json()) as {
      ok: boolean;
      frame_b64?: string;
      error?: string;
      mode?: string | null;
    };
    if (!response.ok && !data.error) {
      throw new Error(`mobile-frame failed: ${response.status}`);
    }
    return data;
  });
}
