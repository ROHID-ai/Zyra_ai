"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import AiCore from "@/components/AiCore";
import ResponsePanel from "@/components/ResponsePanel";
import VisionPanel from "@/components/VisionPanel";
import {
  askZyra,
  fetchHealth,
  fetchStatus,
  getVideoFeedUrl,
  sendCommand,
  startCurrencyDetection,
  startObjectDetection,
  stopDetection,
} from "@/lib/api";
import type { DetectionMode, RecentDetection, SystemStatus } from "@/types";

function useSpeech() {
  const speak = useCallback((text: string) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.85;
    window.speechSynthesis.speak(utterance);
  }, []);
  return { speak };
}

function useVoiceCommands(onAction: (action: string) => void) {
  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.lang = "en-US";
      recognition.interimResults = false;

      recognition.onresult = (event) => {
        const transcript =
          event.results[event.results.length - 1][0].transcript.toLowerCase();
        if (
          transcript.includes("what do you see") ||
          transcript.includes("what's ahead") ||
          transcript.includes("whats ahead") ||
          transcript.includes("where is") ||
          transcript.includes("read the scene") ||
          transcript.includes("is the path")
        ) {
          onAction(`ask:${transcript}`);
        } else if (
          transcript.includes("start vision") ||
          transcript.includes("object") ||
          transcript.includes("detect")
        ) {
          onAction("object");
        } else if (
          transcript.includes("currency") ||
          transcript.includes("money") ||
          transcript.includes("rupee")
        ) {
          onAction("currency");
        } else if (transcript.includes("stop")) {
          onAction("stop");
        }
      };

      recognition.start();
      return () => {
        try {
          recognition.abort();
        } catch {
          /* ignore */
        }
      };
    } catch {
      /* unavailable */
    }
  }, [onAction]);
}

function formatDetectionList(detections: RecentDetection[]): string {
  if (detections.length === 0) return "No objects detected yet.";
  return detections
    .map((d) => {
      const bits = [d.label];
      if (d.position) bits.push(d.position);
      if (d.distance) bits.push(d.distance);
      if (d.motion && d.motion !== "stationary") bits.push(d.motion);
      return bits.join(" · ");
    })
    .join(", ");
}

function buildResponseMessage(
  mode: DetectionMode,
  detections: RecentDetection[],
  lastAnnouncement?: string,
  path?: string,
): string {
  if (lastAnnouncement) return lastAnnouncement;

  if (!mode) {
    return "Zyra AI is ready. Activate vision and I will describe important changes around you.";
  }

  if (detections.length === 0) {
    return mode === "currency"
      ? "Scanning for Indian currency notes…"
      : "Scanning your surroundings…";
  }

  if (mode === "currency") {
    const items = detections.map((d) => d.label).join(", ");
    return `I can see: ${items}.`;
  }

  const count = detections.length;
  const items = detections.map((d) => d.label).join(", ");
  const pathBit =
    path && path !== "clear"
      ? ` Path appears ${path.replaceAll("_", " ")}.`
      : "";
  return `I can see ${count} object${count > 1 ? "s" : ""}: ${items}.${pathBit}`;
}

export default function VisionApp() {
  const [mode, setMode] = useState<DetectionMode>(null);
  const [stats, setStats] = useState<SystemStatus | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [backendReady, setBackendReady] = useState(false);
  const [command, setCommand] = useState("");
  const [askBusy, setAskBusy] = useState(false);
  const { speak } = useSpeech();
  const videoUrl = getVideoFeedUrl();

  const detections = useMemo(
    () => stats?.recent_detections ?? [],
    [stats?.recent_detections],
  );
  const objects = stats?.objects ?? [];
  const path = stats?.path ?? "clear";
  const danger = stats?.danger ?? "low";
  const lastAnnouncement =
    stats?.last_response || stats?.voice_engine_stats?.last_announcement;
  const fps = stats?.camera_stats?.fps_actual ?? stats?.camera?.fps_actual ?? 0;
  const warningMessage =
    danger === "critical" || danger === "high"
      ? stats?.hazard
        ? `${stats.hazard.name ?? "Hazard"} — ${stats.hazard.reason?.replaceAll("_", " ") ?? "nearby"}`
        : lastAnnouncement
      : stats?.recent_events?.find((e) => e.critical)?.message ?? null;

  const responseMessage = useMemo(
    () => buildResponseMessage(mode, detections, lastAnnouncement, path),
    [mode, detections, lastAnnouncement, path],
  );

  const visionSummary = useMemo(() => {
    if (!mode) return "Vision module idle.";
    if (detections.length === 0) return "Scanning…";
    return `I can see: ${formatDetectionList(detections)}`;
  }, [mode, detections]);

  const setDetectionMode = useCallback(
    async (newMode: DetectionMode) => {
      setMode(newMode);
      try {
        if (newMode === "object") {
          await startObjectDetection();
          speak("Vision activated");
        } else if (newMode === "currency") {
          await startCurrencyDetection();
          speak("Currency detection activated");
        } else {
          await stopDetection();
          speak("Vision stopped");
        }
      } catch (error) {
        console.error("Mode change failed:", error);
      }
    },
    [speak],
  );

  const runAsk = useCallback(
    async (question: string) => {
      setAskBusy(true);
      try {
        const { answer } = await askZyra(question);
        speak(answer);
        setStats((prev) =>
          prev
            ? { ...prev, last_response: answer, core_state: "responding" }
            : prev,
        );
      } catch (error) {
        console.error("Ask failed:", error);
        speak("I could not reach the vision assistant right now.");
      } finally {
        setAskBusy(false);
      }
    },
    [speak],
  );

  const handleVoiceAction = useCallback(
    (action: string) => {
      if (action.startsWith("ask:")) {
        runAsk(action.slice(4));
        return;
      }
      if (action === "object") setDetectionMode("object");
      else if (action === "currency") setDetectionMode("currency");
      else if (action === "stop") setDetectionMode(null);
    },
    [setDetectionMode, runAsk],
  );

  useVoiceCommands(handleVoiceAction);

  useEffect(() => {
    const loadStats = async () => {
      try {
        const health = await fetchHealth();
        setBackendOnline(true);
        setBackendReady(Boolean(health.ready));
        if (health.ready) {
          const next = await fetchStatus();
          setStats(next);
          if (next.mode !== undefined) setMode(next.mode);
        }
      } catch {
        setBackendOnline(false);
        setBackendReady(false);
      }
    };

    loadStats();
    const interval = setInterval(loadStats, 1500);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "1") setDetectionMode("object");
      if (e.key === "2") setDetectionMode("currency");
      if (e.key === "0") setDetectionMode(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setDetectionMode]);

  const toggleListening = () => {
    if (mode === "object") setDetectionMode(null);
    else setDetectionMode("object");
  };

  const handleCommand = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = command.trim();
    if (!q) return;
    setCommand("");

    const lower = q.toLowerCase();
    const isQuestion =
      lower.includes("?") ||
      lower.startsWith("what") ||
      lower.startsWith("where") ||
      lower.startsWith("is ") ||
      lower.startsWith("how ") ||
      lower.includes("do you see") ||
      lower.includes("ahead") ||
      lower.includes("path");

    try {
      if (isQuestion) {
        await runAsk(q);
        return;
      }
      if (lower.includes("currency") || lower.includes("money") || lower.includes("rupee")) {
        await setDetectionMode("currency");
        return;
      }
      if (
        lower.includes("start") ||
        lower.includes("object") ||
        lower.includes("vision") ||
        lower.includes("detect")
      ) {
        await setDetectionMode("object");
        return;
      }
      if (lower.includes("stop")) {
        await setDetectionMode(null);
        return;
      }
      const result = await sendCommand(q);
      if (result.result) speak(result.result);
    } catch (error) {
      console.error("Command failed:", error);
      speak(responseMessage);
    }
  };

  const isOnline = backendOnline && backendReady;
  const coreActive = Boolean(mode);
  const isWarning = danger === "critical" || danger === "high" || stats?.core_state === "warning";

  return (
    <div className="veyra-app">
      <div className="starfield" aria-hidden />

      {backendOnline === false && (
        <div className="veyra-alert veyra-alert--error">
          Backend offline — run <code>cd backend && python main.py</code>
        </div>
      )}
      {backendOnline && !backendReady && (
        <div className="veyra-alert veyra-alert--warn">
          Loading vision models and camera…
        </div>
      )}

      <header className="veyra-header">
        <div className="veyra-brand">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/logo-za.png?v=2"
            alt="Zyra AI"
            className="veyra-brand-mark"
          />
          <p className="veyra-subtitle">
            <span className={`veyra-dot ${coreActive ? "veyra-dot--on" : ""}`} />
            CORE {coreActive ? "ACTIVE" : "STANDBY"}
          </p>
        </div>

        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/zyra-wordmark.png?v=2"
          alt="Zyra AI — Vision · Intelligence · Action"
          className="veyra-wordmark"
        />

        <div className="veyra-online">
          <span className={`veyra-dot ${isOnline ? "veyra-dot--on" : ""}`} />
          {isOnline ? "ONLINE" : "CONNECTING"}
        </div>
      </header>

      <main className="veyra-main">
        <ResponsePanel
          message={responseMessage}
          warning={warningMessage}
          objects={objects}
          path={path}
          danger={danger}
          currency={stats?.currency}
          events={stats?.recent_events}
        />

        <section className="veyra-center">
          <AiCore
            active={coreActive}
            listening={mode === "object"}
            online={Boolean(isOnline)}
            processing={Boolean(mode && detections.length === 0) || askBusy}
            responding={Boolean(mode && detections.length > 0)}
            warning={isWarning}
            visionActive={Boolean(mode && backendReady)}
            coreState={stats?.core_state}
            onToggleListening={toggleListening}
            onDeactivate={() => setDetectionMode(null)}
          />

          <form className="veyra-command" onSubmit={handleCommand}>
            <span className="veyra-prompt">&gt;</span>
            <input
              type="text"
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="What do you see? What's ahead? Where is my phone?"
              className="veyra-command-input"
              spellCheck={false}
              aria-label="Ask Zyra"
            />
          </form>

          <div className="veyra-mode-toggles">
            <button
              type="button"
              className={`veyra-mode-btn ${mode === "object" ? "active" : ""}`}
              onClick={() => setDetectionMode("object")}
            >
              Vision
            </button>
            <button
              type="button"
              className={`veyra-mode-btn ${mode === "currency" ? "active" : ""}`}
              onClick={() => setDetectionMode("currency")}
            >
              Currency
            </button>
          </div>
        </section>

        <VisionPanel
          videoUrl={videoUrl}
          summary={visionSummary}
          live={Boolean(mode && backendReady)}
          fps={fps}
        />
      </main>
    </div>
  );
}
