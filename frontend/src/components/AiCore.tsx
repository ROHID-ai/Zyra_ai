"use client";

import HolographicAICore, {
  type CoreVisualState,
} from "@/components/HolographicAICore";

interface AiCoreProps {
  active: boolean;
  listening: boolean;
  online: boolean;
  processing: boolean;
  responding: boolean;
  warning?: boolean;
  visionActive: boolean;
  coreState?: string | null;
  onToggleListening: () => void;
  onDeactivate: () => void;
}

function resolveVisualState({
  online,
  listening,
  processing,
  responding,
  warning,
  active,
  coreState,
}: Pick<
  AiCoreProps,
  | "online"
  | "listening"
  | "processing"
  | "responding"
  | "warning"
  | "active"
  | "coreState"
>): CoreVisualState {
  if (!online) return "offline";
  if (coreState === "warning" || warning) return "warning";
  if (coreState === "listening" || listening) return "listening";
  if (coreState === "responding" || responding) return "responding";
  if (coreState === "thinking" || processing) return "thinking";
  if (coreState === "vision" || active) return "vision";
  return "idle";
}

function statusLabel(state: CoreVisualState): string {
  switch (state) {
    case "listening":
      return "LISTENING...";
    case "thinking":
      return "SCANNING...";
    case "responding":
      return "REPLYING...";
    case "warning":
      return "WARNING";
    case "vision":
      return "VISION ACTIVE";
    case "offline":
      return "CORE OFFLINE";
    default:
      return "STANDBY";
  }
}

export default function AiCore({
  active,
  listening,
  online,
  processing,
  responding,
  warning = false,
  visionActive,
  coreState,
  onToggleListening,
  onDeactivate,
}: AiCoreProps) {
  const visualState = resolveVisualState({
    online,
    listening,
    processing,
    responding,
    warning,
    active,
    coreState,
  });

  return (
    <div className="ai-core-wrap">
      <HolographicAICore
        visualState={visualState}
        visionActive={visionActive}
      />

      <p className="ai-status-label">{statusLabel(visualState)}</p>

      <button
        type="button"
        className={`ai-listen-btn ${listening ? "ai-listen-btn--on" : ""}`}
        onClick={onToggleListening}
        disabled={!online && !active}
      >
        {listening || active ? "VISION ON" : "ACTIVATE VISION"}
      </button>

      {active && (
        <button type="button" className="ai-deactivate" onClick={onDeactivate}>
          DEACTIVATE ZYRA
        </button>
      )}
    </div>
  );
}
