"use client";

export type CoreVisualState =
  | "idle"
  | "listening"
  | "thinking"
  | "responding"
  | "warning"
  | "vision"
  | "offline";

interface HolographicAICoreProps {
  visualState: CoreVisualState;
  visionActive?: boolean;
}

const AI_CORE_GIF = "/ai-core.gif";

/**
 * Accurate Dribbble light-AI core animation (hosted locally).
 * State classes only adjust brightness/opacity — they do not replace the GIF.
 */
export default function HolographicAICore({
  visualState,
  visionActive = false,
}: HolographicAICoreProps) {
  return (
    <div
      className={[
        "holo-core",
        `holo-core--${visualState}`,
        visionActive ? "holo-core--vision" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="holo-gif-glow" aria-hidden />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={AI_CORE_GIF}
        alt="Zyra AI core"
        className="holo-gif"
        draggable={false}
      />
    </div>
  );
}
