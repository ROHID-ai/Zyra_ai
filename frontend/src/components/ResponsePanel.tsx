"use client";

import type { CurrencySummary, LiveObject, SceneEvent } from "@/types";

interface ResponsePanelProps {
  message: string;
  warning?: string | null;
  objects?: LiveObject[];
  path?: string;
  danger?: string;
  currency?: CurrencySummary | null;
  events?: SceneEvent[];
}

function sceneSummary(objects: LiveObject[]): string {
  if (!objects.length) return "No clear objects yet.";
  const counts = new Map<string, number>();
  for (const obj of objects) {
    const name = obj.name || "object";
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .slice(0, 8)
    .map(([name, count]) => (count > 1 ? `${count} ${name}s` : `1 ${name}`))
    .join("\n");
}

export default function ResponsePanel({
  message,
  warning,
  objects = [],
  path = "clear",
  danger = "low",
  currency,
  events = [],
}: ResponsePanelProps) {
  const pathLabel = (path || "clear").replaceAll("_", " ").toUpperCase();
  const hasCurrency = Boolean(currency?.currency?.length);

  return (
    <aside className="veyra-panel veyra-panel--left">
      <div className="veyra-panel-header">
        <span className="veyra-panel-title">ZYRA RESPONSE</span>
        <span className="veyra-badge">AI</span>
      </div>

      {warning && (
        <div className="zyra-warning" role="alert">
          <span className="zyra-warning-label">WARNING</span>
          <p>{warning}</p>
        </div>
      )}

      <p className="veyra-response-text">{message}</p>

      <div className="zyra-meta-block">
        <span className="zyra-meta-label">LIVE SCENE</span>
        <pre className="zyra-meta-body">{sceneSummary(objects)}</pre>
      </div>

      <div className="zyra-meta-block">
        <span className="zyra-meta-label">PATH</span>
        <p
          className={`zyra-path zyra-path--${(path || "clear").replaceAll("_", "-")}`}
        >
          {pathLabel}
          {danger && danger !== "low" ? ` · ${danger.toUpperCase()}` : ""}
        </p>
      </div>

      {hasCurrency && (
        <div className="zyra-meta-block">
          <span className="zyra-meta-label">CURRENCY</span>
          <p className="zyra-meta-body">
            {(currency?.currency ?? [])
              .map((c) => `₹${c.value} × ${c.count}`)
              .join(" · ")}
            {typeof currency?.total === "number" ? ` · total ₹${currency.total}` : ""}
          </p>
        </div>
      )}

      {events.length > 0 && (
        <div className="zyra-meta-block">
          <span className="zyra-meta-label">RECENT</span>
          <ul className="zyra-events">
            {events
              .slice()
              .reverse()
              .slice(0, 4)
              .map((event, index) => (
                <li key={`${event.type ?? "e"}-${index}`}>
                  {event.message}
                </li>
              ))}
          </ul>
        </div>
      )}
    </aside>
  );
}
