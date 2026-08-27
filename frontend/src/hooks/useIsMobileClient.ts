"use client";

import { useEffect, useState } from "react";

/** True when opened on phone/tablet or via LAN IP (mobile testing). */
export function useIsMobileClient(): boolean {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const lanHost = process.env.NEXT_PUBLIC_LAN_HOST ?? "";
    const hostname = window.location.hostname;
    const onLan = Boolean(lanHost && hostname === lanHost);
    const uaMobile = /Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent);
    const narrow = window.matchMedia("(max-width: 768px)").matches;
    setIsMobile(onLan || uaMobile || narrow);
  }, []);

  return isMobile;
}
