import type { Metadata } from "next";
import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  subsets: ["latin"],
  weight: ["400", "600", "700", "800"],
  variable: "--font-outfit",
});

export const metadata: Metadata = {
  title: "Zyra AI | Vision • Intelligence • Action",
  description: "Zyra AI — real-time vision, intelligence, and action",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${outfit.variable} h-full`} suppressHydrationWarning>
      <body
        className="h-full overflow-hidden bg-black text-white antialiased"
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
