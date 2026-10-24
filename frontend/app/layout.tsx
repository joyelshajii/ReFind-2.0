import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import GravitationalField from "@/components/GravitationalField";
import QuickSearchPalette from "@/components/QuickSearchPalette";

const fontSansClass = "font-sans";
const fontMonoClass = "font-mono";

export const metadata: Metadata = {
  title: "ReFind — Find what you remember.",
  description: "A privacy-focused, context-aware digital memory search system developed by BitByBit.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="min-h-full flex flex-col bg-[#070709] text-zinc-100 font-sans selection:bg-[#f4d64e]/25 selection:text-[#fff1ae] relative overflow-x-hidden text-base">
        {/* Deep space gravitational distortion particle canvas */}
        <GravitationalField />

        {/* Ambient space-time perimeter vignette */}
        <div className="fixed inset-0 gravitational-vignette z-[1] pointer-events-none" />

        {/* UI Content Layer above particle canvas */}
        <div className="relative z-10 flex-1 flex flex-col">
          <Navbar />
          <QuickSearchPalette />
          <main className="flex-1 flex flex-col">{children}</main>
          <footer className="border-t border-white/[0.09] bg-[#080809]/75 backdrop-blur-md py-6 text-center text-sm text-zinc-400 font-mono">
            <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
              <span className="tracking-wide">ReFind · Find what you remember</span>
              <span>Built by <strong className="text-zinc-200 font-medium tracking-wider">BitByBit</strong></span>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
