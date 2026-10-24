"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Search, FolderGit2, Settings, Copy } from "lucide-react";
import { getHealth } from "@/lib/api";

export default function Navbar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    getHealth().then((data) => setHealth(data));
  }, []);

  const navItems = [
    { label: "Search", href: "/", icon: Search },
    { label: "Indexing", href: "/indexing", icon: FolderGit2 },
    { label: "Duplicates", href: "/duplicates", icon: Copy },
    { label: "Settings", href: "/settings", icon: Settings },
  ];

  return (
    <header className="sticky top-0 z-40 border-b border-white/[0.08] bg-black/60 backdrop-blur-xl transition-all">
      <div className="max-w-[1440px] mx-auto px-4 sm:px-6 min-h-[4.5rem] py-3 lg:py-0 flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-10 h-10 rounded-2xl bg-[#15140f] border border-[#66551e] flex items-center justify-center text-[#f4d64e] shadow-[0_0_18px_-4px_rgba(244,214,78,0.28)] group-hover:border-[#f4d64e]/70 group-hover:shadow-[0_0_22px_-2px_rgba(244,214,78,0.35)] transition-all">
              <Image src="/favicon.ico" alt="" width={36} height={36} unoptimized className="rounded-xl" />
            </div>
            <div className="flex flex-col">
              <span className="font-semibold text-base tracking-tight text-white flex items-center gap-2">
                ReFind
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/[0.04] text-zinc-300 border border-white/[0.12] font-medium">
                  BitByBit
                </span>
              </span>
            </div>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="order-3 lg:order-none w-full lg:w-auto flex items-center justify-center gap-1 sm:gap-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || (item.href === "/" && pathname.startsWith("/search"));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-sm font-medium transition-all ${
                  isActive
                  ? "bg-[#19170f] text-white border border-[#746126] shadow-[0_0_16px_rgba(244,214,78,0.08)]"
                    : "text-zinc-300 hover:text-white hover:bg-white/[0.04] border border-transparent"
                }`}
              >
                <Icon className="w-4 h-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <button
          type="button"
          onClick={() => window.dispatchEvent(new Event("refind:open-quick-search"))}
          aria-label="Open quick search"
          title="Quick search (Alt+Space or Ctrl+K)"
          className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-2.5 py-1.5 text-xs text-zinc-400 transition-colors hover:border-white/20 hover:text-white"
        >
          <Search className="h-3.5 w-3.5" />
          <span className="hidden xl:inline">Quick search</span>
          <kbd className="hidden rounded border border-white/10 px-1.5 py-0.5 font-mono text-[10px] xl:inline">Alt Space</kbd>
        </button>

        {/* Engine Status indicator */}
        <div className="hidden sm:flex items-center gap-2.5">
          <div className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white/[0.035] border border-white/[0.12] text-xs text-zinc-200 backdrop-blur-md">
            <span className={`w-2 h-2 rounded-full ${health ? "bg-emerald-400 shadow-[0_0_8px_#34d399] animate-pulse" : "bg-zinc-500"}`} />
            <span className="font-mono text-xs text-zinc-300 tracking-wide font-medium">
              {health?.database?.pgvector_enabled ? "PostgreSQL + pgvector" : "Local Vector Engine"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
