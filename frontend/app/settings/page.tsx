"use client";

import { useEffect, useState } from "react";
import {
  HardDrive,
  Database,
  ShieldCheck,
  Clock,
  User,
  LogOut,
  Smartphone,
  Zap,
  Sparkles,
  Crown,
  Check,
  CheckCircle2,
  CreditCard,
  Coins,
  RefreshCw,
  Plus,
  Sliders,
  Laptop,
  CheckCircle,
  AlertCircle,
  X,
  UserCheck
} from "lucide-react";
import { getErrorMessage, getIndexStats, getHealth } from "@/lib/api";
import { IndexStats } from "@/types";
import CatMascot from "@/components/CatMascot";

interface AccountProfile {
  name: string;
  email: string;
  accountId: string;
  joinedDate: string;
  isLoggedIn: boolean;
}

type PlanTier = "free" | "premium" | "pro";

interface PlanConfig {
  id: PlanTier;
  name: string;
  badge?: string;
  price: string;
  pricePeriod: string;
  deviceLimit: number;
  tokenLimit: number;
  tokenFormatted: string;
  tokenRefreshText: string;
  tokenRefreshPeriod: string;
  description: string;
  highlight?: boolean;
  features: string[];
}

const PLANS: PlanConfig[] = [
  {
    id: "free",
    name: "Free Plan",
    price: "Free",
    pricePeriod: "₹0 forever",
    deviceLimit: 1,
    tokenLimit: 250000,
    tokenFormatted: "250K",
    tokenRefreshText: "Refreshes every 5 hours",
    tokenRefreshPeriod: "Rolling 5-hour window",
    description: "Available to every user by default. A generous 250,000 token pool that restores every 5 hours—usable again and again throughout the day.",
    features: [
      "250,000 tokens restored every 5 hours",
      "Rolling ChatGPT-style window (no monthly lockout)",
      "Single local device support",
      "Full local file indexing (PDF, DOCX, PPTX, TXT)",
      "Local OCR & Image understanding",
      "Duplicate file detection & cleanup",
    ],
  },
  {
    id: "premium",
    name: "Premium",
    badge: "Most Popular",
    price: "₹250",
    pricePeriod: "per month",
    deviceLimit: 3,
    tokenLimit: 2500000,
    tokenFormatted: "2.5M",
    tokenRefreshText: "Refreshes every 4 hours",
    tokenRefreshPeriod: "Rolling 4-hour window",
    description: "Rewards you with 10x more tokens (2.5M) and a faster 4-hour reset cycle, cross-synced across up to 3 devices.",
    highlight: true,
    features: [
      "2,500,000 tokens restored every 4 hours",
      "Faster 4-hour rolling refresh cycle",
      "Support for up to 3 connected devices",
      "Priority embedding & semantic indexing",
      "Deep Audio & Video indexing support",
      "Cross-device unified search index",
    ],
  },
  {
    id: "pro",
    name: "Premium Pro",
    badge: "Ultimate Power",
    price: "₹750",
    pricePeriod: "per month",
    deviceLimit: 6,
    tokenLimit: 10000000,
    tokenFormatted: "10M",
    tokenRefreshText: "Refreshes every 2 hours",
    tokenRefreshPeriod: "Rolling 2-hour window",
    description: "Large 10M token pool with rapid 2-hour turnaround and simultaneous mesh sync across up to 6 devices.",
    features: [
      "10,000,000 tokens restored every 2 hours",
      "Rapid 2-hour rolling refresh cycle",
      "Support for up to 6 connected devices",
      "Ultra-fast parallel extraction pipeline",
      "Unlimited deep transcription & search",
      "24/7 dedicated priority developer support",
    ],
  },
];

function formatNumber(num: number): string {
  return Math.round(num).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

const DEFAULT_ACCOUNT: AccountProfile = {
  name: "Joyel Shaji",
  email: "joyel.shaji@refind.local",
  accountId: "usr_rf_9a8f4c21e7",
  joinedDate: "October 2026",
  isLoggedIn: true,
};

export default function SettingsPage() {
  const [stats, setStats] = useState<IndexStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [health, setHealth] = useState<any>(null);

  // Account State (Persisted in localStorage)
  const [account, setAccount] = useState<AccountProfile>(DEFAULT_ACCOUNT);
  const [activePlan, setActivePlan] = useState<PlanTier>("free");

  // Modals & Feedback
  const [showAccountModal, setShowAccountModal] = useState(false);
  const [showDeviceModal, setShowDeviceModal] = useState(false);
  const [modalName, setModalName] = useState("");
  const [modalEmail, setModalEmail] = useState("");
  const [feedback, setFeedback] = useState<{ type: "success" | "info"; message: string } | null>(null);

  // Load persisted user & plan preference on mount
  useEffect(() => {
    try {
      const storedAccount = localStorage.getItem("refind_account_profile");
      if (storedAccount) {
        setAccount(JSON.parse(storedAccount));
      }
      const storedPlan = localStorage.getItem("refind_active_plan") as PlanTier | null;
      if (storedPlan && (storedPlan === "free" || storedPlan === "premium" || storedPlan === "pro")) {
        setActivePlan(storedPlan);
      }
    } catch {
      // Ignore localStorage errors in restricted environments
    }
  }, []);

  useEffect(() => {
    getIndexStats()
      .then((data) => {
        setStats(data);
        setStatsError(null);
      })
      .catch((err: unknown) => {
        setStatsError(getErrorMessage(err, "Index statistics are unavailable."));
      });

    getHealth()
      .then((data) => setHealth(data))
      .catch((err) => console.error("Error fetching health:", err));
  }, []);

  const handleSelectPlan = (tier: PlanTier) => {
    setActivePlan(tier);
    try {
      localStorage.setItem("refind_active_plan", tier);
    } catch {}
    const plan = PLANS.find((p) => p.id === tier);
    setFeedback({
      type: "success",
      message: `Plan updated to ${plan?.name}! You now have up to ${plan?.deviceLimit} device(s) and ${plan?.tokenFormatted} tokens.`,
    });
    setTimeout(() => setFeedback(null), 4000);
  };

  const handleOpenAccountModal = () => {
    setModalName(account.name);
    setModalEmail(account.email);
    setShowAccountModal(true);
  };

  const handleSaveAccount = (e: React.FormEvent) => {
    e.preventDefault();
    if (!modalName.trim() || !modalEmail.trim()) return;
    const updated: AccountProfile = {
      ...account,
      name: modalName.trim(),
      email: modalEmail.trim(),
      isLoggedIn: true,
    };
    setAccount(updated);
    try {
      localStorage.setItem("refind_account_profile", JSON.stringify(updated));
    } catch {}
    setShowAccountModal(false);
    setFeedback({ type: "success", message: "Account profile updated successfully." });
    setTimeout(() => setFeedback(null), 3000);
  };

  const handleToggleLogin = () => {
    const updated = {
      ...account,
      isLoggedIn: !account.isLoggedIn,
    };
    setAccount(updated);
    try {
      localStorage.setItem("refind_account_profile", JSON.stringify(updated));
    } catch {}
    setFeedback({
      type: "info",
      message: updated.isLoggedIn
        ? "Signed back into ReFind account."
        : "Logged out. You are currently browsing as a local guest.",
    });
    setTimeout(() => setFeedback(null), 3000);
  };

  const currentPlan = PLANS.find((p) => p.id === activePlan) || PLANS[0];

  // Rolling token usage metrics per plan (restored 250K, 2.5M, 10M limits)
  const tokenUsage = activePlan === "free" ? 142500 : activePlan === "premium" ? 685000 : 1420000;
  const tokenPercent = Math.min(100, Math.round((tokenUsage / currentPlan.tokenLimit) * 100));
  const nextResetTime = activePlan === "free" ? "2h 45m" : activePlan === "premium" ? "1h 18m" : "42m";

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10 space-y-10">
      {/* Toast Feedback */}
      {feedback && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4 py-3 rounded-xl bg-indigo-950/90 border border-indigo-500/50 text-indigo-200 text-sm shadow-[0_8px_32px_rgba(0,0,0,0.8)] backdrop-blur-md animate-in fade-in slide-in-from-bottom-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>{feedback.message}</span>
        </div>
      )}

      {/* Page Title */}
      <div className="page-heading" data-grav-mass="1.2">
        <div className="page-heading__copy space-y-2">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.1] backdrop-blur-md text-xs sm:text-sm text-indigo-400 font-mono font-medium">
            <Sliders className="w-4 h-4" />
            System Preferences & Account
          </div>
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-white">
            Settings & Membership
          </h1>
          <p className="text-base text-zinc-300">
            Manage your account profile, membership plan, rolling token quota, and data privacy.
          </p>
        </div>
        <CatMascot className="page-heading__cat hidden md:block" />
      </div>

      {/* SECTION 1: Account & Profile Management */}
      <div
        data-grav-mass="1.1"
        className="p-6 rounded-2xl bg-black/45 border border-white/[0.08] backdrop-blur-xl space-y-6 shadow-[0_8px_32px_rgba(0,0,0,0.6)]"
      >
        <div className="flex items-center justify-between flex-wrap gap-4 border-b border-white/[0.08] pb-4">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold text-zinc-100 flex items-center gap-2.5">
              <User className="w-5 h-5 text-indigo-400" />
              Account & Profile
            </h2>
            <p className="text-sm text-zinc-300">
              Manage personal identity, active authentication state, and linked devices.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleOpenAccountModal}
              className="px-3.5 py-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.1] text-xs sm:text-sm font-medium text-zinc-200 transition-all"
            >
              Edit Account
            </button>
            <button
              onClick={handleToggleLogin}
              className={`flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs sm:text-sm font-medium border transition-all ${
                account.isLoggedIn
                  ? "bg-red-950/20 hover:bg-red-950/40 border-red-500/30 text-red-300"
                  : "bg-emerald-950/20 hover:bg-emerald-950/40 border-emerald-500/30 text-emerald-300"
              }`}
            >
              <LogOut className="w-3.5 h-3.5" />
              <span>{account.isLoggedIn ? "Log Out" : "Sign In"}</span>
            </button>
          </div>
        </div>

        {account.isLoggedIn ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
            {/* User Identity Pill */}
            <div className="flex items-center gap-4 col-span-1 md:col-span-2">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-indigo-600 border border-indigo-400/40 flex items-center justify-center text-white font-bold text-xl shadow-[0_0_20px_rgba(244,214,78,0.35)] shrink-0">
                {account.name
                  .split(" ")
                  .map((n) => n[0])
                  .join("")
                  .toUpperCase()
                  .slice(0, 2) || "RF"}
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2.5 flex-wrap">
                  <span className="text-base sm:text-lg font-bold text-white tracking-tight">
                    {account.name}
                  </span>
                  <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-emerald-950/60 text-emerald-300 border border-emerald-500/40">
                    <UserCheck className="w-3 h-3" />
                    Verified
                  </span>
                  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-mono font-medium bg-indigo-950/60 text-indigo-300 border border-indigo-500/35">
                    {currentPlan.name}
                  </span>
                </div>
                <p className="text-sm text-zinc-300 font-mono">{account.email}</p>
                <p className="text-xs text-zinc-500 font-mono">
                  Account ID: <span className="text-zinc-400">{account.accountId}</span> • Joined {account.joinedDate}
                </p>
              </div>
            </div>

            {/* Connected Devices Badge */}
            <div className="p-4 rounded-xl bg-black/50 border border-white/[0.08] flex flex-col justify-between gap-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider font-medium flex items-center gap-1.5">
                  <Smartphone className="w-3.5 h-3.5 text-indigo-400" />
                  Connected Devices
                </span>
                <span className="text-xs font-mono font-bold text-emerald-400">
                  1 / {currentPlan.deviceLimit} Active
                </span>
              </div>
              <div className="flex items-center justify-between text-xs sm:text-sm text-zinc-300">
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  Windows PC (This Device)
                </span>
                <button
                  onClick={() => setShowDeviceModal(true)}
                  className="text-xs font-mono text-indigo-400 hover:text-indigo-300 underline font-medium"
                >
                  Manage
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-6 rounded-xl bg-black/40 border border-dashed border-white/[0.15] text-center space-y-3">
            <div className="w-12 h-12 rounded-full bg-white/[0.04] border border-white/[0.1] flex items-center justify-center mx-auto text-zinc-400">
              <User className="w-5 h-5" />
            </div>
            <div className="space-y-1">
              <h3 className="text-base font-semibold text-white">Local Guest Mode</h3>
              <p className="text-sm text-zinc-400 max-w-md mx-auto">
                You are currently using ReFind locally without an authenticated profile. Sign in to link your personal token quota across devices.
              </p>
            </div>
            <button
              onClick={handleToggleLogin}
              className="px-5 py-2.5 rounded-xl bg-indigo-600/40 hover:bg-indigo-600/60 border border-indigo-500/40 text-white text-sm font-medium transition-all shadow-[0_0_15px_rgba(244,214,78,0.25)]"
            >
              Sign In to ReFind
            </button>
          </div>
        )}
      </div>

      {/* SECTION 2: Token Quotas & Usage Meter */}
      <div
        data-grav-mass="1.1"
        className="p-6 rounded-2xl bg-black/45 border border-white/[0.08] backdrop-blur-xl space-y-5 shadow-[0_8px_32px_rgba(0,0,0,0.6)]"
      >
        <div className="flex items-center justify-between flex-wrap gap-4 border-b border-white/[0.08] pb-4">
          <div className="space-y-1">
            <h2 className="text-lg font-semibold text-zinc-100 flex items-center gap-2.5">
              <Coins className="w-5 h-5 text-amber-400" />
              Token Quota & Usage
            </h2>
            <p className="text-sm text-zinc-300">
              ChatGPT-style rolling token window that resets every 5 hours (or faster on paid plans) for continuous daily search.
            </p>
          </div>

          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] text-xs font-mono text-zinc-300 font-medium">
            <RefreshCw className="w-3.5 h-3.5 text-indigo-400" />
            <span>{currentPlan.tokenRefreshText}</span>
          </div>
        </div>

        {/* Progress Meter */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <span className="font-mono text-zinc-300 font-medium">
              Used: <strong className="text-white font-bold">{formatNumber(tokenUsage)}</strong> of{" "}
              <strong className="text-indigo-300">{formatNumber(currentPlan.tokenLimit)} tokens</strong>
            </span>
            <span className="font-mono font-bold text-white">{tokenPercent}%</span>
          </div>

          <div className="w-full h-3 rounded-full bg-white/[0.08] overflow-hidden p-0.5">
            <div
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-indigo-400 to-emerald-400 transition-all duration-500 shadow-[0_0_12px_rgba(244,214,78,0.5)]"
              style={{ width: `${tokenPercent}%` }}
            />
          </div>

          <div className="flex items-center justify-between flex-wrap gap-2 text-xs text-zinc-400 font-mono">
            <span>Cycle: {currentPlan.tokenRefreshPeriod} • Next reset in {nextResetTime}</span>
            <span>
              Remaining: {formatNumber(currentPlan.tokenLimit - tokenUsage)} tokens
            </span>
          </div>
        </div>

        {/* Usage Breakdown Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5 pt-2">
          <div className="p-3.5 rounded-xl bg-black/50 border border-white/[0.06] space-y-1">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider block">
              Search & Retrieval
            </span>
            <span className="text-base font-bold text-white font-mono">
              {formatNumber(tokenUsage * 0.35)} tokens
            </span>
          </div>
          <div className="p-3.5 rounded-xl bg-black/50 border border-white/[0.06] space-y-1">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider block">
              Vision & OCR Extraction
            </span>
            <span className="text-base font-bold text-white font-mono">
              {formatNumber(tokenUsage * 0.45)} tokens
            </span>
          </div>
          <div className="p-3.5 rounded-xl bg-black/50 border border-white/[0.06] space-y-1">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider block">
              Context Reasoning
            </span>
            <span className="text-base font-bold text-white font-mono">
              {formatNumber(tokenUsage * 0.20)} tokens
            </span>
          </div>
        </div>
      </div>

      {/* SECTION 3: Membership Plans (Free, Premium 250, Premium Pro 750) */}
      <div className="space-y-5">
        <div className="space-y-1">
          <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <Crown className="w-6 h-6 text-amber-400" />
            Subscription & Device Plans
          </h2>
          <p className="text-sm text-zinc-300">
            Select the tier matching your workflow. Upgrade anytime to unlock more tokens and multi-device mesh synchronization.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PLANS.map((plan) => {
            const isSelected = activePlan === plan.id;
            return (
              <div
                key={plan.id}
                className={`relative p-6 rounded-2xl flex flex-col justify-between transition-all duration-300 backdrop-blur-xl ${
                  isSelected
                    ? "bg-gradient-to-b from-indigo-950/40 via-black/70 to-black/80 border-2 border-indigo-400/70 shadow-[0_0_35px_rgba(244,214,78,0.25)] ring-1 ring-indigo-400/30"
                    : plan.highlight
                    ? "bg-black/55 border border-purple-500/30 hover:border-purple-400/60 shadow-[0_8px_32px_rgba(0,0,0,0.6)]"
                    : "bg-black/45 border border-white/[0.08] hover:border-white/[0.18] shadow-[0_8px_32px_rgba(0,0,0,0.5)]"
                }`}
              >
                {/* Badge */}
                {plan.badge && (
                  <div className="absolute -top-3 right-6 px-3 py-0.5 rounded-full bg-gradient-to-r from-indigo-500 to-indigo-600 text-white text-xs font-mono font-bold uppercase tracking-wider shadow-[0_0_15px_rgba(244,214,78,0.5)]">
                    {plan.badge}
                  </div>
                )}

                <div className="space-y-4">
                  {/* Title & Price */}
                  <div>
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-bold text-white tracking-tight">{plan.name}</h3>
                      {isSelected && (
                        <span className="inline-flex items-center gap-1 text-xs font-mono font-semibold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded-full border border-emerald-500/30">
                          <Check className="w-3 h-3" />
                          Active
                        </span>
                      )}
                    </div>
                    <div className="flex items-baseline gap-2 mt-2">
                      <span className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight font-mono">
                        {plan.price}
                      </span>
                      <span className="text-xs sm:text-sm text-zinc-400 font-mono">
                        {plan.pricePeriod}
                      </span>
                    </div>
                    <p className="text-xs sm:text-sm text-zinc-400 mt-2.5 leading-relaxed">
                      {plan.description}
                    </p>
                  </div>

                  {/* Highlights (Tokens & Devices) */}
                  <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.06] space-y-2">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-zinc-400">Tokens:</span>
                      <span className="font-bold text-indigo-300">{plan.tokenFormatted} tokens</span>
                    </div>
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-zinc-400">Refresh Cycle:</span>
                      <span className="text-zinc-200">{plan.tokenRefreshText}</span>
                    </div>
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-zinc-400">Supported Devices:</span>
                      <span className="font-bold text-emerald-300">Up to {plan.deviceLimit} device{plan.deviceLimit > 1 ? "s" : ""}</span>
                    </div>
                  </div>

                  {/* Features List */}
                  <div className="space-y-2.5 pt-2">
                    <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider block font-medium">
                      What&apos;s Included:
                    </span>
                    <ul className="space-y-2 text-xs sm:text-sm text-zinc-300">
                      {plan.features.map((feat, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                          <span>{feat}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Plan Action Button */}
                <div className="pt-6">
                  {isSelected ? (
                    <button
                      disabled
                      className="w-full py-2.5 rounded-xl bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 text-sm font-semibold cursor-default"
                    >
                      Current Plan
                    </button>
                  ) : (
                    <button
                      onClick={() => handleSelectPlan(plan.id)}
                      className={`w-full py-2.5 rounded-xl text-sm font-semibold transition-all shadow-md ${
                        plan.id === "pro"
                          ? "bg-gradient-to-r from-amber-500/30 via-purple-600/30 to-indigo-600/30 hover:from-amber-500/50 hover:to-indigo-600/50 text-white border border-amber-500/40"
                          : "bg-indigo-600/30 hover:bg-indigo-600/50 text-white border border-indigo-500/40 hover:shadow-[0_0_20px_rgba(244,214,78,0.3)]"
                      }`}
                    >
                      {activePlan === "pro" && plan.id === "free"
                        ? "Switch to Free"
                        : `Select ${plan.name}`}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* SECTION 4: Data Sources & Local Index connector */}
      <div
        data-grav-mass="1.1"
        className="p-6 rounded-2xl bg-black/45 border border-white/[0.08] backdrop-blur-xl space-y-5 shadow-[0_8px_32px_rgba(0,0,0,0.6)]"
      >
        <div className="space-y-1.5 border-b border-white/[0.08] pb-3.5">
          <h2 className="text-lg font-semibold text-zinc-100 flex items-center gap-2.5">
            <HardDrive className="w-5 h-5 text-indigo-400" />
            Data Sources & Connectors
          </h2>
          <p className="text-sm text-zinc-300">
            ReFind indexes and monitors files directly on your local system with zero third-party leakage.
          </p>
        </div>

        {/* Active Connector: Local Files */}
        <div className="p-4 sm:p-5 rounded-xl bg-black/50 border border-white/[0.08] flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3.5">
            <div className="p-3 rounded-xl bg-emerald-950/30 border border-emerald-500/30 text-emerald-400 shadow-[0_0_12px_rgba(16,185,129,0.2)]">
              <HardDrive className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-base font-semibold text-zinc-100">Local Filesystem</span>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-emerald-950/60 text-emerald-300 border border-emerald-500/40">
                  Active
                </span>
              </div>
              <p className="text-sm text-zinc-300 mt-1">
                Indexes PDF, DOCX, PPTX, TXT, Images (OCR + Vision), and Audio/Video files from your device.
              </p>
            </div>
          </div>

          <div className="text-sm font-mono text-zinc-300 font-medium">
            {stats ? `${stats.total_documents} documents indexed` : statsError ? "Unavailable" : "Ready"}
          </div>
        </div>
      </div>

      {/* SECTION 5: Index & Storage Statistics */}
      <div
        data-grav-mass="1.1"
        className="p-6 rounded-2xl bg-black/45 border border-white/[0.08] backdrop-blur-xl space-y-5 shadow-[0_8px_32px_rgba(0,0,0,0.6)]"
      >
        <div className="space-y-1.5 border-b border-white/[0.08] pb-3.5">
          <h2 className="text-lg font-semibold text-zinc-100 flex items-center gap-2.5">
            <Database className="w-5 h-5 text-indigo-400" />
            Index & Storage Statistics
          </h2>
          <p className="text-sm text-zinc-300">
            Real-time status of the relational SQLite database and dense embedding vector store.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3.5">
          <div className="p-4 rounded-xl bg-black/50 border border-white/[0.08]">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider block font-medium">
              Total Documents
            </span>
            <span className="text-2xl sm:text-3xl font-bold text-white mt-1.5 block font-mono">
              {stats?.total_documents ?? (statsError ? "—" : 0)}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-black/50 border border-white/[0.08]">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider block font-medium">
              Indexed Chunks
            </span>
            <span className="text-2xl sm:text-3xl font-bold text-white mt-1.5 block font-mono">
              {stats?.total_chunks ?? (statsError ? "—" : 0)}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-black/50 border border-white/[0.08]">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider block font-medium">
              Vector Dimension
            </span>
            <span className="text-2xl sm:text-3xl font-bold text-white mt-1.5 block font-mono">
              {stats ? `${stats.embedding_dimension}d` : statsError ? "—" : "768d"}
            </span>
          </div>

          <div className="p-4 rounded-xl bg-black/50 border border-white/[0.08]">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider block font-medium">
              Engine Status
            </span>
            <span className="text-sm font-mono font-semibold text-emerald-400 mt-2 block truncate">
              {health?.database?.pgvector_enabled ? "PostgreSQL + pgvector" : health ? "Local Vector Store" : "Ready"}
            </span>
          </div>
        </div>

        <div className="p-3.5 rounded-xl bg-black/50 border border-white/[0.08] flex items-center justify-between text-sm text-zinc-300 font-mono">
          <span className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-zinc-400" />
            Last Indexed:
          </span>
          <span suppressHydrationWarning className="text-zinc-100 font-medium">
            {stats?.last_indexed_at ? new Date(stats.last_indexed_at).toLocaleString() : statsError ? "Unavailable" : "Never"}
          </span>
        </div>
      </div>

      {/* SECTION 6: Privacy & Data Sovereignty */}
      <div
        data-grav-mass="1.0"
        className="p-6 rounded-2xl bg-black/45 border border-white/[0.08] backdrop-blur-xl space-y-4 shadow-[0_8px_32px_rgba(0,0,0,0.6)]"
      >
        <div className="space-y-1.5 border-b border-white/[0.08] pb-3.5">
          <h2 className="text-lg font-semibold text-zinc-100 flex items-center gap-2.5">
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
            Privacy & Data Sovereignty
          </h2>
          <p className="text-sm text-zinc-300">
            ReFind operates strictly on local-first privacy principles.
          </p>
        </div>

        <div className="space-y-3.5 text-sm sm:text-base text-zinc-200 leading-relaxed font-sans">
          <p>
            • <strong className="text-white font-semibold">Local-First Indexing:</strong> All documents, OCR frames, embeddings, and relational metadata are processed on your local machine.
          </p>
          <p>
            • <strong className="text-white font-semibold">Zero Surveillance:</strong> ReFind does not scrape your web browsing, background apps, or personal accounts. No file uploads occur to third-party servers.
          </p>
          <p>
            • <strong className="text-white font-semibold">Grounded Evidence:</strong> Search explanations only quote text and metadata that actually exist in your indexed content, avoiding hallucinations.
          </p>
        </div>
      </div>

      {/* MODAL 1: Edit Account Details */}
      {showAccountModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-in fade-in">
          <div className="w-full max-w-md p-6 rounded-2xl bg-zinc-950 border border-white/[0.12] shadow-[0_16px_48px_rgba(0,0,0,0.8)] space-y-5">
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <User className="w-5 h-5 text-indigo-400" />
                Edit Account Profile
              </h3>
              <button
                onClick={() => setShowAccountModal(false)}
                className="p-1 rounded-lg hover:bg-white/[0.08] text-zinc-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleSaveAccount} className="space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs font-mono text-zinc-300 uppercase tracking-wider block">
                  Full Name:
                </label>
                <input
                  type="text"
                  required
                  value={modalName}
                  onChange={(e) => setModalName(e.target.value)}
                  placeholder="e.g. Joyel Shaji"
                  className="w-full px-4 py-2.5 rounded-xl bg-black/60 border border-white/[0.1] text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-indigo-400 transition-all font-sans"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-mono text-zinc-300 uppercase tracking-wider block">
                  Email Address:
                </label>
                <input
                  type="email"
                  required
                  value={modalEmail}
                  onChange={(e) => setModalEmail(e.target.value)}
                  placeholder="e.g. joyel@example.com"
                  className="w-full px-4 py-2.5 rounded-xl bg-black/60 border border-white/[0.1] text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-indigo-400 transition-all font-mono"
                />
              </div>

              <div className="pt-2 flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setShowAccountModal(false)}
                  className="flex-1 px-4 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.1] text-sm font-medium text-zinc-300 border border-white/[0.08] transition-all"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="flex-1 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-sm font-medium text-white transition-all shadow-[0_0_15px_rgba(244,214,78,0.4)]"
                >
                  Save Changes
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* MODAL 2: Manage Connected Devices */}
      {showDeviceModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-in fade-in">
          <div className="w-full max-w-lg p-6 rounded-2xl bg-zinc-950 border border-white/[0.12] shadow-[0_16px_48px_rgba(0,0,0,0.8)] space-y-5">
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <Smartphone className="w-5 h-5 text-indigo-400" />
                Connected Devices ({currentPlan.name})
              </h3>
              <button
                onClick={() => setShowDeviceModal(false)}
                className="p-1 rounded-lg hover:bg-white/[0.08] text-zinc-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-sm text-zinc-300">
              Your active plan supports up to <strong className="text-white">{currentPlan.deviceLimit} device{currentPlan.deviceLimit > 1 ? "s" : ""}</strong>.
            </p>

            <div className="space-y-3">
              <div className="p-3.5 rounded-xl bg-white/[0.03] border border-white/[0.08] flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Laptop className="w-5 h-5 text-indigo-400" />
                  <div>
                    <span className="text-sm font-semibold text-white block">Windows PC</span>
                    <span className="text-xs font-mono text-zinc-400">DESKTOP-WIN11 • Local Node</span>
                  </div>
                </div>
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-emerald-950/60 border border-emerald-500/40 text-emerald-300">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                  Current Device
                </span>
              </div>

              {currentPlan.deviceLimit > 1 && (
                <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.06] flex items-center justify-between opacity-80">
                  <div className="flex items-center gap-3">
                    <Smartphone className="w-5 h-5 text-zinc-400" />
                    <div>
                      <span className="text-sm font-semibold text-zinc-200 block">Mobile Companion</span>
                      <span className="text-xs font-mono text-zinc-500">Device Slot Available</span>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 rounded text-xs font-mono text-indigo-300 bg-indigo-950/40 border border-indigo-500/30">
                    Slot Ready
                  </span>
                </div>
              )}
            </div>

            <div className="p-3.5 rounded-xl bg-black/60 border border-white/[0.08] text-xs text-zinc-400 font-mono space-y-1">
              <span className="text-indigo-300 block font-semibold">Device Pairing Code:</span>
              <p className="tracking-widest text-base font-bold text-white">RF-MESH-4829-X7</p>
              <p className="text-zinc-500 text-xs">Enter this code on your companion device to join the mesh network.</p>
            </div>

            <button
              onClick={() => setShowDeviceModal(false)}
              className="w-full py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.1] text-sm font-medium text-zinc-200 border border-white/[0.1] transition-all"
            >
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
