"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import Link from "next/link";
import {
  Copy,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Search,
  Sparkles,
  ExternalLink,
  Download,
  Calendar,
  HardDrive,
  Check,
  ShieldAlert,
  ArrowRight,
  Filter,
  Layers,
  FileText,
  FileCode,
  Image as ImageIcon,
  Presentation,
  Music,
  Video,
  Info,
  Clock,
  FolderOpen
} from "lucide-react";
import {
  getDuplicates,
  deleteDuplicateFile,
  resolveDuplicateGroup,
  resolveAllDuplicates,
  getDocumentDownloadUrl,
  getErrorMessage,
} from "@/lib/api";
import { DuplicateOverviewResponse, DuplicateGroup, DuplicateFileItem } from "@/types";

export default function DuplicatesPage() {
  const [data, setData] = useState<DuplicateOverviewResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<{ type: "success" | "error" | "info"; message: string } | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState<string>("all");

  // Action states
  const [deletingFileId, setDeletingFileId] = useState<string | null>(null);
  const [resolvingGroupId, setResolvingGroupId] = useState<string | null>(null);
  const [isResolvingAll, setIsResolvingAll] = useState(false);
  const [copiedPath, setCopiedPath] = useState<string | null>(null);

  // Modals state
  const [groupToResolve, setGroupToResolve] = useState<DuplicateGroup | null>(null);
  const [fileToDelete, setFileToDelete] = useState<{ group: DuplicateGroup; file: DuplicateFileItem } | null>(null);
  const [showResolveAllModal, setShowResolveAllModal] = useState(false);
  const [deleteFromDisk, setDeleteFromDisk] = useState(true);

  const loadData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const overview = await getDuplicates();
      setData(overview);
    } catch (err: unknown) {
      setError(getErrorMessage(err, "Failed to scan for duplicate files. Make sure the backend is running."));
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedPath(text);
    setTimeout(() => setCopiedPath(null), 2000);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateStr: string) => {
    try {
      const d = new Date(dateStr);
      return d.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return dateStr;
    }
  };

  const getFileIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case "pdf":
        return <FileText className="w-4 h-4 text-red-400" />;
      case "docx":
      case "doc":
        return <FileText className="w-4 h-4 text-blue-400" />;
      case "pptx":
      case "ppt":
        return <Presentation className="w-4 h-4 text-amber-400" />;
      case "png":
      case "jpg":
      case "jpeg":
      case "webp":
      case "gif":
      case "bmp":
      case "tiff":
      case "heic":
      case "heif":
        return <ImageIcon className="w-4 h-4 text-emerald-400" />;
      case "mp3":
      case "wav":
      case "m4a":
      case "flac":
        return <Music className="w-4 h-4 text-purple-400" />;
      case "mp4":
      case "mkv":
      case "avi":
      case "mov":
      case "webm":
        return <Video className="w-4 h-4 text-rose-400" />;
      default:
        return <FileCode className="w-4 h-4 text-zinc-400" />;
    }
  };

  const getMatchBadge = (type: string, label: string) => {
    switch (type) {
      case "exact_hash":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-emerald-950/40 border border-emerald-500/35 text-emerald-300">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            {label}
          </span>
        );
      case "exact_text":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-cyan-950/40 border border-cyan-500/35 text-cyan-300">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            {label}
          </span>
        );
      case "filename_and_content":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-indigo-950/40 border border-indigo-500/35 text-indigo-300">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
            {label}
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-purple-950/40 border border-purple-500/35 text-purple-300">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
            {label}
          </span>
        );
    }
  };

  // Filter groups
  const filteredGroups = useMemo(() => {
    if (!data?.groups) return [];
    return data.groups.filter((group) => {
      // Type filter
      if (selectedType !== "all" && group.match_type !== selectedType) {
        return false;
      }
      // Query filter
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesNameOrPath = group.files.some(
          (f) => f.filename.toLowerCase().includes(q) || f.filepath.toLowerCase().includes(q)
        );
        const matchesDiff = group.differences.toLowerCase().includes(q);
        const matchesWhy = group.why_duplicate.toLowerCase().includes(q);
        return matchesNameOrPath || matchesDiff || matchesWhy;
      }
      return true;
    });
  }, [data, selectedType, searchQuery]);

  // Execute Individual Delete
  const handleExecuteDeleteFile = async () => {
    if (!fileToDelete) return;
    const { file } = fileToDelete;
    setDeletingFileId(file.id);
    setFileToDelete(null);
    try {
      const res = await deleteDuplicateFile(file.id, deleteFromDisk);
      setFeedback({ type: "success", message: res.message });
      await loadData();
    } catch (err: unknown) {
      setFeedback({ type: "error", message: getErrorMessage(err, "Failed to delete file") });
    } finally {
      setDeletingFileId(null);
    }
  };

  // Execute Group Resolution (Keep Best & Delete Others)
  const handleExecuteResolveGroup = async () => {
    if (!groupToResolve) return;
    const group = groupToResolve;
    setResolvingGroupId(group.group_id);
    setGroupToResolve(null);
    try {
      const res = await resolveDuplicateGroup(group.recommended_document_id, undefined, deleteFromDisk);
      setFeedback({ type: "success", message: res.message });
      await loadData();
    } catch (err: unknown) {
      setFeedback({ type: "error", message: getErrorMessage(err, "Failed to resolve duplicate group") });
    } finally {
      setResolvingGroupId(null);
    }
  };

  // Execute Resolve All
  const handleExecuteResolveAll = async () => {
    setShowResolveAllModal(false);
    setIsResolvingAll(true);
    try {
      const res = await resolveAllDuplicates(deleteFromDisk);
      setFeedback({ type: "success", message: res.message });
      await loadData();
    } catch (err: unknown) {
      setFeedback({ type: "error", message: getErrorMessage(err, "Failed to resolve all duplicates") });
    } finally {
      setIsResolvingAll(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-10 space-y-8">
      {/* Page Title & Header */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.1] backdrop-blur-md text-xs sm:text-sm text-indigo-400 font-mono font-medium">
          <Copy className="w-4 h-4" />
          Deduplication & Storage Optimizer
        </div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-white">
              Duplicate Files Manager
            </h1>
            <p className="text-sm sm:text-base text-zinc-300 mt-1">
              Identify redundant files indexed in your system via content hashing, extracted text, and filename similarity.
            </p>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={loadData}
              disabled={isLoading}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.10] text-sm font-mono text-zinc-200 border border-white/[0.1] transition-all disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
              <span>Refresh</span>
            </button>

            {data && data.total_groups > 0 && (
              <button
                onClick={() => setShowResolveAllModal(true)}
                disabled={isLoading || isResolvingAll}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600/40 to-purple-600/40 hover:from-indigo-600/60 hover:to-purple-600/60 text-white text-sm font-medium border border-indigo-500/40 transition-all shadow-[0_0_20px_rgba(99,102,241,0.25)]"
              >
                <Sparkles className="w-4 h-4 text-indigo-300" />
                <span>Keep All Best & Clean</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* KPI Stats Grid */}
      {data && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="p-5 rounded-2xl bg-black/45 border border-white/[0.08] backdrop-blur-xl space-y-1 shadow-[0_8px_32px_rgba(0,0,0,0.5)]">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider">Duplicate Sets</span>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-white font-mono">{data.total_groups}</span>
              <span className="text-xs text-zinc-400">groups detected</span>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-black/45 border border-white/[0.08] backdrop-blur-xl space-y-1 shadow-[0_8px_32px_rgba(0,0,0,0.5)]">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider">Redundant Files</span>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-amber-300 font-mono">{data.total_duplicate_files}</span>
              <span className="text-xs text-zinc-400">files eligible to delete</span>
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-black/45 border border-white/[0.08] backdrop-blur-xl space-y-1 shadow-[0_8px_32px_rgba(0,0,0,0.5)]">
            <span className="text-xs font-mono text-zinc-400 uppercase tracking-wider">Reclaimable Storage</span>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-emerald-400 font-mono">
                {formatFileSize(data.total_reclaimable_bytes)}
              </span>
              <span className="text-xs text-zinc-400">recoverable disk space</span>
            </div>
          </div>
        </div>
      )}

      {/* Feedback Banner */}
      {feedback && (
        <div
          className={`p-4 rounded-xl border backdrop-blur-md text-sm font-mono flex items-center justify-between gap-3 ${
            feedback.type === "success"
              ? "bg-emerald-950/30 border-emerald-500/30 text-emerald-200"
              : feedback.type === "error"
              ? "bg-red-950/30 border-red-500/30 text-red-200"
              : "bg-indigo-950/30 border-indigo-500/30 text-indigo-200"
          }`}
        >
          <div className="flex items-center gap-2.5">
            {feedback.type === "success" ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
            )}
            <span>{feedback.message}</span>
          </div>
          <button onClick={() => setFeedback(null)} className="text-zinc-400 hover:text-white">
            ✕
          </button>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-5 rounded-2xl bg-red-950/20 border border-red-500/30 backdrop-blur-md text-sm text-red-200 font-mono flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Search & Match Type Filter Bar */}
      <div className="flex flex-col sm:flex-row items-center gap-3">
        <div className="relative flex-1 w-full">
          <Search className="w-4 h-4 text-zinc-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search duplicates by filename, directory path, or difference..."
            className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-black/60 border border-white/[0.1] text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-indigo-400 font-mono transition-all"
          />
        </div>

        <div className="flex items-center gap-1.5 w-full sm:w-auto overflow-x-auto pb-1 sm:pb-0">
          {[
            { id: "all", label: "All" },
            { id: "exact_hash", label: "Exact Hash" },
            { id: "exact_text", label: "Same Text" },
            { id: "filename_and_content", label: "Name Variants" },
            { id: "similar_content", label: "Near-Duplicates" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedType(tab.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono uppercase tracking-wider transition-all border whitespace-nowrap font-medium ${
                selectedType === tab.id
                  ? "bg-indigo-600/30 text-indigo-200 border-indigo-500/45 shadow-[0_0_10px_rgba(99,102,241,0.2)]"
                  : "bg-white/[0.02] text-zinc-300 hover:text-white border-white/[0.06] hover:border-white/[0.14]"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Groups List */}
      {isLoading ? (
        <div className="p-16 text-center text-sm text-zinc-400 font-mono flex flex-col items-center justify-center gap-3.5 rounded-2xl bg-black/40 border border-white/[0.06]">
          <RefreshCw className="w-6 h-6 animate-spin text-indigo-400" />
          <span>Scanning indexed files and computing content hashes...</span>
        </div>
      ) : filteredGroups.length > 0 ? (
        <div className="space-y-6">
          {filteredGroups.map((group) => {
            const recommended = group.files.find((f) => f.is_recommended);
            const duplicates = group.files.filter((f) => !f.is_recommended);

            return (
              <div
                key={group.group_id}
                className="p-6 rounded-2xl bg-black/50 border border-white/[0.08] hover:border-white/[0.16] backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.6)] space-y-5 transition-all"
              >
                {/* Group Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/[0.08] pb-4">
                  <div className="flex items-center gap-2.5 flex-wrap">
                    {getMatchBadge(group.match_type, group.match_label)}
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-white/[0.06] text-zinc-300 border border-white/[0.1]">
                      {group.files.length} copies
                    </span>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-950/40 text-emerald-300 border border-emerald-500/30">
                      Save {formatFileSize(group.potential_savings_bytes)}
                    </span>
                  </div>

                  <button
                    onClick={() => setGroupToResolve(group)}
                    disabled={resolvingGroupId === group.group_id}
                    className="self-start sm:self-auto flex items-center gap-2 px-3.5 py-1.5 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 hover:text-white border border-indigo-500/40 text-xs sm:text-sm font-medium transition-all shadow-[0_0_12px_rgba(99,102,241,0.2)] disabled:opacity-50"
                  >
                    <CheckCircle2 className="w-4 h-4 text-indigo-300" />
                    <span>Keep Best & Delete Others</span>
                  </button>
                </div>

                {/* Context & Differences Explanation Box */}
                <div className="p-4 rounded-xl bg-white/[0.02] border border-white/[0.06] space-y-2 text-xs sm:text-sm">
                  <div className="flex items-start gap-2">
                    <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
                    <div className="space-y-1">
                      <p className="text-zinc-200">
                        <strong className="text-indigo-300 font-mono">Why Duplicate: </strong>
                        {group.why_duplicate}
                      </p>
                      <p className="text-zinc-400">
                        <strong className="text-zinc-300 font-mono">Differences: </strong>
                        {group.differences}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Files Grid in this Group */}
                <div className="space-y-3">
                  <span className="text-xs font-mono uppercase tracking-wider text-zinc-400 font-medium">
                    Files in this group:
                  </span>

                  <div className="grid grid-cols-1 gap-3">
                    {group.files.map((file) => (
                      <div
                        key={file.id}
                        className={`p-4 rounded-xl transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3.5 ${
                          file.is_recommended
                            ? "bg-emerald-950/15 border-2 border-emerald-500/40 shadow-[0_0_20px_rgba(16,185,129,0.12)]"
                            : "bg-black/40 border border-white/[0.06] hover:border-white/[0.12]"
                        }`}
                      >
                        <div className="flex items-start gap-3 min-w-0">
                          <div className="p-2.5 rounded-lg bg-black/60 border border-white/[0.08] shrink-0 mt-0.5 sm:mt-0">
                            {getFileIcon(file.file_type)}
                          </div>

                          <div className="flex flex-col min-w-0 space-y-1">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-semibold text-sm sm:text-base text-zinc-100 truncate">
                                {file.filename}
                              </span>

                              {file.is_recommended ? (
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 animate-pulse">
                                  <Sparkles className="w-3 h-3 text-emerald-400" />
                                  RECOMMENDED TO KEEP (Latest Updated)
                                </span>
                              ) : (
                                <span className="text-xs font-mono px-2 py-0.5 rounded bg-red-950/40 text-red-300 border border-red-800/35">
                                  Redundant Duplicate
                                </span>
                              )}
                            </div>

                            <p className="text-xs text-zinc-400 font-mono">
                              {file.recommendation_reason}
                            </p>

                            <div className="flex items-center gap-3 text-xs text-zinc-400 font-mono flex-wrap pt-0.5">
                              <span className="flex items-center gap-1.5 text-zinc-300">
                                <Calendar className="w-3.5 h-3.5 text-zinc-400" />
                                {formatDate(file.modified_at)}
                              </span>
                              <span className="text-zinc-600">•</span>
                              <span className="text-zinc-300">{formatFileSize(file.file_size)}</span>
                              <span className="text-zinc-600">•</span>
                              <span
                                className="truncate max-w-[280px] sm:max-w-[360px] text-zinc-400 cursor-pointer hover:text-indigo-300 transition-colors"
                                title={file.filepath}
                                onClick={() => copyToClipboard(file.filepath)}
                              >
                                {file.filepath}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* File Action Buttons */}
                        <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                          <button
                            onClick={() => copyToClipboard(file.filepath)}
                            className="p-2.5 rounded-lg bg-white/[0.05] hover:bg-white/[0.10] text-zinc-300 hover:text-white border border-white/[0.1] transition-all"
                            title={copiedPath === file.filepath ? "Path copied!" : "Copy file path"}
                          >
                            {copiedPath === file.filepath ? (
                              <Check className="w-4 h-4 text-emerald-400" />
                            ) : (
                              <Copy className="w-4 h-4" />
                            )}
                          </button>

                          <a
                            href={getDocumentDownloadUrl(file.id)}
                            target="_blank"
                            rel="noreferrer"
                            className="p-2.5 rounded-lg bg-white/[0.05] hover:bg-white/[0.10] text-zinc-300 hover:text-white border border-white/[0.1] transition-all"
                            title="Open original file"
                          >
                            <Download className="w-4 h-4" />
                          </a>

                          <button
                            onClick={() => setFileToDelete({ group, file })}
                            disabled={deletingFileId === file.id}
                            className={`p-2.5 rounded-lg transition-all ${
                              file.is_recommended
                                ? "bg-red-950/20 hover:bg-red-950/40 text-red-400/80 hover:text-red-300 border border-red-800/30"
                                : "bg-red-950/30 hover:bg-red-950/50 text-red-300 hover:text-red-200 border border-red-800/40 shadow-[0_0_10px_rgba(239,68,68,0.15)]"
                            }`}
                            title={file.is_recommended ? "Delete recommended file" : "Delete this duplicate"}
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="p-16 rounded-2xl bg-black/40 border border-white/[0.06] text-center space-y-3">
          <div className="w-12 h-12 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 mx-auto flex items-center justify-center">
            <CheckCircle2 className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-semibold text-white">No Duplicates Found</h3>
          <p className="text-sm text-zinc-400 max-w-md mx-auto">
            {searchQuery
              ? "No duplicate files match your search criteria. Try clearing the filter."
              : "All indexed files in your digital memory are unique. Your storage is tidy and clean!"}
          </p>
          <div className="pt-2">
            <Link
              href="/indexing"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.10] text-sm font-mono text-zinc-200 border border-white/[0.1] transition-all"
            >
              <HardDrive className="w-4 h-4 text-indigo-400" />
              <span>Go to Indexing</span>
            </Link>
          </div>
        </div>
      )}

      {/* Confirmation Modal: Keep Best & Delete Others for Group */}
      {groupToResolve && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
          <div className="max-w-lg w-full p-6 rounded-2xl bg-zinc-950 border border-white/[0.14] shadow-[0_16px_48px_rgba(0,0,0,0.8)] space-y-5">
            <div className="flex items-center gap-3 text-indigo-300 font-semibold text-lg">
              <Sparkles className="w-5 h-5 text-indigo-400" />
              <span>Keep Best & Delete Others</span>
            </div>

            <p className="text-sm text-zinc-300">
              ReFind will retain the latest updated file and delete all redundant copies in this group:
            </p>

            <div className="space-y-3 text-xs font-mono">
              <div className="p-3.5 rounded-xl bg-emerald-950/20 border border-emerald-500/30 text-emerald-200 space-y-1">
                <strong className="text-emerald-400 block uppercase">File to Keep:</strong>
                <span className="font-semibold">{groupToResolve.files.find((f) => f.is_recommended)?.filename}</span>
                <span className="block text-zinc-400 truncate">{groupToResolve.files.find((f) => f.is_recommended)?.filepath}</span>
              </div>

              <div className="p-3.5 rounded-xl bg-red-950/20 border border-red-500/30 text-red-200 space-y-1">
                <strong className="text-red-400 block uppercase">
                  {groupToResolve.files.filter((f) => !f.is_recommended).length} Duplicate(s) to Delete:
                </strong>
                <ul className="list-disc list-inside text-zinc-300 space-y-1 max-h-36 overflow-y-auto">
                  {groupToResolve.files
                    .filter((f) => !f.is_recommended)
                    .map((f) => (
                      <li key={f.id} className="truncate" title={f.filepath}>
                        {f.filename} ({formatFileSize(f.file_size)})
                      </li>
                    ))}
                </ul>
              </div>
            </div>

            {/* Checkbox: delete from disk */}
            <label className="flex items-center gap-2.5 cursor-pointer text-xs sm:text-sm text-zinc-300 pt-1">
              <input
                type="checkbox"
                checked={deleteFromDisk}
                onChange={(e) => setDeleteFromDisk(e.target.checked)}
                className="w-4 h-4 rounded bg-black border-white/20 text-indigo-600 focus:ring-0"
              />
              <span>Also delete physical duplicate files from system hard drive</span>
            </label>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={() => setGroupToResolve(null)}
                className="px-4 py-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.10] text-sm text-zinc-300 border border-white/[0.1] transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleExecuteResolveGroup}
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-all shadow-[0_0_15px_rgba(99,102,241,0.3)]"
              >
                Confirm Keep Best
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal: Delete Individual File */}
      {fileToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
          <div className="max-w-md w-full p-6 rounded-2xl bg-zinc-950 border border-white/[0.14] shadow-[0_16px_48px_rgba(0,0,0,0.8)] space-y-4">
            <div className="flex items-center gap-3 text-red-400 font-semibold text-lg">
              <Trash2 className="w-5 h-5 text-red-400" />
              <span>Delete Duplicate File</span>
            </div>

            <p className="text-sm text-zinc-300">
              Are you sure you want to delete <strong className="text-white">&ldquo;{fileToDelete.file.filename}&rdquo;</strong>?
            </p>

            <div className="p-3 rounded-xl bg-black/60 border border-white/[0.08] text-xs font-mono text-zinc-400 truncate">
              {fileToDelete.file.filepath}
            </div>

            {/* Checkbox: delete from disk */}
            <label className="flex items-center gap-2.5 cursor-pointer text-xs sm:text-sm text-zinc-300 pt-1">
              <input
                type="checkbox"
                checked={deleteFromDisk}
                onChange={(e) => setDeleteFromDisk(e.target.checked)}
                className="w-4 h-4 rounded bg-black border-white/20 text-indigo-600 focus:ring-0"
              />
              <span>Delete original file from system disk</span>
            </label>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={() => setFileToDelete(null)}
                className="px-4 py-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.10] text-sm text-zinc-300 border border-white/[0.1] transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleExecuteDeleteFile}
                className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-500 text-white text-sm font-medium transition-all shadow-[0_0_15px_rgba(239,68,68,0.3)]"
              >
                Delete File
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal: Resolve All Duplicates */}
      {showResolveAllModal && data && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
          <div className="max-w-lg w-full p-6 rounded-2xl bg-zinc-950 border border-white/[0.14] shadow-[0_16px_48px_rgba(0,0,0,0.8)] space-y-5">
            <div className="flex items-center gap-3 text-indigo-300 font-semibold text-lg">
              <ShieldAlert className="w-5 h-5 text-indigo-400" />
              <span>Clean All Duplicate Files</span>
            </div>

            <p className="text-sm text-zinc-300">
              This action will retain the recommended (latest updated) file in all{" "}
              <strong className="text-white font-mono">{data.total_groups}</strong> duplicate groups and remove all{" "}
              <strong className="text-amber-300 font-mono">{data.total_duplicate_files}</strong> redundant copies,
              recovering <strong className="text-emerald-400 font-mono">{formatFileSize(data.total_reclaimable_bytes)}</strong>.
            </p>

            {/* Checkbox: delete from disk */}
            <label className="flex items-center gap-2.5 cursor-pointer text-xs sm:text-sm text-zinc-300 pt-1">
              <input
                type="checkbox"
                checked={deleteFromDisk}
                onChange={(e) => setDeleteFromDisk(e.target.checked)}
                className="w-4 h-4 rounded bg-black border-white/20 text-indigo-600 focus:ring-0"
              />
              <span>Also delete physical redundant files from disk drive</span>
            </label>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                onClick={() => setShowResolveAllModal(false)}
                className="px-4 py-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.10] text-sm text-zinc-300 border border-white/[0.1] transition-all"
              >
                Cancel
              </button>
              <button
                onClick={handleExecuteResolveAll}
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium transition-all shadow-[0_0_15px_rgba(99,102,241,0.3)]"
              >
                Proceed with Cleaning
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
