"use client";

import Link from "next/link";
import { useEffect, useState, useRef, useCallback, ChangeEvent, DragEvent, useTransition } from "react";
import {
  FolderGit2,
  UploadCloud,
  RefreshCw,
  FolderOpen,
  CheckCircle2,
  HardDrive,
  FileText,
  FileCode,
  Image as ImageIcon,
  Presentation,
  Download,
  Trash2,
  Search,
  ExternalLink,
  Layers,
  Calendar,
  Clock,
  Filter,
  AlertTriangle,
  Music,
  Video,
  Copy
} from "lucide-react";
import IndexProgress from "@/components/IndexProgress";
import MediaIndexingQueue from "@/components/MediaIndexingQueue";
import CatMascot from "@/components/CatMascot";
import {
  getIndexStatus,
  triggerIndexing,
  uploadFiles,
  getIndexedDocuments,
  deleteDocument,
  deleteAllDocuments,
  getDocumentDownloadUrl,
  getErrorMessage,
} from "@/lib/api";
import { IndexProgressStatus, IndexedDocumentItem } from "@/types";

const SUPPORTED_FORMATS = ["PDF", "DOCX", "PPTX", "TXT", "PNG", "JPG", "JPEG", "WEBP", "GIF", "BMP", "TIFF", "HEIC", "HEIF", "MP4", "MKV", "AVI", "MOV", "WEBM", "MP3", "WAV", "M4A", "FLAC", "AAC", "OGG"];

export default function IndexingPage() {
  const [status, setStatus] = useState<IndexProgressStatus>({
    is_indexing: false,
    current_stage: "idle",
    total_files: 0,
    processed_files: 0,
    documents_count: 0,
    images_count: 0,
    other_count: 0,
  });

  const [folderPathInput, setFolderPathInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);
  const [feedbackIsError, setFeedbackIsError] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Indexed files viewer state
  const [documents, setDocuments] = useState<IndexedDocumentItem[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [docSearchQuery, setDocSearchQuery] = useState("");
  const [selectedFormat, setSelectedFormat] = useState("all");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [isClearingIndex, setIsClearingIndex] = useState(false);
  const [backendConnected, setBackendConnected] = useState(true);
  const documentRequest = useRef(0);

  // Fetch indexed documents list
  const loadDocuments = useCallback(async () => {
    const request = ++documentRequest.current;
    setIsLoadingDocs(true);
    try {
      const docs = await getIndexedDocuments(docSearchQuery, selectedFormat);
      if (request !== documentRequest.current) return;
      setDocuments(docs);
      setBackendConnected(true);
    } catch (err) {
      if (request === documentRequest.current) setBackendConnected(false);
    } finally {
      if (request === documentRequest.current) setIsLoadingDocs(false);
    }
  }, [docSearchQuery, selectedFormat]);

  const refreshStatus = async () => {
    try {
      const updated = await getIndexStatus();
      setStatus(updated);
      setBackendConnected(true);
    } catch {
      setBackendConnected(false);
    }
  };

  const isBusy = status.is_indexing || isUploading;

  // Poll lightweight progress; fast polling during active indexing/upload, relaxed when idle.
  useEffect(() => {
    let active = true;
    let inFlight = false;
    let previous: IndexProgressStatus | null = null;
    const fetchStatus = async () => {
      if (inFlight || document.hidden) return;
      inFlight = true;
      try {
        const data = await getIndexStatus();
        if (!active) return;
        setStatus(data);
        setBackendConnected(true);
        if (previous && !data.is_indexing && (
          previous.is_indexing || previous.total_files !== data.total_files ||
          previous.last_indexed_at !== data.last_indexed_at
        )) {
          void loadDocuments();
        }
        previous = data;
      } catch {
        if (active) setBackendConnected(false);
      } finally {
        inFlight = false;
      }
    };
    void fetchStatus();
    const pollInterval = isBusy ? 400 : 3000;
    const interval = setInterval(fetchStatus, pollInterval);
    document.addEventListener("visibilitychange", fetchStatus);
    return () => {
      active = false;
      clearInterval(interval);
      document.removeEventListener("visibilitychange", fetchStatus);
    };
  }, [isBusy, loadDocuments]);

  // One initial request, and debounce subsequent filter edits.
  useEffect(() => {
    const timer = setTimeout(() => { void loadDocuments(); }, 250);
    return () => clearTimeout(timer);
  }, [loadDocuments]);

  const handleIndexFolder = async () => {
    const selectedPath = folderPathInput.trim();
    if (!selectedPath) {
      setFeedbackMessage("Enter the full path of the folder you want to index, or select files below.");
      setFeedbackIsError(true);
      return;
    }
    if (isBusy) return;
    setFeedbackIsError(false);
    try {
      setFeedbackMessage("Starting directory indexing...");
      setStatus((prev) => ({
        ...prev,
        is_indexing: true,
        current_stage: "Reading",
        current_file: "Scanning local directory...",
        processed_files: 0,
        error_message: null,
      }));
      const res = await triggerIndexing(selectedPath);
      setFeedbackMessage(res.message);
      await refreshStatus();
      loadDocuments();
    } catch (err: any) {
      setFeedbackMessage(err.message || "Failed to start indexing");
      setFeedbackIsError(true);
      await refreshStatus();
    }
  };

  const handleFileUpload = async (files: FileList | null) => {
    if (isBusy || !files || files.length === 0) return;
    setIsUploading(true);
    setFeedbackIsError(false);
    setFeedbackMessage(`Uploading and indexing ${files.length} file(s)...`);

    const fileNames = Array.from(files).map((f) => f.name);
    const initialFileName = fileNames[0] || "file";

    // Immediately trigger the visual loading screen with Stage 01 Reading
    setStatus((prev) => ({
      ...prev,
      is_indexing: true,
      current_stage: "Reading",
      current_file: initialFileName,
      processed_files: 0,
      total_files: files.length,
      error_message: null,
    }));

    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
      }
      const res = await uploadFiles(formData);

      const failed = res.failed_files ?? [];
      setFeedbackIsError(failed.length > 0);
      const failureSummary = failed.slice(0, 3)
        .map((item) => `${item.filename}: ${item.error}`)
        .join(" | ");
      const remainingFailures = failed.length > 3 ? ` | ${failed.length - 3} more failed` : "";
      setFeedbackMessage([res.message, failureSummary + remainingFailures].filter(Boolean).join(" "));

      // Refresh status from database and list
      await refreshStatus();
      await loadDocuments();
    } catch (err: unknown) {
      setStatus((prev) => ({
        ...prev,
        is_indexing: false,
        current_stage: "Error",
        error_message: getErrorMessage(err, "Failed to upload files"),
      }));
      setFeedbackMessage(getErrorMessage(err, "Failed to upload files"));
      setFeedbackIsError(true);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDeleteDoc = async (id: string, name: string) => {
    if (!confirm(`Remove "${name}" from indexed memory and delete its saved ReFind upload copy, if any?\n\nSource files outside ReFind upload storage will be preserved.`)) return;
    setDeletingId(id);
    setFeedbackIsError(false);
    try {
      const res = await deleteDocument(id);
      setFeedbackMessage(res.message);
      await refreshStatus();
      loadDocuments();
    } catch (err: any) {
      setFeedbackMessage(err.message || "Failed to delete document");
      setFeedbackIsError(true);
    } finally {
      setDeletingId(null);
    }
  };

  const handleClearIndex = async () => {
    setFeedbackIsError(false);
    if (!confirm(
      "Clear the entire index, delete all saved ReFind upload copies, and cancel queued indexing jobs?\n\n" +
      "This includes previous and failed uploads. Source files outside ReFind upload storage will be preserved."
    )) return;

    setIsClearingIndex(true);
    try {
      const res = await deleteAllDocuments();
      setDocuments([]);
      setFeedbackMessage(res.message);
      await refreshStatus();
      await loadDocuments();
    } catch (err: any) {
      setFeedbackMessage(err.message || "Failed to clear indexed files");
      setFeedbackIsError(true);
    } finally {
      setIsClearingIndex(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files);
    }
  };

  const getFileIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case "pdf":
        return <FileText className="w-4 h-4 text-red-400/90" />;
      case "docx":
      case "doc":
        return <FileText className="w-4 h-4 text-blue-400/90" />;
      case "pptx":
      case "ppt":
        return <Presentation className="w-4 h-4 text-amber-400/90" />;
      case "png":
      case "jpg":
      case "jpeg":
      case "webp":
      case "gif":
      case "bmp":
      case "tiff":
      case "heic":
      case "heif":
        return <ImageIcon className="w-4 h-4 text-emerald-400/90" />;
      case "mp3":
      case "wav":
      case "m4a":
      case "flac":
      case "aac":
      case "ogg":
      case "wma":
      case "opus":
      case "audio":
        return <Music className="w-4 h-4 text-purple-400/90" />;
      case "mp4":
      case "mkv":
      case "avi":
      case "mov":
      case "wmv":
      case "webm":
      case "flv":
      case "m4v":
      case "video":
        return <Video className="w-4 h-4 text-rose-400/90" />;
      default:
        return <FileCode className="w-4 h-4 text-zinc-400" />;
    }
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
      });
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10 space-y-8">
      {/* Title Header */}
      <div className="page-heading" data-grav-mass="1.2">
        <div className="page-heading__copy space-y-2.5">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.1] backdrop-blur-md text-xs sm:text-sm text-indigo-400 font-mono font-medium">
            <HardDrive className="w-4 h-4" />
            Local Files Connector
          </div>
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-white">
            Digital Memory Indexing
          </h1>
          <p className="text-base text-zinc-300">
            Index local files into the high-dimensional vector space and relational metadata store.
          </p>
        </div>
        <CatMascot className="page-heading__cat hidden md:block" />
      </div>

      {/* Main Live Progress Indicator */}
      <IndexProgress status={status} />
      <MediaIndexingQueue />

      {/* Index Local Directory */}
      <div className="grid grid-cols-1 gap-6">
        <div
          data-grav-mass="1.1"
          className="p-6 rounded-2xl bg-black/45 border border-white/[0.08] backdrop-blur-xl flex flex-col justify-between gap-5 hover:border-white/[0.16] transition-all shadow-[0_8px_32px_rgba(0,0,0,0.6)]"
        >
          <div className="space-y-3.5">
            <div className="flex items-center gap-2.5 text-zinc-100 font-semibold text-base">
              <FolderOpen className="w-4 h-4 text-indigo-400" />
              Index Local Directory
            </div>
            <p className="text-sm text-zinc-300 leading-relaxed">
              Index files in the folder you specify, including its subfolders. Choose individual files below if you only want to index selected files.
            </p>

            <div className="space-y-2 pt-2">
              <label htmlFor="index-folder-path" className="text-xs font-mono text-zinc-300 uppercase tracking-wider font-medium">
                Directory Path (required):
              </label>
              <input
                id="index-folder-path"
                type="text"
                required
                value={folderPathInput}
                onChange={(e) => setFolderPathInput(e.target.value)}
                placeholder="Full path of the folder you want to index"
                className="w-full px-4 py-2.5 rounded-xl bg-black/60 border border-white/[0.1] text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-indigo-400 font-mono transition-all"
              />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleIndexFolder}
              disabled={isBusy || !folderPathInput.trim()}
              className="flex-1 flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/45 disabled:opacity-50 text-white text-sm font-medium border border-indigo-500/40 transition-all shadow-[0_0_15px_-3px_rgba(244,214,78,0.18)]"
            >
              <FolderGit2 className="w-4 h-4" />
              <span>{status.is_indexing ? "Indexing..." : "Scan & Index Folder"}</span>
            </button>
          </div>
        </div>

      </div>

      {/* Drag & Drop File Ingestion Zone */}
      <div
        data-grav-mass="1.0"
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => { if (!isBusy) fileInputRef.current?.click(); }}
        className={`p-8 sm:p-10 rounded-2xl border-2 border-dashed cursor-pointer transition-all flex flex-col items-center justify-center gap-3.5 text-center backdrop-blur-md ${
          dragActive
            ? "border-indigo-400/80 bg-indigo-950/20 shadow-[0_0_30px_rgba(244,214,78,0.14)]"
            : "border-white/[0.12] hover:border-white/[0.24] bg-black/40 hover:bg-black/60"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          disabled={isBusy}
          className="hidden"
          onChange={(e: ChangeEvent<HTMLInputElement>) => handleFileUpload(e.target.files)}
        />
        <div className="w-14 h-14 rounded-2xl bg-white/[0.05] border border-white/[0.1] flex items-center justify-center text-indigo-400 shadow-[0_0_15px_-3px_rgba(244,214,78,0.14)]">
          <UploadCloud className="w-7 h-7" />
        </div>
        <div className="space-y-1">
          <p className="text-base sm:text-lg font-semibold text-zinc-100">
            {isUploading
              ? `Indexing in progress: ${status.current_stage || "Processing"}...`
              : "Drag and drop local files here"}
          </p>
          <p className="text-sm text-zinc-400">
            {isUploading
              ? (status.current_file ? `Processing ${status.current_file}` : "Analyzing digital memory...")
              : "or click to browse from your filesystem"}
          </p>
        </div>

        <p className="max-w-xl text-sm text-zinc-400">
          Include matching subtitles or transcripts for faster media indexing, such as movie.mkv and movie.srt. SRT, VTT, ASS, SSA, SUB, LRC, and TXT companions are supported.
        </p>

        {/* Supported Formats Pills */}
        <div className="flex items-center gap-2 flex-wrap justify-center pt-2">
          <span className="text-xs text-zinc-400 uppercase tracking-wider font-mono mr-1 font-medium">
            Supported:
          </span>
          {SUPPORTED_FORMATS.map((fmt) => (
            <span
              key={fmt}
              className="px-2.5 py-1 rounded-md bg-white/[0.04] border border-white/[0.08] text-xs font-mono text-zinc-300"
            >
              {fmt}
            </span>
          ))}
        </div>
      </div>

      {feedbackMessage && (
        <div className="p-4 rounded-xl bg-black/60 border border-white/[0.1] backdrop-blur-md text-sm text-zinc-200 font-mono flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            {feedbackIsError
              ? <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              : <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />}
            <span>{feedbackMessage}</span>
          </div>
          <button
            onClick={() => setFeedbackMessage(null)}
            className="text-zinc-400 hover:text-white text-sm"
          >
            ✕
          </button>
        </div>
      )}

      {!backendConnected && (
        <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/20 backdrop-blur-md text-sm text-amber-200 font-mono flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
          <span>Backend server at <code className="text-white font-semibold">http://localhost:8000</code> is currently unreachable. Make sure the FastAPI backend is running.</span>
        </div>
      )}

      {/* NEW: Indexed Documents Explorer Panel */}
      <div
        data-grav-mass="1.3"
        className="p-6 rounded-2xl bg-black/50 border border-white/[0.08] backdrop-blur-xl shadow-[0_8px_32px_rgba(0,0,0,0.7)] space-y-5"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/[0.08] pb-4">
          <div>
            <div className="flex items-center gap-2.5">
              <h2 className="text-lg font-semibold text-white">Indexed Memory Store</h2>
              <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-white/[0.06] text-indigo-300 border border-white/[0.1] font-medium">
                {documents.length} files
              </span>
            </div>
            <p className="text-sm text-zinc-300 mt-1">
              Browse, search, inspect chunks & entities, or download indexed files.
            </p>
          </div>

          <div className="self-start sm:self-auto flex items-center gap-2">
            <Link
              href="/duplicates"
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-indigo-600/25 hover:bg-indigo-600/40 text-sm font-mono text-indigo-200 border border-indigo-500/35 transition-all shadow-[0_0_12px_rgba(99,102,241,0.2)]"
              title="Detect and manage duplicate files"
            >
              <Copy className="w-4 h-4 text-indigo-300" />
              <span>Duplicates</span>
            </Link>
            <button
              onClick={loadDocuments}
              disabled={isLoadingDocs || isClearingIndex}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white/[0.05] hover:bg-white/[0.10] text-sm font-mono text-zinc-200 border border-white/[0.1] transition-all disabled:opacity-50"
              title="Refresh files list"
            >
              <RefreshCw className={`w-4 h-4 ${isLoadingDocs ? "animate-spin" : ""}`} />
              <span>Refresh</span>
            </button>
            <button
              onClick={handleClearIndex}
              disabled={isBusy || isLoadingDocs || isClearingIndex}
              className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-red-950/30 hover:bg-red-950/50 text-sm font-mono text-red-300 border border-red-800/40 transition-all disabled:opacity-50"
              title="Clear all indexed data, saved upload copies, and queued jobs"
            >
              <Trash2 className={`w-4 h-4 ${isClearingIndex ? "animate-pulse" : ""}`} />
              <span>{isClearingIndex ? "Clearing..." : "Clear Index"}</span>
            </button>
          </div>
        </div>

        {/* Filter & Search Bar for Files */}
        <div className="flex flex-col sm:flex-row items-center gap-3">
          <div className="relative flex-1 w-full">
            <Search className="w-4 h-4 text-zinc-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={docSearchQuery}
              onChange={(e) => setDocSearchQuery(e.target.value)}
              placeholder="Filter by filename..."
              className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-black/60 border border-white/[0.1] text-sm text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-indigo-400 font-mono transition-all"
            />
          </div>

          <div className="flex items-center gap-1.5 w-full sm:w-auto overflow-x-auto pb-1 sm:pb-0">
            {["all", "pdf", "docx", "pptx", "png", "jpg", "video", "audio", "txt"].map((fmt) => (
              <button
                key={fmt}
                onClick={() => setSelectedFormat(fmt)}
                className={`px-3 py-1.5 rounded-lg text-xs font-mono uppercase tracking-wider transition-all border font-medium ${
                  selectedFormat === fmt
                    ? "bg-indigo-600/30 text-indigo-200 border-indigo-500/45 shadow-[0_0_10px_rgba(244,214,78,0.14)]"
                    : "bg-white/[0.02] text-zinc-300 hover:text-white border-white/[0.06] hover:border-white/[0.14]"
                }`}
              >
                {fmt}
              </button>
            ))}
          </div>
        </div>

        {/* Files Grid / List */}
        {isLoadingDocs && documents.length === 0 ? (
          <div className="p-8 text-center text-sm text-zinc-400 font-mono flex items-center justify-center gap-2.5">
            <RefreshCw className="w-4 h-4 animate-spin text-indigo-400" />
            <span>Loading indexed files...</span>
          </div>
        ) : documents.length > 0 ? (
          <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="p-4 rounded-xl bg-white/[0.02] hover:bg-white/[0.05] border border-white/[0.06] hover:border-white/[0.14] transition-all flex flex-col sm:flex-row sm:items-center justify-between gap-3.5"
              >
                <div className="flex items-start gap-3.5 min-w-0">
                  <div className="p-2.5 rounded-lg bg-black/40 border border-white/[0.08] shrink-0 mt-0.5 sm:mt-0">
                    {getFileIcon(doc.file_type)}
                  </div>

                  <div className="flex flex-col min-w-0 space-y-1.5">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-sm sm:text-base text-zinc-100 truncate">
                        {doc.filename}
                      </span>
                      <span className="text-xs uppercase font-mono px-2.5 py-0.5 rounded bg-white/[0.06] text-zinc-200 border border-white/[0.1] font-medium">
                        {doc.file_type}
                      </span>
                      <span className="text-xs font-mono text-indigo-300 bg-indigo-950/40 border border-indigo-500/30 px-2.5 py-0.5 rounded font-medium">
                        {doc.chunk_count} chunk{doc.chunk_count !== 1 ? "s" : ""}
                      </span>
                    </div>

                    <div className="flex items-center gap-3 text-xs sm:text-sm text-zinc-300 font-mono flex-wrap">
                      <span className="flex items-center gap-1.5">
                        <Calendar className="w-3.5 h-3.5 text-zinc-400" />
                        {formatDate(doc.modified_at)}
                      </span>
                      <span className="text-zinc-600">•</span>
                      <span>{formatFileSize(doc.file_size)}</span>
                      <span className="text-zinc-600">•</span>
                      <span className="truncate max-w-[240px] text-zinc-400" title={doc.filepath}>
                        {doc.filepath}
                      </span>
                    </div>

                    {/* Entities if any */}
                    {doc.entities && doc.entities.length > 0 && (
                      <div className="flex items-center gap-1.5 flex-wrap pt-0.5">
                        <span className="text-xs text-zinc-400 font-mono font-medium">Entities:</span>
                        {doc.entities.slice(0, 4).map((ent, i) => (
                          <span
                            key={i}
                            className="text-xs font-mono px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.08] text-zinc-200"
                          >
                            {ent.entity_value}
                          </span>
                        ))}
                        {doc.entities.length > 4 && (
                          <span className="text-xs font-mono text-zinc-400">
                            +{doc.entities.length - 4} more
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-2 shrink-0 self-end sm:self-center">
                  <a
                    href={getDocumentDownloadUrl(doc.id)}
                    target="_blank"
                    rel="noreferrer"
                    className="p-2.5 rounded-lg bg-white/[0.05] hover:bg-white/[0.10] text-zinc-200 hover:text-white border border-white/[0.1] transition-all"
                    title="Open original file"
                  >
                    <Download className="w-4 h-4" />
                  </a>

                  <button
                    onClick={() => handleDeleteDoc(doc.id, doc.filename)}
                    disabled={isBusy || deletingId === doc.id}
                    className="p-2.5 rounded-lg bg-red-950/25 hover:bg-red-950/45 text-red-300 hover:text-red-200 border border-red-800/35 transition-all disabled:opacity-50"
                    title="Remove from index and delete its saved upload copy"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 rounded-xl bg-black/40 border border-white/[0.06] text-center space-y-2">
            <p className="text-sm text-zinc-300">No indexed documents match your filter.</p>
            <p className="text-xs text-zinc-400 font-mono">
              Upload files above or scan a local folder to add them to your index.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
