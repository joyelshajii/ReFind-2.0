"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ArrowDown, ArrowUp, CornerDownLeft, FileText, Image as ImageIcon, LoaderCircle, Presentation, Search, X } from "lucide-react";
import { getErrorMessage, openDocument, searchDocuments } from "@/lib/api";
import type { SearchResultItem } from "@/types";

const MAX_RESULTS = 8;

function FileIcon({ type }: { type: string }) {
  const name = type.toLowerCase();
  if (["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic", "heif"].includes(name)) {
    return <ImageIcon className="h-4 w-4 text-emerald-300" aria-hidden="true" />;
  }
  if (["ppt", "pptx"].includes(name)) {
    return <Presentation className="h-4 w-4 text-amber-300" aria-hidden="true" />;
  }
  return <FileText className="h-4 w-4 text-indigo-300" aria-hidden="true" />;
}

export default function QuickSearchPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [openingId, setOpeningId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const searchRequest = useRef(0);

  const show = useCallback(() => {
    previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setOpen(true);
  }, []);

  const hide = useCallback(() => {
    searchRequest.current += 1;
    setOpen(false);
    setQuery("");
    setResults([]);
    setLoading(false);
    setOpeningId(null);
    setError(null);
    previousFocus.current?.focus();
  }, []);

  const openResult = useCallback(async (result: SearchResultItem) => {
    if (openingId) return;
    setOpeningId(result.document_id);
    setError(null);
    try {
      await openDocument(result.document_id);
      hide();
    } catch (cause) {
      setError(getErrorMessage(cause, "Could not open this file."));
      setOpeningId(null);
    }
  }, [hide, openingId]);

  useEffect(() => {
    const onOpen = () => show();
    window.addEventListener("refind:open-quick-search", onOpen);
    return () => window.removeEventListener("refind:open-quick-search", onOpen);
  }, [show]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const shortcut = (event.altKey && !event.ctrlKey && !event.metaKey && event.code === "Space")
        || ((event.ctrlKey || event.metaKey) && !event.altKey && event.key.toLowerCase() === "k");
      if (shortcut) {
        event.preventDefault();
        event.stopPropagation();
        if (open) hide();
        else show();
        return;
      }
      if (!open) return;
      if (event.key === "Escape") {
        event.preventDefault();
        hide();
      } else if (event.key === "ArrowDown" && results.length) {
        event.preventDefault();
        setSelected((current) => (current + 1) % results.length);
      } else if (event.key === "ArrowUp" && results.length) {
        event.preventDefault();
        setSelected((current) => (current - 1 + results.length) % results.length);
      } else if (event.key === "Enter" && results.length && !openingId) {
        event.preventDefault();
        void openResult(results[selected]);
      } else if (event.key === "Tab") {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [hide, open, openResult, openingId, results, selected, show]);

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    const request = ++searchRequest.current;
    const trimmed = query.trim();
    if (!open || !trimmed) {
      setResults([]);
      setSelected(0);
      setLoading(false);
      setError(null);
      return;
    }

    setResults([]);
    setSelected(0);
    setLoading(true);
    setError(null);
    const timer = window.setTimeout(async () => {
      try {
        const response = await searchDocuments(trimmed, MAX_RESULTS);
        if (request !== searchRequest.current) return;
        setResults(response.results);
        setSelected(0);
      } catch (cause) {
        if (request !== searchRequest.current) return;
        setResults([]);
        setError(getErrorMessage(cause, "Search is unavailable."));
      } finally {
        if (request === searchRequest.current) setLoading(false);
      }
    }, 250);
    return () => window.clearTimeout(timer);
  }, [open, query]);

  useEffect(() => {
    listRef.current?.querySelector(`[data-result-index="${selected}"]`)?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  if (!open) return null;

  return createPortal(
    <div className="fixed inset-0 z-[100] flex justify-center bg-black/75 px-3 pt-[min(16vh,140px)] backdrop-blur-md sm:px-6" onMouseDown={(event) => {
      if (event.target === event.currentTarget) hide();
    }}>
      <section role="dialog" aria-modal="true" aria-label="Quick search" className="flex h-fit max-h-[min(70vh,650px)] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-white/15 bg-[#111217] shadow-[0_24px_90px_rgba(0,0,0,0.8)]">
        <div className="flex items-center gap-3 border-b border-white/10 px-5 py-4">
          <Search className="h-5 w-5 shrink-0 text-indigo-300" aria-hidden="true" />
          <input
            ref={inputRef}
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search your files and memories..."
            aria-label="Search indexed files"
            aria-controls="quick-search-results"
            autoComplete="off"
            className="min-w-0 flex-1 bg-transparent text-base text-white outline-none placeholder:text-zinc-500 [appearance:textfield]"
          />
          {loading ? <LoaderCircle className="h-4 w-4 animate-spin text-zinc-400" aria-label="Searching" /> : null}
          <button type="button" onClick={hide} aria-label="Close quick search" className="rounded-md p-1 text-zinc-400 hover:bg-white/10 hover:text-white"><X className="h-4 w-4" /></button>
        </div>

        <div id="quick-search-results" ref={listRef} className="min-h-28 overflow-y-auto p-2" aria-live="polite">
          {error ? <p role="alert" className="px-4 py-5 text-sm text-red-300">{error}</p> : null}
          {!query.trim() ? <p className="px-4 py-8 text-center text-sm text-zinc-500">Type a filename or describe what you remember</p> : null}
          {query.trim() && loading ? <p className="px-4 py-8 text-center text-sm text-zinc-500">Searching your index...</p> : null}
          {query.trim() && !loading && !error && results.length === 0 ? <p className="px-4 py-8 text-center text-sm text-zinc-500">No matching files</p> : null}
          {results.length > 0 ? (
            <>
              <div className="px-3 pb-2 pt-2 text-[11px] font-medium uppercase tracking-[0.18em] text-zinc-500">Files</div>
              {results.map((result, index) => (
                <button
                  key={result.document_id}
                  data-result-index={index}
                  type="button"
                  onMouseEnter={() => setSelected(index)}
                  onClick={() => void openResult(result)}
                  disabled={openingId !== null}
                  aria-selected={selected === index}
                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left outline-none transition-colors ${selected === index ? "bg-indigo-500/15 text-white" : "text-zinc-200 hover:bg-white/[0.06]"}`}
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04]"><FileIcon type={result.file_type} /></span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium">{result.filename}</span>
                    <span className="mt-0.5 block truncate text-xs text-zinc-500">{result.snippet || result.filepath}</span>
                  </span>
                  <span className="shrink-0 text-[11px] font-medium uppercase tracking-wide text-zinc-500">{openingId === result.document_id ? "Opening" : result.file_type}</span>
                </button>
              ))}
            </>
          ) : null}
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-white/10 px-4 py-3 text-[11px] text-zinc-500">
          <span className="font-semibold tracking-wide text-zinc-400">ReFind</span>
          <span className="flex items-center gap-2"><ArrowUp className="h-3 w-3" /><ArrowDown className="h-3 w-3" /> Select <CornerDownLeft className="ml-2 h-3 w-3" /> Open <span className="ml-2 rounded border border-white/10 px-1.5 py-0.5">Esc</span> Close</span>
        </div>
      </section>
    </div>,
    document.body
  );
}
