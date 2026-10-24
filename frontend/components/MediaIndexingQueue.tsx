"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Clock, Loader2 } from "lucide-react";
import { decideMediaIndexJob, getErrorMessage, getMediaIndexJobs } from "@/lib/api";
import { MediaIndexJob } from "@/types";

export default function MediaIndexingQueue() {
  const [jobs, setJobs] = useState<MediaIndexJob[]>([]);
  const [dates, setDates] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const load = useCallback(async () => {
    try {
      setJobs(await getMediaIndexJobs());
      setLoadError(null);
    } catch (err) {
      setLoadError(getErrorMessage(err, "Unable to check pending media."));
    }
  }, []);
  useEffect(() => {
    let active = true;
    let inFlight = false;
    const poll = async () => {
      if (inFlight || document.hidden) return;
      inFlight = true;
      try {
        const next = await getMediaIndexJobs();
        if (active) { setJobs(next); setLoadError(null); }
      } catch (err) {
        if (active) setLoadError(getErrorMessage(err, "Unable to check pending media."));
      } finally { inFlight = false; }
    };
    void poll();
    const timer = setInterval(poll, 3000);
    document.addEventListener("visibilitychange", poll);
    return () => { active = false; clearInterval(timer); document.removeEventListener("visibilitychange", poll); };
  }, []);

  const decide = async (job: MediaIndexJob, action: "continue" | "schedule" | "skip") => {
    setError(null);
    let scheduledAt: string | undefined;
    if (action === "schedule") {
      const date = new Date(dates[job.id] || "");
      if (!Number.isFinite(date.getTime()) || date.getTime() <= Date.now()) {
        setError("Choose a future date and time before scheduling.");
        return;
      }
      scheduledAt = date.toISOString();
    }
    setBusy(job.id);
    try {
      await decideMediaIndexJob(job.id, action, scheduledAt);
    } catch (err) {
      setError(getErrorMessage(err, "Unable to update media indexing."));
    } finally {
      await load();
      setBusy(null);
    }
  };

  if (!jobs.length && !loadError) return null;
  return (
    <section className="space-y-4 rounded-2xl border border-amber-400/25 bg-black/50 p-6" aria-label="Media indexing choices">
      <h2 className="flex items-center gap-2 text-lg font-semibold text-amber-200"><Clock className="h-5 w-5" />Media indexing choices</h2>
      <p className="text-sm text-zinc-300">ReFind checks for readable subtitles, transcripts, and lyrics first. Recordings longer than 15 minutes without usable text need your choice before full processing starts.</p>
      {(error || loadError) && <p role="alert" className="text-sm text-red-300">{error || loadError}</p>}
      {jobs.map((job) => (
        <article key={job.id} className="space-y-3 rounded-xl border border-white/10 p-4">
          <div className="flex items-start gap-3">
            {job.status === "running" ? <Loader2 className="mt-1 h-4 w-4 shrink-0 animate-spin text-indigo-300" /> : <AlertTriangle className="mt-1 h-4 w-4 shrink-0 text-amber-300" />}
            <div className="min-w-0">
              <h3 className="break-words font-medium text-zinc-100">{job.filename}</h3>
              <p className="text-sm text-zinc-400">{job.duration_seconds == null ? "Duration could not be determined" : `${Math.ceil(job.duration_seconds / 60)} minutes`}</p>
            </div>
          </div>
          <p className="text-sm text-zinc-300">
            {job.status === "running" ? "Full media indexing is running."
              : job.status === "queued" ? "Queued to start when the current indexing finishes."
              : job.status === "scheduled" && job.scheduled_at ? `Scheduled for ${new Date(job.scheduled_at).toLocaleString()}.`
              : "No usable transcript was found. Processing the audio and video content can take considerably longer. Continue now, choose a later time, or skip for now."}
          </p>
          {job.error_message && <p role="alert" className="text-sm text-red-300">{job.error_message}</p>}
          {job.status !== "running" && (
            <div className="flex flex-wrap items-end gap-3">
              <button type="button" disabled={!!busy || job.status === "queued"} onClick={() => void decide(job, "continue")} className="rounded-lg bg-indigo-600 px-4 py-2 text-sm text-white disabled:opacity-50">Continue now</button>
              <label className="flex flex-col gap-1 text-xs text-zinc-300" htmlFor={`schedule-${job.id}`}>
                Later date and time (your local time)
                <input id={`schedule-${job.id}`} type="datetime-local" value={dates[job.id] || ""} onChange={(event) => setDates((prev) => ({ ...prev, [job.id]: event.target.value }))} className="rounded-lg border border-white/20 bg-zinc-900 px-3 py-2 text-sm text-white [color-scheme:dark]" />
              </label>
              <button type="button" disabled={!!busy} onClick={() => void decide(job, "schedule")} className="rounded-lg border border-indigo-400/40 px-4 py-2 text-sm text-indigo-200 disabled:opacity-50">Schedule later</button>
              <button type="button" disabled={!!busy} onClick={() => void decide(job, "skip")} className="rounded-lg px-3 py-2 text-sm text-zinc-300 hover:bg-white/10 disabled:opacity-50">{job.status === "scheduled" || job.status === "queued" ? "Cancel" : "Skip for now"}</button>
            </div>
          )}
        </article>
      ))}
      <p className="text-xs text-zinc-400">Scheduled indexing requires ReFind to be running. If it is closed at the chosen time, the job starts when ReFind next runs. Files with usable transcripts use the faster text path.</p>
    </section>
  );
}
