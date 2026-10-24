"use client";

import { useEffect } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

export default function ErrorPage({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  useEffect(() => {
    console.error("Page rendering failed:", error);
  }, [error]);

  return (
    <div className="flex-1 flex items-center justify-center px-4 py-16">
      <section role="alert" className="w-full max-w-lg rounded-2xl border border-red-500/20 bg-black/60 p-8 text-center shadow-xl">
        <AlertTriangle className="mx-auto h-8 w-8 text-amber-400" />
        <h1 className="mt-4 text-xl font-semibold text-white">This page ran into a problem</h1>
        <p className="mt-2 text-sm text-zinc-300">
          ReFind could not finish displaying this page. You can try loading it again.
        </p>
        {error.digest && <p className="mt-3 text-xs text-zinc-500">Reference: {error.digest}</p>}
        <button
          onClick={() => retry()}
          className="mt-6 inline-flex items-center gap-2 rounded-xl border border-indigo-400/30 bg-indigo-500/15 px-4 py-2 text-sm font-medium text-indigo-100 hover:bg-indigo-500/25"
        >
          <RefreshCw className="h-4 w-4" />
          Try again
        </button>
      </section>
    </div>
  );
}
