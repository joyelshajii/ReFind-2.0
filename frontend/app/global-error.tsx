"use client";

import { useEffect } from "react";

export default function GlobalError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  useEffect(() => {
    console.error("Root layout rendering failed:", error);
  }, [error]);

  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#050505", color: "#f4f4f5", fontFamily: "sans-serif" }}>
        <main style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: "24px" }}>
          <section role="alert" style={{ maxWidth: "480px", textAlign: "center" }}>
            <h1>ReFind could not start this page</h1>
            <p style={{ color: "#a1a1aa", lineHeight: 1.6 }}>
              An unexpected error stopped the app from loading. Try again, or check the browser console for details.
            </p>
            {error.digest && <p style={{ color: "#71717a", fontSize: "12px" }}>Reference: {error.digest}</p>}
            <button
              onClick={() => retry()}
              style={{ marginTop: "16px", padding: "10px 16px", borderRadius: "10px", border: "1px solid #6366f1", background: "#312e81", color: "white", cursor: "pointer" }}
            >
              Try again
            </button>
          </section>
        </main>
      </body>
    </html>
  );
}
