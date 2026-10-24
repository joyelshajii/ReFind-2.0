import {
  SearchResponse,
  ContextDetailResponse,
  IndexProgressStatus,
  IndexStats,
  IndexedDocumentItem
} from "@/types";

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export class ApiError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback;
}

function responseMessage(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object") return fallback;
  const detail = (body as { detail?: unknown; message?: unknown }).detail
    ?? (body as { message?: unknown }).message;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item && typeof item === "object" && "msg" in item
        ? String((item as { msg: unknown }).msg)
        : String(item))
      .filter(Boolean);
    if (messages.length) return messages.join("; ");
  }
  return fallback;
}

async function requestJson<T>(path: string, init: RequestInit, fallback: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new ApiError("Could not reach the ReFind backend. Check that it is running and try again.");
  }

  let responseText: string;
  try {
    responseText = await response.text();
  } catch {
    throw new ApiError("Could not read the response from the ReFind backend.", response.status);
  }

  let body: unknown = null;
  if (responseText.trim()) {
    try {
      body = JSON.parse(responseText);
    } catch {
      if (response.ok) {
        throw new ApiError("The ReFind backend returned an invalid response.", response.status);
      }
      const contentType = response.headers.get("content-type") ?? "";
      body = contentType.includes("text/plain")
        ? { detail: responseText.slice(0, 500) }
        : null;
    }
  }

  if (!response.ok) {
    throw new ApiError(responseMessage(body, `${fallback} (HTTP ${response.status})`), response.status);
  }
  return body as T;
}

export async function searchDocuments(query: string, topK: number = 10): Promise<SearchResponse> {
  return requestJson<SearchResponse>("/api/search", {
    method: "POST",
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  }, "Search request failed");
}

export async function getDocumentContext(
  documentId: string,
  query: string,
  score: number = 0.94,
  scoreLabel: string = "Strong match"
): Promise<ContextDetailResponse> {
  const params = new URLSearchParams({
    q: query,
    score: score.toString(),
    score_label: scoreLabel,
  });
  return requestJson<ContextDetailResponse>(
    `/api/documents/${encodeURIComponent(documentId)}/context?${params.toString()}`,
    { cache: "no-store" },
    "Failed to load document context",
  );
}

export async function getIndexStatus(): Promise<IndexProgressStatus> {
  return requestJson<IndexProgressStatus>("/api/index/status", { cache: "no-store" }, "Failed to fetch indexing status");
}

export async function getIndexStats(): Promise<IndexStats> {
  return requestJson<IndexStats>("/api/index/stats", { cache: "no-store" }, "Failed to fetch index stats");
}

export async function getIndexedDocuments(search?: string, fileType?: string): Promise<IndexedDocumentItem[]> {
  const params = new URLSearchParams();
  if (search && search.trim()) params.set("search", search.trim());
  if (fileType && fileType.trim() && fileType !== "all") params.set("file_type", fileType.trim());
  params.set("limit", "100");

  const query = params.toString();
  return requestJson<IndexedDocumentItem[]>(
    `/api/documents${query ? `?${query}` : ""}`,
    { cache: "no-store" },
    "Failed to fetch indexed documents list",
  );
}

export async function deleteDocument(documentId: string): Promise<{ status: string; message: string }> {
  return requestJson<{ status: string; message: string }>(`/api/documents/${encodeURIComponent(documentId)}`, {
    method: "DELETE",
  }, "Failed to remove document");
}

export async function deleteAllDocuments(): Promise<{
  status: string;
  removed_count: number;
  message: string;
}> {
  return requestJson<{ status: string; removed_count: number; message: string }>("/api/documents", {
    method: "DELETE",
  }, "Failed to clear indexed files");
}

export async function triggerIndexing(folderPath?: string): Promise<{ status: string; message: string; files_found: number }> {
  return requestJson<{ status: string; message: string; files_found: number }>("/api/index", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(folderPath ? { folder_path: folderPath } : {}),
  }, "Failed to trigger indexing");
}

export async function uploadFiles(formData: FormData): Promise<{
  status: string;
  message: string;
  files?: Array<{ id: string; filename: string; file_type: string; file_size: number }>;
  failed_files?: Array<{ filename: string; error: string }>;
}> {
  return requestJson<{
    status: string;
    message: string;
    files?: Array<{ id: string; filename: string; file_type: string; file_size: number }>;
    failed_files?: Array<{ filename: string; error: string }>;
  }>("/api/index/upload", {
    method: "POST",
    body: formData,
  }, "Failed to upload files");
}

export async function seedDemoData(): Promise<{ status: string; message: string; demo_scenario: string }> {
  return requestJson<{ status: string; message: string; demo_scenario: string }>("/api/demo/seed", {
    method: "POST",
  }, "Failed to seed demo data");
}

export async function getHealth(): Promise<any> {
  try {
    return await requestJson("/api/health", { cache: "no-store" }, "Health check failed");
  } catch {
    return null;
  }
}

export function getDocumentDownloadUrl(documentId: string): string {
  return `${API_BASE_URL}/api/documents/${encodeURIComponent(documentId)}/download`;
}
