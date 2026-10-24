export interface QueryClues {
  raw_query: string;
  intent: string;
  file_type?: string | null;
  topics: string[];
  people: string[];
  technologies: string[];
  time_clues: string[];
  keywords: string[];
}

export interface EvidenceItem {
  clue_name: string;
  clue_type: "person" | "date" | "technology" | "topic" | "file_type" | string;
  matched_text: string;
  location?: string | null;
  verified: boolean;
}

export interface SearchResultItem {
  document_id: string;
  filename: string;
  filepath: string;
  file_type: string;
  file_size: number;
  modified_at: string;
  score: number;
  score_label: string;
  snippet: string;
  page_or_slide?: string | null;
  matching_clues: string[];
  detected_entities: string[];
  evidence: EvidenceItem[];
  ai_explanation?: string | null;
}

export interface SearchResponse {
  query: string;
  clues: QueryClues;
  total_results: number;
  results: SearchResultItem[];
  processing_time_ms: number;
}

export interface ContextDetailResponse {
  document_id: string;
  filename: string;
  file_type: string;
  score: number;
  score_label: string;
  memory_clues: string[];
  document_evidence: EvidenceItem[];
  snippets: string[];
  ai_explanation?: string | null;
}

export interface IndexProgressStatus {
  is_indexing: boolean;
  current_stage: string;
  total_files: number;
  processed_files: number;
  documents_count: number;
  images_count: number;
  other_count: number;
  current_file?: string | null;
  last_indexed_at?: string | null;
  error_message?: string | null;
}

export interface IndexStats {
  total_documents: number;
  total_chunks: number;
  last_indexed_at?: string | null;
  storage_source: string;
  database_type: string;
  embedding_provider: string;
  embedding_dimension: number;
}

export interface MediaIndexJob {
  id: string;
  filename: string;
  duration_seconds: number | null;
  status: "awaiting" | "queued" | "scheduled" | "running" | "failed" | "completed" | "skipped";
  scheduled_at: string | null;
  error_message: string | null;
}

export interface MediaIndexJob {
  id: string;
  filename: string;
  duration_seconds: number | null;
  status: "awaiting" | "queued" | "scheduled" | "running" | "failed" | "completed" | "skipped";
  scheduled_at: string | null;
  error_message: string | null;
}

export interface IndexedDocumentItem {
  id: string;
  filename: string;
  filepath: string;
  file_type: string;
  file_size: number;
  created_at: string;
  modified_at: string;
  source: string;
  summary?: string;
  indexed_at: string;
  entities: { id?: string; entity_type: string; entity_value: string }[];
  chunk_count: number;
}

export interface DuplicateFileItem {
  id: string;
  filename: string;
  filepath: string;
  file_type: string;
  file_size: number;
  modified_at: string;
  created_at: string;
  is_recommended: boolean;
  recommendation_reason: string;
  exists_on_disk: boolean;
}

export interface DuplicateGroup {
  group_id: string;
  match_type: "exact_hash" | "exact_text" | "filename_and_content" | "similar_content" | string;
  match_label: string;
  confidence: number;
  why_duplicate: string;
  differences: string;
  potential_savings_bytes: number;
  recommended_document_id: string;
  files: DuplicateFileItem[];
}

export interface DuplicateOverviewResponse {
  total_groups: number;
  total_duplicate_files: number;
  total_reclaimable_bytes: number;
  groups: DuplicateGroup[];
}

export interface ResolveGroupResponse {
  status: string;
  message: string;
  kept_document_id: string;
  deleted_count: number;
  freed_bytes: number;
}

export interface ResolveAllResponse {
  status: string;
  message: string;
  groups_resolved: number;
  total_deleted: number;
  freed_bytes: number;
}
