const BASE = "/api";

export interface AnalyzeResponse {
  text: string;
  is_attack: boolean;
  risk_level: "SAFE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  ensemble_score: number;
  short_circuited: boolean;
  explanation: string;
  total_latency_ms: number;
  layers: {
    heuristic_score: number;
    heuristic_triggered_rules: string[];
    heuristic_latency_ms: number;
    embedding_score: number | null;
    embedding_nearest_attacks: { text: string; similarity: number }[];
    embedding_latency_ms: number | null;
    bert_score: number | null;
    bert_shap_tokens: { token: string; importance: number }[];
    bert_latency_ms: number | null;
  };
}

export interface StatsResponse {
  total_analyzed: number;
  attack_count: number;
  benign_count: number;
  risk_distribution: Record<string, number>;
  avg_latency_ms: number;
  dataset_stats: Record<string, { total: number; attacks: number; benign: number }>;
}

export interface MetricsResponse {
  embedding: Record<string, number> | null;
  bert: Record<string, number> | null;
}

export interface DatasetRow {
  id: string;
  prompt: string;
  label: number;
  attack_type: string;
  attack_subtype: string;
  source: string;
  severity: number;
  language: string;
  created_at: string;
}

export interface DatasetResponse {
  rows: DatasetRow[];
  total: number;
  page: number;
  page_size: number;
  attack_count: number;
  benign_count: number;
}

export interface GalleryItem {
  prompt: string;
  label: number;
  risk_level: string;
  ensemble_score: number;
  attack_subtype: string;
  source: string;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined) url.searchParams.set(k, String(v));
    });
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  analyze: (text: string) => post<AnalyzeResponse>("/analyze", { text }),
  stats: () => get<StatsResponse>("/stats"),
  metrics: () => get<MetricsResponse>("/metrics"),
  dataset: (params?: Record<string, string | number | undefined>) =>
    get<DatasetResponse>("/dataset", params),
  gallery: (n = 20) => get<{ items: GalleryItem[] }>("/gallery", { n }),
  health: () => get<{ status: string; models_loaded: boolean; dataset_loaded: boolean }>("/health"),
};
