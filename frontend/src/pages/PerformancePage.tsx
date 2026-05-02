import { useEffect, useState } from "react";
import { BarChart2, TrendingUp } from "lucide-react";
import Spinner from "../components/Spinner";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { api, MetricsResponse, StatsResponse } from "../api/client";

const RISK_COLORS: Record<string, string> = {
  SAFE: "#10b981",
  LOW: "#f59e0b",
  MEDIUM: "#f97316",
  HIGH: "#ef4444",
  CRITICAL: "#dc2626",
};

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-card p-4 text-center">
      <div className="text-2xl font-bold text-primary">{value}</div>
      <div className="text-sm text-muted-foreground mt-0.5">{label}</div>
    </div>
  );
}

export default function PerformancePage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.metrics(), api.stats()])
      .then(([m, s]) => { setMetrics(m); setStats(s); })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const barData = metrics
    ? [
        {
          name: "Embedding",
          ...(metrics.embedding
            ? {
                Accuracy: +(metrics.embedding.accuracy * 100).toFixed(1),
                Precision: +(metrics.embedding.precision * 100).toFixed(1),
                Recall: +(metrics.embedding.recall * 100).toFixed(1),
                F1: +(metrics.embedding.f1 * 100).toFixed(1),
              }
            : {}),
        },
        {
          name: "BERT",
          ...(metrics.bert
            ? {
                Accuracy: +(metrics.bert.accuracy * 100).toFixed(1),
                Precision: +(metrics.bert.precision * 100).toFixed(1),
                Recall: +(metrics.bert.recall * 100).toFixed(1),
                F1: +(metrics.bert.f1 * 100).toFixed(1),
              }
            : {}),
        },
      ]
    : [];

  const radarData = metrics?.bert
    ? [
        { metric: "Accuracy", BERT: +(metrics.bert.accuracy * 100).toFixed(1), Embedding: metrics.embedding ? +(metrics.embedding.accuracy * 100).toFixed(1) : 0 },
        { metric: "Precision", BERT: +(metrics.bert.precision * 100).toFixed(1), Embedding: metrics.embedding ? +(metrics.embedding.precision * 100).toFixed(1) : 0 },
        { metric: "Recall", BERT: +(metrics.bert.recall * 100).toFixed(1), Embedding: metrics.embedding ? +(metrics.embedding.recall * 100).toFixed(1) : 0 },
        { metric: "F1", BERT: +(metrics.bert.f1 * 100).toFixed(1), Embedding: metrics.embedding ? +(metrics.embedding.f1 * 100).toFixed(1) : 0 },
      ]
    : [];

  const pieData = stats
    ? Object.entries(stats.risk_distribution)
        .filter(([, v]) => v > 0)
        .map(([name, value]) => ({ name, value }))
    : [];

  if (loading) return <Spinner label="Loading metrics…" />;

  const noMetrics = !metrics?.embedding && !metrics?.bert;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <BarChart2 className="text-primary" size={32} />
          Model Performance
        </h1>
        <p className="text-muted-foreground mt-1">Test-set evaluation metrics and runtime statistics</p>
      </div>

      {noMetrics ? (
        <div className="rounded-xl border border-border bg-card p-8 text-center text-muted-foreground">
          No metrics found. Run <code className="text-primary">python scripts/train.py --model all</code> to train and evaluate models.
        </div>
      ) : (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {metrics?.bert && (
              <>
                <MetricCard label="BERT Accuracy" value={`${(metrics.bert.accuracy * 100).toFixed(1)}%`} />
                <MetricCard label="BERT F1" value={`${(metrics.bert.f1 * 100).toFixed(1)}%`} />
                <MetricCard label="BERT Precision" value={`${(metrics.bert.precision * 100).toFixed(1)}%`} />
                <MetricCard label="BERT Recall" value={`${(metrics.bert.recall * 100).toFixed(1)}%`} />
              </>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Bar chart */}
            <div className="rounded-xl border border-border bg-card p-5">
              <h2 className="font-semibold mb-4 flex items-center gap-2">
                <TrendingUp size={16} className="text-primary" />
                Model Comparison
              </h2>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={barData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                  <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px" }}
                    formatter={(v: number) => `${v}%`}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Accuracy" fill="#3b82f6" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Precision" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Recall" fill="#10b981" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="F1" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Radar chart */}
            <div className="rounded-xl border border-border bg-card p-5">
              <h2 className="font-semibold mb-4">Performance Radar</h2>
              <ResponsiveContainer width="100%" height={240}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 12 }} />
                  <Radar name="BERT" dataKey="BERT" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                  {metrics?.embedding && (
                    <Radar name="Embedding" dataKey="Embedding" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                  )}
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* Live stats */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="rounded-xl border border-border bg-card p-5">
            <h2 className="font-semibold mb-4">Risk Distribution (live)</h2>
            {pieData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {pieData.map((entry) => (
                      <Cell key={entry.name} fill={RISK_COLORS[entry.name] ?? "#64748b"} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: "8px" }} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-muted-foreground text-sm py-8 text-center">No analyses yet. Try the Analyzer tab.</p>
            )}
          </div>

          <div className="rounded-xl border border-border bg-card p-5 space-y-4">
            <h2 className="font-semibold">Runtime Statistics</h2>
            <div className="space-y-3 text-sm">
              {[
                ["Total Analyzed", stats.total_analyzed.toLocaleString()],
                ["Attacks Detected", stats.attack_count.toLocaleString()],
                ["Benign Requests", stats.benign_count.toLocaleString()],
                ["Avg Latency", `${stats.avg_latency_ms.toFixed(1)}ms`],
              ].map(([label, value]) => (
                <div key={label} className="flex justify-between border-b border-border/50 pb-2">
                  <span className="text-muted-foreground">{label}</span>
                  <span className="font-medium">{value}</span>
                </div>
              ))}
            </div>

            <div>
              <div className="text-sm font-medium mb-2 text-muted-foreground">Dataset Splits</div>
              {Object.entries(stats.dataset_stats).map(([split, info]) => (
                <div key={split} className="flex justify-between text-sm py-1 border-b border-border/30">
                  <span className="capitalize text-muted-foreground">{split}</span>
                  <span>{(info as { total: number }).total?.toLocaleString()} rows</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
