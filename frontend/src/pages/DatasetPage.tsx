import { useEffect, useState } from "react";
import { Database, ChevronLeft, ChevronRight } from "lucide-react";
import { api, DatasetRow } from "../api/client";
import Spinner from "../components/Spinner";

const LABEL_BADGE: Record<number, string> = {
  0: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  1: "bg-red-500/20 text-red-400 border-red-500/30",
};

const SEVERITY_LABEL: Record<number, string> = {
  0: "None",
  1: "Low",
  2: "Medium",
  3: "High",
};

export default function DatasetPage() {
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [total, setTotal] = useState(0);
  const [attackCount, setAttackCount] = useState(0);
  const [benignCount, setBenignCount] = useState(0);
  const [page, setPage] = useState(1);
  const [split, setSplit] = useState("test");
  const [labelFilter, setLabelFilter] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const pageSize = 50;

  useEffect(() => {
    setLoading(true);
    api
      .dataset({
        split,
        page,
        page_size: pageSize,
        label: labelFilter !== "" ? Number(labelFilter) : undefined,
      })
      .then((d) => {
        setRows(d.rows);
        setTotal(d.total);
        setAttackCount(d.attack_count);
        setBenignCount(d.benign_count);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [split, page, labelFilter]);

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Database className="text-primary" size={32} />
          Dataset Explorer
        </h1>
        <p className="text-muted-foreground mt-1">Browse training, validation, and test splits</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Total Rows", value: total },
          { label: "Attacks", value: attackCount },
          { label: "Benign", value: benignCount },
        ].map(({ label, value }) => (
          <div key={label} className="rounded-xl border border-border bg-card p-4 text-center">
            <div className="text-2xl font-bold">{value.toLocaleString()}</div>
            <div className="text-sm text-muted-foreground">{label}</div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex gap-3 flex-wrap">
        <select
          value={split}
          onChange={(e) => { setSplit(e.target.value); setPage(1); }}
          className="rounded-lg border border-border bg-card px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          {["train", "val", "test"].map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <select
          value={labelFilter}
          onChange={(e) => { setLabelFilter(e.target.value); setPage(1); }}
          className="rounded-lg border border-border bg-card px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
        >
          <option value="">All labels</option>
          <option value="0">Benign</option>
          <option value="1">Attack</option>
        </select>
      </div>

      {/* Table */}
      <div className="rounded-xl border border-border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                {["Label", "Prompt", "Type", "Subtype", "Source", "Severity"].map((h) => (
                  <th key={h} className="text-left px-4 py-3 text-muted-foreground font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={6}><Spinner label="Loading dataset…" /></td>
                </tr>
              ) : rows.length === 0 ? (
                <tr>
                  <td colSpan={6} className="text-center py-12 text-muted-foreground">No data found</td>
                </tr>
              ) : (
                rows.map((row) => (
                  <tr key={row.id} className="border-b border-border/50 hover:bg-muted/20 transition-colors">
                    <td className="px-4 py-3">
                      <span className={`text-xs px-2 py-0.5 rounded border ${LABEL_BADGE[row.label]}`}>
                        {row.label === 1 ? "Attack" : "Benign"}
                      </span>
                    </td>
                    <td className="px-4 py-3 max-w-xs">
                      <span className="line-clamp-2 text-muted-foreground">{row.prompt}</span>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">{row.attack_type}</td>
                    <td className="px-4 py-3 text-muted-foreground">{row.attack_subtype}</td>
                    <td className="px-4 py-3 text-muted-foreground">{row.source}</td>
                    <td className="px-4 py-3 text-muted-foreground">{SEVERITY_LABEL[row.severity] ?? row.severity}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">
            Page {page} of {totalPages} · {total.toLocaleString()} rows
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="p-2 rounded-lg border border-border hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <ChevronLeft size={16} />
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="p-2 rounded-lg border border-border hover:bg-muted disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
