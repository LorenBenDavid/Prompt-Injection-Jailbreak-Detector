import { useEffect, useState } from "react";
import { BookOpen, AlertTriangle, CheckCircle } from "lucide-react";
import Spinner from "../components/Spinner";
import { motion } from "framer-motion";
import { api, GalleryItem } from "../api/client";

const RISK_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  SAFE: { bg: "bg-emerald-500/10", text: "text-emerald-400", border: "border-emerald-500/30" },
  LOW: { bg: "bg-yellow-500/10", text: "text-yellow-400", border: "border-yellow-500/30" },
  MEDIUM: { bg: "bg-orange-500/10", text: "text-orange-400", border: "border-orange-500/30" },
  HIGH: { bg: "bg-red-500/10", text: "text-red-400", border: "border-red-500/30" },
  CRITICAL: { bg: "bg-red-400/20", text: "text-red-300", border: "border-red-400/50" },
};

function GalleryCard({ item, index }: { item: GalleryItem; index: number }) {
  const colors = RISK_COLORS[item.risk_level] ?? RISK_COLORS.SAFE;
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className={`rounded-xl border p-4 space-y-3 ${colors.border} ${colors.bg}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className={`flex items-center gap-1.5 text-sm font-semibold ${colors.text}`}>
          {item.label === 1 ? <AlertTriangle size={14} /> : <CheckCircle size={14} />}
          {item.risk_level}
        </div>
        <span className="text-xs text-muted-foreground">{(item.ensemble_score * 100).toFixed(1)}%</span>
      </div>

      <p className="text-sm text-muted-foreground line-clamp-3">{item.prompt}</p>

      <div className="flex flex-wrap gap-1.5">
        {item.attack_subtype !== "none" && (
          <span className="text-xs px-2 py-0.5 rounded-full border border-border text-muted-foreground">
            {item.attack_subtype}
          </span>
        )}
        <span className="text-xs px-2 py-0.5 rounded-full border border-border text-muted-foreground">
          {item.source}
        </span>
      </div>
    </motion.div>
  );
}

export default function GalleryPage() {
  const [items, setItems] = useState<GalleryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    api
      .gallery(30)
      .then((d) => setItems(d.items))
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load gallery"))
      .finally(() => setLoading(false));
  }, []);

  const filtered = filter === "all" ? items : items.filter((i) => i.risk_level === filter);
  const riskLevels = ["all", ...new Set(items.map((i) => i.risk_level))];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <BookOpen className="text-primary" size={32} />
          Example Gallery
        </h1>
        <p className="text-muted-foreground mt-1">Curated examples from the test set with live model scores</p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-2">
        {riskLevels.map((r) => (
          <button
            key={r}
            onClick={() => setFilter(r)}
            className={`text-sm px-3 py-1.5 rounded-full border transition-colors capitalize ${
              filter === r
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:text-foreground hover:border-primary/40"
            }`}
          >
            {r}
          </button>
        ))}
      </div>

      {loading ? (
        <Spinner label="Loading gallery…" />
      ) : error ? (
        <div className="rounded-xl border border-destructive/50 bg-destructive/10 p-6 text-center text-destructive">
          {error}
          <p className="text-sm mt-1 text-muted-foreground">Make sure models are loaded and the backend is running.</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="py-24 text-center text-muted-foreground">No examples found.</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((item, i) => (
            <GalleryCard key={i} item={item} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
