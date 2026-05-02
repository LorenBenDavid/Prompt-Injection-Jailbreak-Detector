import { Routes, Route, NavLink } from "react-router-dom";
import { Shield, Database, BarChart2, BookOpen } from "lucide-react";
import AnalyzerPage from "./pages/AnalyzerPage";
import DatasetPage from "./pages/DatasetPage";
import PerformancePage from "./pages/PerformancePage";
import GalleryPage from "./pages/GalleryPage";

const navItems = [
  { to: "/", label: "Analyzer", Icon: Shield },
  { to: "/dataset", label: "Dataset", Icon: Database },
  { to: "/performance", label: "Performance", Icon: BarChart2 },
  { to: "/gallery", label: "Gallery", Icon: BookOpen },
];

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-border bg-card sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-14 flex items-center gap-8">
          <div className="flex items-center gap-2 font-semibold text-primary">
            <Shield size={20} />
            <span>PromptGuard</span>
          </div>
          <nav className="flex gap-1">
            {navItems.map(({ to, label, Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors ${
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"
                  }`
                }
              >
                <Icon size={14} />
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-8">
        <Routes>
          <Route path="/" element={<AnalyzerPage />} />
          <Route path="/dataset" element={<DatasetPage />} />
          <Route path="/performance" element={<PerformancePage />} />
          <Route path="/gallery" element={<GalleryPage />} />
        </Routes>
      </main>
    </div>
  );
}
