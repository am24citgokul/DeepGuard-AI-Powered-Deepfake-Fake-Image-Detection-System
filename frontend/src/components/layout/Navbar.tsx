import React from "react";
import { Link, useLocation } from "react-router-dom";
import { 
  ShieldCheck, 
  Sun, 
  Moon, 
  Menu, 
  Wifi, 
  AlertCircle,
  Database
} from "lucide-react";
import { isConfiguredBackend } from "@/lib/api";

interface NavbarProps {
  isDarkMode: boolean;
  onToggleDarkMode: () => void;
  onOpenMobileMenu: () => void;
  backendOnline: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  isDarkMode,
  onToggleDarkMode,
  onOpenMobileMenu,
  backendOnline,
}) => {
  const isBackend = isConfiguredBackend() && backendOnline;

  return (
    <header className="sticky top-0 z-30 h-16 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between">
      {/* Left: Mobile hamburger & Brand */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onOpenMobileMenu}
          aria-label="Open navigation menu"
          className="lg:hidden p-2 rounded-xl text-slate-500 hover:text-slate-800 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <Menu className="w-5 h-5" />
        </button>

        <Link to="/" className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-600 via-violet-600 to-teal-500 flex items-center justify-center text-white shadow-sm">
            <ShieldCheck className="w-5 h-5 stroke-[2.5]" />
          </div>
          <div>
            <span className="font-heading font-extrabold text-base tracking-tight text-slate-900 dark:text-white flex items-center gap-1.5">
              DeepGuard <span className="text-violet-600 dark:text-violet-400">X</span>
            </span>
            <span className="text-[10px] font-mono text-slate-400 block -mt-1 tracking-wider uppercase">
              Forensic Screening
            </span>
          </div>
        </Link>
      </div>

      {/* Right: Backend status pill + Theme toggle */}
      <div className="flex items-center gap-3">
        {/* Backend status pill */}
        {isBackend ? (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>Connected</span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono font-medium bg-amber-50 dark:bg-amber-950/40 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-800">
            <Database className="w-3.5 h-3.5" />
            <span>Demo data</span>
          </div>
        )}

        {/* Dark Mode Toggle */}
        <button
          type="button"
          onClick={onToggleDarkMode}
          aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
          className="p-2 rounded-xl text-slate-500 hover:text-slate-800 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 border border-slate-200 dark:border-slate-800 transition-colors"
        >
          {isDarkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-600" />}
        </button>
      </div>
    </header>
  );
};
