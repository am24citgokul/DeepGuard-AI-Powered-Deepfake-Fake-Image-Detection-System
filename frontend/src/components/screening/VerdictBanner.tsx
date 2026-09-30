import React from "react";
import { Printer, Download, Clock, Hash, CheckCircle, AlertTriangle, ShieldAlert } from "lucide-react";
import type { FinalVerdict, ConsistencyType } from "@/lib/types";
import { getVerdictConfig, formatDateTime } from "@/lib/format";

interface VerdictBannerProps {
  verdict: FinalVerdict;
  consistency: ConsistencyType;
  reasoning: string;
  createdAt: string;
  reportId: string;
  className?: string;
}

export const VerdictBanner: React.FC<VerdictBannerProps> = ({
  verdict,
  consistency,
  reasoning,
  createdAt,
  reportId,
  className = "",
}) => {
  const config = getVerdictConfig(verdict);
  const VerdictIcon = config.icon;

  const handlePrint = () => {
    window.print();
  };

  return (
    <div
      className={`rounded-2xl border ${config.borderLight} dark:${config.borderDark} ${config.bgLight} dark:${config.bgDark} p-6 sm:p-8 shadow-sm relative overflow-hidden ${className}`}
    >
      {/* Background Accent Gradient glow */}
      <div
        className="absolute -right-20 -top-20 w-64 h-64 rounded-full blur-3xl opacity-20 pointer-events-none"
        style={{ backgroundColor: config.color }}
      />

      <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
        {/* Left Column: Verdict Badge & Reasoning */}
        <div className="flex-1">
          {/* Top metadata tags */}
          <div className="flex flex-wrap items-center gap-3 mb-3">
            <span
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border shadow-xs bg-white dark:bg-slate-900 ${config.textLight} dark:${config.textDark} ${config.borderLight} dark:${config.borderDark}`}
            >
              <VerdictIcon className="w-4 h-4" />
              FINAL VERDICT: {config.label}
            </span>

            <span className="font-mono text-xs px-2.5 py-1 rounded-full bg-white/80 dark:bg-slate-900/80 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300">
              {consistency.replace(/_/g, " ")}
            </span>
          </div>

          {/* Reasoning Sentence */}
          <h2 className="font-heading font-bold text-xl sm:text-2xl text-slate-900 dark:text-white leading-snug">
            {reasoning}
          </h2>

          {/* Timestamp and Report ID */}
          <div className="mt-3 flex flex-wrap items-center gap-4 text-xs font-mono text-slate-500 dark:text-slate-400">
            <span className="flex items-center gap-1.5">
              <Hash className="w-3.5 h-3.5 text-slate-400" />
              ID: <span className="font-bold text-slate-700 dark:text-slate-200">{reportId}</span>
            </span>
            <span className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
              {formatDateTime(createdAt)}
            </span>
          </div>
        </div>

        {/* Right Column: Actions */}
        <div className="flex items-center gap-3 shrink-0 print:hidden">
          <button
            type="button"
            onClick={handlePrint}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl font-heading font-semibold text-xs bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-slate-700 shadow-xs transition-all hover:shadow"
          >
            <Printer className="w-4 h-4 text-slate-500" />
            Print / Save PDF
          </button>
        </div>
      </div>
    </div>
  );
};
