import React from "react";
import { AlertCircle, TrendingDown, HelpCircle } from "lucide-react";
import type { GeneralizationPoint } from "@/lib/types";

interface GeneralizationGapProps {
  data: GeneralizationPoint[];
  className?: string;
}

export const GeneralizationGap: React.FC<GeneralizationGapProps> = ({
  data,
  className = "",
}) => {
  return (
    <div className={`rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 shadow-sm ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4 pb-3 border-b border-slate-100 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-rose-50 dark:bg-rose-950/60 text-rose-600 dark:text-rose-400">
              <TrendingDown className="w-4 h-4" />
            </span>
            <h3 className="font-heading font-semibold text-slate-900 dark:text-white text-base">
              Generalization Gap Across Evaluation Distributions
            </h3>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Horizontal step chart tracking accuracy drop from benchmark training test distribution to wild unconstrained deepfakes.
          </p>
        </div>

        <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-rose-50 dark:bg-rose-950/40 text-rose-700 dark:text-rose-400 border border-rose-200 dark:border-rose-900">
          Δ -30.1% Total Drop
        </span>
      </div>

      {/* Horizontal Step Chart */}
      <div className="flex flex-col gap-4 py-2">
        {data.map((item, idx) => {
          const pct = item.accuracy * 100;
          // Determine color based on accuracy
          let barBg = "bg-emerald-500";
          let badgeColor = "text-emerald-700 bg-emerald-50 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-400";
          if (pct < 60) {
            barBg = "bg-rose-500";
            badgeColor = "text-rose-700 bg-rose-50 border-rose-200 dark:bg-rose-950/40 dark:text-rose-400";
          } else if (pct < 75) {
            barBg = "bg-amber-500";
            badgeColor = "text-amber-700 bg-amber-50 border-amber-200 dark:bg-amber-950/40 dark:text-amber-400";
          }

          return (
            <div key={idx} className="flex flex-col gap-1.5">
              <div className="flex flex-wrap items-center justify-between text-xs gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-slate-400 font-bold">#{idx + 1}</span>
                  <span className="font-heading font-semibold text-slate-800 dark:text-slate-200">
                    {item.set}
                  </span>
                  <span className="font-mono text-[11px] text-slate-400">
                    (n={item.n.toLocaleString()})
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`px-2 py-0.5 rounded-full text-xs font-mono font-bold border ${badgeColor}`}>
                    {pct.toFixed(1)}% Accuracy
                  </span>
                </div>
              </div>

              {/* Bar track */}
              <div className="w-full h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden relative">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${barBg}`}
                  style={{ width: `${pct}%` }}
                />
              </div>

              {/* Note / explanation */}
              <p className="text-[11px] text-slate-500 dark:text-slate-400 italic pl-1">
                {item.note}
              </p>
            </div>
          );
        })}
      </div>

      {/* Honesty & Research Callout Bar */}
      <div className="mt-5 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200 dark:border-slate-800 flex items-start gap-3">
        <HelpCircle className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
          <strong className="text-slate-900 dark:text-white font-medium">Why the Generalization Gap Matters: </strong>
          Modern deepfake detectors achieve high laboratory scores when testing on familiar generative pipelines (e.g., FaceForensics++), but suffer severe accuracy collapses when confronted with unseen generative models, distinct video codecs, or diverse real-world lighting. This empirical delta is why single-model decisions can never serve as definitive legal proof without human investigator corroboration.
        </div>
      </div>
    </div>
  );
};
