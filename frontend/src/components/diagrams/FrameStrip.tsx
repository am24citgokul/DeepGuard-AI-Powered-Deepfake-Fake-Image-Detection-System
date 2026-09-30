import React, { useState } from "react";
import { Film, AlertCircle, CheckCircle2 } from "lucide-react";
import { MODALITY_COLORS } from "@/lib/format";

interface FrameStripProps {
  frameScores?: number[];
  framesAnalyzed?: number;
  method?: string;
  className?: string;
}

export const FrameStrip: React.FC<FrameStripProps> = ({
  frameScores = [0.85, 0.88, 0.91, 0.94, 0.92, 0.90, 0.88, 0.86, 0.82, 0.79, 0.83, 0.88],
  framesAnalyzed,
  method = "CNN-GRU Temporal Model",
  className = "",
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const totalFrames = framesAnalyzed || frameScores.length;

  return (
    <div className={`rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40 p-4 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <Film className="w-4 h-4 text-teal-500" />
          <span className="font-heading font-semibold text-xs text-slate-800 dark:text-slate-200">
            Temporal Frame Sequence Analysis
          </span>
          <span className="font-mono text-[11px] px-2 py-0.5 rounded-full bg-teal-50 text-teal-700 dark:bg-teal-950/60 dark:text-teal-300 border border-teal-200 dark:border-teal-800">
            {totalFrames} frames sampled
          </span>
        </div>

        <span className="text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
          {method}
        </span>
      </div>

      {/* Frame Strip Carousel / Grid */}
      <div className="flex items-end gap-1.5 overflow-x-auto pb-2 pt-1 scrollbar-thin">
        {frameScores.map((score, index) => {
          // Authentic vs Manipulated score (assume >0.5 manipulation or authentic based on context)
          const isManipulated = score > 0.5;
          const barColor = isManipulated ? "#DC2626" : "#16A34A";
          const percent = (score * 100).toFixed(1);

          return (
            <div
              key={index}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
              className="flex flex-col items-center shrink-0 w-11 sm:w-12 group cursor-pointer"
            >
              {/* Frame Tile (Geometric Silhouette Neutral Placeholder) */}
              <div
                className={`w-full aspect-square rounded-md overflow-hidden border transition-all ${
                  hoveredIndex === index
                    ? "border-teal-500 shadow-sm scale-105"
                    : "border-slate-300 dark:border-slate-700"
                } bg-slate-100 dark:bg-slate-800 flex items-center justify-center relative`}
              >
                <svg viewBox="0 0 40 40" className="w-full h-full text-slate-400 dark:text-slate-600 p-1">
                  <rect x="6" y="6" width="28" height="28" rx="4" fill="none" stroke="currentColor" strokeWidth="1.5" />
                  <circle cx="20" cy="17" r="6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="2 2" />
                  <path d="M12 32 C12 25, 28 25, 28 32" fill="none" stroke="currentColor" strokeWidth="1.5" />
                </svg>
                <span className="absolute bottom-0.5 right-1 text-[8px] font-mono font-bold text-slate-500">
                  #{index + 1}
                </span>
              </div>

              {/* Per-frame Score Bar */}
              <div className="w-full mt-1.5 flex flex-col gap-0.5">
                <div className="w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${score * 100}%`,
                      backgroundColor: barColor,
                    }}
                  />
                </div>
                <span className="text-[9px] font-mono text-center text-slate-500 dark:text-slate-400 truncate">
                  {percent}%
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {hoveredIndex !== null && (
        <div className="mt-2 text-xs font-mono flex items-center justify-between text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-850 p-2 rounded-lg border border-slate-200 dark:border-slate-700 animate-fadeIn">
          <span>Frame #{hoveredIndex + 1} Saliency Anomaly</span>
          <span className="font-bold">
            Risk: {(frameScores[hoveredIndex] * 100).toFixed(1)}% ({frameScores[hoveredIndex] > 0.5 ? "Manipulated" : "Authentic"})
          </span>
        </div>
      )}
    </div>
  );
};
