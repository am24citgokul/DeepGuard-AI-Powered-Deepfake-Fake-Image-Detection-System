import React, { useState } from "react";
import type { ConfusionMatrixData } from "@/lib/types";

interface ConfusionHeatmapProps {
  data: ConfusionMatrixData;
  className?: string;
}

export const ConfusionHeatmap: React.FC<ConfusionHeatmapProps> = ({
  data,
  className = "",
}) => {
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
    count: number;
    pct: number;
  } | null>(null);

  const { labels, matrix } = data;

  // Compute row totals for percentage calculation
  const rowTotals = matrix.map((row) => row.reduce((sum, val) => sum + val, 0));
  
  // Find global max value for intensity scaling
  const maxVal = Math.max(...matrix.flatMap((row) => row));

  return (
    <div className={`rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 shadow-sm ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4 pb-3 border-b border-slate-100 dark:border-slate-800">
        <div>
          <h3 className="font-heading font-semibold text-slate-900 dark:text-white text-base">
            Multi-Class Confusion Matrix Heatmap
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Row indicates Ground Truth class; Column indicates Model Prediction.
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
          <span>0</span>
          <div className="w-24 h-2.5 rounded bg-gradient-to-r from-blue-50 to-blue-700 dark:from-slate-800 dark:to-blue-500 border border-slate-200 dark:border-slate-700"></div>
          <span>{maxVal}</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[500px]">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-2 text-[10px] font-mono text-slate-400 uppercase tracking-wider text-left">
                  Actual \ Pred
                </th>
                {labels.map((lbl, idx) => (
                  <th key={idx} className="p-2 text-xs font-heading font-semibold text-slate-700 dark:text-slate-300 text-center">
                    <span className="truncate block max-w-[70px] sm:max-w-none">{lbl}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, rIdx) => {
                const total = rowTotals[rIdx] || 1;
                return (
                  <tr key={rIdx}>
                    <td className="p-2 text-xs font-heading font-semibold text-slate-700 dark:text-slate-300 text-left">
                      {labels[rIdx]}
                    </td>
                    {row.map((val, cIdx) => {
                      const isDiagonal = rIdx === cIdx;
                      const ratio = val / maxVal;
                      const pct = (val / total) * 100;

                      // Calculate cell background
                      const isHovered = hoveredCell?.row === rIdx && hoveredCell?.col === cIdx;

                      return (
                        <td
                          key={cIdx}
                          onMouseEnter={() => setHoveredCell({ row: rIdx, col: cIdx, count: val, pct })}
                          onMouseLeave={() => setHoveredCell(null)}
                          className="p-1 text-center cursor-pointer transition-all"
                        >
                          <div
                            className={`py-3 px-2 rounded-lg font-mono text-xs transition-all ${
                              isHovered ? "ring-2 ring-violet-500 scale-105 z-10 relative" : ""
                            } ${isDiagonal ? "font-bold ring-1 ring-blue-400/40" : ""}`}
                            style={{
                              backgroundColor: `rgba(59, 130, 246, ${Math.max(0.06, ratio)})`,
                              color: ratio > 0.4 ? "#FFFFFF" : "inherit",
                            }}
                          >
                            <div>{val}</div>
                            <div className="text-[10px] opacity-75 font-sans">
                              {pct.toFixed(0)}%
                            </div>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {hoveredCell && (
        <div className="mt-4 p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs flex items-center justify-between animate-fadeIn font-mono">
          <div>
            <span className="text-slate-400">Actual: </span>
            <span className="font-bold text-slate-800 dark:text-white">{labels[hoveredCell.row]}</span>
            <span className="text-slate-400 mx-2">➔ Pred: </span>
            <span className="font-bold text-slate-800 dark:text-white">{labels[hoveredCell.col]}</span>
          </div>
          <div>
            <span className="font-bold text-blue-600 dark:text-blue-400">{hoveredCell.count} samples </span>
            <span className="text-slate-500">({hoveredCell.pct.toFixed(1)}% of row)</span>
          </div>
        </div>
      )}
    </div>
  );
};
