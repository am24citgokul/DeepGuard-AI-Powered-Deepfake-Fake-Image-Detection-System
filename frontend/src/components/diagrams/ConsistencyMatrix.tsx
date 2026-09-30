import React from "react";
import { Check, X, Minus, HelpCircle } from "lucide-react";
import type { ScreeningResult, Verdict } from "@/lib/types";
import { MODALITY_COLORS } from "@/lib/format";

interface ConsistencyMatrixProps {
  screening: ScreeningResult;
  className?: string;
}

export const ConsistencyMatrix: React.FC<ConsistencyMatrixProps> = ({
  screening,
  className = "",
}) => {
  const { modalities } = screening;

  const modalitiesList = [
    { key: "image" as const, label: "Image", color: MODALITY_COLORS.image, res: modalities.image },
    { key: "video" as const, label: "Video", color: MODALITY_COLORS.video, res: modalities.video },
    { key: "audio" as const, label: "Audio", color: MODALITY_COLORS.audio, res: modalities.audio },
  ];

  function getPairStatus(v1?: Verdict, v2?: Verdict) {
    if (!v1 || !v2) {
      return { status: "NA", label: "N/A", note: "One or both modalities were not provided." };
    }
    if (v1 === "UNCERTAIN" || v2 === "UNCERTAIN") {
      return { status: "UNCERTAIN", label: "Excluded", note: "Uncertain stream is excluded from voting agreement." };
    }
    if (v1 === v2) {
      return { status: "AGREE", label: "Agree", note: `Both branches concordantly flagged ${v1}.` };
    }
    return { status: "DISAGREE", label: "Disagree", note: `Contradiction: Branch 1 flagged ${v1} while Branch 2 flagged ${v2}.` };
  }

  return (
    <div className={`rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 shadow-sm ${className}`}>
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100 dark:border-slate-800">
        <div>
          <h3 className="font-heading font-semibold text-slate-900 dark:text-white text-base">
            Cross-Modal Pairwise Consistency Matrix
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Evaluates concordance between spatial, temporal, and acoustic decision vectors.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400 font-medium">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span> Agree
          </span>
          <span className="flex items-center gap-1 text-rose-600 dark:text-rose-400 font-medium">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span> Disagree
          </span>
          <span className="flex items-center gap-1 text-slate-400 font-medium">
            <span className="w-2.5 h-2.5 rounded-full bg-slate-300 dark:bg-slate-700"></span> N/A
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-center border-collapse">
          <thead>
            <tr>
              <th className="p-2 text-xs font-mono text-slate-400 text-left">Input Pair</th>
              {modalitiesList.map((m) => (
                <th key={m.key} className="p-3 text-xs font-heading font-semibold" style={{ color: m.color }}>
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {modalitiesList.map((row) => (
              <tr key={row.key} className="border-t border-slate-100 dark:border-slate-800/80">
                <td className="p-3 text-xs font-heading font-semibold text-left" style={{ color: row.color }}>
                  {row.label}
                </td>
                {modalitiesList.map((col) => {
                  const isDiagonal = row.key === col.key;
                  if (isDiagonal) {
                    const hasData = Boolean(row.res);
                    return (
                      <td key={col.key} className="p-2">
                        <div className="py-2.5 px-3 rounded-xl bg-slate-100/70 dark:bg-slate-800/50 text-slate-400 dark:text-slate-500 text-xs font-mono">
                          {hasData ? (row.res?.verdict || "—") : "Omitted"}
                        </div>
                      </td>
                    );
                  }

                  const pair = getPairStatus(row.res?.verdict, col.res?.verdict);

                  return (
                    <td key={col.key} className="p-2">
                      <div
                        title={pair.note}
                        className={`py-2 px-3 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition-transform hover:scale-105 cursor-help ${
                          pair.status === "AGREE"
                            ? "bg-emerald-50 text-emerald-700 border border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-400 dark:border-emerald-800/60"
                            : pair.status === "DISAGREE"
                            ? "bg-rose-50 text-rose-700 border border-rose-200 dark:bg-rose-950/40 dark:text-rose-400 dark:border-rose-800/60"
                            : pair.status === "UNCERTAIN"
                            ? "bg-slate-100 text-slate-600 border border-slate-300 uncertain-hatch dark:bg-slate-800 dark:text-slate-400"
                            : "bg-slate-50 text-slate-400 border border-dashed border-slate-200 dark:bg-slate-850 dark:text-slate-600 dark:border-slate-800"
                        }`}
                      >
                        {pair.status === "AGREE" && <Check className="w-3.5 h-3.5" />}
                        {pair.status === "DISAGREE" && <X className="w-3.5 h-3.5" />}
                        {pair.status === "UNCERTAIN" && <HelpCircle className="w-3.5 h-3.5" />}
                        {pair.status === "NA" && <Minus className="w-3.5 h-3.5" />}
                        <span>{pair.label}</span>
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
