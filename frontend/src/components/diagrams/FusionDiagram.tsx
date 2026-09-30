import React from "react";
import { 
  Image as ImageIcon, 
  Video, 
  Mic, 
  GitMerge,
  ShieldCheck,
  AlertTriangle,
  ShieldAlert,
  HelpCircle,
  Slash
} from "lucide-react";
import type { ScreeningResult, ModalityResult } from "@/lib/types";
import { MODALITY_COLORS, VERDICT_COLORS, getVerdictConfig } from "@/lib/format";
import { ConfidenceGauge } from "./ConfidenceGauge";

interface FusionDiagramProps {
  screening: ScreeningResult;
  className?: string;
}

export const FusionDiagram: React.FC<FusionDiagramProps> = ({
  screening,
  className = "",
}) => {
  const { modalities, final_verdict } = screening;
  const finalConfig = getVerdictConfig(final_verdict);

  const lanes = [
    {
      key: "image" as const,
      label: "Image Branch",
      modalityColor: MODALITY_COLORS.image,
      icon: ImageIcon,
      result: modalities.image,
    },
    {
      key: "video" as const,
      label: "Video Branch",
      modalityColor: MODALITY_COLORS.video,
      icon: Video,
      result: modalities.video,
    },
    {
      key: "audio" as const,
      label: "Audio Branch",
      modalityColor: MODALITY_COLORS.audio,
      icon: Mic,
      result: modalities.audio,
    },
  ];

  return (
    <div className={`rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 shadow-sm ${className}`}>
      <div className="flex items-center justify-between mb-6 pb-3 border-b border-slate-100 dark:border-slate-800">
        <div>
          <h3 className="font-heading font-semibold text-slate-900 dark:text-white text-base">
            Cross-Modal Evidence Fusion Architecture
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Active inputs stream into the Bayesian fusion kernel. Unprovided or uncertain modalities are systematically isolated.
          </p>
        </div>
        <span className="font-mono text-xs px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
          Convergence Engine
        </span>
      </div>

      {/* Main Diagram Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* Left Side: Three Lanes (Span 7) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          {lanes.map((lane) => {
            const hasData = Boolean(lane.result);
            const verdict = lane.result?.verdict;
            const isUncertain = verdict === "UNCERTAIN";
            const verdictConf = verdict ? getVerdictConfig(verdict) : null;
            const VerdictIcon = verdictConf?.icon || HelpCircle;

            if (!hasData) {
              // Lane with no file is drawn dashed and grey ("not provided")
              return (
                <div
                  key={lane.key}
                  className="rounded-xl border-2 border-dashed border-slate-300 dark:border-slate-700 bg-slate-50/50 dark:bg-slate-900/40 p-4 flex items-center justify-between opacity-70"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-slate-200 dark:bg-slate-800 flex items-center justify-center text-slate-400">
                      <lane.icon className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-slate-500 dark:text-slate-400">
                        {lane.label}
                      </div>
                      <div className="text-xs text-slate-400 dark:text-slate-500 font-mono">
                        Not provided in this screening
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-200/80 dark:bg-slate-800 text-slate-500 text-xs font-mono">
                    <Slash className="w-3.5 h-3.5" />
                    <span>Omitted</span>
                  </div>
                </div>
              );
            }

            // Lane with UNCERTAIN verdict is hollow/hatched with caption "excluded from vote"
            if (isUncertain) {
              return (
                <div
                  key={lane.key}
                  className="rounded-xl border border-slate-400 dark:border-slate-600 uncertain-hatch p-4 flex items-center justify-between relative overflow-hidden"
                  style={{ borderLeft: `5px solid ${lane.modalityColor}` }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-10 h-10 rounded-lg flex items-center justify-center text-white"
                      style={{ backgroundColor: lane.modalityColor }}
                    >
                      <lane.icon className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-slate-900 dark:text-white flex items-center gap-2">
                        {lane.label}
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-200 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                          {lane.result?.model}
                        </span>
                      </div>
                      <div className="text-xs text-slate-600 dark:text-slate-300 font-medium mt-0.5 flex items-center gap-1">
                        <HelpCircle className="w-3.5 h-3.5 text-slate-500" />
                        <span>UNCERTAIN</span>
                        <span className="font-mono text-slate-400">• {(lane.result!.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-col items-end">
                    <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border border-slate-300 dark:border-slate-600">
                      Excluded from vote
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono mt-1">
                      Low confidence margin
                    </span>
                  </div>
                </div>
              );
            }

            // Normal Lane with data
            return (
              <div
                key={lane.key}
                className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-850 p-3.5 sm:p-4 flex items-center justify-between shadow-xs transition-all hover:border-slate-300 dark:hover:border-slate-600"
                style={{ borderLeft: `5px solid ${lane.modalityColor}` }}
              >
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center text-white shrink-0 shadow-xs"
                    style={{ backgroundColor: lane.modalityColor }}
                  >
                    <lane.icon className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-slate-900 dark:text-white flex items-center gap-2">
                      {lane.label}
                    </div>
                    <div className="text-[11px] text-slate-500 dark:text-slate-400 font-mono truncate max-w-[200px] sm:max-w-xs">
                      {lane.result?.model}
                    </div>
                    {/* Verdict Pill */}
                    <div className="mt-1.5 flex items-center gap-2">
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold border ${verdictConf?.bgLight} ${verdictConf?.textLight} ${verdictConf?.borderLight} dark:${verdictConf?.bgDark} dark:${verdictConf?.textDark} dark:${verdictConf?.borderDark}`}
                      >
                        <VerdictIcon className="w-3 h-3" />
                        {verdictConf?.label}
                      </span>
                      {lane.result?.predicted_class && (
                        <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                          [{lane.result.predicted_class}]
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Semicircle confidence gauge */}
                <div className="shrink-0 pl-2">
                  <ConfidenceGauge
                    confidence={lane.result!.confidence}
                    color={verdictConf?.color}
                    size={84}
                    label="Score"
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Center: Convergence SVG graphic (Span 2) */}
        <div className="hidden lg:flex lg:col-span-2 items-center justify-center h-full">
          <svg width="100" height="240" viewBox="0 0 100 240" className="overflow-visible">
            {/* Image line */}
            <path
              d="M 0,40 C 50,40 50,120 100,120"
              fill="none"
              stroke={lanes[0].result ? (lanes[0].result.verdict === "UNCERTAIN" ? "#64748B" : MODALITY_COLORS.image) : "#CBD5E1"}
              strokeWidth="2.5"
              strokeDasharray={!lanes[0].result || lanes[0].result.verdict === "UNCERTAIN" ? "4 4" : undefined}
            />
            {/* Video line */}
            <path
              d="M 0,120 L 100,120"
              fill="none"
              stroke={lanes[1].result ? (lanes[1].result.verdict === "UNCERTAIN" ? "#64748B" : MODALITY_COLORS.video) : "#CBD5E1"}
              strokeWidth="2.5"
              strokeDasharray={!lanes[1].result || lanes[1].result.verdict === "UNCERTAIN" ? "4 4" : undefined}
            />
            {/* Audio line */}
            <path
              d="M 0,200 C 50,200 50,120 100,120"
              fill="none"
              stroke={lanes[2].result ? (lanes[2].result.verdict === "UNCERTAIN" ? "#64748B" : MODALITY_COLORS.audio) : "#CBD5E1"}
              strokeWidth="2.5"
              strokeDasharray={!lanes[2].result || lanes[2].result.verdict === "UNCERTAIN" ? "4 4" : undefined}
            />
            {/* Node entry arrow */}
            <polygon points="94,116 100,120 94,124" fill={MODALITY_COLORS.fusion} />
          </svg>
        </div>

        {/* Right Side: Fusion Node with ring colour = final verdict (Span 3) */}
        <div className="lg:col-span-3 flex flex-col items-center justify-center text-center p-6 rounded-2xl bg-gradient-to-b from-slate-50 to-slate-100/60 dark:from-slate-800/50 dark:to-slate-900 border border-slate-200 dark:border-slate-800">
          <div className="relative mb-4">
            {/* Outer Ring with Verdict Color */}
            <div
              className="w-24 h-24 rounded-full flex items-center justify-center transition-all shadow-md"
              style={{
                border: `5px solid ${finalConfig.color}`,
                backgroundColor: "rgba(139, 92, 246, 0.12)",
              }}
            >
              <div
                className="w-16 h-16 rounded-full flex items-center justify-center text-white shadow-inner"
                style={{ backgroundColor: MODALITY_COLORS.fusion }}
              >
                <GitMerge className="w-8 h-8" />
              </div>
            </div>
            
            {/* Ring Badge */}
            <div
              className="absolute -bottom-2 left-1/2 -translate-x-1/2 px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold text-white shadow-xs"
              style={{ backgroundColor: finalConfig.color }}
            >
              FUSED
            </div>
          </div>

          <div className="font-heading font-bold text-base text-slate-900 dark:text-white">
            Multimodal Fusion Node
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1 font-mono">
            {screening.consistency.replace(/_/g, " ")}
          </div>

          <div className="mt-4 w-full pt-4 border-t border-slate-200 dark:border-slate-700/80">
            <span
              className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${finalConfig.bgLight} ${finalConfig.textLight} ${finalConfig.borderLight} dark:${finalConfig.bgDark} dark:${finalConfig.textDark} dark:${finalConfig.borderDark}`}
            >
              <finalConfig.icon className="w-3.5 h-3.5" />
              {finalConfig.label}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
