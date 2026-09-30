import React from "react";
import { 
  Image as ImageIcon, 
  Video, 
  Mic, 
  Layers, 
  Eye, 
  Flame, 
  Sparkles,
  HelpCircle,
  AlertTriangle
} from "lucide-react";
import type { ModalityResult, ModalityType } from "@/lib/types";
import { 
  MODALITY_COLORS, 
  getVerdictConfig, 
  getMaturityChip,
  formatPercent 
} from "@/lib/format";
import { ConfidenceGauge } from "../diagrams/ConfidenceGauge";
import { FrameStrip } from "../diagrams/FrameStrip";
import { Waveform } from "../diagrams/Waveform";
import { SVG_SILHOUETTE_ORIGINAL } from "@/lib/mockData";

interface EvidenceCardProps {
  modality: ModalityType;
  title: string;
  result?: ModalityResult;
  className?: string;
}

export const EvidenceCard: React.FC<EvidenceCardProps> = ({
  modality,
  title,
  result,
  className = "",
}) => {
  const color = MODALITY_COLORS[modality];
  const Icon = modality === "image" ? ImageIcon : modality === "video" ? Video : Mic;

  if (!result) {
    return (
      <div
        className={`rounded-2xl border border-dashed border-slate-300 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 p-6 flex flex-col items-center justify-center text-center ${className}`}
        style={{ borderTop: `4px solid ${color}` }}
      >
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center text-white mb-3 opacity-60"
          style={{ backgroundColor: color }}
        >
          <Icon className="w-6 h-6" />
        </div>
        <h4 className="font-heading font-semibold text-sm text-slate-600 dark:text-slate-400">
          {title} — Not Evaluated
        </h4>
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-1 max-w-xs">
          No file was provided for this stream. The fusion kernel operated with remaining inputs.
        </p>
      </div>
    );
  }

  const verdictConf = getVerdictConfig(result.verdict);
  const maturityChip = getMaturityChip(result.maturity);
  const VerdictIcon = verdictConf.icon;

  return (
    <div
      className={`rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm overflow-hidden flex flex-col ${className}`}
      style={{ borderTop: `4px solid ${color}` }}
    >
      {/* Header */}
      <div className="p-4 sm:p-5 border-b border-slate-100 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center text-white shrink-0 shadow-xs"
            style={{ backgroundColor: color }}
          >
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-heading font-bold text-base text-slate-900 dark:text-white">
                {title}
              </h4>
              <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${maturityChip.color}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${maturityChip.dot}`}></span>
                {maturityChip.label}
              </span>
            </div>
            <p className="text-xs font-mono text-slate-500 dark:text-slate-400 mt-0.5">
              Model: {result.model}
            </p>
          </div>
        </div>

        {/* Verdict Badge */}
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${verdictConf.bgLight} ${verdictConf.textLight} ${verdictConf.borderLight} dark:${verdictConf.bgDark} dark:${verdictConf.textDark} dark:${verdictConf.borderDark}`}
          >
            <VerdictIcon className="w-3.5 h-3.5" />
            {verdictConf.label}
          </span>
          {result.predicted_class && (
            <span className="font-mono text-xs px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
              Class: {result.predicted_class}
            </span>
          )}
        </div>
      </div>

      {/* Main Body */}
      <div className="p-4 sm:p-6 flex-1 flex flex-col gap-5">
        {/* Top Summary Row: Confidence Gauge + Note */}
        <div className="flex flex-col sm:flex-row items-center sm:items-start justify-between gap-4 p-4 rounded-xl bg-slate-50/70 dark:bg-slate-800/40 border border-slate-200/80 dark:border-slate-800">
          <div className="flex-1">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
              Forensic Observation & Rationale
            </span>
            <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed font-sans">
              {result.note || "Forensic vector computed from normalized model inference pass."}
            </p>
            {result.verdict === "UNCERTAIN" && (
              <div className="mt-2 text-xs font-medium text-slate-500 dark:text-slate-400 flex items-center gap-1">
                <HelpCircle className="w-3.5 h-3.5" />
                Confidence score below voting certainty threshold (0.55). Excluded from final consensus.
              </div>
            )}
          </div>

          <div className="shrink-0 flex items-center gap-3">
            <ConfidenceGauge
              confidence={result.confidence}
              color={verdictConf.color}
              size={96}
              isUncertain={result.verdict === "UNCERTAIN"}
              label="Confidence"
            />
          </div>
        </div>

        {/* Modality-Specific Evidence Displays */}
        {modality === "image" && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-heading font-semibold text-slate-800 dark:text-slate-200 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-blue-500" />
                Explainability Triptych (Original | Grad-CAM | ELA Map)
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                Pixel-level saliency localization
              </span>
            </div>

            {/* Three panels side by side */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {/* Panel 1: Original */}
              <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-850 flex flex-col">
                <div className="px-3 py-1.5 bg-slate-200/60 dark:bg-slate-800 text-[11px] font-mono font-medium text-slate-700 dark:text-slate-300 flex items-center justify-between">
                  <span className="flex items-center gap-1">
                    <Eye className="w-3 h-3 text-blue-500" /> Original Input
                  </span>
                  <span className="text-[10px] text-slate-400">224x224</span>
                </div>
                <div className="h-44 flex items-center justify-center p-2 bg-slate-50 dark:bg-slate-900">
                  <img
                    src={SVG_SILHOUETTE_ORIGINAL}
                    alt="Original neutral geometric illustration"
                    className="h-full w-auto object-contain rounded-md"
                  />
                </div>
              </div>

              {/* Panel 2: Grad-CAM heatmap */}
              <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-850 flex flex-col">
                <div className="px-3 py-1.5 bg-slate-200/60 dark:bg-slate-800 text-[11px] font-mono font-medium text-slate-700 dark:text-slate-300 flex items-center justify-between">
                  <span className="flex items-center gap-1">
                    <Flame className="w-3 h-3 text-rose-500" /> Grad-CAM Heatmap
                  </span>
                  <span className="text-[10px] text-slate-400">Target Layer</span>
                </div>
                <div className="h-44 flex items-center justify-center p-2 bg-slate-900">
                  {result.gradcam_url ? (
                    <img
                      src={result.gradcam_url}
                      alt="Grad-CAM activation saliency heatmap"
                      className="h-full w-auto object-contain rounded-md"
                    />
                  ) : (
                    <div className="text-center p-4 text-xs font-mono text-slate-400">
                      Explainability unavailable for this prediction
                    </div>
                  )}
                </div>
              </div>

              {/* Panel 3: ELA error map */}
              <div className="rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-850 flex flex-col">
                <div className="px-3 py-1.5 bg-slate-200/60 dark:bg-slate-800 text-[11px] font-mono font-medium text-slate-700 dark:text-slate-300 flex items-center justify-between">
                  <span className="flex items-center gap-1">
                    <Sparkles className="w-3 h-3 text-teal-500" /> ELA Error Map
                  </span>
                  <span className="text-[10px] text-slate-400">10x Residual</span>
                </div>
                <div className="h-44 flex items-center justify-center p-2 bg-slate-950">
                  {result.ela_url ? (
                    <img
                      src={result.ela_url}
                      alt="Error level analysis residual map"
                      className="h-full w-auto object-contain rounded-md"
                    />
                  ) : (
                    <div className="text-center p-4 text-xs font-mono text-slate-400">
                      Explainability unavailable for this prediction
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {modality === "video" && (
          <FrameStrip
            frameScores={result.frame_scores}
            framesAnalyzed={result.frames_analyzed}
            method={result.model}
          />
        )}

        {modality === "audio" && (
          <div className="space-y-4">
            {/* Probability Breakdown Bar */}
            {result.real_probability !== undefined && result.fake_probability !== undefined && (
              <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs font-heading font-semibold">
                  <span className="text-emerald-600 dark:text-emerald-400">
                    REAL {(result.real_probability * 100).toFixed(1)}%
                  </span>
                  <span className="text-[11px] font-mono text-slate-400">
                    Operating Threshold: {(result.threshold ?? 0.50).toFixed(2)}
                  </span>
                  <span className="text-rose-600 dark:text-rose-400">
                    FAKE {(result.fake_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden flex">
                  <div
                    className="h-full bg-emerald-500 transition-all"
                    style={{ width: `${result.real_probability * 100}%` }}
                    title={`Real: ${(result.real_probability * 100).toFixed(1)}%`}
                  />
                  <div
                    className="h-full bg-rose-500 transition-all"
                    style={{ width: `${result.fake_probability * 100}%` }}
                    title={`Fake: ${(result.fake_probability * 100).toFixed(1)}%`}
                  />
                </div>
              </div>
            )}

            <Waveform
              features={result.features}
              duration={result.audio_duration_sec}
              sampleRate={result.sampling_rate}
              maturity={result.maturity}
            />
          </div>
        )}
      </div>
    </div>
  );
};
