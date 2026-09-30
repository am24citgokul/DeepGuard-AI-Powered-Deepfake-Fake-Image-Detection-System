import React from "react";
import { Mic, Activity, AlertTriangle } from "lucide-react";
import type { Maturity } from "@/lib/types";
import { MODALITY_COLORS } from "@/lib/format";

interface WaveformProps {
  features?: Record<string, number>;
  duration?: number;
  sampleRate?: number;
  channels?: string;
  maturity?: Maturity;
  className?: string;
}

export const Waveform: React.FC<WaveformProps> = ({
  features = {
    mean_spectral_flatness: 0.0245,
    mean_zcr: 0.119,
    mean_high_mfcc_var: 142.1,
  },
  duration,
  sampleRate = 16000,
  channels = "Mono",
  maturity = "TRAINED",
  className = "",
}) => {
  // Generate pseudo-waveform bars based on deterministic pattern
  const bars = [
    25, 45, 70, 85, 60, 40, 65, 90, 75, 50, 30, 60, 80, 95, 85, 60, 45, 75, 90, 65, 40, 20, 45, 70, 85,
    60, 40, 65, 80, 55, 35, 60, 85, 70, 45, 25, 50, 75, 90, 60, 40, 30, 55, 75, 95, 70, 50, 35, 20, 15,
  ];

  const featureLabels: Record<string, { label: string; max: number; unit?: string }> = {
    mean_spectral_flatness: { label: "Spectral Flatness", max: 0.05 },
    mean_zcr: { label: "Zero Crossing Rate (ZCR)", max: 0.25 },
    mean_high_mfcc_var: { label: "High MFCC Variance", max: 200 },
    formant_jitter: { label: "Glottal Jitter", max: 0.1 },
  };

  return (
    <div className={`rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/40 p-4 relative overflow-hidden ${className}`}>
      {/* Supporting Evidence Badge */}
      <div className="mb-3 px-3 py-1.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-700 dark:text-indigo-300 text-xs flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <Activity className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
          <span className="font-semibold uppercase tracking-wider text-[10px]">SUPPORTING AUDIO FORENSICS</span>
        </div>
        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-600 dark:text-indigo-300">
          SUPPORTING EVIDENCE (Non-verdict)
        </span>
      </div>

      {/* Header with audio metadata */}
      <div className="flex flex-wrap items-center justify-between mb-3 gap-2">
        <div className="flex items-center gap-2">
          <Mic className="w-4 h-4 text-pink-500" />
          <span className="font-heading font-semibold text-xs text-slate-800 dark:text-slate-200">
            Acoustic Spectrum & Waveform Forensics
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] font-mono text-slate-500 dark:text-slate-400">
          <span>{sampleRate / 1000} kHz</span>
          <span>•</span>
          <span>{channels}</span>
          {duration !== undefined && (
            <>
              <span>•</span>
              <span>{duration.toFixed(2)}s</span>
            </>
          )}
        </div>
      </div>

      {/* Waveform Visualization */}
      <div className="w-full h-16 bg-white dark:bg-slate-850 rounded-lg border border-slate-200 dark:border-slate-800 p-2 flex items-center justify-between gap-1 overflow-hidden">
        {bars.map((h, i) => (
          <div
            key={i}
            className="flex-1 bg-pink-400/80 dark:bg-pink-500/80 rounded-full transition-all hover:bg-pink-600"
            style={{
              height: `${h}%`,
              opacity: 0.5 + (h / 200),
            }}
          />
        ))}
      </div>

      {/* Extracted Features Bar Chart */}
      {features && Object.keys(features).length > 0 && (
        <div className="mt-4 pt-3 border-t border-slate-200 dark:border-slate-800">
          <div className="text-[11px] font-mono text-slate-500 dark:text-slate-400 mb-2 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-pink-500" />
            Extracted DSP Feature Metrics:
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {Object.entries(features).map(([key, val]) => {
              const meta = featureLabels[key] || { label: key, max: Math.max(val * 1.5, 1) };
              const ratio = Math.min(1, Math.max(0, val / meta.max));
              return (
                <div key={key} className="bg-white dark:bg-slate-850 p-2.5 rounded-lg border border-slate-200/80 dark:border-slate-800">
                  <div className="flex justify-between items-center text-[10px] text-slate-500 mb-1">
                    <span className="truncate pr-1">{meta.label}</span>
                    <span className="font-mono font-bold text-slate-800 dark:text-slate-200">
                      {typeof val === "number" ? val.toFixed(4) : val}
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-pink-500 rounded-full transition-all"
                      style={{ width: `${ratio * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
