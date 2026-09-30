import React from "react";
import { Check, Loader2 } from "lucide-react";

export type StepState = "pending" | "current" | "completed";

export interface StepperStep {
  id: string;
  label: string;
  sublabel: string;
}

const DEFAULT_STEPS: StepperStep[] = [
  { id: "upload", label: "Upload Received", sublabel: "Multipart payload verified" },
  { id: "preprocess", label: "Preprocessing", sublabel: "Face alignment & resampling" },
  { id: "inference", label: "Model Inference", sublabel: "Evaluating active modality branches" },
  { id: "evidence", label: "Evidence Extraction", sublabel: "Grad-CAM & ELA residual mapping" },
  { id: "fusion", label: "Cross-Modal Fusion", sublabel: "Probabilistic consensus calculation" },
];

interface StepperProps {
  currentStepIndex: number; // 0 to 4
  steps?: StepperStep[];
  className?: string;
}

export const Stepper: React.FC<StepperProps> = ({
  currentStepIndex,
  steps = DEFAULT_STEPS,
  className = "",
}) => {
  return (
    <div className={`p-6 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm ${className}`}>
      <div className="flex items-center justify-between mb-6 pb-2 border-b border-slate-100 dark:border-slate-800">
        <div>
          <h4 className="font-heading font-semibold text-sm text-slate-900 dark:text-white">
            Forensic Screening Execution Pipeline
          </h4>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Real-time inference stepper tracking data flow across evaluation lanes.
          </p>
        </div>
        <span className="font-mono text-xs px-2.5 py-1 rounded-full bg-violet-50 text-violet-700 dark:bg-violet-950 dark:text-violet-300">
          Step {Math.min(currentStepIndex + 1, steps.length)} of {steps.length}
        </span>
      </div>

      <div className="relative">
        {/* Progress track */}
        <div className="hidden md:block absolute top-4 left-6 right-6 h-0.5 bg-slate-200 dark:bg-slate-800 -z-0">
          <div
            className="h-full bg-violet-600 transition-all duration-500"
            style={{ width: `${(currentStepIndex / (steps.length - 1)) * 100}%` }}
          />
        </div>

        {/* Steps Grid */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 relative z-10">
          {steps.map((step, idx) => {
            const isCompleted = idx < currentStepIndex;
            const isCurrent = idx === currentStepIndex;

            return (
              <div
                key={step.id}
                className="flex md:flex-col items-center md:items-center gap-3 text-left md:text-center"
              >
                {/* Step Circle */}
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center font-mono text-xs font-bold transition-all shrink-0 ${
                    isCompleted
                      ? "bg-emerald-500 text-white shadow-xs"
                      : isCurrent
                      ? "bg-violet-600 text-white ring-4 ring-violet-100 dark:ring-violet-950/60 animate-pulse"
                      : "bg-slate-100 dark:bg-slate-800 text-slate-400 border border-slate-200 dark:border-slate-700"
                  }`}
                >
                  {isCompleted ? (
                    <Check className="w-4 h-4 stroke-[3]" />
                  ) : isCurrent ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    idx + 1
                  )}
                </div>

                {/* Step Labels */}
                <div>
                  <div
                    className={`font-heading font-semibold text-xs transition-colors ${
                      isCurrent
                        ? "text-violet-600 dark:text-violet-400"
                        : isCompleted
                        ? "text-slate-900 dark:text-white"
                        : "text-slate-400 dark:text-slate-600"
                    }`}
                  >
                    {step.label}
                  </div>
                  <div className="text-[11px] text-slate-400 dark:text-slate-500 mt-0.5 line-clamp-1">
                    {step.sublabel}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
