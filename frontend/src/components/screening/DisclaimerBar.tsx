import React from "react";
import { ShieldAlert, Info } from "lucide-react";

interface DisclaimerBarProps {
  className?: string;
  variant?: "result" | "general";
}

export const DisclaimerBar: React.FC<DisclaimerBarProps> = ({
  className = "",
  variant = "result",
}) => {
  return (
    <div
      className={`rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-900/60 p-4 flex items-start gap-3 text-xs leading-relaxed text-slate-600 dark:text-slate-400 ${className}`}
    >
      <div className="p-1 rounded-md bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5">
        <ShieldAlert className="w-4 h-4" />
      </div>

      <div className="flex-1">
        <strong className="font-heading font-semibold text-slate-800 dark:text-slate-200 block mb-0.5">
          AI Forensics Boundary & Responsible Use Disclaimer
        </strong>
        {variant === "result" ? (
          <p>
            This is an AI-assisted screening result, not proof of fraud. <strong>REVIEW REQUIRED</strong> and{" "}
            <strong>HIGH RISK</strong> cases need human verification. Statistical classifiers are susceptible to compression artifacts, lighting anomalies, and out-of-distribution biometric shifts.
          </p>
        ) : (
          <p>
            DeepGuard X is an AI screening aid that supports human review. Deepfake detection models cannot guarantee 100% detection of zero-day synthesis techniques and must be corroborated by forensic personnel.
          </p>
        )}
      </div>
    </div>
  );
};
