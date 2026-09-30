import { 
  CheckCircle, 
  AlertTriangle, 
  ShieldAlert, 
  HelpCircle,
  LucideIcon 
} from "lucide-react";
import type { Verdict, FinalVerdict, Maturity, ModalityType } from "./types";

export const MODALITY_COLORS = {
  image: "#3B82F6",
  video: "#14B8A6",
  audio: "#EC4899",
  fusion: "#8B5CF6",
} as const;

export const VERDICT_COLORS = {
  AUTHENTIC: "#16A34A",
  TRUSTED: "#16A34A",
  REVIEW_REQUIRED: "#F59E0B",
  MANIPULATED: "#DC2626",
  HIGH_RISK: "#DC2626",
  UNCERTAIN: "#64748B",
} as const;

export interface VerdictConfig {
  label: string;
  color: string;
  bgLight: string;
  bgDark: string;
  borderLight: string;
  borderDark: string;
  textLight: string;
  textDark: string;
  icon: LucideIcon;
  description: string;
}

export function getVerdictConfig(verdict: Verdict | FinalVerdict | string): VerdictConfig {
  switch (verdict) {
    case "TRUSTED":
    case "AUTHENTIC":
      return {
        label: verdict === "TRUSTED" ? "TRUSTED" : "AUTHENTIC",
        color: "#16A34A",
        bgLight: "bg-emerald-50",
        bgDark: "dark:bg-emerald-950/40",
        borderLight: "border-emerald-200",
        borderDark: "dark:border-emerald-800/60",
        textLight: "text-emerald-700",
        textDark: "dark:text-emerald-400",
        icon: CheckCircle,
        description: "Consistent signals indicate authentic capture characteristics.",
      };
    case "REVIEW_REQUIRED":
      return {
        label: "REVIEW REQUIRED",
        color: "#F59E0B",
        bgLight: "bg-amber-50",
        bgDark: "dark:bg-amber-950/40",
        borderLight: "border-amber-200",
        borderDark: "dark:border-amber-800/60",
        textLight: "text-amber-700",
        textDark: "dark:text-amber-400",
        icon: AlertTriangle,
        description: "Mixed or marginal evidence signals require human forensic assessment.",
      };
    case "HIGH_RISK":
    case "MANIPULATED":
      return {
        label: verdict === "HIGH_RISK" ? "HIGH RISK" : "MANIPULATED",
        color: "#DC2626",
        bgLight: "bg-rose-50",
        bgDark: "dark:bg-rose-950/40",
        borderLight: "border-rose-200",
        borderDark: "dark:border-rose-800/60",
        textLight: "text-rose-700",
        textDark: "dark:text-rose-400",
        icon: ShieldAlert,
        description: "Deepfake synthesis or manipulation artifacts flagged across evaluation.",
      };
    case "UNCERTAIN":
    default:
      return {
        label: "UNCERTAIN",
        color: "#64748B",
        bgLight: "bg-slate-100 uncertain-hatch",
        bgDark: "dark:bg-slate-900 uncertain-hatch",
        borderLight: "border-slate-300",
        borderDark: "dark:border-slate-700",
        textLight: "text-slate-700",
        textDark: "dark:text-slate-300",
        icon: HelpCircle,
        description: "Low model confidence — human review recommended.",
      };
  }
}

export function getMaturityChip(maturity: Maturity) {
  switch (maturity) {
    case "TRAINED":
      return {
        label: "Trained model",
        color: "bg-sky-50 text-sky-700 border-sky-200 dark:bg-sky-950/50 dark:text-sky-300 dark:border-sky-800",
        dot: "bg-sky-500",
      };
    case "TRAINED_FALLBACK":
      return {
        label: "Trained · fallback active",
        color: "bg-teal-50 text-teal-700 border-teal-200 dark:bg-teal-950/50 dark:text-teal-300 dark:border-teal-800",
        dot: "bg-teal-500",
      };
    case "PROTOTYPE":
    default:
      return {
        label: "Heuristic prototype",
        color: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/50 dark:text-amber-300 dark:border-amber-800",
        dot: "bg-amber-500",
      };
  }
}

export function formatPercent(value: number, decimals: number = 1): string {
  if (value === undefined || value === null || isNaN(value)) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatDateTime(isoOrDateStr: string): string {
  if (!isoOrDateStr) return "—";
  try {
    const d = new Date(isoOrDateStr);
    if (isNaN(d.getTime())) return isoOrDateStr;
    return d.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoOrDateStr;
  }
}
