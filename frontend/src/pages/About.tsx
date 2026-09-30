import React from "react";
import { 
  Info, 
  Layers, 
  GitMerge, 
  ShieldAlert, 
  CheckCircle2, 
  AlertTriangle,
  HelpCircle,
  ArrowDown,
  ArrowRight
} from "lucide-react";
import { PipelineDiagram } from "@/components/diagrams/PipelineDiagram";
import { DisclaimerBar } from "@/components/screening/DisclaimerBar";
import { MODALITY_COLORS, VERDICT_COLORS } from "@/lib/format";

export const About: React.FC = () => {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="header-gradient-banner rounded-3xl p-6 sm:p-8 border border-slate-200/80 dark:border-slate-800 shadow-xs">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-200/80 dark:bg-slate-800 text-slate-700 dark:text-slate-300 text-xs font-mono font-medium mb-3">
            <Info className="w-3.5 h-3.5" />
            System Architecture & Methodology
          </div>
          <h1 className="font-heading font-extrabold text-2xl sm:text-3xl text-slate-900 dark:text-white tracking-tight">
            Methodology & Technical Boundaries
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 font-sans leading-relaxed">
            Detailed operational documentation on how multimodal streams are extracted, normalized, probabilistic fusion rules are calculated, and where automated boundaries end.
          </p>
        </div>
      </div>

      {/* 1. Large Architecture Diagram */}
      <div className="space-y-3">
        <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white flex items-center gap-2">
          <Layers className="w-4 h-4 text-violet-600" />
          End-to-End System Pipeline
        </h3>
        <PipelineDiagram size="large" />
      </div>

      {/* 2. "How the verdict is decided" Flowchart */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 sm:p-8 shadow-sm space-y-6">
        <div>
          <div className="flex items-center gap-2">
            <GitMerge className="w-4 h-4 text-violet-600" />
            <h3 className="font-heading font-bold text-lg text-slate-900 dark:text-white">
              How the Verdict is Decided
            </h3>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Transparent deterministic consensus logic combining independent spatial, temporal, and acoustic decision vectors.
          </p>
        </div>

        {/* Visual Flowchart */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-stretch">
          {/* Step 1: Intake & Isolation */}
          <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-850/60 flex flex-col justify-between">
            <div>
              <span className="font-mono text-[10px] font-bold text-violet-600 dark:text-violet-400 block mb-1">
                RULE 1: ISOLATION
              </span>
              <h4 className="font-heading font-bold text-xs text-slate-900 dark:text-white mb-2">
                Omitted & Uncertain Inputs
              </h4>
              <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed font-sans">
                Any stream not uploaded is marked <em>omitted</em>. Any stream yielding low confidence (&lt;55%) is classified <strong>UNCERTAIN</strong> and immediately isolated from the vote.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-200/60 dark:border-slate-700/60 font-mono text-[10px] text-slate-500">
              ➔ Yields Active Vote Pool (1–3 streams)
            </div>
          </div>

          {/* Step 2: Unanimous Agreement */}
          <div className="p-4 rounded-xl border border-emerald-200 dark:border-emerald-900/60 bg-emerald-50/50 dark:bg-emerald-950/20 flex flex-col justify-between">
            <div>
              <span className="font-mono text-[10px] font-bold text-emerald-600 dark:text-emerald-400 block mb-1">
                RULE 2: CONCORDANCE
              </span>
              <h4 className="font-heading font-bold text-xs text-emerald-950 dark:text-emerald-200 mb-2">
                Unanimous Consensus
              </h4>
              <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed font-sans">
                If all active streams concordantly flag authentic, the system issues <strong>TRUSTED</strong>. If all active streams flag manipulation, the system issues <strong>HIGH RISK</strong>.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-emerald-200/60 dark:border-emerald-800/60 font-mono text-[10px] text-emerald-700 dark:text-emerald-400 font-bold">
              ➔ TRUSTED or HIGH RISK
            </div>
          </div>

          {/* Step 3: Divergence */}
          <div className="p-4 rounded-xl border border-amber-200 dark:border-amber-900/60 bg-amber-50/50 dark:bg-amber-950/20 flex flex-col justify-between">
            <div>
              <span className="font-mono text-[10px] font-bold text-amber-600 dark:text-amber-400 block mb-1">
                RULE 3: CONFLICT
              </span>
              <h4 className="font-heading font-bold text-xs text-amber-950 dark:text-amber-200 mb-2">
                Divergent / Mixed Signals
              </h4>
              <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed font-sans">
                If one branch flags authentic while another flags manipulated, a cross-modal contradiction is registered. The system automatically issues <strong>REVIEW REQUIRED</strong>.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-amber-200/60 dark:border-amber-800/60 font-mono text-[10px] text-amber-700 dark:text-amber-400 font-bold">
              ➔ REVIEW REQUIRED
            </div>
          </div>

          {/* Step 4: Fallback */}
          <div className="p-4 rounded-xl border border-slate-300 dark:border-slate-700 bg-slate-100/70 dark:bg-slate-800/50 flex flex-col justify-between">
            <div>
              <span className="font-mono text-[10px] font-bold text-slate-600 dark:text-slate-400 block mb-1">
                RULE 4: INSUFFICIENT DATA
              </span>
              <h4 className="font-heading font-bold text-xs text-slate-900 dark:text-white mb-2">
                All Streams Uncertain
              </h4>
              <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed font-sans">
                If all provided inputs fall below statistical margin, no consensus can be formed. The system flags <strong>REVIEW REQUIRED (ALL UNCERTAIN)</strong> for manual forensic inspection.
              </p>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-300/60 dark:border-slate-700/60 font-mono text-[10px] text-slate-600 dark:text-slate-300 font-bold">
              ➔ Mandatory Human Review
            </div>
          </div>
        </div>
      </div>

      {/* 3. Plain "Limitations" Section */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 sm:p-8 shadow-sm space-y-4">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-amber-500" />
          <h3 className="font-heading font-bold text-lg text-slate-900 dark:text-white">
            System Limitations & Boundary Conditions
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-sans">
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200 dark:border-slate-800">
            <h4 className="font-heading font-bold text-xs text-slate-900 dark:text-white mb-1">
              Social Media & Messaging App Re-Compression
            </h4>
            <p>
              Platforms like WhatsApp, Telegram, and WeChat re-encode videos and images with aggressive rate-control quantization. This strips high-frequency ELA noise and smooths fine facial pores, which can trigger false positives on innocent authentic media.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200 dark:border-slate-800">
            <h4 className="font-heading font-bold text-xs text-slate-900 dark:text-white mb-1">
              Novel Generative AI Paradigms
            </h4>
            <p>
              Our image model was trained on FaceForensics++ c23 (Deepfakes, FaceSwap, Face2Face, NeuralTextures, FaceShifter). High-resolution diffusion generation (e.g. Midjourney v6, Flux, Sora) or voice cloning models trained on unrepresented architectures will experience lower detection sensitivity.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200 dark:border-slate-800">
            <h4 className="font-heading font-bold text-xs text-slate-900 dark:text-white mb-1">
              Audio Deepfake Detector & Robustness Limitations
            </h4>
            <p>
              The audio branch utilizes a trained 4-block 2D CNN with Log-Mel Spectrogram representations achieving 98.47% test accuracy on clean speech. However, additive background noise (SNR ≤ 15 dB) causes severe degradation in synthetic detection recall. Noisy recordings require manual auditory review.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200 dark:border-slate-800">
            <h4 className="font-heading font-bold text-xs text-slate-900 dark:text-white mb-1">
              Legal & Forensic Standing
            </h4>
            <p>
              DeepGuard X is an investigatory triage tool designed to assist forensic analysts in triaging high-volume identity streams. It does not provide legally definitive verification without expert human testimony and cryptographic metadata verification.
            </p>
          </div>
        </div>
      </div>

      {/* Persistent Disclaimer */}
      <DisclaimerBar variant="general" />
    </div>
  );
};
