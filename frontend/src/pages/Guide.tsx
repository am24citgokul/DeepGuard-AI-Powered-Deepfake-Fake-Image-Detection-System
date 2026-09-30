import React from "react";
import { 
  BookOpen, 
  Eye, 
  Flame, 
  Sparkles, 
  ShieldAlert, 
  Info, 
  CheckCircle2,
  HelpCircle
} from "lucide-react";
import { 
  SVG_SILHOUETTE_ORIGINAL, 
  SVG_SILHOUETTE_GRADCAM, 
  SVG_SILHOUETTE_ELA 
} from "@/lib/mockData";

export const Guide: React.FC = () => {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="header-gradient-banner rounded-3xl p-6 sm:p-8 border border-slate-200/80 dark:border-slate-800 shadow-xs">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-100/70 dark:bg-violet-950/60 text-violet-700 dark:text-violet-300 text-xs font-mono font-medium mb-3">
            <BookOpen className="w-3.5 h-3.5" />
            Interpretability Standard
          </div>
          <h1 className="font-heading font-extrabold text-2xl sm:text-3xl text-slate-900 dark:text-white tracking-tight">
            Evidence Guide & Forensic Interpretability
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 font-sans leading-relaxed">
            Understand how to read Grad-CAM activation saliency maps and Error Level Analysis (ELA) error maps to verify algorithmic findings.
          </p>
        </div>
      </div>

      {/* 1. Annotated Diagram of Triptych with Numbered Callouts */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 sm:p-8 shadow-sm space-y-6">
        <div>
          <h3 className="font-heading font-bold text-lg text-slate-900 dark:text-white">
            How to Read the Forensic Triptych
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Every image screening provides three parallel views to decouple semantic face structure from mathematical synthesis artifacts.
          </p>
        </div>

        {/* Triptych Visual with numbered badges */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Panel 1: Original */}
          <div className="relative rounded-2xl border border-slate-200 dark:border-slate-800 overflow-hidden bg-slate-50 dark:bg-slate-850 flex flex-col">
            <div className="p-3 bg-slate-100 dark:bg-slate-800 flex items-center justify-between border-b border-slate-200 dark:border-slate-700">
              <span className="font-mono text-xs font-bold text-slate-700 dark:text-slate-200 flex items-center gap-1.5">
                <Eye className="w-3.5 h-3.5 text-blue-500" />
                Panel 1: Original Input
              </span>
              <span className="w-6 h-6 rounded-full bg-blue-600 text-white font-mono font-bold text-xs flex items-center justify-center">
                1
              </span>
            </div>
            <div className="p-4 flex items-center justify-center h-52 bg-slate-100 dark:bg-slate-900">
              <img
                src={SVG_SILHOUETTE_ORIGINAL}
                alt="Original geometric reference"
                className="h-full object-contain"
              />
            </div>
            <div className="p-3 bg-white dark:bg-slate-900 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-400 leading-normal">
              <strong>Callout 1: </strong>Provides geometric landmark anchoring. Use this to verify head orientation, natural skin pores, and eye gaze consistency.
            </div>
          </div>

          {/* Panel 2: Grad-CAM */}
          <div className="relative rounded-2xl border border-slate-200 dark:border-slate-800 overflow-hidden bg-slate-50 dark:bg-slate-850 flex flex-col">
            <div className="p-3 bg-slate-100 dark:bg-slate-800 flex items-center justify-between border-b border-slate-200 dark:border-slate-700">
              <span className="font-mono text-xs font-bold text-slate-700 dark:text-slate-200 flex items-center gap-1.5">
                <Flame className="w-3.5 h-3.5 text-rose-500" />
                Panel 2: Grad-CAM Heatmap
              </span>
              <span className="w-6 h-6 rounded-full bg-rose-600 text-white font-mono font-bold text-xs flex items-center justify-center">
                2
              </span>
            </div>
            <div className="p-4 flex items-center justify-center h-52 bg-slate-900">
              <img
                src={SVG_SILHOUETTE_GRADCAM}
                alt="Grad-CAM activation heatmap"
                className="h-full object-contain"
              />
            </div>
            <div className="p-3 bg-white dark:bg-slate-900 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-400 leading-normal">
              <strong>Callout 2: </strong>Highlights convolutional layer gradients. Red and yellow clusters show exactly where the neural network detected manipulation clues.
            </div>
          </div>

          {/* Panel 3: ELA */}
          <div className="relative rounded-2xl border border-slate-200 dark:border-slate-800 overflow-hidden bg-slate-50 dark:bg-slate-850 flex flex-col">
            <div className="p-3 bg-slate-100 dark:bg-slate-800 flex items-center justify-between border-b border-slate-200 dark:border-slate-700">
              <span className="font-mono text-xs font-bold text-slate-700 dark:text-slate-200 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-teal-500" />
                Panel 3: ELA Error Map
              </span>
              <span className="w-6 h-6 rounded-full bg-teal-600 text-white font-mono font-bold text-xs flex items-center justify-center">
                3
              </span>
            </div>
            <div className="p-4 flex items-center justify-center h-52 bg-slate-950">
              <img
                src={SVG_SILHOUETTE_ELA}
                alt="ELA compression error map"
                className="h-full object-contain"
              />
            </div>
            <div className="p-3 bg-white dark:bg-slate-900 border-t border-slate-100 dark:border-slate-800 text-[11px] text-slate-600 dark:text-slate-400 leading-normal">
              <strong>Callout 3: </strong>Multiplies compression difference by 10x. Spliced or pasted facial regions show distinct brightness from the surrounding background.
            </div>
          </div>
        </div>
      </div>

      {/* 2. Two Info Cards: What Grad-CAM shows & What ELA shows */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Card 1: What Grad-CAM shows */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 shadow-xs space-y-3">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-rose-50 dark:bg-rose-950/50 text-rose-600">
              <Flame className="w-5 h-5" />
            </div>
            <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white">
              What Grad-CAM Shows
            </h3>
          </div>

          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-sans">
            Gradient-weighted Class Activation Mapping (Grad-CAM) computes the gradients of the target manipulation score with respect to the final convolutional feature maps of EfficientNet-B0.
          </p>

          <ul className="space-y-2 text-xs text-slate-600 dark:text-slate-300">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
              <span><strong>Hotspots (Red/Orange):</strong> Spatial regions that contributed most strongly to a "Manipulated" or "Authentic" classification.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
              <span><strong>Boundary Focus:</strong> In FaceSwap and Face2Face, activation frequently surrounds the jawline boundary where blending occurs.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
              <span><strong>Texture Focus:</strong> In NeuralTextures or GAN synthesis, heatmaps typically cover the mouth interior and periorbital eye regions.</span>
            </li>
          </ul>
        </div>

        {/* Card 2: What ELA shows */}
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 shadow-xs space-y-3">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-teal-50 dark:bg-teal-950/50 text-teal-600">
              <Sparkles className="w-5 h-5" />
            </div>
            <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white">
              What Error Level Analysis (ELA) Shows
            </h3>
          </div>

          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed font-sans">
            Error Level Analysis highlights disparities in compression levels. When an image is resaved at 90% JPEG quality, regions that have been edited or digitally inserted compress at a different rate than pristine image zones.
          </p>

          <ul className="space-y-2 text-xs text-slate-600 dark:text-slate-300">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
              <span><strong>Uniform Grain:</strong> An authentic, unmodified capture typically exhibits an evenly distributed, low-contrast noise floor.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
              <span><strong>Glowing Discontinuities:</strong> Pasted faces or re-rendered mouth textures will appear distinctly brighter or darker than the canvas.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
              <span><strong>High-Frequency Outlines:</strong> Splice boundaries frequently glow intensely due to sharp contrast gradients between differing compression blocks.</span>
            </li>
          </ul>
        </div>
      </div>

      {/* 3. A Red-Outlined Card: "What this evidence does NOT prove" */}
      <div className="rounded-2xl border-2 border-rose-500 dark:border-rose-600 bg-rose-50/40 dark:bg-rose-950/20 p-6 sm:p-8 space-y-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-rose-600 text-white shrink-0">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-heading font-extrabold text-base sm:text-lg text-rose-950 dark:text-rose-100">
              What This Evidence Does NOT Prove
            </h3>
            <span className="text-xs font-mono text-rose-700 dark:text-rose-400">
              Critical Forensic Boundary & Legal Standards
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <div className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-rose-200 dark:border-rose-900/60 shadow-xs">
            <h4 className="font-heading font-bold text-xs text-rose-900 dark:text-rose-200 mb-1">
              1. A heatmap is not proof of manipulation
            </h4>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed font-sans">
              Neural network heatmaps indicate mathematical correlation, not causal evidence. High saliency can be triggered by eye glasses, sharp specular reflections, or unusual makeup.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-rose-200 dark:border-rose-900/60 shadow-xs">
            <h4 className="font-heading font-bold text-xs text-rose-900 dark:text-rose-200 mb-1">
              2. ELA alone cannot prove an image is fake
            </h4>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed font-sans">
              Innocent operations like cropping, resizing, WhatsApp recompression, or social media uploads cause natural ELA variance without any synthetic face alteration.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-white dark:bg-slate-900 border border-rose-200 dark:border-rose-900/60 shadow-xs">
            <h4 className="font-heading font-bold text-xs text-rose-900 dark:text-rose-200 mb-1">
              3. Clean-looking evidence does not prove authenticity
            </h4>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-relaxed font-sans">
              Advanced diffusion models and zero-shot generators can craft seamless synthetic faces without generating traditional JPEG block artifacts. Absence of evidence is not evidence of absence.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
