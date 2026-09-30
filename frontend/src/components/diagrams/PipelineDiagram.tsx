import React, { useState } from "react";
import { 
  Upload, 
  Cpu, 
  Image as ImageIcon, 
  Video, 
  Mic, 
  Layers, 
  GitMerge, 
  CheckCircle2,
  Info,
  X
} from "lucide-react";
import { MODALITY_COLORS } from "@/lib/format";

interface NodeData {
  id: string;
  title: string;
  subtitle: string;
  category: "input" | "prep" | "branch" | "evidence" | "fusion" | "verdict";
  color?: string;
  description: string;
  techDetails: string[];
}

const PIPELINE_NODES: Record<string, NodeData> = {
  upload: {
    id: "upload",
    title: "1. Media Intake",
    subtitle: "Identity Artifacts",
    category: "input",
    description: "Accepts 1 to 3 isolated digital identity artifacts (still image portrait, facial video, voice recording). Never sent to third parties.",
    techDetails: ["Client-side integrity validation", "Multipart boundary packaging", "Zero-retention temporary buffer"],
  },
  preprocess: {
    id: "preprocess",
    title: "2. Preprocessing",
    subtitle: "Standardization",
    category: "prep",
    description: "Decouples input streams, performs MTCNN / OpenCV face landmark localization, aligns head pose, extracts uniform temporal audio frames.",
    techDetails: ["Face bounding box crop (224x224)", "16-frame uniform temporal sampler", "16kHz mono audio resampling & STFT"],
  },
  imageLane: {
    id: "imageLane",
    title: "Image Branch",
    subtitle: "Spatial Forensics",
    category: "branch",
    color: MODALITY_COLORS.image,
    description: "Class-weighted EfficientNet-B0 trained on FaceForensics++ c23, identifying pixel-level blending artifacts, GAN checkerboard patterns, and boundary splices.",
    techDetails: ["EfficientNet-B0 backbone (5.3M params)", "Inverse class-frequency loss", "81.3% test binary accuracy"],
  },
  videoLane: {
    id: "videoLane",
    title: "Video Branch",
    subtitle: "Temporal Forensics",
    category: "branch",
    color: MODALITY_COLORS.video,
    description: "Bidirectional CNN-GRU sequence architecture detecting frame-to-frame temporal incoherence, facial flicker, and biometric pulse consistency.",
    techDetails: ["Spatial feature maps + GRU temporal recurrence", "16-frame sequence aggregation", "Optical flow continuity validation"],
  },
  audioLane: {
    id: "audioLane",
    title: "Audio Branch",
    subtitle: "Baseline CNN",
    category: "branch",
    color: MODALITY_COLORS.audio,
    description: "4-Block 2D CNN with Log-Mel Spectrogram frontend trained on 16 kHz audio (64,000 samples) detecting voice cloning and deepfake synthesis artifacts.",
    techDetails: ["Trained 4-Block 2D CNN (128 Mels, 16 kHz)", "Operating Decision Threshold: 0.50", "Supporting acoustic forensics (ZCR, spectral flatness)"],
  },
  evidence: {
    id: "evidence",
    title: "4. Evidence Extraction",
    subtitle: "Interpretability",
    category: "evidence",
    description: "Generates interpretable forensic heatmaps: Grad-CAM gradient saliency maps and Error Level Analysis (ELA) compression discrepancy maps.",
    techDetails: ["Target layer Grad-CAM saliency", "10x ELA residual compression map", "Frame-by-frame temporal anomaly track"],
  },
  fusion: {
    id: "fusion",
    title: "5. Cross-Modal Fusion",
    subtitle: "Consistency Engine",
    category: "fusion",
    color: MODALITY_COLORS.fusion,
    description: "Probabilistic fusion kernel combining available modality vectors. Automatically isolates uncertain streams from the voting consensus.",
    techDetails: ["Weighted Bayesian evidence pooling", "Uncertain stream exclusion (<55%)", "Cross-modal pairwise divergence check"],
  },
  verdict: {
    id: "verdict",
    title: "6. Screening Verdict",
    subtitle: "Decision Aid",
    category: "verdict",
    description: "Outputs final classification: TRUSTED, REVIEW REQUIRED, or HIGH RISK, accompanied by full rationale and audit-ready report metadata.",
    techDetails: ["Three-tier forensic risk triage", "Zero-hallucination calibrated decision", "Immutable audit event logging"],
  },
};

interface PipelineDiagramProps {
  isAnalyzing?: boolean;
  onNodeClick?: (nodeId: string) => void;
  className?: string;
  size?: "default" | "large";
}

export const PipelineDiagram: React.FC<PipelineDiagramProps> = ({
  isAnalyzing = false,
  className = "",
  size = "default",
}) => {
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null);

  return (
    <div className={`relative w-full rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 sm:p-6 shadow-sm overflow-hidden ${className}`}>
      {/* Header Bar */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100 dark:border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center p-1.5 rounded-lg bg-violet-100 dark:bg-violet-950/60 text-violet-600 dark:text-violet-400">
              <Layers className="w-4 h-4" />
            </span>
            <h3 className="font-heading font-semibold text-slate-900 dark:text-white text-base">
              End-to-End Multimodal Forensic Pipeline
            </h3>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Click any node to inspect internal model layers, DSP algorithms, and verification logic.
          </p>
        </div>

        {isAnalyzing && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-violet-50 dark:bg-violet-950/50 border border-violet-200 dark:border-violet-800">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-violet-600"></span>
            </span>
            <span className="text-xs font-medium text-violet-700 dark:text-violet-300">
              Pipeline Active
            </span>
          </div>
        )}
      </div>

      {/* Responsive Diagram Container: Horizontal on desktop (grid-cols-6 or flex), stacked on mobile */}
      <div className="relative py-2 overflow-x-auto">
        <div className="min-w-[760px] flex items-center justify-between gap-3">
          {/* 1. Media Intake */}
          <div
            onClick={() => setSelectedNode(PIPELINE_NODES.upload)}
            className="flex-1 cursor-pointer transition-all transform hover:-translate-y-0.5 group"
          >
            <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-800/60 hover:border-slate-400 dark:hover:border-slate-600 hover:shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-2">
                <Upload className="w-4 h-4 text-slate-600 dark:text-slate-300" />
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-200/60 dark:bg-slate-700 text-slate-600 dark:text-slate-300">IN</span>
              </div>
              <div className="font-heading font-semibold text-xs text-slate-800 dark:text-slate-200 leading-tight">
                Media Intake
              </div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 truncate mt-0.5">
                Img / Vid / Aud
              </div>
            </div>
          </div>

          {/* Connector Arrow */}
          <div className="w-6 flex items-center justify-center shrink-0">
            <svg width="24" height="12" viewBox="0 0 24 12" className="text-slate-300 dark:text-slate-600">
              <line x1="0" y1="6" x2="20" y2="6" stroke="currentColor" strokeWidth="2" strokeDasharray={isAnalyzing ? "3 3" : undefined} className={isAnalyzing ? "animate-flow-dash stroke-violet-500" : ""} />
              <polygon points="18,3 24,6 18,9" fill="currentColor" className={isAnalyzing ? "fill-violet-500" : ""} />
            </svg>
          </div>

          {/* 2. Preprocessing */}
          <div
            onClick={() => setSelectedNode(PIPELINE_NODES.preprocess)}
            className="flex-1 cursor-pointer transition-all transform hover:-translate-y-0.5 group"
          >
            <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-800/60 hover:border-slate-400 dark:hover:border-slate-600 hover:shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-2">
                <Cpu className="w-4 h-4 text-slate-600 dark:text-slate-300" />
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-200/60 dark:bg-slate-700 text-slate-600 dark:text-slate-300">PRE</span>
              </div>
              <div className="font-heading font-semibold text-xs text-slate-800 dark:text-slate-200 leading-tight">
                Preprocessing
              </div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 truncate mt-0.5">
                Crop & Align
              </div>
            </div>
          </div>

          {/* Fan-out Connector */}
          <div className="w-8 flex items-center justify-center shrink-0">
            <svg width="32" height="110" viewBox="0 0 32 110" className="text-slate-300 dark:text-slate-600 overflow-visible">
              <path d="M0,55 C16,55 16,18 30,18" fill="none" stroke={MODALITY_COLORS.image} strokeWidth="2" strokeDasharray={isAnalyzing ? "4 4" : undefined} className={isAnalyzing ? "animate-flow-dash" : ""} />
              <path d="M0,55 L30,55" fill="none" stroke={MODALITY_COLORS.video} strokeWidth="2" strokeDasharray={isAnalyzing ? "4 4" : undefined} className={isAnalyzing ? "animate-flow-dash" : ""} />
              <path d="M0,55 C16,55 16,92 30,92" fill="none" stroke={MODALITY_COLORS.audio} strokeWidth="2" strokeDasharray={isAnalyzing ? "4 4" : undefined} className={isAnalyzing ? "animate-flow-dash" : ""} />
            </svg>
          </div>

          {/* 3. Three Parallel Lanes (Stacked vertically within this column) */}
          <div className="flex flex-col gap-2 w-52 shrink-0">
            {/* Image Lane */}
            <div
              onClick={() => setSelectedNode(PIPELINE_NODES.imageLane)}
              className="p-2.5 rounded-xl border border-blue-200 dark:border-blue-900/60 bg-blue-50/50 dark:bg-blue-950/30 hover:border-blue-400 cursor-pointer transition-all flex items-center gap-2.5 border-t-[3px] border-t-blue-500"
            >
              <div className="p-1.5 rounded-lg bg-blue-100 dark:bg-blue-900/50 text-blue-600">
                <ImageIcon className="w-3.5 h-3.5" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="font-heading font-semibold text-xs text-blue-950 dark:text-blue-200 truncate">
                  Image (EfficientNet)
                </div>
                <div className="text-[10px] text-blue-600/80 dark:text-blue-400 font-mono">
                  Spatial Artifacts
                </div>
              </div>
            </div>

            {/* Video Lane */}
            <div
              onClick={() => setSelectedNode(PIPELINE_NODES.videoLane)}
              className="p-2.5 rounded-xl border border-teal-200 dark:border-teal-900/60 bg-teal-50/50 dark:bg-teal-950/30 hover:border-teal-400 cursor-pointer transition-all flex items-center gap-2.5 border-t-[3px] border-t-teal-500"
            >
              <div className="p-1.5 rounded-lg bg-teal-100 dark:bg-teal-900/50 text-teal-600">
                <Video className="w-3.5 h-3.5" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="font-heading font-semibold text-xs text-teal-950 dark:text-teal-200 truncate">
                  Video (CNN-GRU)
                </div>
                <div className="text-[10px] text-teal-600/80 dark:text-teal-400 font-mono">
                  Temporal Coherence
                </div>
              </div>
            </div>

            {/* Audio Lane */}
            <div
              onClick={() => setSelectedNode(PIPELINE_NODES.audioLane)}
              className="p-2.5 rounded-xl border border-pink-200 dark:border-pink-900/60 bg-pink-50/50 dark:bg-pink-950/30 hover:border-pink-400 cursor-pointer transition-all flex items-center gap-2.5 border-t-[3px] border-t-pink-500"
            >
              <div className="p-1.5 rounded-lg bg-pink-100 dark:bg-pink-900/50 text-pink-600">
                <Mic className="w-3.5 h-3.5" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="font-heading font-semibold text-xs text-pink-950 dark:text-pink-200 truncate">
                  Audio (Baseline CNN)
                </div>
                <div className="text-[10px] text-pink-600/80 dark:text-pink-400 font-mono">
                  Trained 2D CNN
                </div>
              </div>
            </div>
          </div>

          {/* Convergence Connector */}
          <div className="w-8 flex items-center justify-center shrink-0">
            <svg width="32" height="110" viewBox="0 0 32 110" className="text-slate-300 dark:text-slate-600 overflow-visible">
              <path d="M2,18 C16,18 16,55 30,55" fill="none" stroke={MODALITY_COLORS.image} strokeWidth="2" strokeDasharray={isAnalyzing ? "4 4" : undefined} className={isAnalyzing ? "animate-flow-dash" : ""} />
              <path d="M2,55 L30,55" fill="none" stroke={MODALITY_COLORS.video} strokeWidth="2" strokeDasharray={isAnalyzing ? "4 4" : undefined} className={isAnalyzing ? "animate-flow-dash" : ""} />
              <path d="M2,92 C16,92 16,55 30,55" fill="none" stroke={MODALITY_COLORS.audio} strokeWidth="2" strokeDasharray={isAnalyzing ? "4 4" : undefined} className={isAnalyzing ? "animate-flow-dash" : ""} />
            </svg>
          </div>

          {/* 4. Cross-Modal Fusion */}
          <div
            onClick={() => setSelectedNode(PIPELINE_NODES.fusion)}
            className="flex-1 cursor-pointer transition-all transform hover:-translate-y-0.5 group"
          >
            <div className="p-3.5 rounded-xl border border-violet-200 dark:border-violet-900 bg-violet-50/60 dark:bg-violet-950/40 hover:border-violet-400 dark:hover:border-violet-600 hover:shadow-sm border-t-[3px] border-t-violet-500">
              <div className="flex items-center justify-between text-violet-600 mb-2">
                <GitMerge className="w-4 h-4" />
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-violet-200/60 dark:bg-violet-900 text-violet-700 dark:text-violet-300">FUSE</span>
              </div>
              <div className="font-heading font-semibold text-xs text-violet-950 dark:text-violet-200 leading-tight">
                Evidence Fusion
              </div>
              <div className="text-[11px] text-violet-600/80 dark:text-violet-400 truncate mt-0.5">
                Consensus Engine
              </div>
            </div>
          </div>

          {/* Connector Arrow */}
          <div className="w-6 flex items-center justify-center shrink-0">
            <svg width="24" height="12" viewBox="0 0 24 12" className="text-slate-300 dark:text-slate-600">
              <line x1="0" y1="6" x2="20" y2="6" stroke="#8B5CF6" strokeWidth="2" strokeDasharray={isAnalyzing ? "3 3" : undefined} className={isAnalyzing ? "animate-flow-dash" : ""} />
              <polygon points="18,3 24,6 18,9" fill="#8B5CF6" />
            </svg>
          </div>

          {/* 5. Verdict Output */}
          <div
            onClick={() => setSelectedNode(PIPELINE_NODES.verdict)}
            className="flex-1 cursor-pointer transition-all transform hover:-translate-y-0.5 group"
          >
            <div className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-700 bg-emerald-50/50 dark:bg-emerald-950/30 border-t-[3px] border-t-emerald-500 hover:shadow-sm">
              <div className="flex items-center justify-between text-emerald-600 mb-2">
                <CheckCircle2 className="w-4 h-4" />
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-emerald-900 text-emerald-700 dark:text-emerald-300">RESULT</span>
              </div>
              <div className="font-heading font-semibold text-xs text-slate-900 dark:text-white leading-tight">
                Authenticity Verdict
              </div>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 truncate mt-0.5">
                Risk + Rationale
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Selected Node Drawer / Tooltip Modal */}
      {selectedNode && (
        <div className="mt-4 p-4 rounded-xl bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 transition-all animate-fadeIn">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-heading font-bold text-sm text-slate-900 dark:text-white">
                  {selectedNode.title}
                </span>
                <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                  [{selectedNode.subtitle}]
                </span>
              </div>
              <p className="text-xs text-slate-600 dark:text-slate-300 mt-1 leading-relaxed max-w-2xl">
                {selectedNode.description}
              </p>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="p-1 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-400 hover:text-slate-600"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="mt-3 pt-3 border-t border-slate-200/60 dark:border-slate-700/60 flex flex-wrap gap-2">
            {selectedNode.techDetails.map((detail, idx) => (
              <span
                key={idx}
                className="text-[11px] font-mono px-2 py-0.5 rounded-md bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300"
              >
                • {detail}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
