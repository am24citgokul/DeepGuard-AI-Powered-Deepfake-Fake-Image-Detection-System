import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { 
  ShieldCheck, 
  ArrowRight, 
  Layers, 
  Image as ImageIcon, 
  Video as VideoIcon, 
  Mic as MicIcon, 
  Camera, 
  CheckCircle2, 
  AlertTriangle, 
  FileText,
  Activity,
  Sparkles
} from "lucide-react";
import { PipelineDiagram } from "@/components/diagrams/PipelineDiagram";
import { ConfidenceGauge } from "@/components/diagrams/ConfidenceGauge";
import { api } from "@/lib/api";
import { 
  MODALITY_COLORS, 
  getVerdictConfig, 
  getMaturityChip,
  formatPercent,
  formatDateTime 
} from "@/lib/format";
import type { HealthResponse, ScreeningResult, ModelDetail } from "@/lib/types";

export const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelDetail[]>([]);
  const [latestScreening, setLatestScreening] = useState<ScreeningResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [hData, mData, reports] = await Promise.all([
          api.getHealth(),
          api.getModels(),
          api.getReports(),
        ]);
        setHealth(hData);
        setModels(mData);
        if (reports && reports.length > 0) {
          setLatestScreening(reports[0]);
        }
      } catch (err) {
        console.error("Dashboard data load error:", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const latestVerdictConf = latestScreening
    ? getVerdictConfig(latestScreening.final_verdict)
    : null;

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* 1. Soft Gradient Header Banner */}
      <div className="header-gradient-banner rounded-3xl p-6 sm:p-10 border border-slate-200/80 dark:border-slate-800 shadow-xs relative overflow-hidden">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-100/70 dark:bg-violet-950/60 text-violet-700 dark:text-violet-300 text-xs font-mono font-medium mb-3">
              <Sparkles className="w-3.5 h-3.5" />
              Multimodal Identity Verification
            </div>
            <h1 className="font-heading font-extrabold text-2xl sm:text-3xl lg:text-4xl text-slate-900 dark:text-white tracking-tight leading-tight">
              AI Identity Authenticity Screening
            </h1>
            <p className="text-sm sm:text-base text-slate-600 dark:text-slate-300 mt-2 font-sans leading-relaxed">
              Research-grade multimodal forensic analytics combining spatial neural feature maps, video temporal kinetic coherence, and acoustic spectral signatures into one unified audit.
            </p>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <Link
              to="/screening"
              className="inline-flex items-center gap-2.5 px-6 py-3.5 rounded-2xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-700 hover:to-indigo-700 text-white font-heading font-bold text-sm shadow-md hover:shadow-lg transition-all transform hover:-translate-y-0.5"
            >
              <span>Start Screening</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* 2. Hero Interactive Pipeline Diagram */}
      <PipelineDiagram />

      {/* 3. Three Model Maturity Cards (Image / Video / Audio) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Image Model Card */}
        {(() => {
          const imgM = models.find((m) => m.modality === "image");
          const chip = getMaturityChip(imgM?.maturity || "TRAINED");
          const ready = health?.models?.image?.ready ?? true;

          return (
            <div
              className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-5 shadow-xs flex flex-col justify-between"
              style={{ borderTop: `4px solid ${MODALITY_COLORS.image}` }}
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950/50 text-blue-600">
                      <ImageIcon className="w-4 h-4" />
                    </div>
                    <span className="font-heading font-bold text-sm text-slate-900 dark:text-white">
                      Image Branch
                    </span>
                  </div>

                  {/* Ready Dot */}
                  <div className="flex items-center gap-1.5 text-[11px] font-mono">
                    <span className={`w-2 h-2 rounded-full ${ready ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`}></span>
                    <span className="text-slate-500">{ready ? "Ready" : "Offline"}</span>
                  </div>
                </div>

                <h4 className="font-heading font-semibold text-sm text-slate-800 dark:text-slate-200">
                  {imgM?.name || "EfficientNet-B0 (Rebalanced)"}
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
                  {imgM?.dataset || "FaceForensics++ c23 (4,500 test evaluations)"}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${chip.color}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${chip.dot}`}></span>
                  {chip.label}
                </span>
                <Link to="/models" className="text-xs text-blue-600 dark:text-blue-400 font-medium hover:underline">
                  View Metrics →
                </Link>
              </div>
            </div>
          );
        })()}

        {/* Video Model Card */}
        {(() => {
          const vidM = models.find((m) => m.modality === "video");
          const chip = getMaturityChip(vidM?.maturity || "TRAINED");
          const ready = health?.models?.video?.ready ?? true;

          return (
            <div
              className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-5 shadow-xs flex flex-col justify-between"
              style={{ borderTop: `4px solid ${MODALITY_COLORS.video}` }}
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-xl bg-teal-50 dark:bg-teal-950/50 text-teal-600">
                      <VideoIcon className="w-4 h-4" />
                    </div>
                    <span className="font-heading font-bold text-sm text-slate-900 dark:text-white">
                      Video Branch
                    </span>
                  </div>

                  {/* Ready Dot */}
                  <div className="flex items-center gap-1.5 text-[11px] font-mono">
                    <span className={`w-2 h-2 rounded-full ${ready ? "bg-emerald-500 animate-pulse" : "bg-rose-500"}`}></span>
                    <span className="text-slate-500">{ready ? "Ready" : "Offline"}</span>
                  </div>
                </div>

                <h4 className="font-heading font-semibold text-sm text-slate-800 dark:text-slate-200">
                  {vidM?.name || "CNN-GRU Temporal Sequence"}
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
                  {vidM?.dataset || "16-frame uniform temporal sampling & kinetic coherence"}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${chip.color}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${chip.dot}`}></span>
                  {chip.label}
                </span>
                <Link to="/models" className="text-xs text-teal-600 dark:text-teal-400 font-medium hover:underline">
                  View Metrics →
                </Link>
              </div>
            </div>
          );
        })()}

        {/* Audio Model Card */}
        {(() => {
          const audM = models.find((m) => m.modality === "audio");
          const chip = getMaturityChip(audM?.maturity || "PROTOTYPE");
          const ready = health?.models?.audio?.ready ?? true;

          return (
            <div
              className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-5 shadow-xs flex flex-col justify-between"
              style={{ borderTop: `4px solid ${MODALITY_COLORS.audio}` }}
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-xl bg-pink-50 dark:bg-pink-950/50 text-pink-600">
                      <MicIcon className="w-4 h-4" />
                    </div>
                    <span className="font-heading font-bold text-sm text-slate-900 dark:text-white">
                      Audio Branch
                    </span>
                  </div>

                  {/* Ready Dot */}
                  <div className="flex items-center gap-1.5 text-[11px] font-mono">
                    <span className={`w-2 h-2 rounded-full ${ready ? "bg-emerald-500 animate-pulse" : "bg-amber-500"}`}></span>
                    <span className="text-slate-500">{ready ? "Ready" : "Prototype"}</span>
                  </div>
                </div>

                <h4 className="font-heading font-semibold text-sm text-slate-800 dark:text-slate-200">
                  {audM?.name || "Baseline CNN (Log-Mel Spectrogram)"}
                </h4>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
                  {audM?.dataset || "In-the-Wild Audio Dataset (3,540 balanced test samples)"}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
                <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border ${chip.color}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${chip.dot}`}></span>
                  {chip.label}
                </span>
                <Link to="/models" className="text-xs text-pink-600 dark:text-pink-400 font-medium hover:underline">
                  View Metrics →
                </Link>
              </div>
            </div>
          );
        })()}
      </div>

      {/* 4. Latest Screening Summary Card */}
      {latestScreening && latestVerdictConf && (
        <div className="rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-5 pb-3 border-b border-slate-100 dark:border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-violet-600" />
                <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white">
                  Latest Screening Audit Record
                </h3>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 font-mono">
                ID: {latestScreening.id} • {formatDateTime(latestScreening.created_at)}
              </p>
            </div>

            <div className="flex items-center gap-3">
              <span
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${latestVerdictConf.bgLight} ${latestVerdictConf.textLight} ${latestVerdictConf.borderLight} dark:${latestVerdictConf.bgDark} dark:${latestVerdictConf.textDark} dark:${latestVerdictConf.borderDark}`}
              >
                <latestVerdictConf.icon className="w-3.5 h-3.5" />
                {latestVerdictConf.label}
              </span>

              <Link
                to={`/reports/${latestScreening.id}`}
                className="text-xs font-semibold text-violet-600 dark:text-violet-400 hover:underline flex items-center gap-1"
              >
                Full Inspection <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>

          <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 font-sans mb-6">
            "{latestScreening.reasoning}"
          </p>

          {/* Three mini gauges side by side */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 pt-2">
            {/* Image gauge */}
            <div className="p-4 rounded-xl bg-slate-50/60 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-800 flex flex-col items-center">
              <div className="text-xs font-heading font-semibold text-blue-600 dark:text-blue-400 mb-2 flex items-center gap-1.5">
                <ImageIcon className="w-3.5 h-3.5" /> Image Branch
              </div>
              {latestScreening.modalities.image ? (
                <ConfidenceGauge
                  confidence={latestScreening.modalities.image.confidence}
                  color={MODALITY_COLORS.image}
                  size={100}
                  label="Confidence"
                  sublabel={latestScreening.modalities.image.verdict}
                />
              ) : (
                <div className="text-xs font-mono text-slate-400 py-6">Omitted</div>
              )}
            </div>

            {/* Video gauge */}
            <div className="p-4 rounded-xl bg-slate-50/60 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-800 flex flex-col items-center">
              <div className="text-xs font-heading font-semibold text-teal-600 dark:text-teal-400 mb-2 flex items-center gap-1.5">
                <VideoIcon className="w-3.5 h-3.5" /> Video Branch
              </div>
              {latestScreening.modalities.video ? (
                <ConfidenceGauge
                  confidence={latestScreening.modalities.video.confidence}
                  color={MODALITY_COLORS.video}
                  size={100}
                  label="Confidence"
                  sublabel={latestScreening.modalities.video.verdict}
                />
              ) : (
                <div className="text-xs font-mono text-slate-400 py-6">Omitted</div>
              )}
            </div>

            {/* Audio gauge */}
            <div className="p-4 rounded-xl bg-slate-50/60 dark:bg-slate-800/40 border border-slate-200/60 dark:border-slate-800 flex flex-col items-center">
              <div className="text-xs font-heading font-semibold text-pink-600 dark:text-pink-400 mb-2 flex items-center gap-1.5">
                <MicIcon className="w-3.5 h-3.5" /> Audio Branch
              </div>
              {latestScreening.modalities.audio ? (
                <ConfidenceGauge
                  confidence={latestScreening.modalities.audio.confidence}
                  color={MODALITY_COLORS.audio}
                  size={100}
                  label="Confidence"
                  sublabel={latestScreening.modalities.audio.verdict}
                />
              ) : (
                <div className="text-xs font-mono text-slate-400 py-6">Omitted</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 5. Quick-Action Tiles */}
      <div className="space-y-3">
        <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white">
          Quick Verification Actions
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
          <Link
            to="/screening"
            className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-violet-400 dark:hover:border-violet-600 shadow-xs hover:shadow-md transition-all group flex flex-col items-center text-center col-span-2 sm:col-span-1"
          >
            <div className="w-10 h-10 rounded-xl bg-violet-100 dark:bg-violet-950 text-violet-600 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
              <Sparkles className="w-5 h-5" />
            </div>
            <span className="font-heading font-bold text-xs text-slate-900 dark:text-white">
              Screen All
            </span>
            <span className="text-[10px] text-slate-400 font-mono mt-0.5">
              Full Multimodal
            </span>
          </Link>

          <Link
            to="/analyze?tab=image"
            className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-blue-400 dark:hover:border-blue-600 shadow-xs hover:shadow-md transition-all group flex flex-col items-center text-center"
          >
            <div className="w-10 h-10 rounded-xl bg-blue-100 dark:bg-blue-950 text-blue-600 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
              <ImageIcon className="w-5 h-5" />
            </div>
            <span className="font-heading font-bold text-xs text-slate-900 dark:text-white">
              Analyze Image
            </span>
            <span className="text-[10px] text-slate-400 font-mono mt-0.5">
              Grad-CAM & ELA
            </span>
          </Link>

          <Link
            to="/analyze?tab=video"
            className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-teal-400 dark:hover:border-teal-600 shadow-xs hover:shadow-md transition-all group flex flex-col items-center text-center"
          >
            <div className="w-10 h-10 rounded-xl bg-teal-100 dark:bg-teal-950 text-teal-600 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
              <VideoIcon className="w-5 h-5" />
            </div>
            <span className="font-heading font-bold text-xs text-slate-900 dark:text-white">
              Analyze Video
            </span>
            <span className="text-[10px] text-slate-400 font-mono mt-0.5">
              CNN-GRU Sequence
            </span>
          </Link>

          <Link
            to="/analyze?tab=audio"
            className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-pink-400 dark:hover:border-pink-600 shadow-xs hover:shadow-md transition-all group flex flex-col items-center text-center"
          >
            <div className="w-10 h-10 rounded-xl bg-pink-100 dark:bg-pink-950 text-pink-600 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
              <MicIcon className="w-5 h-5" />
            </div>
            <span className="font-heading font-bold text-xs text-slate-900 dark:text-white">
              Analyze Audio
            </span>
            <span className="text-[10px] text-slate-400 font-mono mt-0.5">
              Baseline CNN
            </span>
          </Link>

          <Link
            to="/webcam"
            className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-amber-400 dark:hover:border-amber-600 shadow-xs hover:shadow-md transition-all group flex flex-col items-center text-center"
          >
            <div className="w-10 h-10 rounded-xl bg-amber-100 dark:bg-amber-950 text-amber-600 flex items-center justify-center mb-2 group-hover:scale-110 transition-transform">
              <Camera className="w-5 h-5" />
            </div>
            <span className="font-heading font-bold text-xs text-slate-900 dark:text-white">
              Open Webcam
            </span>
            <span className="text-[10px] text-slate-400 font-mono mt-0.5">
              Live Biometrics
            </span>
          </Link>
        </div>
      </div>
    </div>
  );
};
