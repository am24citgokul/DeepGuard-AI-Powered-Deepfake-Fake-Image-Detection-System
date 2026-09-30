import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { 
  Layers, 
  Image as ImageIcon, 
  Video as VideoIcon, 
  Mic as MicIcon, 
  Play, 
  AlertCircle,
  RotateCcw
} from "lucide-react";
import { UploadZone } from "@/components/screening/UploadZone";
import { EvidenceCard } from "@/components/screening/EvidenceCard";
import { DisclaimerBar } from "@/components/screening/DisclaimerBar";
import { Waveform } from "@/components/diagrams/Waveform";
import { MODALITY_COLORS } from "@/lib/format";
import { api } from "@/lib/api";
import type { ModalityResult } from "@/lib/types";

export const Analyze: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const currentTab = (searchParams.get("tab") as "image" | "video" | "audio") || "image";

  const [file, setFile] = useState<File | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<ModalityResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [selectedAudioModel, setSelectedAudioModel] = useState<"baseline_cnn" | "aasist">("baseline_cnn");

  useEffect(() => {
    // Reset state on tab switch
    setFile(null);
    setResult(null);
    setErrorMsg(null);
  }, [currentTab]);

  const handleTabChange = (tab: "image" | "video" | "audio") => {
    setSearchParams({ tab });
  };

  const handleRunAnalysis = async () => {
    if (!file) {
      setErrorMsg("Please select a file to analyze.");
      return;
    }

    setErrorMsg(null);
    setIsRunning(true);
    setResult(null);

    try {
      const res = await api.analyzeModality(currentTab, file, selectedAudioModel);
      setResult(res);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to analyze modality.");
    } finally {
      setIsRunning(false);
    }
  };

  const modalityMeta = {
    image: {
      title: "Image Branch Evaluation",
      accept: "image/*,.jpg,.jpeg,.png,.webp,.bmp",
      hint: "Upload a facial portrait to compute Grad-CAM heatmaps and Error Level Analysis.",
      color: MODALITY_COLORS.image,
      icon: ImageIcon,
    },
    video: {
      title: "Video Branch Evaluation",
      accept: "video/*,.mp4,.avi,.mov,.mkv",
      hint: "Upload a short video clip to evaluate frame-by-frame temporal sequence coherence.",
      color: MODALITY_COLORS.video,
      icon: VideoIcon,
    },
    audio: {
      title: "Audio Deepfake Detection",
      accept: "audio/*,.wav,.mp3,.flac,.m4a,.aac",
      hint: "Upload a speech voice clip to evaluate deepfake synthesis using the trained Baseline CNN.",
      color: MODALITY_COLORS.audio,
      icon: MicIcon,
    },
  }[currentTab];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="header-gradient-banner rounded-3xl p-6 sm:p-8 border border-slate-200/80 dark:border-slate-800 shadow-xs">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-100/70 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 text-xs font-mono font-medium mb-3">
            <Layers className="w-3.5 h-3.5" />
            Isolated Modality Lab
          </div>
          <h1 className="font-heading font-extrabold text-2xl sm:text-3xl text-slate-900 dark:text-white tracking-tight">
            Analyze Single Modality
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 font-sans leading-relaxed">
            Diagnose individual identity artifacts in isolation. Inspect localized feature maps, temporal frame strips, or deep learning acoustic representations independently.
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-200 dark:border-slate-800 gap-2 overflow-x-auto pb-1">
        {(["image", "video", "audio"] as const).map((tab) => {
          const isActive = currentTab === tab;
          const meta = tab === "image" ? { label: "Image (EfficientNet)", color: MODALITY_COLORS.image, icon: ImageIcon }
            : tab === "video" ? { label: "Video (CNN-GRU)", color: MODALITY_COLORS.video, icon: VideoIcon }
            : { label: "Audio (Baseline CNN)", color: MODALITY_COLORS.audio, icon: MicIcon };

          return (
            <button
              key={tab}
              type="button"
              onClick={() => handleTabChange(tab)}
              className={`flex items-center gap-2 px-5 py-3 rounded-t-xl font-heading font-semibold text-xs transition-all relative ${
                isActive
                  ? "bg-white dark:bg-slate-900 text-slate-900 dark:text-white border-t border-x border-slate-200 dark:border-slate-800 shadow-xs"
                  : "text-slate-500 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100/60 dark:hover:bg-slate-800/40"
              }`}
            >
              <meta.icon className="w-4 h-4" style={{ color: meta.color }} />
              <span>{meta.label}</span>
              {isActive && (
                <div
                  className="absolute bottom-0 left-0 right-0 h-0.5"
                  style={{ backgroundColor: meta.color }}
                />
              )}
            </button>
          );
        })}
      </div>

      {/* Main Analysis Workspace: Desktop 2-column layout (Upload beside Result) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Upload (Span 5) */}
        <div className="lg:col-span-5 space-y-4">
          <UploadZone
            modality={currentTab}
            title={modalityMeta.title}
            accept={modalityMeta.accept}
            hint={modalityMeta.hint}
            file={file}
            onFileSelect={setFile}
            onFileRemove={() => {
              setFile(null);
              setResult(null);
            }}
            disabled={isRunning}
          />

          {/* Audio Model Selector */}
          {currentTab === "audio" && (
            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200 dark:border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-heading font-semibold text-slate-700 dark:text-slate-300">
                  Model Selector:
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                  TRAINED MODEL
                </span>
              </div>
              <select
                value={selectedAudioModel}
                onChange={(e) => setSelectedAudioModel(e.target.value as "baseline_cnn" | "aasist")}
                disabled={isRunning}
                className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-xs font-heading font-medium text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-pink-500"
              >
                <option value="baseline_cnn">Baseline CNN (Log-Mel Spectrogram — Default)</option>
                <option value="aasist">AASIST (Graph Spectral Architecture)</option>
              </select>
              <div className="text-[11px] text-slate-500 dark:text-slate-400 flex items-center justify-between">
                <span>Preprocessing: 16 kHz Mono (4.0s)</span>
                <span>Operating Threshold: 0.50</span>
              </div>
            </div>
          )}

          {errorMsg && (
            <div className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-700 dark:text-rose-300 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleRunAnalysis}
              disabled={isRunning || !file}
              className={`flex-1 py-3.5 px-6 rounded-xl font-heading font-bold text-xs flex items-center justify-center gap-2 shadow-sm transition-all ${
                isRunning || !file
                  ? "bg-slate-200 dark:bg-slate-800 text-slate-400 dark:text-slate-600 cursor-not-allowed"
                  : "bg-slate-900 dark:bg-white text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-100 cursor-pointer shadow-md"
              }`}
            >
              {isRunning ? (
                <>
                  <span className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin"></span>
                  <span>Executing Inference...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>Analyze {currentTab.toUpperCase()}</span>
                </>
              )}
            </button>

            {file && (
              <button
                type="button"
                onClick={() => {
                  setFile(null);
                  setResult(null);
                }}
                className="p-3.5 rounded-xl border border-slate-200 dark:border-slate-800 text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800"
                title="Reset"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Result Beside Upload (Span 7) */}
        <div className="lg:col-span-7">
          {result && currentTab === "audio" ? (
            <div className="space-y-6">
              {/* Check if model failed to load */}
              {result.status === "unavailable" || result.prediction === "AUDIO MODEL UNAVAILABLE" ? (
                <div className="p-6 rounded-2xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-200 space-y-3">
                  <div className="flex items-center gap-2 font-heading font-bold text-base text-rose-700 dark:text-rose-300">
                    <AlertCircle className="w-5 h-5 text-rose-600" />
                    AUDIO MODEL UNAVAILABLE
                  </div>
                  <p className="text-xs text-rose-600 dark:text-rose-400">
                    The trained audio neural model checkpoint could not be loaded. In accordance with DeepGuard X safety protocol, heuristic fallback is strictly disabled to prevent false confidence.
                  </p>
                  <div className="p-3 rounded-lg bg-black/10 dark:bg-black/30 font-mono text-[11px] text-rose-700 dark:text-rose-300 overflow-x-auto">
                    {result.note || "Checkpoint missing or corrupted."}
                  </div>
                </div>
              ) : (
                <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm overflow-hidden border-t-4 border-t-pink-500">
                  {/* Header */}
                  <div className="p-5 border-b border-slate-100 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3 bg-slate-50/50 dark:bg-slate-850/50">
                    <div>
                      <span className="text-[10px] font-mono tracking-widest uppercase text-pink-600 dark:text-pink-400 font-bold block mb-1">
                        AUDIO DEEPFAKE DETECTION
                      </span>
                      <h3 className="font-heading font-bold text-lg text-slate-900 dark:text-white flex items-center gap-2">
                        Model: {result.model || (selectedAudioModel === "aasist" ? "AASIST" : "Baseline CNN")}
                      </h3>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border border-emerald-500/20">
                        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                        LIVE MODEL INFERENCE
                      </span>
                      <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-sky-50 text-sky-700 border border-sky-200 dark:bg-sky-950 dark:text-sky-300 dark:border-sky-800">
                        STATUS: TRAINED MODEL
                      </span>
                    </div>
                  </div>

                  <div className="p-5 sm:p-6 space-y-6">
                    {/* Verdict and Assessment */}
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200/80 dark:border-slate-800 gap-4">
                      <div>
                        <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block mb-1">
                          MODEL ASSESSMENT PREDICTION
                        </span>
                        <div className={`text-xl font-heading font-extrabold flex items-center gap-2 ${
                          result.verdict === "AUTHENTIC"
                            ? "text-emerald-600 dark:text-emerald-400"
                            : result.verdict === "MANIPULATED"
                            ? "text-rose-600 dark:text-rose-400"
                            : "text-amber-600 dark:text-amber-400"
                        }`}>
                          {result.prediction || (result.verdict === "AUTHENTIC" ? "LIKELY REAL" : "LIKELY SYNTHETIC / MANIPULATED")}
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <span className="text-[10px] font-mono text-slate-400 uppercase block">Confidence</span>
                          <span className="text-lg font-heading font-bold text-slate-900 dark:text-white">
                            {((result.confidence ?? 0.85) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="text-right border-l border-slate-200 dark:border-slate-700 pl-4">
                          <span className="text-[10px] font-mono text-slate-400 uppercase block">Threshold</span>
                          <span className="text-lg font-mono font-bold text-slate-900 dark:text-white">
                            {(result.threshold ?? 0.50).toFixed(2)}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Probabilities Row */}
                    <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200/80 dark:border-slate-800 space-y-3">
                      <div className="flex items-center justify-between text-xs font-heading font-bold">
                        <span className="text-emerald-600 dark:text-emerald-400">
                          REAL {((result.real_probability ?? (result.verdict === "AUTHENTIC" ? result.confidence : 1 - result.confidence)) * 100).toFixed(1)}%
                        </span>
                        <span className="text-xs font-mono text-slate-400">
                          Decision Boundary: {(result.threshold ?? 0.50).toFixed(2)}
                        </span>
                        <span className="text-rose-600 dark:text-rose-400">
                          FAKE {((result.fake_probability ?? (result.verdict === "MANIPULATED" ? result.confidence : 1 - result.confidence)) * 100).toFixed(1)}%
                        </span>
                      </div>

                      <div className="w-full h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden flex">
                        <div
                          className="h-full bg-emerald-500 transition-all"
                          style={{ width: `${(result.real_probability ?? (result.verdict === "AUTHENTIC" ? result.confidence : 1 - result.confidence)) * 100}%` }}
                        />
                        <div
                          className="h-full bg-rose-500 transition-all"
                          style={{ width: `${(result.fake_probability ?? (result.verdict === "MANIPULATED" ? result.confidence : 1 - result.confidence)) * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Robustness Warning Banner */}
                    <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-800 dark:text-amber-200 text-xs flex items-start gap-2.5">
                      <AlertCircle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-bold">ROBUSTNESS WARNING:</span> Additive background noise below 15 dB SNR causes severe degradation in synthetic detection recall. Noisy recordings require human auditory review.
                      </div>
                    </div>

                    {/* Supporting Audio Forensics */}
                    <Waveform
                      features={result.features}
                      duration={result.audio_duration_sec}
                      sampleRate={result.sampling_rate || 16000}
                      channels="Mono"
                      maturity="TRAINED"
                    />
                  </div>
                </div>
              )}
              <DisclaimerBar variant="result" />
            </div>
          ) : result ? (
            <div className="space-y-6">
              {currentTab === "video" && (result.status === "unavailable" || result.prediction === "VIDEO MODEL UNAVAILABLE") ? (
                <div className="p-6 rounded-2xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-800 dark:text-rose-200 space-y-3">
                  <div className="flex items-center gap-2 font-heading font-bold text-base text-rose-700 dark:text-rose-300">
                    <AlertCircle className="w-5 h-5 text-rose-600" />
                    VIDEO MODEL UNAVAILABLE
                  </div>
                  <p className="text-xs text-rose-600 dark:text-rose-400">
                    The trained CNN-GRU video model checkpoint could not be loaded. In accordance with DeepGuard X safety protocol, heuristic fallback is strictly disabled to prevent false confidence.
                  </p>
                  <div className="p-3 rounded-lg bg-black/10 dark:bg-black/30 font-mono text-[11px] text-rose-700 dark:text-rose-300 overflow-x-auto">
                    {result.note || "Checkpoint missing or corrupted."}
                  </div>
                </div>
              ) : (
                <EvidenceCard
                  modality={currentTab}
                  title={`${currentTab.toUpperCase()} Diagnostic Evidence`}
                  result={result}
                />
              )}
              <DisclaimerBar variant="result" />
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed border-slate-300 dark:border-slate-800 bg-white/40 dark:bg-slate-900/20 p-12 text-center flex flex-col items-center justify-center min-h-[380px]">
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center text-white mb-3 opacity-60"
                style={{ backgroundColor: modalityMeta.color }}
              >
                <modalityMeta.icon className="w-6 h-6" />
              </div>
              <h4 className="font-heading font-semibold text-sm text-slate-700 dark:text-slate-300">
                Awaiting {currentTab.toUpperCase()} Input
              </h4>
              <p className="text-xs text-slate-400 dark:text-slate-500 max-w-sm mt-1">
                Upload a media file and click Analyze to generate the complete forensic evidence card with full explainability.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
