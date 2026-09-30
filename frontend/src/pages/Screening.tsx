import React, { useState } from "react";
import { 
  Sparkles, 
  ArrowRight, 
  RotateCcw, 
  AlertCircle, 
  CheckCircle,
  Play
} from "lucide-react";
import { UploadZone } from "@/components/screening/UploadZone";
import { Stepper } from "@/components/screening/Stepper";
import { VerdictBanner } from "@/components/screening/VerdictBanner";
import { FusionDiagram } from "@/components/diagrams/FusionDiagram";
import { ConsistencyMatrix } from "@/components/diagrams/ConsistencyMatrix";
import { EvidenceCard } from "@/components/screening/EvidenceCard";
import { DisclaimerBar } from "@/components/screening/DisclaimerBar";
import { api, ApiError } from "@/lib/api";
import type { ScreeningResult } from "@/lib/types";

export const Screening: React.FC = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);

  const [isRunning, setIsRunning] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<ScreeningResult | null>(null);

  const hasAnyFile = Boolean(imageFile || videoFile || audioFile);

  const handleRunScreening = async () => {
    if (!hasAnyFile) {
      setErrorMsg("Please upload at least one modality artifact (image, video, or audio) to proceed.");
      return;
    }

    setErrorMsg(null);
    setIsRunning(true);
    setStepIndex(0);
    setResult(null);

    // Simulate animated stepper progression
    const stepTimer = setInterval(() => {
      setStepIndex((prev) => (prev < 4 ? prev + 1 : prev));
    }, 450);

    try {
      const formData = new FormData();
      if (imageFile) formData.append("image", imageFile);
      if (videoFile) formData.append("video", videoFile);
      if (audioFile) formData.append("audio", audioFile);

      const res = await api.screen(formData);
      clearInterval(stepTimer);
      setStepIndex(4);
      setResult(res);

      // Smooth scroll to results
      setTimeout(() => {
        document.getElementById("screening-results")?.scrollIntoView({ behavior: "smooth" });
      }, 200);
    } catch (err: any) {
      clearInterval(stepTimer);
      setErrorMsg(err.message || "Screening inference encountered a server error. Please try again.");
    } finally {
      setIsRunning(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setVideoFile(null);
    setAudioFile(null);
    setResult(null);
    setErrorMsg(null);
    setStepIndex(0);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="header-gradient-banner rounded-3xl p-6 sm:p-8 border border-slate-200/80 dark:border-slate-800 shadow-xs">
        <div className="max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-100/70 dark:bg-violet-950/60 text-violet-700 dark:text-violet-300 text-xs font-mono font-medium mb-3">
            <Sparkles className="w-3.5 h-3.5" />
            Core Verification Engine
          </div>
          <h1 className="font-heading font-extrabold text-2xl sm:text-3xl text-slate-900 dark:text-white tracking-tight">
            Multimodal Identity Screening
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 font-sans leading-relaxed">
            Upload any combination of an image, video, and audio file. The engine will evaluate each provided stream in parallel and converge evidence into a consolidated authenticity audit.
          </p>
        </div>
      </div>

      {/* Upload Section: Three zones side by side */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white">
            1. Select Identity Artifacts
          </h3>
          {hasAnyFile && (
            <button
              type="button"
              onClick={handleReset}
              className="text-xs font-mono text-slate-500 hover:text-rose-600 dark:hover:text-rose-400 flex items-center gap-1.5 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset uploads
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <UploadZone
            modality="image"
            title="Image Portrait"
            accept="image/*,.jpg,.jpeg,.png,.webp,.bmp"
            hint="JPEG, PNG, WebP up to 10MB. Neutral front-facing portrait."
            file={imageFile}
            onFileSelect={setImageFile}
            onFileRemove={() => setImageFile(null)}
            disabled={isRunning}
          />

          <UploadZone
            modality="video"
            title="Facial Video"
            accept="video/*,.mp4,.avi,.mov,.mkv"
            hint="MP4, WebM, MOV up to 50MB. 2–10 seconds of facial movement."
            file={videoFile}
            onFileSelect={setVideoFile}
            onFileRemove={() => setVideoFile(null)}
            disabled={isRunning}
          />

          <UploadZone
            modality="audio"
            title="Voice Recording"
            accept="audio/*,.wav,.mp3,.flac,.m4a,.aac"
            hint="WAV, MP3, AAC up to 20MB. Clear speech recording (3–15s)."
            file={audioFile}
            onFileSelect={setAudioFile}
            onFileRemove={() => setAudioFile(null)}
            disabled={isRunning}
          />
        </div>

        {/* Action Button & Error state */}
        <div className="pt-2 flex flex-col items-center gap-3">
          {errorMsg && (
            <div className="p-3 rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900 text-rose-700 dark:text-rose-300 text-xs flex items-center gap-2 max-w-xl text-center">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          <button
            type="button"
            onClick={handleRunScreening}
            disabled={isRunning || !hasAnyFile}
            className={`px-8 py-4 rounded-2xl font-heading font-bold text-sm flex items-center gap-3 shadow-md transition-all ${
              isRunning || !hasAnyFile
                ? "bg-slate-200 dark:bg-slate-800 text-slate-400 dark:text-slate-600 cursor-not-allowed"
                : "bg-gradient-to-r from-violet-600 via-indigo-600 to-teal-600 hover:from-violet-700 hover:to-teal-700 text-white hover:shadow-lg transform hover:-translate-y-0.5 cursor-pointer"
            }`}
          >
            {isRunning ? (
              <>
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-white"></span>
                </span>
                <span>Evaluating Multimodal Streams...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Run Screening</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Stepper while running */}
      {isRunning && (
        <div className="py-2 animate-fadeIn">
          <Stepper currentStepIndex={stepIndex} />
        </div>
      )}

      {/* RESULT LAYOUT (top to bottom) */}
      {result && (
        <div id="screening-results" className="space-y-8 pt-4 border-t border-slate-200 dark:border-slate-800">
          {/* 1) Verdict banner */}
          <VerdictBanner
            verdict={result.final_verdict}
            consistency={result.consistency}
            reasoning={result.reasoning}
            createdAt={result.created_at}
            reportId={result.id}
          />

          {/* 2) FUSION DIAGRAM */}
          <FusionDiagram screening={result} />

          {/* 3) CROSS-MODAL CONSISTENCY MATRIX */}
          <ConsistencyMatrix screening={result} />

          {/* 4) EVIDENCE CARDS, one per modality */}
          <div className="space-y-6">
            <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white">
              Forensic Evidence Breakdown
            </h3>

            <div className="grid grid-cols-1 gap-6">
              {/* Image Evidence Card */}
              <EvidenceCard
                modality="image"
                title="Image Forensic Evidence"
                result={result.modalities.image}
              />

              {/* Video Evidence Card */}
              <EvidenceCard
                modality="video"
                title="Video Temporal Evidence"
                result={result.modalities.video}
              />

              {/* Audio Evidence Card */}
              <EvidenceCard
                modality="audio"
                title="Audio Spectral Evidence"
                result={result.modalities.audio}
              />
            </div>
          </div>

          {/* 5) Calm disclaimer bar */}
          <DisclaimerBar variant="result" />
        </div>
      )}
    </div>
  );
};
