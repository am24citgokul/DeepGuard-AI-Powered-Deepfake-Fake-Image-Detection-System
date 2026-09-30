import React, { useRef, useState, useEffect } from "react";
import { 
  Camera, 
  VideoOff, 
  Play, 
  RotateCcw, 
  Sparkles, 
  AlertCircle, 
  CheckCircle,
  Timer
} from "lucide-react";
import { EvidenceCard } from "@/components/screening/EvidenceCard";
import { DisclaimerBar } from "@/components/screening/DisclaimerBar";
import { api } from "@/lib/api";
import { MODALITY_COLORS } from "@/lib/format";
import type { ModalityResult } from "@/lib/types";

export const Webcam: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [result, setResult] = useState<ModalityResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      // Cleanup camera stream on unmount
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  const handleStartCamera = async () => {
    setErrorMsg(null);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      setStream(mediaStream);
      setCameraActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err: any) {
      console.error("Camera access error:", err);
      setErrorMsg(
        "Camera permission was denied or no camera device is connected. Please enable camera access in your browser settings to perform live screening."
      );
    }
  };

  const handleStopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setCameraActive(false);
  };

  const handleTriggerCaptureWithCountdown = () => {
    if (!cameraActive || countdown !== null) return;
    setCountdown(3);

    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev === null || prev <= 1) {
          clearInterval(timer);
          executeFrameCapture();
          return null;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const executeFrameCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
    setCapturedImage(dataUrl);
    setResult(null);
  };

  const handleAnalyzeSnapshot = async () => {
    if (!capturedImage) return;

    setIsRunning(true);
    setErrorMsg(null);

    try {
      const res = await api.analyzeWebcamFrame(capturedImage);
      setResult(res);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to process webcam snapshot.");
    } finally {
      setIsRunning(false);
    }
  };

  const handleReset = () => {
    setCapturedImage(null);
    setResult(null);
    setErrorMsg(null);
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="header-gradient-banner rounded-3xl p-6 sm:p-8 border border-slate-200/80 dark:border-slate-800 shadow-xs">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-100/70 dark:bg-amber-950/60 text-amber-700 dark:text-amber-300 text-xs font-mono font-medium mb-3">
            <Camera className="w-3.5 h-3.5" />
            Live Biometric Feed
          </div>
          <h1 className="font-heading font-extrabold text-2xl sm:text-3xl text-slate-900 dark:text-white tracking-tight">
            Live Webcam Screening
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 font-sans leading-relaxed">
            Capture a real-time snapshot with our 3-second biometric stabilization countdown and test for facial autoencoding, warping, or deepfake injection in real time.
          </p>
        </div>
      </div>

      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Camera Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left: Viewfinder & Controls (Span 6) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden shadow-xs">
            {/* Viewfinder Bar */}
            <div className="p-4 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-2 font-heading font-semibold text-xs text-slate-900 dark:text-white">
                <span className={`w-2.5 h-2.5 rounded-full ${cameraActive ? "bg-emerald-500 animate-pulse" : "bg-slate-400"}`}></span>
                <span>{cameraActive ? "Camera Stream Active" : "Camera Idle"}</span>
              </div>

              {cameraActive && (
                <button
                  type="button"
                  onClick={handleStopCamera}
                  className="text-xs font-mono text-rose-600 hover:text-rose-700 flex items-center gap-1"
                >
                  <VideoOff className="w-3.5 h-3.5" /> Stop
                </button>
              )}
            </div>

            {/* Video Viewport */}
            <div className="relative aspect-4/3 bg-slate-950 flex items-center justify-center overflow-hidden">
              {/* Live Video Element */}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`w-full h-full object-cover transform scale-x-[-1] ${
                  cameraActive && !capturedImage ? "block" : "hidden"
                }`}
              />

              {/* Frozen Snapshot Preview */}
              {capturedImage && (
                <img
                  src={capturedImage}
                  alt="Captured snapshot"
                  className="w-full h-full object-cover transform scale-x-[-1]"
                />
              )}

              {/* Offline / Idle state */}
              {!cameraActive && !capturedImage && (
                <div className="flex flex-col items-center justify-center p-6 text-center text-slate-500">
                  <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center text-slate-400 mb-3">
                    <Camera className="w-7 h-7" />
                  </div>
                  <p className="text-xs font-heading font-semibold text-slate-300">
                    Camera is currently off
                  </p>
                  <p className="text-[11px] text-slate-500 max-w-xs mt-1">
                    Click "Start Camera" below to grant temporary browser camera permission.
                  </p>
                </div>
              )}

              {/* 3-Second Countdown Overlay */}
              {countdown !== null && (
                <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-xs flex flex-col items-center justify-center text-white z-20 animate-fadeIn">
                  <div className="w-24 h-24 rounded-full border-4 border-violet-500 flex items-center justify-center font-mono font-extrabold text-5xl text-violet-400 animate-pulse">
                    {countdown}
                  </div>
                  <span className="text-xs font-mono mt-3 uppercase tracking-wider text-slate-300">
                    Hold still for biometric capture
                  </span>
                </div>
              )}

              {/* Viewfinder Biometric Reticle */}
              {cameraActive && !capturedImage && countdown === null && (
                <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
                  <div className="w-48 h-60 border-2 border-violet-400/50 rounded-3xl relative">
                    <span className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-violet-400"></span>
                    <span className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-violet-400"></span>
                    <span className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-violet-400"></span>
                    <span className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-violet-400"></span>
                  </div>
                </div>
              )}
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div className="p-3 bg-rose-50 dark:bg-rose-950/40 border-t border-rose-200 dark:border-rose-900 text-rose-700 dark:text-rose-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Action Buttons */}
            <div className="p-4 bg-slate-50/60 dark:bg-slate-850/60 border-t border-slate-100 dark:border-slate-800 flex flex-wrap items-center gap-3">
              {!cameraActive && !capturedImage ? (
                <button
                  type="button"
                  onClick={handleStartCamera}
                  className="flex-1 py-3 px-5 rounded-xl font-heading font-bold text-xs bg-violet-600 hover:bg-violet-700 text-white shadow-sm flex items-center justify-center gap-2"
                >
                  <Camera className="w-4 h-4" />
                  Start Camera
                </button>
              ) : capturedImage ? (
                <>
                  <button
                    type="button"
                    onClick={handleAnalyzeSnapshot}
                    disabled={isRunning}
                    className="flex-1 py-3 px-5 rounded-xl font-heading font-bold text-xs bg-emerald-600 hover:bg-emerald-700 text-white shadow-sm flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    {isRunning ? (
                      <>
                        <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                        Analyzing Snapshot...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-current" />
                        Analyze Captured Frame
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={handleReset}
                    className="p-3 rounded-xl border border-slate-200 dark:border-slate-700 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                    title="Retake photo"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  onClick={handleTriggerCaptureWithCountdown}
                  disabled={countdown !== null}
                  className="flex-1 py-3 px-5 rounded-xl font-heading font-bold text-xs bg-violet-600 hover:bg-violet-700 text-white shadow-sm flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  <Timer className="w-4 h-4" />
                  Capture Snapshot (3s Countdown)
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right: Evidence Card Result (Span 6) */}
        <div className="lg:col-span-6">
          {result ? (
            <div className="space-y-6">
              <EvidenceCard
                modality="image"
                title="Live Biometric Saliency Evidence"
                result={result}
              />
              <DisclaimerBar variant="result" />
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed border-slate-300 dark:border-slate-800 bg-white/40 dark:bg-slate-900/20 p-12 text-center flex flex-col items-center justify-center min-h-[380px]">
              <div className="w-12 h-12 rounded-xl bg-amber-500/10 text-amber-500 flex items-center justify-center mb-3">
                <Sparkles className="w-6 h-6" />
              </div>
              <h4 className="font-heading font-semibold text-sm text-slate-700 dark:text-slate-300">
                Awaiting Frame Capture
              </h4>
              <p className="text-xs text-slate-400 dark:text-slate-500 max-w-sm mt-1">
                Start your camera, position your face within the biometric reticle, and capture a frame to execute instant inference.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
