import React, { useRef, useState, useEffect } from "react";
import { 
  Upload, 
  X, 
  FileCheck, 
  Image as ImageIcon, 
  Video as VideoIcon, 
  Mic as MicIcon,
  Play,
  Pause
} from "lucide-react";
import { MODALITY_COLORS } from "@/lib/format";

interface UploadZoneProps {
  modality: "image" | "video" | "audio";
  title: string;
  accept: string;
  hint: string;
  file: File | null;
  onFileSelect: (file: File) => void;
  onFileRemove: () => void;
  disabled?: boolean;
}

export const UploadZone: React.FC<UploadZoneProps> = ({
  modality,
  title,
  accept,
  hint,
  file,
  onFileSelect,
  onFileRemove,
  disabled = false,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const color = MODALITY_COLORS[modality];
  const Icon = modality === "image" ? ImageIcon : modality === "video" ? VideoIcon : MicIcon;

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const toggleAudio = () => {
    if (!audioRef.current) return;
    if (isPlayingAudio) {
      audioRef.current.pause();
      setIsPlayingAudio(false);
    } else {
      audioRef.current.play();
      setIsPlayingAudio(true);
    }
  };

  return (
    <div
      className={`rounded-2xl border transition-all bg-white dark:bg-slate-900 overflow-hidden flex flex-col h-full ${
        file
          ? "border-slate-300 dark:border-slate-700 shadow-sm"
          : isDragOver
          ? "border-slate-400 dark:border-slate-500 shadow-md ring-2 ring-violet-500/20"
          : "border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700"
      }`}
      style={{ borderTop: `4px solid ${color}` }}
    >
      {/* Header */}
      <div className="p-4 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center text-white shrink-0"
            style={{ backgroundColor: color }}
          >
            <Icon className="w-4 h-4" />
          </div>
          <div>
            <h4 className="font-heading font-semibold text-sm text-slate-900 dark:text-white">
              {title}
            </h4>
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">
              {modality.toUpperCase()} STREAM
            </span>
          </div>
        </div>

        <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400">
          Optional
        </span>
      </div>

      {/* Body: Upload area or preview */}
      <div className="p-4 flex-1 flex flex-col justify-center">
        {file ? (
          /* Preview Mode */
          <div className="flex flex-col h-full justify-between gap-3">
            <div className="rounded-xl overflow-hidden bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 flex items-center justify-center min-h-[170px] relative">
              {modality === "image" && previewUrl && (
                <img
                  src={previewUrl}
                  alt="Upload preview"
                  className="w-full h-44 object-cover"
                />
              )}

              {modality === "video" && previewUrl && (
                <video
                  src={previewUrl}
                  controls
                  className="w-full h-44 object-cover"
                />
              )}

              {modality === "audio" && previewUrl && (
                <div className="w-full p-4 flex flex-col items-center justify-center gap-3">
                  <audio
                    ref={audioRef}
                    src={previewUrl}
                    onEnded={() => setIsPlayingAudio(false)}
                    className="hidden"
                  />
                  <button
                    type="button"
                    onClick={toggleAudio}
                    className="w-12 h-12 rounded-full flex items-center justify-center text-white shadow-md transition-transform hover:scale-105"
                    style={{ backgroundColor: color }}
                  >
                    {isPlayingAudio ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
                  </button>
                  <span className="text-xs font-mono text-slate-500">
                    {isPlayingAudio ? "Playing audio..." : "Click to test audio playback"}
                  </span>
                </div>
              )}

              {/* Remove button */}
              <button
                type="button"
                onClick={onFileRemove}
                className="absolute top-2 right-2 p-1.5 rounded-full bg-slate-900/80 hover:bg-rose-600 text-white transition-colors"
                title="Remove file"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* File info bar */}
            <div className="flex items-center justify-between text-xs bg-slate-50 dark:bg-slate-800/60 p-2.5 rounded-lg border border-slate-200/80 dark:border-slate-700">
              <div className="flex items-center gap-2 truncate pr-2">
                <FileCheck className="w-4 h-4 text-emerald-500 shrink-0" />
                <span className="font-mono text-slate-700 dark:text-slate-300 truncate">
                  {file.name}
                </span>
              </div>
              <span className="font-mono text-[10px] text-slate-400 shrink-0">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </span>
            </div>
          </div>
        ) : (
          /* Dropzone Mode */
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all min-h-[190px] ${
              isDragOver
                ? "border-violet-500 bg-violet-50/50 dark:bg-violet-950/20"
                : "border-slate-200 dark:border-slate-700/80 hover:border-slate-300 dark:hover:border-slate-600 bg-slate-50/50 dark:bg-slate-800/20"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept={accept}
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  onFileSelect(e.target.files[0]);
                }
              }}
              className="hidden"
            />
            <div
              className="w-12 h-12 rounded-full flex items-center justify-center mb-3 transition-transform group-hover:scale-110"
              style={{ backgroundColor: `${color}18`, color }}
            >
              <Upload className="w-6 h-6" />
            </div>
            <p className="font-heading font-semibold text-xs text-slate-800 dark:text-slate-200 mb-1">
              Drag & drop or <span style={{ color }}>browse file</span>
            </p>
            <p className="text-[11px] text-slate-400 dark:text-slate-500 max-w-[200px] leading-tight">
              {hint}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
