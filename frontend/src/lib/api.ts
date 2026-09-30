import type {
  HealthResponse,
  ScreeningResult,
  ModalityResult,
  ModelDetail,
  MetricsResponse,
} from "./types";
import {
  MOCK_HEALTH,
  MOCK_SCREENING_RESULTS,
  MOCK_MODELS,
  MOCK_METRICS,
  SVG_SILHOUETTE_GRADCAM,
  SVG_SILHOUETTE_ELA,
} from "./mockData";

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").trim().replace(/\/$/, "");

export function isConfiguredBackend(): boolean {
  return Boolean(BASE_URL);
}

// In-memory reports store for Demo mode session additions
let demoReports: ScreeningResult[] = [...MOCK_SCREENING_RESULTS];

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      let errorMsg = `Server error (${res.status})`;
      try {
        const errorJson = await res.json();
        if (errorJson?.error || errorJson?.message) {
          errorMsg = errorJson.error || errorJson.message;
        }
      } catch {
        // fallback to text or status
      }
      throw new ApiError(errorMsg, res.status);
    }

    return await res.json();
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === "AbortError") {
      throw new ApiError("Request timed out. Please try again.", 408);
    }
    throw new ApiError(err.message || "Network error. Backend unreachable.", err.status);
  }
}

export const api = {
  // Health check
  async getHealth(): Promise<HealthResponse> {
    if (!isConfiguredBackend()) {
      return MOCK_HEALTH;
    }
    try {
      return await request<HealthResponse>("/api/health");
    } catch {
      // Fallback to mock if configured backend is not currently responding
      return {
        ...MOCK_HEALTH,
        status: "offline",
      };
    }
  },

  // Full multimodal screening
  async screen(formData: FormData): Promise<ScreeningResult> {
    if (!isConfiguredBackend()) {
      // Simulate realistic network delay
      await new Promise((r) => setTimeout(r, 1400));
      
      const hasImage = formData.has("image");
      const hasVideo = formData.has("video");
      const hasAudio = formData.has("audio");

      // Pick a suitable mock or build a synthesized demo result
      let base = MOCK_SCREENING_RESULTS[0]; // Authentic
      if (hasImage && hasVideo && hasAudio) {
        base = MOCK_SCREENING_RESULTS[1]; // Mixed
      } else if (hasImage && !hasVideo && !hasAudio) {
        base = MOCK_SCREENING_RESULTS[3]; // Single image / uncertain test
      } else if (hasImage && hasAudio && !hasVideo) {
        base = MOCK_SCREENING_RESULTS[4]; // Image + audio missing video
      }

      const newResult: ScreeningResult = {
        ...base,
        id: `SCN-DEMO-${Date.now().toString().slice(-6)}`,
        created_at: new Date().toISOString(),
        modalities: {
          ...(hasImage ? { image: base.modalities.image || MOCK_SCREENING_RESULTS[0].modalities.image } : {}),
          ...(hasVideo ? { video: base.modalities.video || MOCK_SCREENING_RESULTS[0].modalities.video } : {}),
          ...(hasAudio ? { audio: base.modalities.audio || MOCK_SCREENING_RESULTS[0].modalities.audio } : {}),
        },
      };

      demoReports.unshift(newResult);
      return newResult;
    }

    return await request<ScreeningResult>("/api/screen", {
      method: "POST",
      body: formData,
    });
  },

  // Single modality analysis: /api/analyze/image | /api/analyze/video | /api/analyze/audio
  async analyzeModality(modality: "image" | "video" | "audio", file: File, modelName?: string): Promise<ModalityResult> {
    if (!isConfiguredBackend()) {
      await new Promise((r) => setTimeout(r, 1000));
      if (modality === "image") {
        return {
          verdict: "AUTHENTIC",
          confidence: 0.912,
          model: "EfficientNet-B0 (Rebalanced)",
          maturity: "TRAINED",
          predicted_class: "Original",
          note: "Demo evaluation: No structural synthesis boundary detected.",
          gradcam_url: SVG_SILHOUETTE_GRADCAM,
          ela_url: SVG_SILHOUETTE_ELA,
        };
      } else if (modality === "video") {
        return {
          verdict: "MANIPULATED",
          confidence: 0.884,
          model: "CNN-GRU Sequence Model",
          maturity: "TRAINED",
          frames_analyzed: 16,
          frame_scores: [0.65, 0.72, 0.89, 0.91, 0.95, 0.92, 0.88, 0.85, 0.82, 0.86, 0.90, 0.88, 0.85, 0.82, 0.76, 0.81],
          note: "Temporal sequence analysis detected boundary jitter across central sequence frames.",
        };
      } else {
        const isAasist = modelName === "aasist";
        return {
          verdict: "AUTHENTIC",
          prediction: "LIKELY REAL",
          confidence: 0.985,
          real_probability: 0.985,
          fake_probability: 0.015,
          threshold: 0.50,
          model: isAasist ? "AASIST" : "Baseline CNN",
          maturity: "TRAINED",
          status: "trained_model",
          features: {
            mean_spectral_flatness: 0.0195,
            mean_zcr: 0.091,
            mean_high_mfcc_var: 154.2,
          },
          supporting_forensics: {
            mean_spectral_flatness: 0.0195,
            mean_zcr: 0.091,
            mean_high_mfcc_var: 154.2,
            note: "Supporting acoustic features calculated for forensic reference only; does NOT determine final model verdict.",
          },
          audio_duration_sec: 4.0,
          sampling_rate: 16000,
          note: `${isAasist ? "AASIST" : "Baseline CNN"} deep learning inference: 98.5% model confidence. Status: Trained Model.`,
        };
      }
    }

    const formData = new FormData();
    formData.append("file", file);
    if (modality === "audio" && modelName) {
      formData.append("model", modelName);
    }
    return await request<ModalityResult>(`/api/analyze/${modality}`, {
      method: "POST",
      body: formData,
    });
  },

  async getAudioMetrics(): Promise<any> {
    if (!isConfiguredBackend()) {
      return {
        accuracy: 0.9847,
        fake_recall: 0.9751,
        real_recall: 0.9944,
        f1: 0.9846,
        roc_auc: 0.9965,
        fpr: 0.0056,
        fnr: 0.0249,
      };
    }
    return await request<any>("/api/audio/metrics");
  },

  // Live webcam frame snapshot: /api/webcam
  async analyzeWebcamFrame(imageBase64: string): Promise<ModalityResult> {
    if (!isConfiguredBackend()) {
      await new Promise((r) => setTimeout(r, 900));
      return {
        verdict: "AUTHENTIC",
        confidence: 0.928,
        model: "EfficientNet-B0 (Live Webcam Mode)",
        maturity: "TRAINED",
        predicted_class: "Original",
        note: "Live biometric scan verified authentic skin texture and optical naturalness.",
        gradcam_url: SVG_SILHOUETTE_GRADCAM,
        ela_url: SVG_SILHOUETTE_ELA,
      };
    }

    return await request<ModalityResult>("/api/webcam", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: imageBase64 }),
    });
  },

  // Get reports list: /api/reports
  async getReports(): Promise<ScreeningResult[]> {
    if (!isConfiguredBackend()) {
      return [...demoReports];
    }
    return await request<ScreeningResult[]>("/api/reports");
  },

  // Get single report by ID: /api/reports/:id
  async getReportById(id: string): Promise<ScreeningResult> {
    if (!isConfiguredBackend()) {
      const found = demoReports.find((r) => r.id === id);
      if (found) return found;
      return demoReports[0];
    }
    return await request<ScreeningResult>(`/api/reports/${encodeURIComponent(id)}`);
  },

  // Get models information: /api/models
  async getModels(): Promise<ModelDetail[]> {
    if (!isConfiguredBackend()) {
      return MOCK_MODELS;
    }
    return await request<ModelDetail[]>("/api/models");
  },

  // Get evaluation metrics: /api/metrics
  async getMetrics(): Promise<MetricsResponse> {
    if (!isConfiguredBackend()) {
      return MOCK_METRICS;
    }
    return await request<MetricsResponse>("/api/metrics");
  },
};
