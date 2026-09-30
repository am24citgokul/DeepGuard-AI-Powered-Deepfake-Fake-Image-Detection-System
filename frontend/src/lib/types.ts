export type Verdict = "AUTHENTIC" | "MANIPULATED" | "UNCERTAIN";
export type FinalVerdict = "TRUSTED" | "REVIEW_REQUIRED" | "HIGH_RISK";
export type Maturity = "TRAINED" | "TRAINED_FALLBACK" | "PROTOTYPE";

export type ModalityType = "image" | "video" | "audio" | "fusion";

export type ConsistencyType = 
  | "UNANIMOUS_AUTHENTIC" 
  | "UNANIMOUS_MANIPULATED" 
  | "MIXED" 
  | "SINGLE_MODALITY" 
  | "ALL_UNCERTAIN";

export interface ModalityResult {
  verdict: Verdict;
  confidence: number; // 0..1
  model: string;
  maturity: Maturity;
  predicted_class?: string;
  note?: string;
  gradcam_url?: string;
  ela_url?: string;
  report_url?: string;
  frames_analyzed?: number;
  frame_scores?: number[]; // 0..1 per frame
  features?: Record<string, number>;
  prediction?: string;
  real_probability?: number;
  fake_probability?: number;
  threshold?: number;
  status?: string;
  checkpoint?: string;
  preprocessing?: Record<string, any>;
  supporting_forensics?: Record<string, any>;
  audio_duration_sec?: number;
  sampling_rate?: number;
}

export interface ScreeningResult {
  id: string;
  created_at: string; // ISO
  final_verdict: FinalVerdict;
  consistency: ConsistencyType;
  reasoning: string;
  modalities: {
    image?: ModalityResult;
    video?: ModalityResult;
    audio?: ModalityResult;
  };
}

export interface HealthResponse {
  status: string;
  models: {
    image: { name: string; ready: boolean };
    video: { name: string; ready: boolean };
    audio: { name: string; ready: boolean };
  };
}

export interface ModelDetail {
  modality: "image" | "video" | "audio";
  name: string;
  maturity: Maturity;
  dataset: string;
  architecture: string;
  limitations: string[];
}

export interface CheckpointComparison {
  name: string;
  binary_accuracy: number;
  real_recall: number;
  fake_recall: number;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface TrainingCurvePoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
}

export interface GeneralizationPoint {
  set: string;
  accuracy: number;
  n: number;
  note: string;
}

export interface MetricsResponse {
  active_checkpoint: string;
  comparison: CheckpointComparison[];
  confusion_matrix: ConfusionMatrixData;
  training_curve: TrainingCurvePoint[];
  generalization: GeneralizationPoint[];
}

export interface ScreeningReportListItem {
  id: string;
  created_at: string;
  final_verdict: FinalVerdict;
  consistency: ConsistencyType;
  modalities_present: ("image" | "video" | "audio")[];
  reasoning: string;
}
