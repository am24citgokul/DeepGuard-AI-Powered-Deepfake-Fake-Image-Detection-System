import type { 
  ScreeningResult, 
  HealthResponse, 
  ModelDetail, 
  MetricsResponse 
} from "./types";

// Neutral geometric silhouette SVG placeholders (No real or fake human faces)
export const SVG_SILHOUETTE_ORIGINAL = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="100%" height="100%">
  <rect width="320" height="320" fill="%23F1F5F9"/>
  <circle cx="160" cy="110" r="54" fill="none" stroke="%233B82F6" stroke-width="3" stroke-dasharray="4 3"/>
  <path d="M90 270 C90 190, 230 190, 230 270" fill="none" stroke="%233B82F6" stroke-width="3"/>
  <!-- Biometric geometric reference grid -->
  <line x1="160" y1="50" x2="160" y2="270" stroke="%2394A3B8" stroke-width="1" stroke-dasharray="2 2"/>
  <line x1="80" y1="110" x2="240" y2="110" stroke="%2394A3B8" stroke-width="1" stroke-dasharray="2 2"/>
  <line x1="100" y1="150" x2="220" y2="150" stroke="%2394A3B8" stroke-width="1" stroke-dasharray="2 2"/>
  <rect x="135" y="95" width="14" height="6" rx="2" fill="%2364748B"/>
  <rect x="171" y="95" width="14" height="6" rx="2" fill="%2364748B"/>
  <circle cx="160" cy="122" r="3" fill="%2364748B"/>
  <path d="M148 140 Q160 148 172 140" fill="none" stroke="%2364748B" stroke-width="2"/>
  <text x="16" y="304" font-family="monospace" font-size="11" fill="%2364748B">GEOMETRIC BIOMETRIC TARGET</text>
</svg>`;

export const SVG_SILHOUETTE_GRADCAM = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="100%" height="100%">
  <defs>
    <radialGradient id="heat" cx="52%" cy="46%" r="35%">
      <stop offset="0%" stop-color="%23DC2626" stop-opacity="0.85"/>
      <stop offset="45%" stop-color="%23F59E0B" stop-opacity="0.75"/>
      <stop offset="75%" stop-color="%2310B981" stop-opacity="0.45"/>
      <stop offset="100%" stop-color="%233B82F6" stop-opacity="0.05"/>
    </radialGradient>
  </defs>
  <rect width="320" height="320" fill="%230F172A"/>
  <circle cx="160" cy="110" r="54" fill="none" stroke="%23475569" stroke-width="2"/>
  <path d="M90 270 C90 190, 230 190, 230 270" fill="none" stroke="%23475569" stroke-width="2"/>
  <rect width="320" height="320" fill="url(%23heat)"/>
  <circle cx="166" cy="135" r="28" fill="%23DC2626" opacity="0.4" filter="blur(6px)"/>
  <text x="16" y="304" font-family="monospace" font-size="11" fill="%23F8FAFC">GRAD-CAM ACTIVATION SALIENCY</text>
</svg>`;

export const SVG_SILHOUETTE_ELA = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="100%" height="100%">
  <rect width="320" height="320" fill="%23050811"/>
  <!-- High-frequency compression artifact noise simulation -->
  <g stroke="%2338BDF8" stroke-width="0.8" opacity="0.35">
    <path d="M40 80 Q80 85 120 78 T200 84 T280 76"/>
    <path d="M30 140 Q90 135 150 145 T250 138"/>
    <path d="M50 210 Q110 205 180 215 T290 208"/>
  </g>
  <circle cx="160" cy="110" r="54" fill="none" stroke="%23F43F5E" stroke-width="1.8" opacity="0.8" stroke-dasharray="2 4"/>
  <ellipse cx="160" cy="138" rx="22" ry="12" fill="none" stroke="%23FBBF24" stroke-width="2" stroke-dasharray="1 2"/>
  <text x="16" y="304" font-family="monospace" font-size="11" fill="%2394A3B8">ELA RESIDUAL ERROR MAP (10x)</text>
</svg>`;

export const MOCK_HEALTH: HealthResponse = {
  status: "ready",
  models: {
    image: { name: "EfficientNet-B0 (Rebalanced)", ready: true },
    video: { name: "CNN-GRU Sequence Model", ready: true },
    audio: { name: "Baseline CNN", ready: true },
  },
};

export const MOCK_SCREENING_RESULTS: ScreeningResult[] = [
  // 1. TRUSTED (Unanimous Authentic)
  {
    id: "SCN-2026-0928-8421",
    created_at: "2026-09-28T14:32:00Z",
    final_verdict: "TRUSTED",
    consistency: "UNANIMOUS_AUTHENTIC",
    reasoning: "Image, video temporal dynamics, and acoustic spectra exhibit authentic biometric features with consistent natural distribution.",
    modalities: {
      image: {
        verdict: "AUTHENTIC",
        confidence: 0.942,
        model: "EfficientNet-B0 (Rebalanced)",
        maturity: "TRAINED",
        predicted_class: "Original",
        note: "Uniform error level distribution across facial landmarks; no boundary gradient discontinuity.",
        gradcam_url: SVG_SILHOUETTE_GRADCAM,
        ela_url: SVG_SILHOUETTE_ELA,
      },
      video: {
        verdict: "AUTHENTIC",
        confidence: 0.918,
        model: "CNN-GRU Sequence Model",
        maturity: "TRAINED",
        frames_analyzed: 16,
        frame_scores: [0.94, 0.92, 0.91, 0.93, 0.95, 0.90, 0.92, 0.91, 0.93, 0.94, 0.92, 0.93, 0.91, 0.90, 0.92, 0.93],
        note: "Temporal sequence continuity across 16 frames shows natural micro-expression kinetics.",
      },
      audio: {
        verdict: "AUTHENTIC",
        prediction: "LIKELY REAL",
        confidence: 0.985,
        real_probability: 0.985,
        fake_probability: 0.015,
        threshold: 0.50,
        model: "Baseline CNN",
        maturity: "TRAINED",
        status: "trained_model",
        features: {
          mean_spectral_flatness: 0.0182,
          mean_zcr: 0.0841,
          mean_high_mfcc_var: 168.4,
          formant_jitter: 0.012,
        },
        supporting_forensics: {
          mean_spectral_flatness: 0.0182,
          mean_zcr: 0.0841,
          mean_high_mfcc_var: 168.4,
          note: "Supporting acoustic features calculated for forensic reference only; does NOT determine final model verdict.",
        },
        note: "Baseline CNN deep learning inference: 98.5% confidence. Status: Trained Model.",
      },
    },
  },

  // 2. REVIEW_REQUIRED (Mixed signals)
  {
    id: "SCN-2026-0928-7914",
    created_at: "2026-09-28T12:15:30Z",
    final_verdict: "REVIEW_REQUIRED",
    consistency: "MIXED",
    reasoning: "Cross-modal conflict: Video temporal sequence flagged synthetic boundary artifacts while still image classifier indicates authentic skin tone.",
    modalities: {
      image: {
        verdict: "AUTHENTIC",
        confidence: 0.724,
        model: "EfficientNet-B0 (Rebalanced)",
        maturity: "TRAINED",
        predicted_class: "Original",
        note: "Static skin texture resembles authentic lighting, but marginal confidence near decision threshold.",
        gradcam_url: SVG_SILHOUETTE_GRADCAM,
        ela_url: SVG_SILHOUETTE_ELA,
      },
      video: {
        verdict: "MANIPULATED",
        confidence: 0.885,
        model: "CNN-GRU Sequence Model",
        maturity: "TRAINED",
        frames_analyzed: 16,
        frame_scores: [0.35, 0.42, 0.78, 0.89, 0.91, 0.94, 0.92, 0.88, 0.79, 0.82, 0.86, 0.90, 0.88, 0.85, 0.82, 0.76],
        note: "Frames 4–12 exhibit significant inter-frame jitter and warping characteristic of Deepfakes blending.",
      },
      audio: {
        verdict: "UNCERTAIN",
        prediction: "UNCERTAIN",
        confidence: 0.518,
        real_probability: 0.518,
        fake_probability: 0.482,
        threshold: 0.50,
        model: "Baseline CNN",
        maturity: "TRAINED",
        status: "trained_model",
        features: {
          mean_spectral_flatness: 0.0245,
          mean_zcr: 0.119,
          mean_high_mfcc_var: 142.1,
          formant_jitter: 0.038,
        },
        supporting_forensics: {
          mean_spectral_flatness: 0.0245,
          mean_zcr: 0.119,
          mean_high_mfcc_var: 142.1,
          note: "Supporting acoustic features calculated for forensic reference only; does NOT determine final model verdict.",
        },
        note: "Audio prediction marginal around 0.50 threshold due to background noise.",
      },
    },
  },

  // 3. HIGH_RISK (Unanimous Manipulated)
  {
    id: "SCN-2026-0928-6502",
    created_at: "2026-09-28T10:48:12Z",
    final_verdict: "HIGH_RISK",
    consistency: "UNANIMOUS_MANIPULATED",
    reasoning: "Deepfake synthesis artifacts detected with elevated probability across both vision and auditory channels.",
    modalities: {
      image: {
        verdict: "MANIPULATED",
        confidence: 0.968,
        model: "EfficientNet-B0 (Rebalanced)",
        maturity: "TRAINED",
        predicted_class: "FaceSwap",
        note: "Strong saliency concentrated at jawline boundary and eye socket synthesis perimeter.",
        gradcam_url: SVG_SILHOUETTE_GRADCAM,
        ela_url: SVG_SILHOUETTE_ELA,
      },
      video: {
        verdict: "MANIPULATED",
        confidence: 0.952,
        model: "CNN-GRU Sequence Model",
        maturity: "TRAINED",
        frames_analyzed: 16,
        frame_scores: [0.92, 0.94, 0.96, 0.98, 0.97, 0.95, 0.94, 0.96, 0.98, 0.99, 0.97, 0.95, 0.96, 0.94, 0.93, 0.95],
        note: "Continuous frame-level manipulation signatures consistent with autoencoder face swapping.",
      },
      audio: {
        verdict: "MANIPULATED",
        prediction: "LIKELY SYNTHETIC / MANIPULATED",
        confidence: 0.974,
        real_probability: 0.026,
        fake_probability: 0.974,
        threshold: 0.50,
        model: "Baseline CNN",
        maturity: "TRAINED",
        status: "trained_model",
        features: {
          mean_spectral_flatness: 0.0412,
          mean_zcr: 0.184,
          mean_high_mfcc_var: 82.3,
          formant_jitter: 0.062,
        },
        supporting_forensics: {
          mean_spectral_flatness: 0.0412,
          mean_zcr: 0.184,
          mean_high_mfcc_var: 82.3,
          note: "Supporting acoustic features calculated for forensic reference only; does NOT determine final model verdict.",
        },
        note: "Baseline CNN deep learning inference: 97.4% fake confidence. Status: Trained Model.",
      },
    },
  },

  // 4. UNCERTAIN image sample (Image uncertain, Video authentic, excluded from vote)
  {
    id: "SCN-2026-0927-4190",
    created_at: "2026-09-27T18:04:19Z",
    final_verdict: "TRUSTED",
    consistency: "SINGLE_MODALITY",
    reasoning: "Image classifier output returned uncertain confidence (0.492) and was excluded from voting; video temporal dynamics verified authentic capture.",
    modalities: {
      image: {
        verdict: "UNCERTAIN",
        confidence: 0.492,
        model: "EfficientNet-B0 (Rebalanced)",
        maturity: "TRAINED",
        predicted_class: "Ambiguous",
        note: "Low model confidence due to extreme motion blur; explainability maps inconclusive.",
        gradcam_url: SVG_SILHOUETTE_GRADCAM,
        ela_url: SVG_SILHOUETTE_ELA,
      },
      video: {
        verdict: "AUTHENTIC",
        confidence: 0.892,
        model: "CNN-GRU Sequence Model",
        maturity: "TRAINED",
        frames_analyzed: 12,
        frame_scores: [0.88, 0.90, 0.91, 0.89, 0.87, 0.90, 0.92, 0.89, 0.88, 0.89, 0.90, 0.91],
        note: "Biometric pulse rate optical flow aligns with natural capillary blood flow.",
      },
    },
  },

  // 5. Missing modality sample (Image + Audio provided, Video omitted)
  {
    id: "SCN-2026-0927-2388",
    created_at: "2026-09-27T09:30:45Z",
    final_verdict: "HIGH_RISK",
    consistency: "UNANIMOUS_MANIPULATED",
    reasoning: "Two modalities evaluated (Video was not provided). Both image and audio branches independently triggered synthesis alerts.",
    modalities: {
      image: {
        verdict: "MANIPULATED",
        confidence: 0.915,
        model: "EfficientNet-B0 (Rebalanced)",
        maturity: "TRAINED",
        predicted_class: "NeuralTextures",
        note: "High-frequency periodic neural texture lattice detected in peri-oral skin patches.",
        gradcam_url: SVG_SILHOUETTE_GRADCAM,
        ela_url: SVG_SILHOUETTE_ELA,
      },
      audio: {
        verdict: "MANIPULATED",
        prediction: "LIKELY SYNTHETIC / MANIPULATED",
        confidence: 0.965,
        real_probability: 0.035,
        fake_probability: 0.965,
        threshold: 0.50,
        model: "Baseline CNN",
        maturity: "TRAINED",
        status: "trained_model",
        features: {
          mean_spectral_flatness: 0.0384,
          mean_zcr: 0.162,
          mean_high_mfcc_var: 94.6,
        },
        supporting_forensics: {
          mean_spectral_flatness: 0.0384,
          mean_zcr: 0.162,
          mean_high_mfcc_var: 94.6,
          note: "Supporting acoustic features calculated for forensic reference only; does NOT determine final model verdict.",
        },
        note: "Baseline CNN deep learning inference: 96.5% fake confidence. Status: Trained Model.",
      },
    },
  },
];

export const MOCK_MODELS: ModelDetail[] = [
  {
    modality: "image",
    name: "EfficientNet-B0 (Class-Weighted Rebalanced)",
    maturity: "TRAINED",
    dataset: "FaceForensics++ c23 (4,500 test images across 6 manipulation categories)",
    architecture: "Pretrained EfficientNet-B0 backbone with custom classification head fine-tuned with inverse-frequency class weights",
    limitations: [
      "Vulnerable to aggressive JPEG double-compression below quality factor 50",
      "Noticeable performance degradation on out-of-distribution deepfakes (e.g. Celeb-DF v2, WildDeepfake)",
      "Subject to false positives on heavy camera sensor grain or low-light sensor noise",
    ],
  },
  {
    modality: "video",
    name: "CNN-GRU Temporal Sequence Model",
    maturity: "TRAINED",
    dataset: "FaceForensics++ frame sequences (16-frame uniform temporal sampling)",
    architecture: "Spatial 2D CNN feature extractor coupled to a bidirectional Gated Recurrent Unit (GRU) with temporal attention pooling",
    limitations: [
      "Inference latency scales with video duration; requires GPU for sub-second responses",
      "Falls back to frame-aggregation heuristic if GPU temporal context tensor is unavailable",
      "Less sensitive to ultra-short (1–2 frame) temporal splice anomalies",
    ],
  },
  {
    modality: "audio",
    name: "Baseline CNN (Log-Mel Spectrogram)",
    maturity: "TRAINED",
    dataset: "In-the-Wild Audio Dataset (3,540 balanced test samples)",
    architecture: "4-Block 2D CNN with Log-Mel Spectrogram Frontend (128 mels, 16 kHz)",
    limitations: [
      "ROBUSTNESS WARNING: Severe accuracy drop under heavy background noise (SNR <= 15 dB)",
      "Performance degradation on volume shifts > +/-6 dB",
      "Requires speech segments; unvoiced silence clips are ambiguous",
    ],
  },
];

export const MOCK_METRICS: MetricsResponse = {
  active_checkpoint: "models/efficientnet_b0_best.pth (Rebalanced)",
  comparison: [
    {
      name: "efficientnet_b0_best.pth (Original Loss)",
      binary_accuracy: 0.8127,
      real_recall: 0.2933,
      fake_recall: 0.9165,
    },
    {
      name: "efficientnet_b0_rebalanced.pth (Class-Weighted)",
      binary_accuracy: 0.7480,
      real_recall: 0.7720,
      fake_recall: 0.7432,
    },
  ],
  confusion_matrix: {
    labels: [
      "Original",
      "FaceSwap",
      "Deepfakes",
      "Face2Face",
      "NeuralTextures",
      "FaceShifter",
    ],
    matrix: [
      [220, 76, 31, 115, 174, 134],
      [103, 405, 45, 49, 66, 82],
      [51, 38, 471, 74, 62, 54],
      [88, 52, 61, 418, 77, 54],
      [112, 64, 48, 69, 397, 60],
      [21, 23, 14, 18, 22, 652],
    ],
  },
  training_curve: [
    { epoch: 1, train_loss: 1.482, val_loss: 1.215, train_acc: 0.428, val_acc: 0.512 },
    { epoch: 2, train_loss: 1.156, val_loss: 1.042, train_acc: 0.534, val_acc: 0.558 },
    { epoch: 3, train_loss: 0.984, val_loss: 0.952, train_acc: 0.612, val_acc: 0.579 },
    { epoch: 4, train_loss: 0.842, val_loss: 0.918, train_acc: 0.678, val_acc: 0.584 },
    { epoch: 5, train_loss: 0.725, val_loss: 0.894, train_acc: 0.736, val_acc: 0.594 },
  ],
  generalization: [
    {
      set: "FF++ In-Distribution Test",
      accuracy: 0.8127,
      n: 4500,
      note: "Standard benchmark test split matching training distribution exactly.",
    },
    {
      set: "FF++ JPEG c40 (Heavy Compression)",
      accuracy: 0.6840,
      n: 1500,
      note: "Significant degradation due to loss of high-frequency manipulation boundaries.",
    },
    {
      set: "Celeb-DF v2 (Out-of-Distribution)",
      accuracy: 0.5842,
      n: 1200,
      note: "Cross-dataset drop demonstrates severe vulnerability to unseen generation pipelines.",
    },
    {
      set: "WildDeepfake (In-The-Wild)",
      accuracy: 0.5120,
      n: 1000,
      note: "Near chance performance on internet deepfakes with diverse compression and lighting.",
    },
  ],
};
