# DeepGuard X — AI Identity Authenticity Screening

Research-grade multimodal forensic analytics web application combining spatial convolutional attention (Grad-CAM), Error Level Analysis (ELA), temporal sequence coherence (CNN-GRU), and acoustic spectral heuristics into an explainable identity verification audit.

---

## 🌟 Architecture & Tech Stack

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, React Router v7, Recharts, Lucide React, Framer Motion
- **Design System**: Lab aesthetic, light/dark themes, custom SVG diagrams, 8px grid, Plus Jakarta Sans, Inter & JetBrains Mono
- **Modality Color Coding**:
  - 🔵 **Image**: `#3B82F6` (EfficientNet-B0 + Grad-CAM + ELA)
  - 🟢 **Video**: `#14B8A6` (CNN-GRU Sequence Model)
  - 🌸 **Audio**: `#EC4899` (Acoustic Feature Heuristics)
  - 🟣 **Cross-Modal Fusion**: `#8B5CF6` (Bayesian Consensus Kernel)
- **Verdict Signals**:
  - 🟢 **TRUSTED / AUTHENTIC**: `#16A34A`
  - 🟡 **REVIEW REQUIRED**: `#F59E0B`
  - 🔴 **HIGH RISK / MANIPULATED**: `#DC2626`
  - ⚪ **UNCERTAIN**: `#64748B` with diagonal-hatch fill

---

## 🚀 Quick Start

### 1. Running the Frontend (Demo Mode by Default)

The frontend runs in **Demo Mode** out-of-the-box with rich sample screenings, models, and empirical benchmark data:

```bash
cd frontend
npm install
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) in your browser. An amber banner (`DEMO DATA — not real results`) indicates demo execution.

### 2. Connecting to the Python/Flask Backend

To connect the React frontend to the live Python/Flask deepfake detection backend:

1. **Start the Flask backend (Port 5001)**:
   ```bash
   python app.py
   ```
   The backend enables CORS across all `/api/*` endpoints.

2. **Configure the Frontend**:
   Create a `.env` file in the `frontend/` directory (or set the environment variable):
   ```env
   VITE_API_BASE_URL=http://localhost:5001
   ```

3. **Restart the Frontend Dev Server**:
   ```bash
   cd frontend
   npm run dev
   ```
   The navbar will update to a green **"Connected"** status pill and send multipart media requests directly to the local Flask models.

---

## 🗺️ Application Routes

| Route | Page | Diagram / Key Features |
|---|---|---|
| `/` | **Dashboard** | Hero interactive Pipeline Diagram with clickable nodes, 3 maturity cards, latest screening summary |
| `/screening` | **Full Screening** | 3 upload zones, execution stepper, Fusion Diagram, Consistency Matrix, Evidence Cards, Disclaimer |
| `/analyze` | **Analyze One Modality** | Tabbed (Image, Video, Audio) single-stream diagnosis beside upload zone |
| `/webcam` | **Live Webcam** | Live camera preview with 3s stabilization countdown reticle & instant inference |
| `/explainability` | **Evidence Guide** | Annotated Grad-CAM/ELA triptych with numbered callouts, boundary rules |
| `/models` | **Models & Metrics** | Multi-class Confusion Heatmap, Recharts training curves, Checkpoint comparison, Generalization Gap |
| `/reports` | **Reports Log** | Audit table with modality dots, search, verdict filters |
| `/reports/:id` | **Report Detail** | Reuses complete 5-layer result audit view |
| `/about` | **Method & Limits** | Pipeline architecture, decision flowchart, and explicit technical boundaries |

---

## ⚖️ Forensic Boundary & Responsible Use

> **Important**: DeepGuard X is an AI screening aid that supports human review. It is not legal proof of fraud. **REVIEW REQUIRED** and **HIGH RISK** cases necessitate trained forensic verification.
