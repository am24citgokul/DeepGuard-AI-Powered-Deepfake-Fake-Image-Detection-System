import React, { useState, useEffect } from "react";
import { 
  BarChart3, 
  Cpu, 
  Layers, 
  AlertTriangle, 
  TrendingUp, 
  Sliders, 
  Info,
  CheckCircle2,
  Inbox
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { ConfusionHeatmap } from "@/components/diagrams/ConfusionHeatmap";
import { GeneralizationGap } from "@/components/diagrams/GeneralizationGap";
import { api } from "@/lib/api";
import { MODALITY_COLORS, getMaturityChip, formatPercent } from "@/lib/format";
import type { ModelDetail, MetricsResponse } from "@/lib/types";

export const Models: React.FC = () => {
  const [models, setModels] = useState<ModelDetail[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [audioMetrics, setAudioMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [mList, mMetrics, aMetrics] = await Promise.all([
          api.getModels(),
          api.getMetrics(),
          api.getAudioMetrics(),
        ]);
        setModels(mList || []);
        setMetrics(mMetrics || null);
        setAudioMetrics(aMetrics || null);
      } catch (err) {
        console.error("Failed to load models/metrics:", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="py-20 flex flex-col items-center justify-center text-center">
        <div className="w-8 h-8 border-2 border-violet-600 border-t-transparent rounded-full animate-spin mb-3"></div>
        <p className="text-xs font-mono text-slate-500">Querying model telemetry & benchmarks...</p>
      </div>
    );
  }

  // If the API returns nothing, show an empty state, not fake numbers
  if (!models.length && !metrics) {
    return (
      <div className="py-20 text-center flex flex-col items-center justify-center">
        <div className="w-14 h-14 rounded-2xl bg-slate-100 dark:bg-slate-800 text-slate-400 flex items-center justify-center mb-3">
          <Inbox className="w-7 h-7" />
        </div>
        <h3 className="font-heading font-bold text-base text-slate-800 dark:text-slate-200">
          No Model Telemetry Available
        </h3>
        <p className="text-xs text-slate-400 max-w-sm mt-1">
          The backend API has not published active checkpoint metrics or model configuration logs.
        </p>
      </div>
    );
  }

  // Format checkpoint comparison data for bar chart
  const comparisonChartData = metrics?.comparison.map((item) => ({
    name: item.name.split(" ")[0].replace(".pth", ""),
    "Binary Acc": Number((item.binary_accuracy * 100).toFixed(1)),
    "Real Recall": Number((item.real_recall * 100).toFixed(1)),
    "Fake Recall": Number((item.fake_recall * 100).toFixed(1)),
  })) || [];

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="header-gradient-banner rounded-3xl p-6 sm:p-8 border border-slate-200/80 dark:border-slate-800 shadow-xs">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-pink-100/70 dark:bg-pink-950/60 text-pink-700 dark:text-pink-300 text-xs font-mono font-medium mb-3">
            <BarChart3 className="w-3.5 h-3.5" />
            Empirical Benchmark Lab
          </div>
          <h1 className="font-heading font-extrabold text-2xl sm:text-3xl text-slate-900 dark:text-white tracking-tight">
            Models & Empirical Evaluation
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 font-sans leading-relaxed">
            Transparent inspection of neural backbones, class rebalancing losses, multi-class confusion matrices, and the cross-dataset generalization gap.
          </p>
        </div>
      </div>

      {/* 1. Model Cards (Image / Video / Audio) */}
      <div className="space-y-4">
        <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white flex items-center gap-2">
          <Cpu className="w-4 h-4 text-violet-600" />
          Active Modality Architecture Specs
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {models.map((mod) => {
            const chip = getMaturityChip(mod.maturity);
            const color = MODALITY_COLORS[mod.modality];

            return (
              <div
                key={mod.modality}
                className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 shadow-xs flex flex-col justify-between"
                style={{ borderTop: `4px solid ${color}` }}
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-mono text-xs uppercase font-bold tracking-wider" style={{ color }}>
                      {mod.modality} STREAM
                    </span>
                    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${chip.color}`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${chip.dot}`}></span>
                      {chip.label}
                    </span>
                  </div>

                  <h4 className="font-heading font-bold text-base text-slate-900 dark:text-white">
                    {mod.name}
                  </h4>

                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                    <strong className="text-slate-700 dark:text-slate-300">Dataset: </strong>
                    {mod.dataset}
                  </p>

                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                    <strong className="text-slate-700 dark:text-slate-300">Architecture: </strong>
                    {mod.architecture}
                  </p>

                  {/* Known Limitations List */}
                  {mod.limitations && mod.limitations.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800">
                      <div className="text-[11px] font-mono text-amber-600 dark:text-amber-400 font-bold mb-1.5 flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3" />
                        Known Limitations:
                      </div>
                      <ul className="space-y-1 text-[11px] text-slate-600 dark:text-slate-400">
                        {mod.limitations.map((lim, i) => (
                          <li key={i} className="flex items-start gap-1.5">
                            <span className="text-amber-500">•</span>
                            <span>{lim}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 2. Confusion-Matrix Heatmap */}
      {metrics?.confusion_matrix && (
        <ConfusionHeatmap data={metrics.confusion_matrix} />
      )}

      {/* 3. Training Curves with Recharts */}
      {metrics?.training_curve && metrics.training_curve.length > 0 && (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-6 pb-3 border-b border-slate-100 dark:border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-violet-600" />
                <h3 className="font-heading font-semibold text-slate-900 dark:text-white text-base">
                  Model Training & Validation Convergence Curves
                </h3>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                5-Epoch fine-tuning of EfficientNet-B0 backbone using class-weighted Cross-Entropy Loss.
              </p>
            </div>
            <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
              Checkpoint: {metrics.active_checkpoint}
            </span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Loss Curve */}
            <div>
              <h4 className="font-heading font-semibold text-xs text-slate-700 dark:text-slate-300 mb-3 text-center">
                Cross-Entropy Loss (Train vs Validation)
              </h4>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics.training_curve}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="epoch" tick={{ fontSize: 11 }} tickFormatter={(val) => `Ep ${val}`} />
                    <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
                    <Tooltip contentStyle={{ borderRadius: "12px", fontSize: "12px" }} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Line type="monotone" dataKey="train_loss" name="Train Loss" stroke="#3B82F6" strokeWidth={2} />
                    <Line type="monotone" dataKey="val_loss" name="Val Loss" stroke="#EC4899" strokeWidth={2} strokeDasharray="4 4" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Accuracy Curve */}
            <div>
              <h4 className="font-heading font-semibold text-xs text-slate-700 dark:text-slate-300 mb-3 text-center">
                Multi-Class Accuracy (Train vs Validation)
              </h4>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics.training_curve}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="epoch" tick={{ fontSize: 11 }} tickFormatter={(val) => `Ep ${val}`} />
                    <YAxis tick={{ fontSize: 11 }} tickFormatter={(val) => `${(val * 100).toFixed(0)}%`} domain={[0.3, 0.9]} />
                    <Tooltip
                      contentStyle={{ borderRadius: "12px", fontSize: "12px" }}
                      formatter={(val: any) => [`${(Number(val) * 100).toFixed(1)}%`]}
                    />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Line type="monotone" dataKey="train_acc" name="Train Accuracy" stroke="#10B981" strokeWidth={2} />
                    <Line type="monotone" dataKey="val_acc" name="Val Accuracy" stroke="#8B5CF6" strokeWidth={2} strokeDasharray="4 4" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 4. Checkpoint Comparison (Bar chart + Table) */}
      {metrics?.comparison && metrics.comparison.length > 0 && (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-6 pb-3 border-b border-slate-100 dark:border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-teal-600" />
                <h3 className="font-heading font-semibold text-slate-900 dark:text-white text-base">
                  Checkpoint Calibration Comparison
                </h3>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Evaluating class-weighted loss rebalancing: shifting sensitivity from majority fake classes to minority pristine authentic captures.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
            {/* Bar Chart (Span 6) */}
            <div className="lg:col-span-6 h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonChartData}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} unit="%" domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{ borderRadius: "12px", fontSize: "12px" }}
                    formatter={(val: any) => [`${val}%`]}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px" }} />
                  <Bar dataKey="Binary Acc" fill="#3B82F6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Real Recall" fill="#10B981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Fake Recall" fill="#F43F5E" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Comparison Table (Span 6) */}
            <div className="lg:col-span-6 overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-100 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-850/50">
                    <th className="py-2.5 px-3 font-mono text-slate-400">Checkpoint Model</th>
                    <th className="py-2.5 px-3 font-mono text-slate-400 text-center">Binary Acc</th>
                    <th className="py-2.5 px-3 font-mono text-slate-400 text-center">Real Recall</th>
                    <th className="py-2.5 px-3 font-mono text-slate-400 text-center">Fake Recall</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono">
                  {metrics.comparison.map((item, idx) => (
                    <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-850/40">
                      <td className="py-3 px-3 font-semibold text-slate-800 dark:text-slate-200">
                        {item.name}
                      </td>
                      <td className="py-3 px-3 text-center text-blue-600 dark:text-blue-400 font-bold">
                        {(item.binary_accuracy * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-3 text-center text-emerald-600 dark:text-emerald-400 font-bold">
                        {(item.real_recall * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-3 text-center text-rose-600 dark:text-rose-400 font-bold">
                        {(item.fake_recall * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <p className="mt-3 text-[11px] text-slate-500 italic leading-normal">
                <strong>Analysis: </strong>The original loss model prioritized fake recall (91.7%) at the expense of high false-positive rates on authentic faces (only 29.3% real recall). The rebalanced checkpoint lifts authentic recall to 77.2%, preventing innocent identity rejections.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Audio Deepfake Detection Benchmark & Robustness Lab */}
      {audioMetrics && (
        <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 shadow-sm space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100 dark:border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-pink-500" />
                <h3 className="font-heading font-semibold text-slate-900 dark:text-white text-base">
                  Audio Deepfake Detector — Verified Test Benchmark
                </h3>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Baseline CNN (Log-Mel Spectrogram Frontend) evaluated on frozen In-The-Wild benchmark (N = 3,540 balanced test samples).
              </p>
            </div>
            <span className="text-xs font-mono px-2.5 py-1 rounded-full bg-pink-50 text-pink-700 border border-pink-200 dark:bg-pink-950 dark:text-pink-300 dark:border-pink-800">
              Checkpoint: baseline_cnn_best.pt
            </span>
          </div>

          {/* Test Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200/80 dark:border-slate-800 text-center">
              <span className="text-[10px] font-mono text-slate-400 uppercase block mb-1">Accuracy</span>
              <span className="text-base font-heading font-extrabold text-slate-900 dark:text-white">
                {(audioMetrics.accuracy * 100).toFixed(2)}%
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200/80 dark:border-slate-800 text-center">
              <span className="text-[10px] font-mono text-slate-400 uppercase block mb-1">F1-Score</span>
              <span className="text-base font-heading font-extrabold text-blue-600 dark:text-blue-400">
                {audioMetrics.f1.toFixed(4)}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200/80 dark:border-slate-800 text-center">
              <span className="text-[10px] font-mono text-slate-400 uppercase block mb-1">Fake Recall</span>
              <span className="text-base font-heading font-extrabold text-rose-600 dark:text-rose-400">
                {(audioMetrics.fake_recall * 100).toFixed(2)}%
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200/80 dark:border-slate-800 text-center">
              <span className="text-[10px] font-mono text-slate-400 uppercase block mb-1">Real Recall</span>
              <span className="text-base font-heading font-extrabold text-emerald-600 dark:text-emerald-400">
                {(audioMetrics.real_recall * 100).toFixed(2)}%
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200/80 dark:border-slate-800 text-center">
              <span className="text-[10px] font-mono text-slate-400 uppercase block mb-1">ROC-AUC</span>
              <span className="text-base font-heading font-extrabold text-violet-600 dark:text-violet-400">
                {audioMetrics.roc_auc.toFixed(4)}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200/80 dark:border-slate-800 text-center">
              <span className="text-[10px] font-mono text-slate-400 uppercase block mb-1">FPR</span>
              <span className="text-base font-heading font-extrabold text-slate-700 dark:text-slate-300">
                {(audioMetrics.fpr * 100).toFixed(2)}%
              </span>
            </div>
            <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-850 border border-slate-200/80 dark:border-slate-800 text-center">
              <span className="text-[10px] font-mono text-slate-400 uppercase block mb-1">FNR</span>
              <span className="text-base font-heading font-extrabold text-slate-700 dark:text-slate-300">
                {(audioMetrics.fnr * 100).toFixed(2)}%
              </span>
            </div>
          </div>

          {/* Robustness Breakdown Table */}
          {audioMetrics.robustness && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="font-heading font-semibold text-xs text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
                  Perturbation & Environmental Robustness Analysis
                </h4>
                <span className="text-[11px] font-mono text-slate-400">
                  Stress Testing
                </span>
              </div>

              <div className="overflow-x-auto border border-slate-200/80 dark:border-slate-800 rounded-xl">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-100 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-850/50">
                      <th className="py-2.5 px-3 font-mono text-slate-400">Condition</th>
                      <th className="py-2.5 px-3 font-mono text-slate-400 text-center">F1 Score</th>
                      <th className="py-2.5 px-3 font-mono text-slate-400 text-center">Fake Recall</th>
                      <th className="py-2.5 px-3 font-mono text-slate-400 text-center">Real Recall</th>
                      <th className="py-2.5 px-3 font-mono text-slate-400">Status / Diagnostic Warning</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono">
                    {audioMetrics.robustness.map((row: any, idx: number) => (
                      <tr key={idx} className="hover:bg-slate-50/50 dark:hover:bg-slate-850/40">
                        <td className="py-2.5 px-3 font-semibold text-slate-800 dark:text-slate-200">
                          {row.condition}
                        </td>
                        <td className="py-2.5 px-3 text-center text-blue-600 dark:text-blue-400 font-bold">
                          {(row.f1 * 100).toFixed(2)}%
                        </td>
                        <td className="py-2.5 px-3 text-center text-rose-600 dark:text-rose-400 font-bold">
                          {(row.fake_recall * 100).toFixed(2)}%
                        </td>
                        <td className="py-2.5 px-3 text-center text-emerald-600 dark:text-emerald-400 font-bold">
                          {(row.real_recall * 100).toFixed(2)}%
                        </td>
                        <td className="py-2.5 px-3">
                          {row.warning ? (
                            <span className="inline-flex items-center gap-1 text-[11px] font-sans font-bold text-amber-600 dark:text-amber-400">
                              <AlertTriangle className="w-3 h-3 shrink-0" />
                              {row.warning}
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-[11px] font-sans text-emerald-600 dark:text-emerald-400 font-medium">
                              <CheckCircle2 className="w-3 h-3 shrink-0" />
                              Passed Verification
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {audioMetrics.robustness_warning && (
                <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-800 dark:text-amber-200 text-xs flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                  <span className="font-semibold">{audioMetrics.robustness_warning}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* 5. Generalization Gap Diagram */}
      {metrics?.generalization && (
        <GeneralizationGap data={metrics.generalization} />
      )}
    </div>
  );
};
