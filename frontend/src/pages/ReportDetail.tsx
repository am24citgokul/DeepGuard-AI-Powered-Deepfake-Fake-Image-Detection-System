import React, { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import { ChevronLeft, AlertCircle, Loader2 } from "lucide-react";
import { VerdictBanner } from "@/components/screening/VerdictBanner";
import { FusionDiagram } from "@/components/diagrams/FusionDiagram";
import { ConsistencyMatrix } from "@/components/diagrams/ConsistencyMatrix";
import { EvidenceCard } from "@/components/screening/EvidenceCard";
import { DisclaimerBar } from "@/components/screening/DisclaimerBar";
import { api } from "@/lib/api";
import type { ScreeningResult } from "@/lib/types";

export const ReportDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [report, setReport] = useState<ScreeningResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    async function loadReport() {
      if (!id) return;
      try {
        const data = await api.getReportById(id);
        setReport(data);
      } catch (err: any) {
        setErrorMsg(err.message || "Failed to locate specified report.");
      } finally {
        setLoading(false);
      }
    }
    loadReport();
  }, [id]);

  if (loading) {
    return (
      <div className="py-20 flex flex-col items-center justify-center text-center">
        <Loader2 className="w-8 h-8 text-violet-600 animate-spin mb-3" />
        <p className="text-xs font-mono text-slate-500">Retrieving audit record #{id}...</p>
      </div>
    );
  }

  if (errorMsg || !report) {
    return (
      <div className="py-16 text-center max-w-md mx-auto">
        <div className="w-12 h-12 rounded-full bg-rose-100 text-rose-600 flex items-center justify-center mx-auto mb-3">
          <AlertCircle className="w-6 h-6" />
        </div>
        <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white">
          Report Not Found
        </h3>
        <p className="text-xs text-slate-500 mt-1 mb-4">
          {errorMsg || `Could not find an audit log matching identifier "${id}".`}
        </p>
        <Link
          to="/reports"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900 text-white dark:bg-white dark:text-slate-900 text-xs font-semibold"
        >
          <ChevronLeft className="w-4 h-4" /> Back to Reports Log
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Breadcrumb */}
      <div className="flex items-center justify-between">
        <Link
          to="/reports"
          className="inline-flex items-center gap-1.5 text-xs font-heading font-semibold text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
          <span>Back to All Reports</span>
        </Link>

        <span className="font-mono text-xs text-slate-400">
          Audit Certificate #{report.id}
        </span>
      </div>

      {/* 1) Verdict banner */}
      <VerdictBanner
        verdict={report.final_verdict}
        consistency={report.consistency}
        reasoning={report.reasoning}
        createdAt={report.created_at}
        reportId={report.id}
      />

      {/* 2) Fusion Diagram */}
      <FusionDiagram screening={report} />

      {/* 3) Cross-Modal Consistency Matrix */}
      <ConsistencyMatrix screening={report} />

      {/* 4) Evidence Cards Breakdown */}
      <div className="space-y-6">
        <h3 className="font-heading font-bold text-base text-slate-900 dark:text-white">
          Forensic Evidence Breakdown
        </h3>

        <div className="grid grid-cols-1 gap-6">
          <EvidenceCard
            modality="image"
            title="Image Forensic Evidence"
            result={report.modalities.image}
          />
          <EvidenceCard
            modality="video"
            title="Video Temporal Evidence"
            result={report.modalities.video}
          />
          <EvidenceCard
            modality="audio"
            title="Audio Spectral Evidence"
            result={report.modalities.audio}
          />
        </div>
      </div>

      {/* 5) Calm disclaimer bar */}
      <DisclaimerBar variant="result" />
    </div>
  );
};
