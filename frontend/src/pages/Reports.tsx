import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { 
  FileText, 
  Search, 
  Filter, 
  ArrowRight, 
  CheckCircle, 
  AlertTriangle, 
  ShieldAlert,
  HelpCircle,
  Inbox
} from "lucide-react";
import { api } from "@/lib/api";
import { 
  MODALITY_COLORS, 
  getVerdictConfig, 
  formatDateTime 
} from "@/lib/format";
import type { ScreeningResult, FinalVerdict } from "@/lib/types";

export const Reports: React.FC = () => {
  const [reports, setReports] = useState<ScreeningResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [verdictFilter, setVerdictFilter] = useState<string>("ALL");
  const [modalityFilter, setModalityFilter] = useState<string>("ALL");

  useEffect(() => {
    async function loadReports() {
      try {
        const data = await api.getReports();
        setReports(data || []);
      } catch (err) {
        console.error("Failed to load reports:", err);
      } finally {
        setLoading(false);
      }
    }
    loadReports();
  }, []);

  const filteredReports = reports.filter((r) => {
    // Search filter: ID or reasoning
    const matchesSearch =
      r.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.reasoning.toLowerCase().includes(searchTerm.toLowerCase());

    // Verdict filter
    const matchesVerdict =
      verdictFilter === "ALL" || r.final_verdict === verdictFilter;

    // Modality filter
    const matchesModality =
      modalityFilter === "ALL" ||
      (modalityFilter === "image" && r.modalities.image) ||
      (modalityFilter === "video" && r.modalities.video) ||
      (modalityFilter === "audio" && r.modalities.audio);

    return matchesSearch && matchesVerdict && matchesModality;
  });

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header Banner */}
      <div className="header-gradient-banner rounded-3xl p-6 sm:p-8 border border-slate-200/80 dark:border-slate-800 shadow-xs">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-100/70 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 text-xs font-mono font-medium mb-3">
            <FileText className="w-3.5 h-3.5" />
            Audit Log Repository
          </div>
          <h1 className="font-heading font-extrabold text-2xl sm:text-3xl text-slate-900 dark:text-white tracking-tight">
            Screening Reports Audit Log
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-300 mt-2 font-sans leading-relaxed">
            Historical audit logs of verified sessions. Filter by forensic verdict, search by unique tracking ID, or filter by evaluated streams.
          </p>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Search input */}
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search report ID or reasoning..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 text-xs rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-violet-500 font-mono"
          />
        </div>

        {/* Filter controls */}
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          {/* Verdict filter */}
          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <Filter className="w-3.5 h-3.5" />
            <span>Verdict:</span>
            <select
              value={verdictFilter}
              onChange={(e) => setVerdictFilter(e.target.value)}
              className="py-1.5 px-3 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs font-mono font-medium text-slate-700 dark:text-slate-200 focus:outline-none"
            >
              <option value="ALL">All Verdicts</option>
              <option value="TRUSTED">TRUSTED</option>
              <option value="REVIEW_REQUIRED">REVIEW REQUIRED</option>
              <option value="HIGH_RISK">HIGH RISK</option>
            </select>
          </div>

          {/* Modality filter */}
          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <span>Modality:</span>
            <select
              value={modalityFilter}
              onChange={(e) => setModalityFilter(e.target.value)}
              className="py-1.5 px-3 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs font-mono font-medium text-slate-700 dark:text-slate-200 focus:outline-none"
            >
              <option value="ALL">All Modalities</option>
              <option value="image">Includes Image</option>
              <option value="video">Includes Video</option>
              <option value="audio">Includes Audio</option>
            </select>
          </div>
        </div>
      </div>

      {/* Reports Table */}
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm overflow-hidden">
        {filteredReports.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-100 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-850/50">
                  <th className="py-3.5 px-4 text-xs font-mono text-slate-400 uppercase tracking-wider">
                    Date & Time
                  </th>
                  <th className="py-3.5 px-4 text-xs font-mono text-slate-400 uppercase tracking-wider">
                    Report ID
                  </th>
                  <th className="py-3.5 px-4 text-xs font-mono text-slate-400 uppercase tracking-wider">
                    Streams
                  </th>
                  <th className="py-3.5 px-4 text-xs font-mono text-slate-400 uppercase tracking-wider">
                    Final Verdict
                  </th>
                  <th className="py-3.5 px-4 text-xs font-mono text-slate-400 uppercase tracking-wider">
                    Consistency
                  </th>
                  <th className="py-3.5 px-4 text-xs font-mono text-slate-400 uppercase tracking-wider text-right">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {filteredReports.map((report) => {
                  const conf = getVerdictConfig(report.final_verdict);
                  const Icon = conf.icon;

                  return (
                    <tr
                      key={report.id}
                      className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                    >
                      {/* Date */}
                      <td className="py-4 px-4 text-xs font-mono text-slate-600 dark:text-slate-400 whitespace-nowrap">
                        {formatDateTime(report.created_at)}
                      </td>

                      {/* ID */}
                      <td className="py-4 px-4 text-xs font-mono font-bold text-slate-900 dark:text-slate-100 whitespace-nowrap">
                        {report.id}
                      </td>

                      {/* Input Types (Colour dots for modalities) */}
                      <td className="py-4 px-4 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <span
                            title={report.modalities.image ? "Image evaluated" : "No image"}
                            className={`w-2.5 h-2.5 rounded-full ${
                              report.modalities.image ? "bg-[#3B82F6]" : "bg-slate-200 dark:bg-slate-750"
                            }`}
                          />
                          <span
                            title={report.modalities.video ? "Video evaluated" : "No video"}
                            className={`w-2.5 h-2.5 rounded-full ${
                              report.modalities.video ? "bg-[#14B8A6]" : "bg-slate-200 dark:bg-slate-750"
                            }`}
                          />
                          <span
                            title={report.modalities.audio ? "Audio evaluated" : "No audio"}
                            className={`w-2.5 h-2.5 rounded-full ${
                              report.modalities.audio ? "bg-[#EC4899]" : "bg-slate-200 dark:bg-slate-750"
                            }`}
                          />
                        </div>
                      </td>

                      {/* Verdict Chip */}
                      <td className="py-4 px-4 whitespace-nowrap">
                        <span
                          className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold border ${conf.bgLight} ${conf.textLight} ${conf.borderLight} dark:${conf.bgDark} dark:${conf.textDark} dark:${conf.borderDark}`}
                        >
                          <Icon className="w-3 h-3" />
                          {conf.label}
                        </span>
                      </td>

                      {/* Consistency */}
                      <td className="py-4 px-4 text-xs font-mono text-slate-500 dark:text-slate-400 whitespace-nowrap">
                        {report.consistency.replace(/_/g, " ")}
                      </td>

                      {/* Action */}
                      <td className="py-4 px-4 text-right whitespace-nowrap">
                        <Link
                          to={`/reports/${report.id}`}
                          className="inline-flex items-center gap-1 px-3 py-1 rounded-lg text-xs font-heading font-semibold text-violet-600 dark:text-violet-400 hover:bg-violet-50 dark:hover:bg-violet-950/40 transition-colors"
                        >
                          View <ArrowRight className="w-3.5 h-3.5" />
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          /* Empty State "No screenings yet" */
          <div className="p-12 text-center flex flex-col items-center justify-center">
            <div className="w-14 h-14 rounded-2xl bg-slate-100 dark:bg-slate-800 text-slate-400 flex items-center justify-center mb-3">
              <Inbox className="w-7 h-7" />
            </div>
            <h4 className="font-heading font-semibold text-sm text-slate-800 dark:text-slate-200">
              No screenings yet
            </h4>
            <p className="text-xs text-slate-400 dark:text-slate-500 max-w-sm mt-1 mb-4">
              {searchTerm || verdictFilter !== "ALL" || modalityFilter !== "ALL"
                ? "No audit records match your search or filter criteria. Try clearing filters."
                : "Execute your first identity verification screening to record immutable audit reports."}
            </p>
            <Link
              to="/screening"
              className="px-4 py-2 rounded-xl bg-violet-600 hover:bg-violet-700 text-white text-xs font-heading font-semibold shadow-xs"
            >
              Start New Screening
            </Link>
          </div>
        )}
      </div>
    </div>
  );
};
