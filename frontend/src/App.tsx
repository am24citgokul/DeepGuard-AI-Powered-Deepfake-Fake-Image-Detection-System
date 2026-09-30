import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import { Dashboard } from "@/pages/Dashboard";
import { Screening } from "@/pages/Screening";
import { Analyze } from "@/pages/Analyze";
import { Webcam } from "@/pages/Webcam";
import { Guide } from "@/pages/Guide";
import { Reports } from "@/pages/Reports";
import { ReportDetail } from "@/pages/ReportDetail";
import { Models } from "@/pages/Models";
import { About } from "@/pages/About";

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/screening" element={<Screening />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/webcam" element={<Webcam />} />
          <Route path="/explainability" element={<Guide />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/reports/:id" element={<ReportDetail />} />
          <Route path="/models" element={<Models />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
