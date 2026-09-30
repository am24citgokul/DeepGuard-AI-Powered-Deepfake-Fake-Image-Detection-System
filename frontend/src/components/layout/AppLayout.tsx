import React, { useState, useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Navbar } from "./Navbar";
import { Sidebar } from "./Sidebar";
import { isConfiguredBackend, api } from "@/lib/api";
import { Database, AlertTriangle } from "lucide-react";

export const AppLayout: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState<boolean>(() => {
    return localStorage.getItem("deepguard_theme") === "dark";
  });
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);
  const [isMobileOpen, setIsMobileOpen] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  const location = useLocation();

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("deepguard_theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("deepguard_theme", "light");
    }
  }, [isDarkMode]);

  useEffect(() => {
    // Check health once on mount
    api.getHealth()
      .then((res) => setBackendOnline(res.status === "ready" || res.status === "ok"))
      .catch(() => setBackendOnline(false));
  }, []);

  const showDemoRibbon = !isConfiguredBackend() || !backendOnline;

  return (
    <div className="min-h-screen bg-[#F7F8FC] dark:bg-[#0B0F19] text-slate-800 dark:text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <Navbar
        isDarkMode={isDarkMode}
        onToggleDarkMode={() => setIsDarkMode(!isDarkMode)}
        onOpenMobileMenu={() => setIsMobileOpen(true)}
        backendOnline={backendOnline}
      />

      <div className="flex-1 flex">
        {/* Left Sidebar */}
        <Sidebar
          isCollapsed={isCollapsed}
          onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
          isMobileOpen={isMobileOpen}
          onCloseMobile={() => setIsMobileOpen(false)}
        />

        {/* Main Content Area */}
        <main
          className={`flex-1 transition-all duration-300 pb-16 min-h-[calc(100vh-4rem)] ${
            isCollapsed ? "lg:ml-20" : "lg:ml-64"
          }`}
        >
          {/* Visible Amber Ribbon in Demo Mode */}
          {showDemoRibbon && (
            <div className="bg-gradient-to-r from-amber-500 via-amber-600 to-amber-500 text-slate-950 font-mono text-xs py-1.5 px-4 font-bold flex items-center justify-center gap-2 shadow-xs select-none">
              <Database className="w-4 h-4 text-slate-950" />
              <span>DEMO DATA — not real results</span>
              <span className="opacity-75 font-sans font-normal hidden sm:inline">
                | Set VITE_API_BASE_URL=http://localhost:5001 to connect live Python/Flask inference
              </span>
            </div>
          )}

          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};
