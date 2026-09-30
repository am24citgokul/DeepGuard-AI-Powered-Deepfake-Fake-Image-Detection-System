import React from "react";
import { NavLink } from "react-router-dom";
import { 
  LayoutDashboard, 
  ShieldAlert, 
  Layers, 
  Camera, 
  BookOpen, 
  FileText, 
  BarChart3, 
  Info,
  ChevronLeft,
  ChevronRight,
  X
} from "lucide-react";
import { MODALITY_COLORS } from "@/lib/format";

interface NavItem {
  to: string;
  label: string;
  icon: React.ElementType;
  badge?: string;
  iconColor: string;
}

const NAV_ITEMS: NavItem[] = [
  {
    to: "/",
    label: "Dashboard",
    icon: LayoutDashboard,
    iconColor: "#6366F1",
  },
  {
    to: "/screening",
    label: "Full Screening",
    icon: ShieldAlert,
    badge: "MAIN",
    iconColor: MODALITY_COLORS.fusion,
  },
  {
    to: "/analyze",
    label: "Analyze One Modality",
    icon: Layers,
    iconColor: MODALITY_COLORS.image,
  },
  {
    to: "/webcam",
    label: "Live Webcam",
    icon: Camera,
    iconColor: MODALITY_COLORS.video,
  },
  {
    to: "/explainability",
    label: "Evidence Guide",
    icon: BookOpen,
    iconColor: "#F59E0B",
  },
  {
    to: "/reports",
    label: "Reports Log",
    icon: FileText,
    iconColor: "#10B981",
  },
  {
    to: "/models",
    label: "Models & Metrics",
    icon: BarChart3,
    iconColor: MODALITY_COLORS.audio,
  },
  {
    to: "/about",
    label: "Method & Limits",
    icon: Info,
    iconColor: "#64748B",
  },
];

interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  isMobileOpen: boolean;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed,
  onToggleCollapse,
  isMobileOpen,
  onCloseMobile,
}) => {
  return (
    <>
      {/* Mobile Backdrop & Drawer */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 bg-slate-900/40 backdrop-blur-xs z-40 lg:hidden"
          onClick={onCloseMobile}
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 transition-all duration-300 flex flex-col ${
          isMobileOpen ? "translate-x-0 w-64" : "-translate-x-full lg:translate-x-0"
        } ${isCollapsed ? "lg:w-20" : "lg:w-64"}`}
      >
        {/* Top Header */}
        <div className="h-16 px-4 flex items-center justify-between border-b border-slate-100 dark:border-slate-800">
          {!isCollapsed && (
            <div className="flex items-center gap-2">
              <span className="font-heading font-bold text-sm tracking-tight text-slate-800 dark:text-slate-100">
                Navigation
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-violet-50 text-violet-600 dark:bg-violet-950 dark:text-violet-300">
                v2.0
              </span>
            </div>
          )}

          {/* Close mobile button */}
          <button
            type="button"
            onClick={onCloseMobile}
            className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            <X className="w-5 h-5" />
          </button>

          {/* Desktop collapse button */}
          <button
            type="button"
            onClick={onToggleCollapse}
            aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="hidden lg:flex p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Navigation list */}
        <div className="flex-1 overflow-y-auto py-4 px-3 flex flex-col gap-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onCloseMobile}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-semibold transition-all group ${
                  isActive
                    ? "bg-violet-50 dark:bg-violet-950/40 text-violet-700 dark:text-violet-300 shadow-xs"
                    : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100"
                }`
              }
              title={isCollapsed ? item.label : undefined}
            >
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-transform group-hover:scale-110"
                style={{ backgroundColor: `${item.iconColor}16`, color: item.iconColor }}
              >
                <item.icon className="w-4 h-4" />
              </div>

              {!isCollapsed && (
                <div className="flex-1 flex items-center justify-between truncate">
                  <span className="truncate">{item.label}</span>
                  {item.badge && (
                    <span className="font-mono text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-violet-600 text-white tracking-wider">
                      {item.badge}
                    </span>
                  )}
                </div>
              )}
            </NavLink>
          ))}
        </div>

        {/* Bottom Modality Legend */}
        {!isCollapsed && (
          <div className="p-4 border-t border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-850/40 text-[11px] font-mono text-slate-500">
            <div className="text-[10px] text-slate-400 uppercase tracking-wider mb-2 font-bold">
              Modality Palette
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <span className="flex items-center gap-1.5 text-blue-600 dark:text-blue-400">
                <span className="w-2 h-2 rounded-full bg-[#3B82F6]"></span> Image
              </span>
              <span className="flex items-center gap-1.5 text-teal-600 dark:text-teal-400">
                <span className="w-2 h-2 rounded-full bg-[#14B8A6]"></span> Video
              </span>
              <span className="flex items-center gap-1.5 text-pink-600 dark:text-pink-400">
                <span className="w-2 h-2 rounded-full bg-[#EC4899]"></span> Audio
              </span>
              <span className="flex items-center gap-1.5 text-violet-600 dark:text-violet-400">
                <span className="w-2 h-2 rounded-full bg-[#8B5CF6]"></span> Fusion
              </span>
            </div>
          </div>
        )}
      </aside>
    </>
  );
};
