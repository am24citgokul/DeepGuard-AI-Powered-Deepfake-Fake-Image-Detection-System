import React from "react";

interface ConfidenceGaugeProps {
  confidence: number; // 0..1
  color?: string;
  size?: number;
  label?: string;
  sublabel?: string;
  isUncertain?: boolean;
}

export const ConfidenceGauge: React.FC<ConfidenceGaugeProps> = ({
  confidence,
  color = "#8B5CF6",
  size = 140,
  label,
  sublabel,
  isUncertain = false,
}) => {
  // Semicircle gauge: 180 degrees (from 180 to 360, or arc from left to right)
  const strokeWidth = Math.max(10, size * 0.08);
  const radius = (size - strokeWidth * 2) / 2;
  const circumference = Math.PI * radius; // Half-circle perimeter
  const clamped = Math.max(0, Math.min(1, confidence));
  const strokeDashoffset = circumference - clamped * circumference;

  const displayColor = isUncertain ? "#64748B" : color;

  return (
    <div className="flex flex-col items-center justify-center select-none">
      <div className="relative" style={{ width: size, height: size * 0.65 }}>
        <svg
          width={size}
          height={size * 0.65}
          viewBox={`0 0 ${size} ${size * 0.65}`}
          className="overflow-visible"
        >
          {/* Background track */}
          <path
            d={`M ${strokeWidth},${size / 2} A ${radius},${radius} 0 0,1 ${size - strokeWidth},${size / 2}`}
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            className="text-slate-100 dark:text-slate-800"
            strokeLinecap="round"
          />

          {/* Active progress arc */}
          <path
            d={`M ${strokeWidth},${size / 2} A ${radius},${radius} 0 0,1 ${size - strokeWidth},${size / 2}`}
            fill="none"
            stroke={displayColor}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />

          {/* Center text for percentage */}
          <text
            x={size / 2}
            y={size / 2 - 4}
            textAnchor="middle"
            className="font-mono font-bold text-slate-800 dark:text-slate-100 transition-colors"
            style={{ fontSize: Math.max(14, size * 0.18) }}
          >
            {isUncertain ? "Uncertain" : `${(clamped * 100).toFixed(1)}%`}
          </text>
          
          <text
            x={size / 2}
            y={size / 2 + 14}
            textAnchor="middle"
            className="font-sans text-[11px] font-medium fill-slate-400 dark:fill-slate-500 uppercase tracking-wider"
          >
            {label || "Confidence"}
          </text>
        </svg>
      </div>

      {sublabel && (
        <span className="text-xs text-slate-500 dark:text-slate-400 mt-1 font-mono text-center">
          {sublabel}
        </span>
      )}
    </div>
  );
};
