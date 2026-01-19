"use client";

import { useState, useEffect } from "react";

interface Hotspot {
  id: number;
  confidence: number;
  area_m2: number;
  lat: number;
  lon: number;
}

interface DetectionData {
  hotspots_count: number;
  avg_confidence: number;
  processing_time_ms: number;
  hotspots: Hotspot[];
}

interface ReportData {
  severity?: string;
  headline?: string;
  primary_stat?: { value: string; label: string; detail: string };
  secondary_stat?: { value: string; label: string; detail: string };
  tertiary_stat?: { value: string; label: string; detail: string };
  insight?: string;
  action?: string;
}

interface ReportGeneratorProps {
  detectionData: DetectionData;
  autoGenerate?: boolean;
}

export default function ReportGenerator({ detectionData, autoGenerate = true }: ReportGeneratorProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [report, setReport] = useState<ReportData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (autoGenerate && !report && !isGenerating && !error) {
      generateReport();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoGenerate]);

  const generateReport = async () => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await fetch("/api/report/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ detection_data: detectionData }),
      });

      const data = await response.json();

      if (data.success && data.report) {
        setReport(data.report);
      } else {
        setError(data.error || "Failed to generate report");
      }
    } catch (err) {
      console.error("Report error:", err);
      setError("Network error");
    } finally {
      setIsGenerating(false);
    }
  };

  // Loading
  if (isGenerating) {
    return (
      <div style={{ padding: 24, background: "rgba(0,200,200,0.1)", borderRadius: 16, border: "1px solid rgba(0,200,200,0.3)" }}>
        <p style={{ color: "white", fontWeight: "bold" }}>⏳ Generating AI Analysis...</p>
      </div>
    );
  }

  // Error
  if (error) {
    return (
      <div style={{ padding: 24, background: "rgba(255,0,0,0.1)", borderRadius: 16, border: "1px solid rgba(255,0,0,0.3)" }}>
        <p style={{ color: "#ff6666" }}>⚠️ {error}</p>
        <button onClick={generateReport} style={{ color: "cyan", marginTop: 8 }}>Try again</button>
      </div>
    );
  }

  // No report
  if (!report) return null;

  const severity = report.severity || "MODERATE";
  const colors: Record<string, string> = {
    CRITICAL: "#ff4444",
    HIGH: "#ff8800", 
    MODERATE: "#ffcc00",
    LOW: "#44ff44"
  };
  const borderColor = colors[severity] || colors.MODERATE;

  return (
    <div style={{ 
      padding: 24, 
      background: "rgba(30,40,60,0.9)", 
      borderRadius: 16, 
      border: `2px solid ${borderColor}`,
      marginBottom: 24
    }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
        <span style={{ fontSize: 28 }}>
          {severity === "CRITICAL" ? "🚨" : severity === "HIGH" ? "⚠️" : severity === "MODERATE" ? "📊" : "✅"}
        </span>
        <div>
          <div style={{ color: borderColor, fontSize: 11, fontWeight: "bold", letterSpacing: 2 }}>
            {severity} PRIORITY
          </div>
          <div style={{ color: "white", fontSize: 18, fontWeight: "bold" }}>
            AI Analysis Report
          </div>
        </div>
        <div style={{ marginLeft: "auto", padding: "4px 8px", background: "rgba(255,255,255,0.1)", borderRadius: 4, fontSize: 11, color: "rgba(255,255,255,0.6)" }}>
          CLAUDE AI
        </div>
      </div>

      {/* Headline */}
      {report.headline && (
        <p style={{ color: "white", fontSize: 20, fontWeight: 300, marginBottom: 20 }}>
          {report.headline}
        </p>
      )}

      {/* Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 20 }}>
        {report.primary_stat && (
          <div style={{ padding: 16, background: "rgba(255,255,255,0.05)", borderRadius: 12, textAlign: "center" }}>
            <div style={{ color: borderColor, fontSize: 24, fontWeight: "bold", fontFamily: "monospace" }}>
              {report.primary_stat.value}
            </div>
            <div style={{ color: "rgba(255,255,255,0.8)", fontSize: 14 }}>{report.primary_stat.label}</div>
            <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 12 }}>{report.primary_stat.detail}</div>
          </div>
        )}
        {report.secondary_stat && (
          <div style={{ padding: 16, background: "rgba(255,255,255,0.05)", borderRadius: 12, textAlign: "center" }}>
            <div style={{ color: "white", fontSize: 24, fontWeight: "bold", fontFamily: "monospace" }}>
              {report.secondary_stat.value}
            </div>
            <div style={{ color: "rgba(255,255,255,0.8)", fontSize: 14 }}>{report.secondary_stat.label}</div>
            <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 12 }}>{report.secondary_stat.detail}</div>
          </div>
        )}
        {report.tertiary_stat && (
          <div style={{ padding: 16, background: "rgba(255,255,255,0.05)", borderRadius: 12, textAlign: "center" }}>
            <div style={{ color: "white", fontSize: 24, fontWeight: "bold", fontFamily: "monospace" }}>
              {report.tertiary_stat.value}
            </div>
            <div style={{ color: "rgba(255,255,255,0.8)", fontSize: 14 }}>{report.tertiary_stat.label}</div>
            <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 12 }}>{report.tertiary_stat.detail}</div>
          </div>
        )}
      </div>

      {/* Insight */}
      {report.insight && (
        <div style={{ padding: 16, background: "rgba(255,255,255,0.05)", borderRadius: 12, marginBottom: 12 }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
            <span>💡</span>
            <div>
              <div style={{ color: "rgba(255,255,255,0.4)", fontSize: 11, fontWeight: "bold", marginBottom: 4 }}>KEY INSIGHT</div>
              <p style={{ color: "rgba(255,255,255,0.9)", fontSize: 14, margin: 0 }}>{report.insight}</p>
            </div>
          </div>
        </div>
      )}

      {/* Action */}
      {report.action && (
        <div style={{ padding: 16, background: `${borderColor}22`, borderRadius: 12, border: `1px solid ${borderColor}44` }}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
            <span>🎯</span>
            <div>
              <div style={{ color: borderColor, fontSize: 11, fontWeight: "bold", marginBottom: 4 }}>NEXT STEP</div>
              <p style={{ color: "white", fontSize: 14, fontWeight: 500, margin: 0 }}>{report.action}</p>
            </div>
          </div>
        </div>
      )}

      {/* Regenerate */}
      <div style={{ textAlign: "center", marginTop: 16 }}>
        <button 
          onClick={generateReport}
          style={{ 
            background: "transparent", 
            border: "none", 
            color: "rgba(255,255,255,0.3)", 
            cursor: "pointer",
            fontSize: 12
          }}
        >
          ↻ Regenerate
        </button>
      </div>
    </div>
  );
}
