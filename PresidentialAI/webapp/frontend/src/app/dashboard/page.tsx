"use client";

import Link from "next/link";
import { useState } from "react";

const mockHotspots = [
  { id: 1, location: "Pacific Gyre - Zone A", coords: "37.7749°N, 122.4194°W", confidence: 94, area: 45000, severity: "high" },
  { id: 2, location: "Caribbean Sector 12", coords: "18.4655°N, 66.1057°W", confidence: 87, area: 32000, severity: "medium" },
  { id: 3, location: "Mediterranean Zone B", coords: "35.8989°N, 14.5146°E", confidence: 82, area: 28500, severity: "medium" },
  { id: 4, location: "South Atlantic Point", coords: "34.6037°S, 58.3816°W", confidence: 76, area: 18200, severity: "low" },
  { id: 5, location: "Indian Ocean Drift", coords: "12.8797°S, 96.9254°E", confidence: 71, area: 15800, severity: "low" },
];

const stats = [
  { label: "Active Hotspots", value: "24", change: "+3", icon: "🎯" },
  { label: "Area Monitored", value: "1.2M", unit: "km²", icon: "📡" },
  { label: "Model Accuracy", value: "94.2", unit: "%", icon: "✓" },
  { label: "Alerts Today", value: "7", change: "+2", icon: "🔔" },
];

const navItems = [
  { icon: "📊", label: "Dashboard", href: "/dashboard", active: true },
  { icon: "🔬", label: "Analyze", href: "/analyze", active: false },
  { icon: "📜", label: "History", href: "/history", active: false },
  { icon: "📋", label: "Reports", href: "/reports", active: false },
  { icon: "ℹ️", label: "About", href: "/about", active: false },
];

export default function DashboardPage() {
  const [selectedHotspot, setSelectedHotspot] = useState<number | null>(null);

  return (
    <div className="flex h-screen bg-[var(--bg-primary)]">
      {/* Sidebar */}
      <aside className="w-20 bg-[var(--bg-secondary)] border-r border-[var(--border)] flex flex-col items-center py-6">
        <Link href="/" className="w-12 h-12 rounded-xl bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-light)] flex items-center justify-center mb-8">
          <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </Link>

        <nav className="flex-1 flex flex-col gap-2">
          {navItems.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all ${
                item.active
                  ? "bg-[var(--accent-primary)] text-white"
                  : "text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)] hover:text-white"
              }`}
              title={item.label}
            >
              <span className="text-xl">{item.icon}</span>
            </Link>
          ))}
        </nav>

        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-light)] flex items-center justify-center text-white font-semibold text-sm">
          OG
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-[var(--bg-secondary)] border-b border-[var(--border)] px-6 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-[var(--text-primary)]">Mission Control</h1>
            <p className="text-sm text-[var(--text-secondary)]">Global Marine Debris Monitoring</p>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-4 py-2 bg-[var(--bg-primary)] rounded-lg border border-[var(--border)]">
              <div className="w-2 h-2 rounded-full bg-[var(--success)] animate-pulse"></div>
              <span className="text-sm text-[var(--text-secondary)]">Model Online</span>
            </div>

            <Link 
              href="/analyze"
              className="px-4 py-2 bg-[var(--accent-primary)] text-white rounded-lg text-sm font-medium hover:bg-[var(--accent-hover)] transition-colors"
            >
              + New Analysis
            </Link>
          </div>
        </header>

        {/* Content Grid */}
        <main className="flex-1 p-6 overflow-auto">
          <div className="grid grid-cols-12 gap-6 h-full">
            {/* Map Section */}
            <div className="col-span-8 flex flex-col gap-6">
              {/* Map */}
              <div className="flex-1 min-h-[400px] bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] relative overflow-hidden">
                {/* Map Placeholder */}
                <div className="absolute inset-0 bg-gradient-to-br from-[#0a1628] via-[#132337] to-[#1a3a5c]">
                  <div 
                    className="absolute inset-0 opacity-20"
                    style={{
                      backgroundImage: `
                        linear-gradient(rgba(74,144,217,0.3) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(74,144,217,0.3) 1px, transparent 1px)
                      `,
                      backgroundSize: '60px 60px'
                    }}
                  ></div>

                  {/* Mock hotspot markers */}
                  <div className="absolute top-1/4 left-1/3 w-4 h-4 rounded-full bg-[var(--danger)] animate-pulse cursor-pointer" title="High severity"></div>
                  <div className="absolute top-1/2 left-1/2 w-3 h-3 rounded-full bg-[var(--warning)] animate-pulse cursor-pointer" title="Medium severity"></div>
                  <div className="absolute top-2/3 left-2/3 w-3 h-3 rounded-full bg-[var(--warning)] animate-pulse cursor-pointer" title="Medium severity"></div>
                  <div className="absolute top-1/3 right-1/4 w-2 h-2 rounded-full bg-[var(--success)] animate-pulse cursor-pointer" title="Low severity"></div>

                  {/* Center content */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <p className="text-[var(--text-secondary)] text-lg mb-2">Interactive Map</p>
                      <p className="text-[var(--text-muted)] text-sm">Satellite imagery with debris overlay</p>
                    </div>
                  </div>
                </div>

                {/* Map Controls */}
                <div className="absolute top-4 right-4 flex flex-col gap-2">
                  <button className="w-10 h-10 bg-[var(--bg-primary)] border border-[var(--border)] rounded-lg flex items-center justify-center text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]">+</button>
                  <button className="w-10 h-10 bg-[var(--bg-primary)] border border-[var(--border)] rounded-lg flex items-center justify-center text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)]">−</button>
                </div>

                {/* Layer Controls */}
                <div className="absolute bottom-4 left-4 flex gap-2">
                  <button className="px-3 py-1.5 bg-[var(--accent-primary)] text-white rounded-lg text-sm">Satellite</button>
                  <button className="px-3 py-1.5 bg-[var(--bg-primary)] border border-[var(--border)] text-[var(--text-secondary)] rounded-lg text-sm hover:bg-[var(--bg-tertiary)]">Heatmap</button>
                  <button className="px-3 py-1.5 bg-[var(--bg-primary)] border border-[var(--border)] text-[var(--text-secondary)] rounded-lg text-sm hover:bg-[var(--bg-tertiary)]">Markers</button>
                </div>

                {/* Coordinates */}
                <div className="absolute bottom-4 right-4 px-3 py-1.5 bg-black/50 backdrop-blur rounded-lg">
                  <span className="text-xs text-[var(--text-secondary)] font-mono">37.7749°N, 122.4194°W</span>
                </div>
              </div>

              {/* Stats Bar */}
              <div className="grid grid-cols-4 gap-4">
                {stats.map((stat) => (
                  <div key={stat.label} className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-2xl">{stat.icon}</span>
                      {stat.change && (
                        <span className="text-xs px-2 py-1 rounded-full bg-[var(--success)]/20 text-[var(--success)]">
                          {stat.change}
                        </span>
                      )}
                    </div>
                    <p className="text-2xl font-bold text-[var(--text-primary)]">
                      {stat.value}
                      {stat.unit && <span className="text-sm text-[var(--text-secondary)] ml-1">{stat.unit}</span>}
                    </p>
                    <p className="text-sm text-[var(--text-secondary)]">{stat.label}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Hotspots Panel */}
            <div className="col-span-4 bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] flex flex-col overflow-hidden">
              <div className="px-4 py-3 border-b border-[var(--border)] flex items-center justify-between">
                <h3 className="font-semibold text-[var(--text-primary)]">Detected Hotspots</h3>
                <span className="text-xs px-2 py-1 bg-[var(--danger)]/20 text-[var(--danger)] rounded-full">
                  {mockHotspots.length} Active
                </span>
              </div>

              <div className="flex-1 overflow-auto p-3">
                <div className="space-y-2">
                  {mockHotspots.map((spot) => (
                    <button
                      key={spot.id}
                      onClick={() => setSelectedHotspot(spot.id)}
                      className={`w-full text-left p-3 rounded-xl transition-colors ${
                        selectedHotspot === spot.id
                          ? "bg-[var(--accent-primary)]/20 border border-[var(--accent-primary)]"
                          : "bg-[var(--bg-primary)] border border-[var(--border)] hover:border-[var(--accent-primary)]/50"
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-lg font-bold text-[var(--text-primary)]">#{spot.id}</span>
                          <span className={`w-2 h-2 rounded-full ${
                            spot.severity === 'high' ? 'bg-[var(--danger)]' :
                            spot.severity === 'medium' ? 'bg-[var(--warning)]' : 'bg-[var(--success)]'
                          }`}></span>
                        </div>
                        <span className="text-sm font-medium text-[var(--accent-light)]">{spot.confidence}%</span>
                      </div>
                      <p className="text-sm font-medium text-[var(--text-primary)] mb-1">{spot.location}</p>
                      <p className="text-xs text-[var(--text-muted)] font-mono">{spot.coords}</p>
                      <div className="flex items-center justify-between mt-2 pt-2 border-t border-[var(--border)]">
                        <span className="text-xs text-[var(--text-secondary)]">Area: {spot.area.toLocaleString()} m²</span>
                        <span className="text-xs text-[var(--accent-primary)]">View →</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="px-4 py-3 border-t border-[var(--border)]">
                <Link 
                  href="/analyze"
                  className="block w-full py-2 bg-[var(--bg-primary)] text-[var(--text-secondary)] rounded-lg text-sm text-center hover:bg-[var(--bg-tertiary)] transition-colors"
                >
                  Analyze New Image →
                </Link>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

