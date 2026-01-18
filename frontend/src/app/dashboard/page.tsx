"use client";

import Link from "next/link";
import { motion } from "framer-motion";

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-black">
      {/* Background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-1/3 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/3 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl" />
        <div className="absolute inset-0 grid-lines opacity-20" />
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 nav-blur border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <span className="text-xl font-semibold">OCEANGUARD</span>
            </Link>
            <span className="text-white/20">/</span>
            <span className="text-white/60">DASHBOARD</span>
          </div>

          <div className="flex items-center gap-6">
            <Link href="/analyze" className="btn-primary text-sm py-2 px-4">
              NEW ANALYSIS
            </Link>
            <Link href="/about" className="text-sm text-white/60 hover:text-white transition-colors">
              ABOUT
            </Link>
          </div>
        </div>
      </nav>

      <main className="relative z-10 pt-24 pb-12 px-6">
        <div className="max-w-7xl mx-auto">
          
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-12"
          >
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              MISSION <span className="text-cyan-400">CONTROL</span>
            </h1>
            <p className="text-white/50 text-lg">
              Global marine debris monitoring overview
            </p>
          </motion.div>

          {/* Stats Row */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
          >
            {[
              { label: "ACTIVE ALERTS", value: "12", trend: "+3", trendUp: false, color: "red" },
              { label: "AREA MONITORED", value: "1.2M", suffix: "km²", trend: "+10%", trendUp: true, color: "cyan" },
              { label: "HOTSPOTS FOUND", value: "2,450", trend: "+50", trendUp: false, color: "amber" },
              { label: "SYSTEM UPTIME", value: "99.9%", trend: "", trendUp: true, color: "green" },
            ].map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + i * 0.05 }}
                className="stat-card"
              >
                <div className="flex items-start justify-between mb-2">
                  <span className="text-xs text-white/40 tracking-widest">{stat.label}</span>
                  {stat.trend && (
                    <span className={`text-xs ${stat.trendUp ? "text-green-400" : "text-red-400"}`}>
                      {stat.trend}
                    </span>
                  )}
                </div>
                <div className={`text-3xl font-bold font-mono text-${stat.color}-400`}>
                  {stat.value}
                  {stat.suffix && <span className="text-lg text-white/40 ml-1">{stat.suffix}</span>}
                </div>
              </motion.div>
            ))}
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-6 mb-8">
            {/* Map Placeholder */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="lg:col-span-2 glass rounded-2xl overflow-hidden"
            >
              <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
                <h2 className="font-semibold text-white">GLOBAL DETECTION MAP</h2>
                <span className="px-2 py-1 bg-cyan-500/20 rounded text-xs text-cyan-400 animate-pulse">LIVE</span>
              </div>
              <div className="aspect-video bg-gradient-to-br from-cyan-900/20 to-indigo-900/20 flex items-center justify-center relative">
                {/* Simulated map grid */}
                <div className="absolute inset-0 grid-lines opacity-30" />
                
                {/* Animated dots representing detections */}
                {[
                  { x: "25%", y: "40%", size: 3, delay: 0 },
                  { x: "45%", y: "35%", size: 2, delay: 0.5 },
                  { x: "65%", y: "55%", size: 4, delay: 1 },
                  { x: "30%", y: "60%", size: 2, delay: 1.5 },
                  { x: "70%", y: "30%", size: 3, delay: 2 },
                  { x: "55%", y: "70%", size: 2, delay: 2.5 },
                ].map((dot, i) => (
                  <motion.div
                    key={i}
                    className="absolute"
                    style={{ left: dot.x, top: dot.y }}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: dot.delay, duration: 0.5 }}
                  >
                    <div 
                      className="relative"
                      style={{ width: dot.size * 8, height: dot.size * 8 }}
                    >
                      <div className="absolute inset-0 bg-cyan-400 rounded-full animate-ping opacity-30" />
                      <div className="absolute inset-0 bg-cyan-400 rounded-full" />
                    </div>
                  </motion.div>
                ))}
                
                <div className="text-center z-10">
                  <span className="text-6xl block mb-4">🌍</span>
                  <p className="text-white/50">Interactive map coming soon</p>
                  <p className="text-xs text-white/30 mt-1">Real-time debris tracking</p>
                </div>
              </div>
            </motion.div>

            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="glass rounded-2xl p-6"
            >
              <h2 className="font-semibold text-white mb-4">QUICK ACTIONS</h2>
              <div className="space-y-3">
                {[
                  { label: "New Analysis", href: "/analyze", icon: "🔍", color: "cyan" },
                  { label: "View Reports", href: "#", icon: "📊", color: "blue" },
                  { label: "Export Data", href: "#", icon: "📤", color: "indigo" },
                  { label: "Settings", href: "#", icon: "⚙️", color: "gray" },
                ].map((action, i) => (
                  <Link
                    key={i}
                    href={action.href}
                    className="flex items-center gap-4 p-4 bg-white/5 rounded-xl hover:bg-white/10 transition-colors group"
                  >
                    <span className="text-2xl">{action.icon}</span>
                    <span className="font-medium text-white group-hover:text-cyan-400 transition-colors">
                      {action.label}
                    </span>
                    <svg className="w-5 h-5 text-white/20 ml-auto group-hover:text-cyan-400 group-hover:translate-x-1 transition-all" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Recent Detections */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="glass rounded-2xl overflow-hidden"
          >
            <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
              <h2 className="font-semibold text-white">RECENT DETECTIONS</h2>
              <button className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors">
                VIEW ALL →
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/5 text-xs text-white/40 tracking-widest">
                    <th className="text-left py-3 px-6">ID</th>
                    <th className="text-left py-3 px-6">LOCATION</th>
                    <th className="text-left py-3 px-6">CONFIDENCE</th>
                    <th className="text-left py-3 px-6">AREA</th>
                    <th className="text-left py-3 px-6">DETECTED</th>
                    <th className="text-left py-3 px-6">STATUS</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { id: "MD-2026-001", loc: "34.05°N, 118.25°W", conf: 98, area: "52,000", date: "2 hours ago", status: "critical" },
                    { id: "MD-2026-002", loc: "25.76°N, 80.19°W", conf: 95, area: "38,000", date: "5 hours ago", status: "high" },
                    { id: "MD-2026-003", loc: "37.77°N, 122.41°W", conf: 92, area: "29,000", date: "8 hours ago", status: "high" },
                    { id: "MD-2026-004", loc: "18.46°N, 66.11°W", conf: 88, area: "15,000", date: "12 hours ago", status: "medium" },
                    { id: "MD-2026-005", loc: "21.31°N, 157.86°W", conf: 82, area: "11,000", date: "1 day ago", status: "medium" },
                  ].map((detection, i) => (
                    <motion.tr
                      key={i}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.5 + i * 0.05 }}
                      className="border-b border-white/5 hover:bg-white/5 transition-colors cursor-pointer"
                    >
                      <td className="py-4 px-6 font-mono text-cyan-400">{detection.id}</td>
                      <td className="py-4 px-6 text-white/60 font-mono text-sm">{detection.loc}</td>
                      <td className="py-4 px-6">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                          detection.conf >= 95 
                            ? "bg-red-500/20 text-red-400" 
                            : detection.conf >= 90 
                              ? "bg-amber-500/20 text-amber-400" 
                              : "bg-green-500/20 text-green-400"
                        }`}>
                          {detection.conf}%
                        </span>
                      </td>
                      <td className="py-4 px-6 text-white/60 font-mono">{detection.area} m²</td>
                      <td className="py-4 px-6 text-white/40 text-sm">{detection.date}</td>
                      <td className="py-4 px-6">
                        <span className={`px-2 py-1 rounded text-xs font-medium uppercase tracking-wider ${
                          detection.status === "critical" 
                            ? "bg-red-500/20 text-red-400" 
                            : detection.status === "high" 
                              ? "bg-amber-500/20 text-amber-400" 
                              : "bg-blue-500/20 text-blue-400"
                        }`}>
                          {detection.status}
                        </span>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
