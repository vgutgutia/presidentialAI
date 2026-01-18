"use client";

import Link from "next/link";
import { motion } from "framer-motion";

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-black">
      {/* Background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/3 left-1/4 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl" />
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
            <span className="text-white/60">ABOUT</span>
          </div>

          <div className="flex items-center gap-6">
            <Link href="/dashboard" className="text-sm text-white/60 hover:text-white transition-colors">
              DASHBOARD
            </Link>
            <Link href="/analyze" className="btn-primary text-sm py-2 px-4">
              ANALYZE
            </Link>
          </div>
        </div>
      </nav>

      <main className="relative z-10 pt-24 pb-20 px-6">
        <div className="max-w-5xl mx-auto">
          
          {/* Hero */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-20"
          >
            <span className="inline-block px-4 py-2 rounded-full glass text-cyan-400 text-sm font-medium mb-6 tracking-wider">
              PRESIDENTIAL AI CHALLENGE 2026
            </span>
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              ABOUT <span className="text-cyan-400">OCEANGUARD</span>
            </h1>
            <p className="text-xl text-white/50 max-w-3xl mx-auto">
              Harnessing the power of artificial intelligence and satellite technology 
              to protect our oceans from the growing threat of marine debris.
            </p>
          </motion.div>

          {/* Mission Section */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-20"
          >
            <div className="glass rounded-3xl p-8 md:p-12">
              <h2 className="text-3xl font-bold mb-6">THE MISSION</h2>
              <p className="text-lg text-white/70 leading-relaxed mb-6">
                Every year, over 14 million tons of plastic enter our oceans. Traditional monitoring 
                methods—boats, planes, manual surveys—are expensive, slow, and can only cover a 
                fraction of the world&apos;s waters. Marine debris accumulates offshore, often undetected, 
                until it devastates coastlines and marine ecosystems.
              </p>
              <p className="text-lg text-white/70 leading-relaxed">
                OceanGuard AI changes the game. By combining freely available Sentinel-2 satellite 
                imagery with cutting-edge deep learning, we can detect marine debris hotspots from 
                space—providing early warning systems for environmental agencies and cleanup operations 
                worldwide.
              </p>
            </div>
          </motion.section>

          {/* Technology Section */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-20"
          >
            <h2 className="text-3xl font-bold mb-8">THE TECHNOLOGY</h2>
            
            <div className="grid md:grid-cols-2 gap-6">
              {[
                {
                  title: "SegFormer Architecture",
                  desc: "State-of-the-art transformer-based semantic segmentation model, adapted for 11-band multispectral satellite imagery.",
                  icon: "🧠",
                },
                {
                  title: "Sentinel-2 Imagery",
                  desc: "Free, open-source satellite data from ESA with 10m resolution and global coverage every 5 days.",
                  icon: "🛰️",
                },
                {
                  title: "MARIDA Dataset",
                  desc: "Trained on the Marine Debris Archive—the largest labeled dataset of marine debris in satellite imagery.",
                  icon: "📊",
                },
                {
                  title: "Focal Loss Training",
                  desc: "Custom loss function handles extreme class imbalance, allowing detection of rare debris against vast ocean backgrounds.",
                  icon: "🎯",
                },
              ].map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="glass rounded-2xl p-6 group hover:bg-white/5 transition-colors"
                >
                  <span className="text-4xl block mb-4">{item.icon}</span>
                  <h3 className="text-xl font-semibold mb-2 group-hover:text-cyan-400 transition-colors">
                    {item.title}
                  </h3>
                  <p className="text-white/50">{item.desc}</p>
                </motion.div>
              ))}
            </div>
          </motion.section>

          {/* Architecture Details */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-20"
          >
            <h2 className="text-3xl font-bold mb-8">SYSTEM ARCHITECTURE</h2>
            
            <div className="glass rounded-3xl p-8 overflow-x-auto">
              <div className="flex items-center justify-between min-w-[600px] gap-4">
                {[
                  { label: "Sentinel-2", sub: "Satellite Data", color: "cyan" },
                  { label: "Preprocessing", sub: "11-Band Normalization", color: "blue" },
                  { label: "SegFormer", sub: "AI Inference", color: "indigo" },
                  { label: "Post-Process", sub: "Hotspot Detection", color: "purple" },
                  { label: "Output", sub: "GeoTIFF + GeoJSON", color: "pink" },
                ].map((step, i) => (
                  <div key={i} className="flex items-center">
                    <div className="text-center">
                      <div className={`w-20 h-20 rounded-2xl bg-${step.color}-500/20 flex items-center justify-center mb-2 mx-auto`}>
                        <span className="text-2xl font-bold font-mono text-white/80">{i + 1}</span>
                      </div>
                      <div className="font-semibold text-white text-sm">{step.label}</div>
                      <div className="text-xs text-white/40">{step.sub}</div>
                    </div>
                    {i < 4 && (
                      <svg className="w-8 h-8 text-white/20 mx-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </motion.section>

          {/* Performance Stats */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-20"
          >
            <h2 className="text-3xl font-bold mb-8">PERFORMANCE</h2>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { value: "85%+", label: "IoU Score", desc: "Intersection over Union" },
                { value: "0.89", label: "F1 Score", desc: "Precision-Recall Balance" },
                { value: "256px", label: "Patch Size", desc: "Processing Tile" },
                { value: "<2s", label: "Inference", desc: "Per Patch (GPU)" },
              ].map((stat, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="stat-card text-center"
                >
                  <div className="text-4xl font-bold text-cyan-400 font-mono mb-1">{stat.value}</div>
                  <div className="font-semibold text-white text-sm">{stat.label}</div>
                  <div className="text-xs text-white/40">{stat.desc}</div>
                </motion.div>
              ))}
            </div>
          </motion.section>

          {/* Team Section */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-20"
          >
            <h2 className="text-3xl font-bold mb-8">THE TEAM</h2>
            
            <div className="glass rounded-3xl p-8 text-center">
              <p className="text-lg text-white/70 mb-6">
                OceanGuard AI was developed for the Presidential AI Challenge 2026 
                by a passionate team dedicated to environmental protection through technology.
              </p>
              <div className="flex justify-center gap-4">
                <a 
                  href="https://github.com/vgutgutia/presidentialAI" 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-secondary"
                >
                  VIEW ON GITHUB
                </a>
              </div>
            </div>
          </motion.section>

          {/* CTA */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center"
          >
            <h2 className="text-4xl font-bold mb-6">
              READY TO <span className="text-cyan-400">START?</span>
            </h2>
            <p className="text-white/50 text-lg mb-8">
              Upload satellite imagery and detect marine debris in seconds.
            </p>
            <Link href="/analyze" className="btn-primary text-lg">
              LAUNCH DETECTION →
            </Link>
          </motion.div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8 px-6">
        <div className="max-w-7xl mx-auto text-center text-white/40 text-sm">
          Presidential AI Challenge 2026 • OceanGuard AI
        </div>
      </footer>
    </div>
  );
}
