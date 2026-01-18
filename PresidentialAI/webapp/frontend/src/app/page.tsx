import Link from "next/link";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[var(--bg-primary)]">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-light)] flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className="text-xl font-semibold text-[var(--text-primary)]">OceanGuard AI</span>
          </div>
          
          <div className="hidden md:flex items-center gap-8">
            <Link href="/dashboard" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
              Dashboard
            </Link>
            <Link href="/analyze" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
              Analyze
            </Link>
            <Link href="/about" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
              Methodology
            </Link>
          </div>

          <Link 
            href="/analyze"
            className="px-5 py-2.5 bg-[var(--accent-primary)] text-white rounded-lg font-medium hover:bg-[var(--accent-hover)] transition-colors"
          >
            Try Demo
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--bg-secondary)] border border-[var(--border)] mb-6">
                <span className="w-2 h-2 rounded-full bg-[var(--success)] animate-pulse"></span>
                <span className="text-sm text-[var(--text-secondary)]">Presidential AI Challenge 2026</span>
              </div>
              
              <h1 className="text-5xl lg:text-6xl font-bold text-[var(--text-primary)] leading-tight mb-6">
                Marine Debris
                <span className="block text-[var(--accent-light)]">Early Warning System</span>
              </h1>
              
              <p className="text-xl text-[var(--text-secondary)] leading-relaxed mb-8 max-w-xl">
                Leveraging Sentinel-2 satellite imagery and deep learning to detect floating marine debris 
                before it reaches our coastlines. Protecting ecosystems, fisheries, and communities.
              </p>

              <div className="flex flex-wrap gap-4">
                <Link 
                  href="/analyze"
                  className="px-8 py-4 bg-[var(--accent-primary)] text-white rounded-xl font-semibold text-lg hover:bg-[var(--accent-hover)] transition-all hover:scale-105"
                >
                  Test the Model →
                </Link>
                <Link 
                  href="/dashboard"
                  className="px-8 py-4 bg-[var(--bg-secondary)] text-[var(--text-primary)] border border-[var(--border)] rounded-xl font-semibold text-lg hover:bg-[var(--bg-tertiary)] transition-colors"
                >
                  View Dashboard
                </Link>
              </div>
            </div>

            {/* Hero Visual */}
            <div className="relative">
              <div className="aspect-square rounded-2xl bg-gradient-to-br from-[var(--bg-secondary)] to-[var(--bg-tertiary)] border border-[var(--border)] overflow-hidden">
                {/* Animated Globe/Satellite Visual */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative w-64 h-64">
                    {/* Orbit rings */}
                    <div className="absolute inset-0 rounded-full border border-[var(--border)] opacity-30"></div>
                    <div className="absolute inset-4 rounded-full border border-[var(--border)] opacity-40"></div>
                    <div className="absolute inset-8 rounded-full border border-[var(--border)] opacity-50"></div>
                    
                    {/* Center globe */}
                    <div className="absolute inset-12 rounded-full bg-gradient-to-br from-[var(--accent-primary)] to-[var(--bg-tertiary)] flex items-center justify-center">
                      <svg className="w-16 h-16 text-white/80" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>

                    {/* Orbiting satellite */}
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-2 w-4 h-4 bg-[var(--accent-light)] rounded-full animate-pulse"></div>
                  </div>
                </div>

                {/* Grid overlay */}
                <div 
                  className="absolute inset-0 opacity-10"
                  style={{
                    backgroundImage: `linear-gradient(var(--border) 1px, transparent 1px), linear-gradient(90deg, var(--border) 1px, transparent 1px)`,
                    backgroundSize: '40px 40px'
                  }}
                ></div>
              </div>

              {/* Floating stats cards */}
              <div className="absolute -bottom-4 -left-4 px-4 py-3 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl">
                <p className="text-2xl font-bold text-[var(--accent-light)]">94.2%</p>
                <p className="text-sm text-[var(--text-secondary)]">Detection Accuracy</p>
              </div>
              <div className="absolute -top-4 -right-4 px-4 py-3 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl">
                <p className="text-2xl font-bold text-[var(--success)]">Real-time</p>
                <p className="text-sm text-[var(--text-secondary)]">Satellite Processing</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section className="py-20 px-6 bg-[var(--bg-secondary)]">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-[var(--text-primary)] mb-4">The Marine Debris Crisis</h2>
            <p className="text-[var(--text-secondary)] max-w-2xl mx-auto">
              Millions of tons of plastic and debris enter our oceans annually, threatening marine life 
              and coastal communities. Traditional monitoring is costly and limited in scope.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { value: "14M", label: "Tons of plastic enter oceans yearly", icon: "📊" },
              { value: "$13B", label: "Annual economic damage to marine industries", icon: "💰" },
              { value: "100K+", label: "Marine animals harmed by debris yearly", icon: "🐋" },
            ].map((stat, i) => (
              <div key={i} className="text-center p-8 rounded-2xl bg-[var(--bg-primary)] border border-[var(--border)]">
                <span className="text-4xl mb-4 block">{stat.icon}</span>
                <p className="text-4xl font-bold text-[var(--accent-light)] mb-2">{stat.value}</p>
                <p className="text-[var(--text-secondary)]">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-[var(--text-primary)] mb-4">Our Solution</h2>
            <p className="text-[var(--text-secondary)] max-w-2xl mx-auto">
              Combining free satellite data with state-of-the-art AI for scalable, 
              cost-effective debris detection and early warning.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { 
                title: "Sentinel-2 Imagery", 
                desc: "Free, global satellite coverage with 11 spectral bands at 10m resolution",
                icon: "🛰️"
              },
              { 
                title: "SegFormer AI", 
                desc: "Transformer-based deep learning for precise semantic segmentation",
                icon: "🧠"
              },
              { 
                title: "Georeferenced Output", 
                desc: "GPS coordinates, heatmaps, and GIS-ready formats for field teams",
                icon: "📍"
              },
              { 
                title: "Early Warning", 
                desc: "Detect debris offshore before it impacts coastlines and ecosystems",
                icon: "⚡"
              },
            ].map((item, i) => (
              <div key={i} className="p-6 rounded-2xl bg-[var(--bg-secondary)] border border-[var(--border)] hover:border-[var(--accent-primary)] transition-colors">
                <span className="text-3xl mb-4 block">{item.icon}</span>
                <h3 className="text-lg font-semibold text-[var(--text-primary)] mb-2">{item.title}</h3>
                <p className="text-sm text-[var(--text-secondary)]">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6 bg-gradient-to-b from-[var(--bg-primary)] to-[var(--bg-secondary)]">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-[var(--text-primary)] mb-6">
            Ready to see it in action?
          </h2>
          <p className="text-xl text-[var(--text-secondary)] mb-8">
            Upload your own satellite imagery or use our sample data to test the marine debris detection model.
          </p>
          <Link 
            href="/analyze"
            className="inline-flex items-center gap-2 px-10 py-5 bg-[var(--accent-primary)] text-white rounded-xl font-semibold text-xl hover:bg-[var(--accent-hover)] transition-all hover:scale-105"
          >
            Launch Analysis Tool
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-[var(--border)]">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-light)] flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className="font-semibold text-[var(--text-primary)]">OceanGuard AI</span>
          </div>
          
          <p className="text-sm text-[var(--text-secondary)]">
            Presidential AI Challenge 2026 • Marine Debris Early Warning System
          </p>

          <div className="flex items-center gap-6 text-sm text-[var(--text-secondary)]">
            <Link href="/about" className="hover:text-[var(--text-primary)] transition-colors">About</Link>
            <Link href="/dashboard" className="hover:text-[var(--text-primary)] transition-colors">Dashboard</Link>
            <Link href="/analyze" className="hover:text-[var(--text-primary)] transition-colors">Analyze</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
