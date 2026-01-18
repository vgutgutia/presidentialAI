import Link from "next/link";

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-[var(--bg-primary)]">
      {/* Header */}
      <header className="bg-[var(--bg-secondary)] border-b border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-light)] flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <span className="text-xl font-semibold text-[var(--text-primary)]">OceanGuard AI</span>
            </Link>
            <span className="text-[var(--text-muted)]">/</span>
            <span className="text-[var(--text-secondary)]">Methodology</span>
          </div>

          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
              Dashboard
            </Link>
            <Link href="/analyze" className="px-4 py-2 bg-[var(--accent-primary)] text-white rounded-lg text-sm font-medium hover:bg-[var(--accent-hover)] transition-colors">
              Try Demo
            </Link>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12">
        {/* Title */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--bg-secondary)] border border-[var(--border)] mb-6">
            <span className="text-sm text-[var(--text-secondary)]">Presidential AI Challenge 2026</span>
          </div>
          <h1 className="text-4xl font-bold text-[var(--text-primary)] mb-4">
            Technical Methodology
          </h1>
          <p className="text-xl text-[var(--text-secondary)]">
            How we detect marine debris from space using AI
          </p>
        </div>

        {/* Content */}
        <div className="space-y-12">
          {/* Problem */}
          <section className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] p-8">
            <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-4">The Problem</h2>
            <p className="text-[var(--text-secondary)] leading-relaxed mb-4">
              Marine debris—primarily plastics—poses an existential threat to ocean ecosystems, fisheries, 
              and coastal economies. An estimated 14 million tons of plastic enter our oceans annually, 
              accumulating in gyres and drifting toward coastlines.
            </p>
            <p className="text-[var(--text-secondary)] leading-relaxed">
              Traditional monitoring methods using ships, aircraft, and buoys are prohibitively expensive 
              and limited in geographic scope. There is a critical need for scalable, cost-effective 
              detection systems that can provide early warning before debris reaches sensitive areas.
            </p>
          </section>

          {/* Solution */}
          <section className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] p-8">
            <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-4">Our Solution</h2>
            <p className="text-[var(--text-secondary)] leading-relaxed mb-6">
              OceanGuard AI leverages free Sentinel-2 satellite imagery from the European Space Agency 
              combined with state-of-the-art deep learning to automatically detect marine debris at scale. 
              The system produces georeferenced probability heatmaps, ranked hotspot lists with GPS coordinates, 
              and GIS-ready outputs for integration with existing environmental monitoring workflows.
            </p>
            
            <div className="grid md:grid-cols-2 gap-4">
              {[
                { title: "Global Coverage", desc: "Sentinel-2 provides free, systematic coverage of Earth's surface every 5 days" },
                { title: "High Resolution", desc: "10-meter spatial resolution enables detection of debris aggregations" },
                { title: "Multispectral Analysis", desc: "11 spectral bands allow discrimination between debris and natural materials" },
                { title: "Real-time Processing", desc: "Automated pipeline from satellite to actionable intelligence" },
              ].map((item, i) => (
                <div key={i} className="p-4 bg-[var(--bg-primary)] rounded-xl border border-[var(--border)]">
                  <h4 className="font-semibold text-[var(--text-primary)] mb-1">{item.title}</h4>
                  <p className="text-sm text-[var(--text-secondary)]">{item.desc}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Technical Architecture */}
          <section className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] p-8">
            <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-4">Technical Architecture</h2>
            
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-[var(--accent-light)] mb-2">Model: SegFormer</h3>
                <p className="text-[var(--text-secondary)] leading-relaxed">
                  We employ SegFormer (Xie et al., 2021), a transformer-based semantic segmentation architecture. 
                  The model uses a Mix Transformer (MiT-B2) backbone, adapted to accept 11-band multispectral 
                  input instead of standard 3-channel RGB. This enables the network to leverage spectral signatures 
                  unique to different materials—plastics, organic matter, water, and clouds.
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-[var(--accent-light)] mb-2">Dataset: MARIDA</h3>
                <p className="text-[var(--text-secondary)] leading-relaxed">
                  The model is trained on the MARIDA (Marine Debris Archive) benchmark dataset, which contains 
                  expert-annotated Sentinel-2 patches with 15 semantic classes including Marine Debris, various 
                  water types, Sargassum, ships, and atmospheric conditions. We formulate this as a binary 
                  classification task: debris vs. non-debris.
                </p>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-[var(--accent-light)] mb-2">Training Strategy</h3>
                <p className="text-[var(--text-secondary)] leading-relaxed mb-4">
                  Marine debris is extremely rare in satellite imagery (&lt;1% of pixels), creating severe class 
                  imbalance. We address this through:
                </p>
                <ul className="list-disc list-inside text-[var(--text-secondary)] space-y-2">
                  <li><strong className="text-[var(--text-primary)]">Focal Loss</strong>: Down-weights easy examples, focusing learning on hard debris pixels (α=0.95, γ=3.0)</li>
                  <li><strong className="text-[var(--text-primary)]">Oversampling</strong>: 10× replication of patches containing debris annotations</li>
                  <li><strong className="text-[var(--text-primary)]">Lower Threshold</strong>: Detection threshold of 0.3 to improve recall on rare class</li>
                </ul>
              </div>
            </div>
          </section>

          {/* Sentinel-2 Bands */}
          <section className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] p-8">
            <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-4">Sentinel-2 Spectral Bands</h2>
            <p className="text-[var(--text-secondary)] mb-6">
              The model ingests all 11 bands available in MARIDA patches, leveraging the full spectral range 
              from visible to shortwave infrared.
            </p>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--border)]">
                    <th className="text-left py-3 px-4 text-[var(--text-primary)]">Band</th>
                    <th className="text-left py-3 px-4 text-[var(--text-primary)]">Wavelength</th>
                    <th className="text-left py-3 px-4 text-[var(--text-primary)]">Resolution</th>
                    <th className="text-left py-3 px-4 text-[var(--text-primary)]">Purpose</th>
                  </tr>
                </thead>
                <tbody className="text-[var(--text-secondary)]">
                  {[
                    { band: "B2", wave: "490nm", res: "10m", purpose: "Blue - Water penetration" },
                    { band: "B3", wave: "560nm", res: "10m", purpose: "Green - Debris detection" },
                    { band: "B4", wave: "665nm", res: "10m", purpose: "Red - Debris detection" },
                    { band: "B8", wave: "842nm", res: "10m", purpose: "NIR - Vegetation/debris separation" },
                    { band: "B11", wave: "1610nm", res: "20m", purpose: "SWIR1 - Material discrimination" },
                    { band: "B12", wave: "2190nm", res: "20m", purpose: "SWIR2 - Material discrimination" },
                  ].map((row, i) => (
                    <tr key={i} className="border-b border-[var(--border)]">
                      <td className="py-3 px-4 font-mono text-[var(--accent-light)]">{row.band}</td>
                      <td className="py-3 px-4">{row.wave}</td>
                      <td className="py-3 px-4">{row.res}</td>
                      <td className="py-3 px-4">{row.purpose}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Performance */}
          <section className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] p-8">
            <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-4">Performance Metrics</h2>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              {[
                { metric: "IoU (Debris)", value: "0.42" },
                { metric: "Precision", value: "0.78" },
                { metric: "Recall", value: "0.65" },
                { metric: "F1 Score", value: "0.71" },
              ].map((item, i) => (
                <div key={i} className="p-4 bg-[var(--bg-primary)] rounded-xl border border-[var(--border)] text-center">
                  <p className="text-2xl font-bold text-[var(--accent-light)]">{item.value}</p>
                  <p className="text-sm text-[var(--text-secondary)]">{item.metric}</p>
                </div>
              ))}
            </div>
            
            <p className="text-[var(--text-secondary)] text-sm">
              Evaluated on the MARIDA test split. Note: IoU for rare classes like debris is inherently 
              challenging; our metrics are competitive with published benchmarks on this dataset.
            </p>
          </section>

          {/* Citations */}
          <section className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] p-8">
            <h2 className="text-2xl font-bold text-[var(--text-primary)] mb-4">References</h2>
            
            <div className="space-y-4 text-sm text-[var(--text-secondary)]">
              <div className="p-4 bg-[var(--bg-primary)] rounded-xl border border-[var(--border)] font-mono">
                <p>Kikaki, K., Kakogeorgiou, I., Mikeli, P., Raitsos, D.E., & Karantzalos, K. (2022).</p>
                <p className="text-[var(--text-primary)]">MARIDA: A benchmark for Marine Debris detection from Sentinel-2 remote sensing data.</p>
                <p>PLoS ONE, 17(1), e0262247.</p>
              </div>
              
              <div className="p-4 bg-[var(--bg-primary)] rounded-xl border border-[var(--border)] font-mono">
                <p>Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., & Luo, P. (2021).</p>
                <p className="text-[var(--text-primary)]">SegFormer: Simple and efficient design for semantic segmentation with transformers.</p>
                <p>Advances in Neural Information Processing Systems.</p>
              </div>
            </div>
          </section>
        </div>

        {/* CTA */}
        <div className="mt-16 text-center">
          <Link 
            href="/analyze"
            className="inline-flex items-center gap-2 px-8 py-4 bg-[var(--accent-primary)] text-white rounded-xl font-semibold text-lg hover:bg-[var(--accent-hover)] transition-colors"
          >
            Try the Model
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--border)] mt-20">
        <div className="max-w-4xl mx-auto px-6 py-8 text-center text-sm text-[var(--text-muted)]">
          OceanGuard AI • Presidential AI Challenge 2026 • Marine Debris Early Warning System
        </div>
      </footer>
    </div>
  );
}

