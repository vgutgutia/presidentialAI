import { Link } from 'react-router-dom'

export default function Hero() {
  return (
    <section className="hero">
      <div className="hero-grid" aria-hidden="true" />
      
      <div className="container">
        <div className="hero-content">
          <div className="hero-badges animate-slide-up">
            <span className="badge">Presidential AI Challenge</span>
            <span className="badge badge-outline">NEON + Earth Engine</span>
          </div>

          <h1 className="hero-title animate-slide-up delay-1">
            Detect Environmental <span className="accent">Waste</span> from Space
          </h1>
          
          <p className="hero-description animate-slide-up delay-2">
            EcoSight AI leverages 0.1-meter resolution satellite imagery from 
            <strong> NEON's Airborne Observation Platform</strong> to identify 
            illegal dumping sites, landfill overflow, and debris accumulation — 
            enabling rapid environmental response.
          </p>

          <div className="hero-stats animate-slide-up delay-3">
            <div className="hero-stat">
              <span className="stat-value">0.1m</span>
              <span className="stat-label">Resolution</span>
            </div>
            <div className="hero-stat">
              <span className="stat-value">81</span>
              <span className="stat-label">NEON Sites</span>
            </div>
            <div className="hero-stat">
              <span className="stat-value">95%</span>
              <span className="stat-label">Detection Accuracy</span>
            </div>
          </div>

          <div className="hero-actions animate-slide-up delay-4">
            <Link to="/demo" className="btn btn-primary btn-lg">
              Try Live Demo
            </Link>
            <a href="#about" className="btn btn-secondary btn-lg">
              How It Works
            </a>
          </div>
        </div>
      </div>

      <style>{`
        .hero-badges {
          display: flex;
          gap: var(--space-3);
          margin-bottom: var(--space-6);
          flex-wrap: wrap;
        }

        .hero-stats {
          display: flex;
          gap: var(--space-8);
          margin-bottom: var(--space-10);
          flex-wrap: wrap;
        }

        .hero-stat {
          display: flex;
          flex-direction: column;
        }

        .hero-stat .stat-value {
          font-family: var(--font-mono);
          font-size: var(--text-3xl);
          font-weight: 600;
          color: var(--accent-primary);
          line-height: 1;
        }

        .hero-stat .stat-label {
          font-size: var(--text-sm);
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
          margin-top: var(--space-1);
        }
      `}</style>
    </section>
  )
}
