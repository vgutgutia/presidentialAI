export default function About() {
  const features = [
    {
      icon: '🛰️',
      title: 'NEON Satellite Imagery',
      description: 'Access 0.1-meter resolution RGB orthomosaics from the National Ecological Observatory Network via Google Earth Engine.'
    },
    {
      icon: '🤖',
      title: 'AI Waste Detection',
      description: 'Deep learning models trained to identify trash accumulation patterns, illegal dumping, and debris fields.'
    },
    {
      icon: '📊',
      title: 'Real-Time Analytics',
      description: 'Instant severity classification, confidence scoring, and area estimation for detected waste zones.'
    },
    {
      icon: '🗺️',
      title: 'Interactive Mapping',
      description: 'Explore detected areas on an interactive map with coordinates, zoom to high-priority zones.'
    }
  ]

  const techStack = [
    { name: 'Google Earth Engine', desc: 'Satellite data platform' },
    { name: 'NEON AOP', desc: '0.1m RGB imagery' },
    { name: 'TensorFlow', desc: 'ML detection model' },
    { name: 'React + Vite', desc: 'Frontend framework' },
  ]

  return (
    <section id="about" className="section">
      <div className="container">
        {/* Section Header */}
        <div className="text-center" style={{ marginBottom: 'var(--space-12)' }}>
          <span className="badge" style={{ marginBottom: 'var(--space-3)' }}>How It Works</span>
          <h2>AI-Powered Environmental Monitoring</h2>
          <p style={{ margin: '0 auto', marginTop: 'var(--space-4)', maxWidth: '650px' }}>
            EcoSight AI combines high-resolution satellite imagery from the 
            <a href="https://developers.google.com/earth-engine/datasets/catalog/projects_neon-prod-earthengine_assets_RGB_001" 
               target="_blank" 
               rel="noopener noreferrer"
               style={{ margin: '0 4px' }}>
              NEON Airborne Observation Platform
            </a>
            with machine learning to detect and classify waste at scale.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-4 gap-6" style={{ marginBottom: 'var(--space-16)' }}>
          {features.map((feature, index) => (
            <div 
              key={feature.title} 
              className={`card card-glow animate-fade-in delay-${index + 1}`}
            >
              <div className="analytics-card-icon" style={{ fontSize: '1.5rem' }}>
                {feature.icon}
              </div>
              <h4>{feature.title}</h4>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Tech Stack & Data Pipeline */}
        <div className="card card-glass" style={{ maxWidth: '900px', margin: '0 auto' }}>
          <div className="flex items-center justify-between flex-wrap gap-4" style={{ marginBottom: 'var(--space-6)' }}>
            <div>
              <h4>Data Pipeline</h4>
              <p className="text-muted" style={{ marginTop: 'var(--space-1)' }}>
                From satellite capture to actionable insights
              </p>
            </div>
            <span className="badge badge-success">Active</span>
          </div>

          <div className="pipeline">
            {techStack.map((tech, index) => (
              <div key={tech.name} className="pipeline-step">
                <div className="pipeline-number">{index + 1}</div>
                <div className="pipeline-content">
                  <strong>{tech.name}</strong>
                  <span className="text-muted">{tech.desc}</span>
                </div>
                {index < techStack.length - 1 && <div className="pipeline-arrow">→</div>}
              </div>
            ))}
          </div>

          <div style={{ marginTop: 'var(--space-6)', paddingTop: 'var(--space-4)', borderTop: '1px solid var(--border-subtle)' }}>
            <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>
              <strong>NEON RGB Camera Imagery</strong> provides orthorectified mosaics at 0.1-meter 
              spatial resolution, acquired by the Airborne Observation Platform across 81 ecological 
              field sites. Data is available from 2013-present under CC0 public domain license.
            </p>
          </div>
        </div>
      </div>

      <style>{`
        .pipeline {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: var(--space-4);
        }

        .pipeline-step {
          display: flex;
          align-items: center;
          gap: var(--space-3);
        }

        .pipeline-number {
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--accent-primary);
          color: var(--text-inverse);
          border-radius: 50%;
          font-weight: 600;
          font-size: var(--text-sm);
        }

        .pipeline-content {
          display: flex;
          flex-direction: column;
        }

        .pipeline-content strong {
          font-size: var(--text-sm);
          color: var(--text-primary);
        }

        .pipeline-content span {
          font-size: var(--text-xs);
        }

        .pipeline-arrow {
          color: var(--text-muted);
          font-size: var(--text-xl);
          margin-left: var(--space-4);
        }

        @media (max-width: 768px) {
          .pipeline {
            flex-direction: column;
            align-items: flex-start;
          }

          .pipeline-arrow {
            transform: rotate(90deg);
            margin: var(--space-2) 0;
            margin-left: var(--space-3);
          }
        }
      `}</style>
    </section>
  )
}
