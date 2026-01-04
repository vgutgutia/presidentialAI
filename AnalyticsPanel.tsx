interface AnalyticsStat {
  label: string
  value: string | number
  icon: string
  change?: {
    value: number
    isPositive: boolean
  }
}

interface AnalyticsPanelProps {
  stats: AnalyticsStat[]
  title?: string
}

export default function AnalyticsPanel({ stats, title = 'Analytics' }: AnalyticsPanelProps) {
  return (
    <div className="results-panel">
      <div className="results-header">
        <h4>{title}</h4>
        <span className="badge">Live</span>
      </div>
      
      <div className="results-body">
        <div className="analytics-grid">
          {stats.map((stat) => (
            <div key={stat.label} className="analytics-card">
              <div className="analytics-card-icon">
                {stat.icon}
              </div>
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
              {stat.change && (
                <span 
                  className={`badge ${stat.change.isPositive ? 'badge-success' : 'badge-danger'}`}
                  style={{ alignSelf: 'flex-start', marginTop: 'var(--space-2)' }}
                >
                  {stat.change.isPositive ? '↑' : '↓'} {Math.abs(stat.change.value)}%
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}


