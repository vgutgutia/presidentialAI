interface Detection {
  id: string
  location: string
  severity: 'high' | 'medium' | 'low'
  area: string
  confidence: number
  coordinates: [number, number]
}

interface ResultsCardProps {
  detections: Detection[]
  onDetectionClick?: (detection: Detection) => void
}

export default function ResultsCard({ detections, onDetectionClick }: ResultsCardProps) {
  const getSeverityLabel = (severity: Detection['severity']) => {
    switch (severity) {
      case 'high': return 'High Priority'
      case 'medium': return 'Medium Priority'
      case 'low': return 'Low Priority'
    }
  }

  return (
    <div className="results-panel">
      <div className="results-header">
        <h4>Detected Areas</h4>
        <span className="badge">{detections.length} found</span>
      </div>
      
      <div className="results-body">
        {detections.length === 0 ? (
          <p className="text-muted text-center">
            No detections yet. Upload an image to start scanning.
          </p>
        ) : (
          detections.map((detection) => (
            <div 
              key={detection.id} 
              className="detection-item"
              onClick={() => onDetectionClick?.(detection)}
              style={{ cursor: onDetectionClick ? 'pointer' : 'default' }}
            >
              <div className={`detection-dot ${detection.severity}`} />
              <div style={{ flex: 1 }}>
                <p className="text-primary" style={{ fontWeight: 500 }}>
                  {detection.location}
                </p>
                <p className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>
                  {detection.area} • {detection.confidence}% confidence
                </p>
              </div>
              <span className={`badge badge-${detection.severity === 'high' ? 'danger' : detection.severity === 'medium' ? 'warning' : 'success'}`}>
                {getSeverityLabel(detection.severity)}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}


