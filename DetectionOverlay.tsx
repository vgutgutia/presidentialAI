import { useRef, useState } from 'react'

export interface DetectionBox {
  id: string
  label: string
  severity: 'high' | 'medium' | 'low'
  confidence: number
  area: string
  // Bounding box as percentage of image dimensions (0-100)
  boundingBox: {
    x: number      // left position %
    y: number      // top position %
    width: number  // width %
    height: number // height %
  }
}

interface DetectionOverlayProps {
  imageSrc: string
  detections: DetectionBox[]
  isAnalyzing?: boolean
  onDetectionClick?: (detection: DetectionBox) => void
}

const severityColors = {
  high: { bg: 'rgba(239, 68, 68, 0.25)', border: '#ef4444', text: '#ef4444' },
  medium: { bg: 'rgba(245, 158, 11, 0.25)', border: '#f59e0b', text: '#f59e0b' },
  low: { bg: 'rgba(34, 197, 94, 0.25)', border: '#22c55e', text: '#22c55e' }
}

export default function DetectionOverlay({ 
  imageSrc, 
  detections, 
  isAnalyzing = false,
  onDetectionClick 
}: DetectionOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [imageLoaded, setImageLoaded] = useState(false)

  const handleBoxClick = (detection: DetectionBox) => {
    setSelectedId(detection.id === selectedId ? null : detection.id)
    onDetectionClick?.(detection)
  }

  return (
    <div className="detection-overlay-container" ref={containerRef}>
      <div className="image-wrapper">
        <img 
          src={imageSrc} 
          alt="Satellite imagery for analysis"
          className="detection-image"
          onLoad={() => setImageLoaded(true)}
          onError={() => setImageLoaded(false)}
        />

        {/* Scanning animation */}
        {isAnalyzing && (
          <div className="scan-overlay">
            <div className="scan-line" />
            <div className="scan-grid" />
            <div className="scan-text">
              <span className="scan-icon">🛰️</span>
              <span>Analyzing satellite imagery...</span>
            </div>
          </div>
        )}

        {/* Detection bounding boxes - using percentage positioning */}
        {!isAnalyzing && imageLoaded && detections.map((detection) => {
          const colors = severityColors[detection.severity]
          const isSelected = detection.id === selectedId

          return (
            <div
              key={detection.id}
              className={`detection-box ${isSelected ? 'selected' : ''}`}
              style={{
                left: `${detection.boundingBox.x}%`,
                top: `${detection.boundingBox.y}%`,
                width: `${detection.boundingBox.width}%`,
                height: `${detection.boundingBox.height}%`,
                backgroundColor: colors.bg,
                borderColor: colors.border,
              }}
              onClick={() => handleBoxClick(detection)}
            >
              {/* Corner markers */}
              <div className="corner-marker tl" style={{ borderColor: colors.border }} />
              <div className="corner-marker tr" style={{ borderColor: colors.border }} />
              <div className="corner-marker bl" style={{ borderColor: colors.border }} />
              <div className="corner-marker br" style={{ borderColor: colors.border }} />
              
              {/* Label */}
              <div 
                className="detection-label"
                style={{ backgroundColor: colors.border }}
              >
                <span className="label-text">{detection.label}</span>
                <span className="label-confidence">{detection.confidence}%</span>
              </div>

              {/* Area badge */}
              <div 
                className="detection-area"
                style={{ backgroundColor: colors.border }}
              >
                {detection.area}
              </div>
            </div>
          )
        })}

        {/* No detections message */}
        {!isAnalyzing && imageLoaded && detections.length === 0 && (
          <div className="no-detections">
            <span>✓ No waste detected in this area</span>
          </div>
        )}
      </div>

      {/* Detection summary */}
      {!isAnalyzing && detections.length > 0 && (
        <div className="detection-summary">
          <div className="summary-stat">
            <span className="summary-value">{detections.length}</span>
            <span className="summary-label">Zones Detected</span>
          </div>
          <div className="summary-stat">
            <span className="summary-value high">{detections.filter(d => d.severity === 'high').length}</span>
            <span className="summary-label">Critical</span>
          </div>
          <div className="summary-stat">
            <span className="summary-value">{Math.round(detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length)}%</span>
            <span className="summary-label">Avg Confidence</span>
          </div>
        </div>
      )}

      <style>{`
        .detection-overlay-container {
          position: relative;
          width: 100%;
          border-radius: var(--radius-xl);
          overflow: hidden;
          background: var(--bg-secondary);
        }

        .image-wrapper {
          position: relative;
          width: 100%;
        }

        .detection-image {
          display: block;
          width: 100%;
          height: auto;
          max-height: 450px;
          object-fit: contain;
          background: var(--bg-tertiary);
        }

        .scan-overlay {
          position: absolute;
          inset: 0;
          background: rgba(10, 14, 20, 0.7);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          overflow: hidden;
        }

        .scan-line {
          position: absolute;
          left: 0;
          right: 0;
          height: 3px;
          background: linear-gradient(90deg, 
            transparent 0%, 
            var(--accent-primary) 20%, 
            var(--accent-primary) 80%, 
            transparent 100%
          );
          box-shadow: 0 0 30px var(--accent-primary), 0 0 60px var(--accent-primary);
          animation: scan 2s ease-in-out infinite;
        }

        .scan-grid {
          position: absolute;
          inset: 0;
          background-image: 
            linear-gradient(rgba(0, 212, 170, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 170, 0.1) 1px, transparent 1px);
          background-size: 30px 30px;
          animation: grid-pulse 2s ease-in-out infinite;
        }

        @keyframes grid-pulse {
          0%, 100% { opacity: 0.3; }
          50% { opacity: 0.6; }
        }

        @keyframes scan {
          0% { top: 0; opacity: 0; }
          5% { opacity: 1; }
          95% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }

        .scan-text {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          color: var(--accent-primary);
          font-weight: 500;
          font-size: var(--text-lg);
          z-index: 10;
          background: var(--bg-glass);
          padding: var(--space-3) var(--space-5);
          border-radius: var(--radius-full);
          backdrop-filter: blur(8px);
        }

        .scan-icon {
          font-size: 1.5rem;
          animation: pulse 1s ease-in-out infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.7; transform: scale(1.1); }
        }

        .detection-box {
          position: absolute;
          border: 2px solid;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .detection-box:hover {
          z-index: 10;
          filter: brightness(1.1);
        }

        .detection-box.selected {
          z-index: 20;
          box-shadow: 0 0 30px currentColor;
        }

        .corner-marker {
          position: absolute;
          width: 12px;
          height: 12px;
          border-style: solid;
          border-width: 0;
        }

        .corner-marker.tl { top: -2px; left: -2px; border-top-width: 3px; border-left-width: 3px; }
        .corner-marker.tr { top: -2px; right: -2px; border-top-width: 3px; border-right-width: 3px; }
        .corner-marker.bl { bottom: -2px; left: -2px; border-bottom-width: 3px; border-left-width: 3px; }
        .corner-marker.br { bottom: -2px; right: -2px; border-bottom-width: 3px; border-right-width: 3px; }

        .detection-label {
          position: absolute;
          top: -26px;
          left: 0;
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: 3px 8px;
          border-radius: 4px;
          font-size: 11px;
          font-weight: 600;
          color: white;
          white-space: nowrap;
          box-shadow: var(--shadow-md);
        }

        .label-text {
          max-width: 150px;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .label-confidence {
          opacity: 0.85;
          font-family: var(--font-mono);
          font-size: 10px;
        }

        .detection-area {
          position: absolute;
          bottom: -22px;
          right: 0;
          padding: 2px 6px;
          border-radius: 3px;
          font-size: 10px;
          font-family: var(--font-mono);
          color: white;
          opacity: 0.9;
        }

        .no-detections {
          position: absolute;
          bottom: var(--space-4);
          left: 50%;
          transform: translateX(-50%);
          background: var(--eco-green-dim);
          color: var(--eco-green);
          padding: var(--space-2) var(--space-4);
          border-radius: var(--radius-full);
          font-size: var(--text-sm);
          font-weight: 500;
          border: 1px solid var(--eco-green);
        }

        .detection-summary {
          display: flex;
          justify-content: space-around;
          padding: var(--space-4);
          background: var(--bg-tertiary);
          border-top: 1px solid var(--border-subtle);
        }

        .summary-stat {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
        }

        .summary-value {
          font-family: var(--font-mono);
          font-size: var(--text-xl);
          font-weight: 700;
          color: var(--accent-primary);
        }

        .summary-value.high {
          color: var(--status-danger);
        }

        .summary-label {
          font-size: var(--text-xs);
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
      `}</style>
    </div>
  )
}
