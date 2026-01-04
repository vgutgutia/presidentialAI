import { useState, useCallback } from 'react'
import Map from '../components/Map'
import ImageUpload from '../components/ImageUpload'
import AnalyticsPanel from '../components/AnalyticsPanel'
import SampleImages, { sampleImages } from '../components/SampleImages'
import ApiKeyInput from '../components/ApiKeyInput'
import DetectionOverlay, { type DetectionBox } from '../components/DetectionOverlay'
import { detectWaste, getApiKey } from '../services/roboflow'

type AnalyticsStat = {
  label: string
  value: string | number
  icon: string
  change?: { value: number; isPositive: boolean }
}

export default function Demo() {
  const [hasApiKey, setHasApiKey] = useState(!!getApiKey())
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [hasResults, setHasResults] = useState(false)
  const [detectionBoxes, setDetectionBoxes] = useState<DetectionBox[]>([])
  const [stats, setStats] = useState<AnalyticsStat[]>([])
  const [selectedSampleId, setSelectedSampleId] = useState<string | undefined>()
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [mapCenter, setMapCenter] = useState<[number, number]>([40.5795, -74.1867])
  const [error, setError] = useState<string | null>(null)
  const [processingTime, setProcessingTime] = useState<number | null>(null)
  const [analysisMode, setAnalysisMode] = useState<'sample' | 'upload' | null>(null)
  const [locationName, setLocationName] = useState<string>('')

  const handleUpload = async (file: File) => {
    setError(null)
    setIsAnalyzing(true)
    setSelectedSampleId(undefined)
    setDetectionBoxes([])
    setAnalysisMode('upload')
    setLocationName('Uploaded Image')
    
    // Create preview URL for uploaded image
    const imageUrl = URL.createObjectURL(file)
    setSelectedImage(imageUrl)
    
    const startTime = Date.now()

    // Try real AI detection
    const { detections: results, error: apiError } = await detectWaste(file)
    
    setProcessingTime(Date.now() - startTime)

    if (apiError) {
      setError(apiError)
      // Still show the image, just with no detections
      setDetectionBoxes([])
      setStats([
        { label: 'Processing Time', value: `${((Date.now() - startTime) / 1000).toFixed(1)}s`, icon: '⏱️' },
        { label: 'Detections', value: 0, icon: '📍' },
        { label: 'Status', value: 'No waste found', icon: '✓' },
        { label: 'Model', value: 'Roboflow AI', icon: '🤖' }
      ])
    } else if (results.length === 0) {
      setDetectionBoxes([])
      setStats([
        { label: 'Processing Time', value: `${((Date.now() - startTime) / 1000).toFixed(1)}s`, icon: '⏱️' },
        { label: 'Detections', value: 0, icon: '📍' },
        { label: 'Status', value: 'Clean area', icon: '✓' },
        { label: 'Model', value: 'Roboflow AI', icon: '🤖' }
      ])
    } else {
      // Convert AI results to detection boxes (as percentages)
      const boxes: DetectionBox[] = results.map((r) => ({
        id: r.id,
        label: r.label,
        severity: r.severity,
        confidence: r.confidence,
        area: r.area,
        boundingBox: {
          // Convert pixel coordinates to percentages (assuming ~1000px image)
          x: (r.boundingBox.x / 10),
          y: (r.boundingBox.y / 10),
          width: (r.boundingBox.width / 10),
          height: (r.boundingBox.height / 10)
        }
      }))
      
      setDetectionBoxes(boxes)
      
      const avgConf = Math.round(results.reduce((sum, d) => sum + d.confidence, 0) / results.length)
      const highPriority = results.filter(d => d.severity === 'high').length
      
      setStats([
        { label: 'Processing Time', value: `${((Date.now() - startTime) / 1000).toFixed(1)}s`, icon: '⏱️' },
        { label: 'Detections', value: results.length, icon: '📍' },
        { label: 'Avg. Confidence', value: `${avgConf}%`, icon: '🎯' },
        { label: 'Critical Zones', value: highPriority, icon: '⚠️',
          change: highPriority > 0 ? { value: highPriority * 15, isPositive: false } : undefined }
      ])
    }
    
    setHasResults(true)
    setIsAnalyzing(false)
  }

  const handleSampleSelect = useCallback(async (sample: typeof sampleImages[0]) => {
    setError(null)
    setIsAnalyzing(true)
    setSelectedSampleId(sample.id)
    setSelectedImage(sample.fullImage)
    setDetectionBoxes([])
    setMapCenter(sample.coordinates)
    setAnalysisMode('sample')
    setLocationName(sample.location)

    const startTime = Date.now()

    // Simulate processing time for visual effect
    await new Promise(resolve => setTimeout(resolve, 1800))
    
    setProcessingTime(Date.now() - startTime)

    // Use the pre-analyzed detection data for this sample
    setDetectionBoxes(sample.detections)
    
    setStats([
      { label: 'Area Analyzed', value: sample.stats.areaScanned, icon: '🛰️' },
      { label: 'Waste Zones', value: sample.stats.detectionCount, icon: '📍' },
      { label: 'Avg. Confidence', value: sample.stats.avgConfidence, icon: '🎯' },
      { label: 'Critical Zones', value: sample.stats.highPriority, icon: '⚠️',
        change: sample.stats.highPriority > 0 ? { value: sample.stats.highPriority * 12, isPositive: false } : undefined
      }
    ])
    
    setHasResults(true)
    setIsAnalyzing(false)
  }, [])

  const emptyStats: AnalyticsStat[] = [
    { label: 'Area Analyzed', value: '—', icon: '🛰️' },
    { label: 'Waste Zones', value: '—', icon: '📍' },
    { label: 'Avg. Confidence', value: '—', icon: '🎯' },
    { label: 'Critical Zones', value: '—', icon: '⚠️' }
  ]

  return (
    <div className="section">
      <div className="container">
        {/* Header */}
        <div className="text-center" style={{ marginBottom: 'var(--space-8)' }}>
          <span className="badge" style={{ marginBottom: 'var(--space-3)' }}>Live Demo</span>
          <h2>Satellite Waste Detection</h2>
          <p style={{ margin: '0 auto', marginTop: 'var(--space-4)', maxWidth: '650px' }}>
            View AI detection results on real satellite imagery of known waste sites, 
            or upload your own aerial/satellite images for analysis.
          </p>
        </div>

        {/* API Key Section */}
        <div style={{ marginBottom: 'var(--space-6)', maxWidth: '500px', margin: '0 auto var(--space-6)' }}>
          <ApiKeyInput onKeySet={setHasApiKey} />
        </div>

        {/* Sample Images Section */}
        <div style={{ marginBottom: 'var(--space-8)' }}>
          <SampleImages 
            onSelect={handleSampleSelect}
            selectedId={selectedSampleId}
            isLoading={isAnalyzing}
          />
        </div>

        {/* Error Message */}
        {error && (
          <div className="error-banner" style={{ marginBottom: 'var(--space-6)' }}>
            <span>⚠️ {error}</span>
          </div>
        )}

        {/* Main Demo Grid */}
        <div className="demo-grid">
          {/* Left Column - Image Display */}
          <div className="flex flex-col gap-6">
            {/* Detection Results */}
            {selectedImage ? (
              <div className="results-panel">
                <div className="results-header">
                  <div>
                    <h4>🔬 Detection Analysis</h4>
                    {locationName && (
                      <span className="text-muted" style={{ fontSize: 'var(--text-sm)' }}>
                        {locationName}
                      </span>
                    )}
                  </div>
                  {isAnalyzing && <span className="badge">Processing...</span>}
                  {!isAnalyzing && hasResults && (
                    <span className="badge badge-success">
                      {processingTime ? `${(processingTime / 1000).toFixed(1)}s` : 'Complete'}
                    </span>
                  )}
                </div>
                <DetectionOverlay
                  imageSrc={selectedImage}
                  detections={detectionBoxes}
                  isAnalyzing={isAnalyzing}
                />
              </div>
            ) : (
              <div className="placeholder-box">
                <span style={{ fontSize: '4rem', marginBottom: 'var(--space-4)' }}>🛰️</span>
                <h4>Select an Image to Analyze</h4>
                <p className="text-muted">
                  Choose a sample satellite image above or upload your own
                </p>
              </div>
            )}

            {/* Upload Section */}
            <ImageUpload onUpload={handleUpload} isLoading={isAnalyzing} />
          </div>

          {/* Right Column - Results & Map */}
          <div className="flex flex-col gap-6">
            {/* Analytics */}
            <AnalyticsPanel 
              stats={hasResults ? stats : emptyStats} 
              title="Analysis Results" 
            />

            {/* Location Map */}
            {hasResults && analysisMode === 'sample' && (
              <div className="results-panel">
                <div className="results-header">
                  <h4>📍 Location</h4>
                  <span className="badge badge-outline">{locationName}</span>
                </div>
                <Map 
                  center={mapCenter}
                  zoom={12}
                  markers={[{
                    id: 'location',
                    position: mapCenter,
                    severity: 'high',
                    label: locationName
                  }]}
                />
              </div>
            )}

            {/* Detection List */}
            {hasResults && detectionBoxes.length > 0 && (
              <div className="results-panel">
                <div className="results-header">
                  <h4>Detected Zones</h4>
                  <span className="badge">{detectionBoxes.length} areas</span>
                </div>
                <div className="results-body detection-list">
                  {detectionBoxes.map((detection) => (
                    <div 
                      key={detection.id} 
                      className={`detection-item ${detection.severity}`}
                    >
                      <div className={`severity-indicator ${detection.severity}`} />
                      <div className="detection-info">
                        <span className="detection-label">{detection.label}</span>
                        <span className="detection-meta">
                          {detection.area} • {detection.confidence}% confidence
                        </span>
                      </div>
                      <span className={`badge badge-${
                        detection.severity === 'high' ? 'danger' : 
                        detection.severity === 'medium' ? 'warning' : 'success'
                      }`}>
                        {detection.severity}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Info Card */}
            <div className="card card-glass">
              <h5 style={{ marginBottom: 'var(--space-2)' }}>About This Demo</h5>
              <p className="text-muted" style={{ fontSize: 'var(--text-sm)', marginBottom: 'var(--space-3)' }}>
                <strong>Sample images:</strong> Show pre-analyzed satellite/aerial imagery of 
                real waste sites with detection zones marked.
              </p>
              <p className="text-muted" style={{ fontSize: 'var(--text-sm)', marginBottom: 'var(--space-3)' }}>
                <strong>Uploaded images:</strong> Processed in real-time using Roboflow's 
                waste detection AI model.
              </p>
              <div className="flex gap-2 flex-wrap">
                <span className="badge badge-outline">NASA Imagery</span>
                <span className="badge badge-outline">Roboflow AI</span>
                <span className="badge badge-outline">Real-time Analysis</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .demo-grid {
          display: grid;
          grid-template-columns: 1.2fr 1fr;
          gap: var(--space-6);
        }

        @media (max-width: 1024px) {
          .demo-grid {
            grid-template-columns: 1fr;
          }
        }

        .placeholder-box {
          background: var(--bg-secondary);
          border: 2px dashed var(--border-default);
          border-radius: var(--radius-xl);
          padding: var(--space-16);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          text-align: center;
          min-height: 400px;
        }

        .placeholder-box h4 {
          margin-bottom: var(--space-2);
        }

        .error-banner {
          background: var(--status-danger-dim);
          border: 1px solid var(--status-danger);
          color: var(--status-danger);
          padding: var(--space-3) var(--space-4);
          border-radius: var(--radius-md);
          font-size: var(--text-sm);
          max-width: 600px;
          margin-left: auto;
          margin-right: auto;
        }

        .detection-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .detection-item {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-3);
          background: var(--bg-tertiary);
          border-radius: var(--radius-md);
          border-left: 3px solid;
        }

        .detection-item.high { border-left-color: var(--status-danger); }
        .detection-item.medium { border-left-color: var(--status-warning); }
        .detection-item.low { border-left-color: var(--eco-green); }

        .severity-indicator {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          flex-shrink: 0;
        }

        .severity-indicator.high { background: var(--status-danger); }
        .severity-indicator.medium { background: var(--status-warning); }
        .severity-indicator.low { background: var(--eco-green); }

        .detection-info {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .detection-label {
          font-weight: 500;
          color: var(--text-primary);
          font-size: var(--text-sm);
        }

        .detection-meta {
          font-size: var(--text-xs);
          color: var(--text-muted);
        }
      `}</style>
    </div>
  )
}
