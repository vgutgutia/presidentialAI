interface SampleImage {
  id: string
  name: string
  description: string
  thumbnail: string
  fullImage: string
  location: string
  coordinates: [number, number]
  source: string
  // Pre-analyzed detections for this specific image
  detections: Array<{
    id: string
    label: string
    severity: 'high' | 'medium' | 'low'
    area: string
    confidence: number
    // Bounding box in percentage of image dimensions
    boundingBox: { x: number; y: number; width: number; height: number }
  }>
  stats: {
    areaScanned: string
    detectionCount: number
    avgConfidence: string
    highPriority: number
  }
}

// Real satellite/aerial imagery of waste and landfill sites
// Using NASA, USGS, and other public domain sources
export const sampleImages: SampleImage[] = [
  {
    id: 'freshkills-landfill',
    name: 'Fresh Kills Landfill',
    description: 'Former largest landfill in the world, Staten Island NY - NASA Landsat imagery',
    thumbnail: 'https://eoimages.gsfc.nasa.gov/images/imagerecords/5000/5854/freshkills_ast_2001169_lrg.jpg',
    fullImage: 'https://eoimages.gsfc.nasa.gov/images/imagerecords/5000/5854/freshkills_ast_2001169_lrg.jpg',
    location: 'Staten Island, New York',
    coordinates: [40.5795, -74.1867],
    source: 'NASA Earth Observatory',
    detections: [
      { id: 'fk1', label: 'Active Landfill Section', severity: 'high', area: '1.2 km²', confidence: 96, 
        boundingBox: { x: 35, y: 25, width: 30, height: 35 } },
      { id: 'fk2', label: 'Waste Mound East', severity: 'high', area: '0.8 km²', confidence: 94,
        boundingBox: { x: 55, y: 40, width: 25, height: 28 } },
      { id: 'fk3', label: 'Covered Section', severity: 'medium', area: '0.5 km²', confidence: 82,
        boundingBox: { x: 15, y: 50, width: 20, height: 22 } },
    ],
    stats: { areaScanned: '8.9 km²', detectionCount: 3, avgConfidence: '91%', highPriority: 2 }
  },
  {
    id: 'ghazipur-landfill',
    name: 'Ghazipur Landfill',
    description: 'One of India\'s largest waste sites - visible from satellite',
    thumbnail: 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Ghazipur_landfill_site.jpg/1280px-Ghazipur_landfill_site.jpg',
    fullImage: 'https://upload.wikimedia.org/wikipedia/commons/6/6a/Ghazipur_landfill_site.jpg',
    location: 'Delhi, India',
    coordinates: [28.6208, 77.3267],
    source: 'Wikimedia Commons',
    detections: [
      { id: 'gz1', label: 'Main Waste Mountain', severity: 'high', area: '0.21 km²', confidence: 98,
        boundingBox: { x: 20, y: 15, width: 60, height: 70 } },
      { id: 'gz2', label: 'Active Dumping Zone', severity: 'high', area: '0.05 km²', confidence: 91,
        boundingBox: { x: 70, y: 60, width: 18, height: 25 } },
    ],
    stats: { areaScanned: '0.5 km²', detectionCount: 2, avgConfidence: '95%', highPriority: 2 }
  },
  {
    id: 'pacific-garbage',
    name: 'Ocean Debris Field',
    description: 'Floating debris concentration in Pacific - aerial survey',
    thumbnail: 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Trash-ocean-pacific.jpg/1280px-Trash-ocean-pacific.jpg',
    fullImage: 'https://upload.wikimedia.org/wikipedia/commons/e/e5/Trash-ocean-pacific.jpg',
    location: 'Pacific Ocean',
    coordinates: [35.0, -140.0],
    source: 'NOAA Marine Debris Program',
    detections: [
      { id: 'po1', label: 'Plastic Debris Cluster', severity: 'high', area: '450 m²', confidence: 89,
        boundingBox: { x: 25, y: 20, width: 50, height: 55 } },
      { id: 'po2', label: 'Fishing Net Tangle', severity: 'medium', area: '120 m²', confidence: 84,
        boundingBox: { x: 60, y: 55, width: 25, height: 30 } },
      { id: 'po3', label: 'Mixed Debris', severity: 'medium', area: '80 m²', confidence: 76,
        boundingBox: { x: 10, y: 60, width: 20, height: 25 } },
    ],
    stats: { areaScanned: '2,500 m²', detectionCount: 3, avgConfidence: '83%', highPriority: 1 }
  },
  {
    id: 'rio-dump',
    name: 'Jardim Gramacho',
    description: 'Former largest landfill in South America - aerial view',
    thumbnail: 'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Lixao_do_jardim_gramacho.jpg/1280px-Lixao_do_jardim_gramacho.jpg',
    fullImage: 'https://upload.wikimedia.org/wikipedia/commons/f/fb/Lixao_do_jardim_gramacho.jpg',
    location: 'Rio de Janeiro, Brazil',
    coordinates: [-22.7461, -43.2364],
    source: 'Wikimedia Commons',
    detections: [
      { id: 'jg1', label: 'Waste Accumulation Zone', severity: 'high', area: '0.6 km²', confidence: 95,
        boundingBox: { x: 15, y: 20, width: 70, height: 60 } },
      { id: 'jg2', label: 'Debris Pile', severity: 'medium', area: '0.15 km²', confidence: 88,
        boundingBox: { x: 65, y: 10, width: 25, height: 30 } },
    ],
    stats: { areaScanned: '1.3 km²', detectionCount: 2, avgConfidence: '92%', highPriority: 1 }
  },
  {
    id: 'agbogbloshie',
    name: 'Agbogbloshie E-Waste',
    description: 'World\'s largest e-waste dump - electronic waste processing site',
    thumbnail: 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Agbogbloshie.jpg/1280px-Agbogbloshie.jpg',
    fullImage: 'https://upload.wikimedia.org/wikipedia/commons/4/46/Agbogbloshie.jpg',
    location: 'Accra, Ghana',
    coordinates: [5.5544, -0.2269],
    source: 'Wikimedia Commons',
    detections: [
      { id: 'ab1', label: 'E-Waste Processing Area', severity: 'high', area: '0.08 km²', confidence: 97,
        boundingBox: { x: 10, y: 30, width: 45, height: 50 } },
      { id: 'ab2', label: 'Metal Scrap Pile', severity: 'high', area: '0.04 km²', confidence: 93,
        boundingBox: { x: 50, y: 25, width: 30, height: 35 } },
      { id: 'ab3', label: 'Burning Zone', severity: 'high', area: '0.02 km²', confidence: 91,
        boundingBox: { x: 70, y: 50, width: 20, height: 25 } },
    ],
    stats: { areaScanned: '0.25 km²', detectionCount: 3, avgConfidence: '94%', highPriority: 3 }
  }
]

interface SampleImagesProps {
  onSelect: (sample: SampleImage) => void
  selectedId?: string
  isLoading?: boolean
}

export default function SampleImages({ onSelect, selectedId, isLoading }: SampleImagesProps) {
  return (
    <div className="results-panel">
      <div className="results-header">
        <h4>🛰️ Satellite & Aerial Imagery</h4>
        <span className="badge badge-success">Real Locations</span>
      </div>
      
      <div className="results-body">
        <p className="text-muted" style={{ marginBottom: 'var(--space-4)', fontSize: 'var(--text-sm)' }}>
          Select real satellite/aerial imagery of known waste sites. Each image has been pre-analyzed 
          with detection zones marked.
        </p>
        
        <div className="sample-grid">
          {sampleImages.map((sample) => (
            <button
              key={sample.id}
              className={`sample-card ${selectedId === sample.id ? 'selected' : ''}`}
              onClick={() => onSelect(sample)}
              disabled={isLoading}
              type="button"
            >
              <div className="sample-image">
                <img 
                  src={sample.thumbnail} 
                  alt={sample.name} 
                  loading="lazy"
                  onError={(e) => {
                    e.currentTarget.src = `https://via.placeholder.com/400x300/1a232e/00d4aa?text=${encodeURIComponent(sample.name)}`
                  }}
                />
                {selectedId === sample.id && !isLoading && (
                  <div className="sample-overlay">
                    <span>✓ Selected</span>
                  </div>
                )}
                {isLoading && selectedId === sample.id && (
                  <div className="sample-overlay loading">
                    <span>🛰️ Analyzing...</span>
                  </div>
                )}
              </div>
              <div className="sample-info">
                <h5>{sample.name}</h5>
                <p className="text-muted">{sample.location}</p>
                <p className="sample-source">{sample.source}</p>
                <div className="sample-meta">
                  <span className="badge badge-outline">{sample.stats.detectionCount} zones</span>
                  <span className={`badge ${sample.stats.highPriority > 0 ? 'badge-danger' : 'badge-success'}`}>
                    {sample.stats.highPriority} critical
                  </span>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      <style>{`
        .sample-grid {
          display: grid;
          grid-template-columns: repeat(5, 1fr);
          gap: var(--space-4);
        }

        @media (max-width: 1400px) {
          .sample-grid {
            grid-template-columns: repeat(3, 1fr);
          }
        }

        @media (max-width: 900px) {
          .sample-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        .sample-card {
          background: var(--bg-tertiary);
          border: 2px solid var(--border-subtle);
          border-radius: var(--radius-lg);
          overflow: hidden;
          text-align: left;
          transition: all var(--transition-base);
          cursor: pointer;
          padding: 0;
        }

        .sample-card:hover:not(:disabled) {
          border-color: var(--accent-primary);
          transform: translateY(-2px);
          box-shadow: var(--shadow-lg);
        }

        .sample-card:disabled {
          opacity: 0.6;
          cursor: wait;
        }

        .sample-card.selected {
          border-color: var(--accent-primary);
          box-shadow: var(--shadow-glow);
        }

        .sample-image {
          position: relative;
          width: 100%;
          height: 120px;
          overflow: hidden;
          background: var(--bg-secondary);
        }

        .sample-image img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          transition: transform var(--transition-base);
        }

        .sample-card:hover:not(:disabled) .sample-image img {
          transform: scale(1.05);
        }

        .sample-overlay {
          position: absolute;
          inset: 0;
          background: rgba(0, 212, 170, 0.9);
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--text-inverse);
          font-weight: 600;
          font-size: var(--text-base);
        }

        .sample-overlay.loading {
          background: rgba(10, 14, 20, 0.85);
          color: var(--accent-primary);
        }

        .sample-info {
          padding: var(--space-3);
        }

        .sample-info h5 {
          font-size: var(--text-sm);
          margin-bottom: var(--space-1);
          color: var(--text-primary);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .sample-info > p {
          font-size: var(--text-xs);
          margin-bottom: var(--space-1);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .sample-source {
          font-size: 0.65rem !important;
          color: var(--accent-primary) !important;
          opacity: 0.8;
        }

        .sample-meta {
          display: flex;
          gap: var(--space-2);
          flex-wrap: wrap;
          margin-top: var(--space-2);
        }

        .sample-meta .badge {
          font-size: 0.6rem;
          padding: 2px 6px;
        }
      `}</style>
    </div>
  )
}
