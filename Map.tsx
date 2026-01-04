import { useEffect, useRef } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

interface MapProps {
  center?: [number, number]
  zoom?: number
  markers?: Array<{
    id: string
    position: [number, number]
    severity: 'high' | 'medium' | 'low'
    label: string
  }>
}

// Severity color mapping
const severityColors = {
  high: '#ef4444',
  medium: '#f59e0b', 
  low: '#22c55e'
}

export default function Map({ 
  center = [37.7749, -122.4194], 
  zoom = 10,
  markers = [] 
}: MapProps) {
  const mapRef = useRef<HTMLDivElement>(null)
  const mapInstanceRef = useRef<L.Map | null>(null)
  const markersLayerRef = useRef<L.LayerGroup | null>(null)

  // Initialize map
  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return

    // Create map instance
    const map = L.map(mapRef.current, {
      center: center,
      zoom: zoom,
      zoomControl: true,
    })

    // Add ESRI World Imagery (satellite tiles - FREE, no API key)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
      maxZoom: 19,
    }).addTo(map)

    // Add labels overlay
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 19,
    }).addTo(map)

    // Create markers layer group
    markersLayerRef.current = L.layerGroup().addTo(map)

    mapInstanceRef.current = map

    return () => {
      map.remove()
      mapInstanceRef.current = null
    }
  }, [])

  // Update center and zoom when props change
  useEffect(() => {
    if (mapInstanceRef.current) {
      mapInstanceRef.current.setView(center, zoom, { animate: true })
    }
  }, [center, zoom])

  // Update markers when they change
  useEffect(() => {
    if (!markersLayerRef.current) return

    // Clear existing markers
    markersLayerRef.current.clearLayers()

    // Add new markers
    markers.forEach((marker) => {
      // Create custom icon based on severity
      const icon = L.divIcon({
        className: 'custom-marker',
        html: `
          <div style="
            width: 24px;
            height: 24px;
            background: ${severityColors[marker.severity]};
            border: 3px solid white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            position: relative;
          ">
            <div style="
              position: absolute;
              bottom: -8px;
              left: 50%;
              transform: translateX(-50%);
              width: 0;
              height: 0;
              border-left: 6px solid transparent;
              border-right: 6px solid transparent;
              border-top: 8px solid ${severityColors[marker.severity]};
            "></div>
          </div>
        `,
        iconSize: [24, 32],
        iconAnchor: [12, 32],
        popupAnchor: [0, -32],
      })

      const leafletMarker = L.marker(marker.position, { icon })
        .bindPopup(`
          <div style="font-family: 'Outfit', sans-serif; min-width: 150px;">
            <strong style="font-size: 14px; color: #1a1a1a;">${marker.label}</strong>
            <div style="
              margin-top: 8px;
              padding: 4px 8px;
              background: ${severityColors[marker.severity]}20;
              color: ${severityColors[marker.severity]};
              border-radius: 4px;
              font-size: 12px;
              font-weight: 500;
              display: inline-block;
            ">
              ${marker.severity.toUpperCase()} PRIORITY
            </div>
          </div>
        `)
        .addTo(markersLayerRef.current!)
    })
  }, [markers])

  return (
    <div className="map-wrapper">
      <div ref={mapRef} className="map-container" />
      
      {/* Map Legend */}
      <div className="map-legend">
        <div className="legend-title">Detection Severity</div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: severityColors.high }}></span>
          <span>High Priority</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: severityColors.medium }}></span>
          <span>Medium Priority</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot" style={{ background: severityColors.low }}></span>
          <span>Low Priority</span>
        </div>
      </div>

      <style>{`
        .map-wrapper {
          position: relative;
          width: 100%;
          height: 500px;
          border-radius: var(--radius-xl);
          overflow: hidden;
          border: 1px solid var(--border-subtle);
        }

        .map-container {
          width: 100%;
          height: 100%;
          background: var(--bg-secondary);
        }

        .map-legend {
          position: absolute;
          bottom: 20px;
          right: 20px;
          background: var(--bg-glass);
          backdrop-filter: blur(8px);
          padding: var(--space-3) var(--space-4);
          border-radius: var(--radius-md);
          border: 1px solid var(--border-subtle);
          z-index: 1000;
        }

        .legend-title {
          font-size: var(--text-xs);
          font-weight: 600;
          color: var(--text-primary);
          margin-bottom: var(--space-2);
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .legend-item {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          font-size: var(--text-xs);
          color: var(--text-secondary);
          margin-bottom: var(--space-1);
        }

        .legend-item:last-child {
          margin-bottom: 0;
        }

        .legend-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          flex-shrink: 0;
        }

        /* Override Leaflet default styles for dark theme */
        .leaflet-container {
          background: var(--bg-secondary);
          font-family: var(--font-display);
        }

        .leaflet-control-zoom {
          border: none !important;
          box-shadow: var(--shadow-md) !important;
        }

        .leaflet-control-zoom a {
          background: var(--bg-elevated) !important;
          color: var(--text-primary) !important;
          border: 1px solid var(--border-subtle) !important;
          width: 32px !important;
          height: 32px !important;
          line-height: 30px !important;
        }

        .leaflet-control-zoom a:hover {
          background: var(--bg-tertiary) !important;
        }

        .leaflet-control-zoom-in {
          border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
        }

        .leaflet-control-zoom-out {
          border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
        }

        .leaflet-popup-content-wrapper {
          background: white;
          border-radius: var(--radius-md);
          box-shadow: var(--shadow-lg);
        }

        .leaflet-popup-tip {
          background: white;
        }

        .custom-marker {
          background: transparent;
          border: none;
        }

        .leaflet-control-attribution {
          background: var(--bg-glass) !important;
          backdrop-filter: blur(4px);
          color: var(--text-muted) !important;
          font-size: 10px !important;
        }

        .leaflet-control-attribution a {
          color: var(--accent-primary) !important;
        }
      `}</style>
    </div>
  )
}
