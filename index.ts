// Detection Types
export interface Detection {
  id: string
  location: string
  severity: 'high' | 'medium' | 'low'
  area: string
  confidence: number
  coordinates: [number, number]
  timestamp?: string
  imageUrl?: string
}

// Map Types
export interface MapMarker {
  id: string
  position: [number, number]
  severity: 'high' | 'medium' | 'low'
  label: string
}

export interface MapBounds {
  north: number
  south: number
  east: number
  west: number
}

// Analytics Types
export interface AnalyticsStat {
  label: string
  value: string | number
  icon: string
  change?: {
    value: number
    isPositive: boolean
  }
}

// API Response Types
export interface AnalysisResult {
  success: boolean
  detections: Detection[]
  metadata: {
    processingTime: number
    imageSize: string
    modelVersion: string
  }
}

export interface SatelliteImageRequest {
  latitude: number
  longitude: number
  zoom: number
  width: number
  height: number
}

export interface SatelliteImageResponse {
  imageUrl: string
  bounds: MapBounds
  captureDate: string
}


