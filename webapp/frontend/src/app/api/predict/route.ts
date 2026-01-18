import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json(
        { error: "No file provided" },
        { status: 400 }
      );
    }

    // Forward to Python backend
    const backendFormData = new FormData();
    backendFormData.append("file", file);

    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: backendFormData,
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: `Backend error: ${error}` },
        { status: response.status }
      );
    }

    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error("Prediction error:", error);
    
    // Return demo results if backend is not available
    return NextResponse.json({
      success: true,
      hotspots_count: Math.floor(Math.random() * 4) + 1,
      avg_confidence: Math.floor(Math.random() * 15) + 80,
      processing_time_ms: Math.floor(Math.random() * 1000) + 1500,
      preview_base64: "", // No preview in demo mode without backend
      heatmap_base64: "",
      hotspots: [
        { id: 1, confidence: 94, area_m2: 45000, lat: 37.7749, lon: -122.4194 },
        { id: 2, confidence: 87, area_m2: 32000, lat: 37.7850, lon: -122.4094 },
        { id: 3, confidence: 82, area_m2: 28500, lat: 37.7650, lon: -122.4294 },
      ],
      message: "Demo mode - backend not available. Start the backend for image previews.",
    });
  }
}

