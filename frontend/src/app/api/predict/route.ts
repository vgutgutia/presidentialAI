import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const sensitivity = formData.get("sensitivity") || "0.5";

    if (!file) {
      return NextResponse.json(
        { error: "No file provided" },
        { status: 400 }
      );
    }

    // Forward to Python backend with sensitivity
    const backendFormData = new FormData();
    backendFormData.append("file", file);

    const response = await fetch(
      `${BACKEND_URL}/predict?sensitivity=${sensitivity}`,
      {
        method: "POST",
        body: backendFormData,
      }
    );

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
    const numHotspots = Math.floor(Math.random() * 8) + 2;
    const hotspots = Array.from({ length: numHotspots }, (_, i) => ({
      id: i + 1,
      confidence: Math.round(95 - i * 5 + Math.random() * 5),
      area_m2: Math.floor(Math.random() * 5000) + 1000,
      lat: 37.77 + (Math.random() - 0.5) * 0.02,
      lon: -122.42 + (Math.random() - 0.5) * 0.02,
      rank: i + 1,
    }));

    return NextResponse.json({
      success: true,
      hotspots_count: numHotspots,
      avg_confidence: Math.round(hotspots.reduce((a, b) => a + b.confidence, 0) / numHotspots),
      processing_time_ms: Math.floor(Math.random() * 800) + 400,
      preview_base64: null,
      heatmap_base64: null,
      hotspots,
      message: "⚠️ Demo mode - start the backend server for real detection with image previews",
    });
  }
}
