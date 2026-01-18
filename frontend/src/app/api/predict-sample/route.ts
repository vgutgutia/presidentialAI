import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  console.log("[predict-sample] Starting request to backend:", BACKEND_URL);
  
  try {
    const body = await request.json();
    const sampleId = body.sample_id || 1;
    const sensitivity = body.sensitivity || 0.5;

    console.log(`[predict-sample] Calling backend for sample ${sampleId}, sensitivity ${sensitivity}`);
    
    // Call the backend predict-sample endpoint with sensitivity
    const response = await fetch(
      `${BACKEND_URL}/predict-sample?sample_id=${sampleId}&sensitivity=${sensitivity}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    console.log(`[predict-sample] Backend response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error("[predict-sample] Backend error:", errorText);
      return NextResponse.json(
        { success: false, error: `Backend error: ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log(`[predict-sample] Backend returned: hotspots=${data.hotspots_count}, has_heatmap=${!!data.heatmap_base64}`);
    return NextResponse.json(data);
  } catch (error) {
    console.error("[predict-sample] FETCH ERROR:", error);
    
    // Return demo results if backend is unavailable
    return NextResponse.json({
      success: true,
      hotspots_count: Math.floor(Math.random() * 8) + 3,
      avg_confidence: Math.floor(Math.random() * 20) + 70,
      processing_time_ms: Math.floor(Math.random() * 500) + 200,
      preview_base64: null,
      heatmap_base64: null,
      hotspots: [
        { id: 1, confidence: 94, area_m2: 4500, lat: 37.7749, lon: -122.4194, rank: 1 },
        { id: 2, confidence: 87, area_m2: 3200, lat: 37.7850, lon: -122.4094, rank: 2 },
        { id: 3, confidence: 82, area_m2: 2850, lat: 37.7650, lon: -122.4294, rank: 3 },
        { id: 4, confidence: 76, area_m2: 1900, lat: 37.7550, lon: -122.4394, rank: 4 },
      ],
      message: "⚠️ Demo mode - backend connection failed",
    });
  }
}
