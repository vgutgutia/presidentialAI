import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const sampleId = body.sample_id || 1;

    // Call the backend predict-sample endpoint
    const response = await fetch(`${BACKEND_URL}/predict-sample?sample_id=${sampleId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Backend error:", errorText);
      return NextResponse.json(
        { success: false, error: `Backend error: ${response.status}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Predict sample error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to connect to backend server" },
      { status: 500 }
    );
  }
}

