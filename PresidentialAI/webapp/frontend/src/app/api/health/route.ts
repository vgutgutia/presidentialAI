import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    
    if (!response.ok) {
      return NextResponse.json({
        status: "degraded",
        backend: "offline",
        frontend: "online",
      });
    }
    
    const health = await response.json();
    return NextResponse.json({
      status: "healthy",
      backend: "online",
      frontend: "online",
      model_loaded: health.model_loaded,
      device: health.device,
    });
  } catch {
    return NextResponse.json({
      status: "degraded", 
      backend: "offline",
      frontend: "online",
      message: "Backend not reachable - running in demo mode",
    });
  }
}

