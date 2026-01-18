import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  
  try {
    const response = await fetch(`${BACKEND_URL}/sample-preview/${id}`);
    
    if (!response.ok) {
      return NextResponse.json({ preview_base64: "", error: "Backend error" });
    }
    
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Sample preview error:", error);
    return NextResponse.json({ preview_base64: "", error: "Backend not available" });
  }
}

