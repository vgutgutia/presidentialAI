"""
Report Generation using Claude AI
Returns structured data for beautiful visual reports
"""

import os
import json

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Try to load API key from config file first, then environment variable
try:
    from config import ANTHROPIC_API_KEY
except ImportError:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Use Sonnet for better quality (still cheap at ~3 cents per report)
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def build_report_prompt(detection_data: dict) -> str:
    """Build the prompt for Claude to generate structured report data."""
    
    hotspots = detection_data.get("hotspots", [])
    hotspots_count = detection_data.get("hotspots_count", len(hotspots))
    avg_confidence = detection_data.get("avg_confidence", 0)
    
    # Get top 3 hotspots
    top_hotspots = sorted(hotspots, key=lambda x: x.get('area_m2', 0), reverse=True)[:3]
    
    # Calculate total area
    total_area = sum(h.get('area_m2', 0) for h in hotspots)
    
    # Determine severity
    if hotspots_count >= 8 or total_area > 100000:
        severity = "CRITICAL"
    elif hotspots_count >= 5 or total_area > 50000:
        severity = "HIGH"
    elif hotspots_count >= 3 or total_area > 20000:
        severity = "MODERATE"
    else:
        severity = "LOW"
    
    # Build hotspots list for JSON
    hotspots_json = json.dumps([{"area": h.get('area_m2', 0), "confidence": h.get('confidence', 0)*100} for h in top_hotspots])
    
    prompt = f"""Analyze this marine debris detection data and return a JSON response.

DETECTION DATA:
- {hotspots_count} debris hotspots detected
- Average confidence: {avg_confidence*100:.0f}%
- Total affected area: {total_area:,.0f} m²
- Severity level: {severity}
- Top hotspots: {hotspots_json}

Return ONLY valid JSON (no markdown, no explanation) in this exact format:
{{
  "severity": "{severity}",
  "headline": "A single impactful sentence about what was found (max 12 words)",
  "total_area_km2": {total_area / 1000000:.2f},
  "primary_stat": {{
    "value": "{hotspots_count}",
    "label": "Hotspots",
    "detail": "detected zones"
  }},
  "secondary_stat": {{
    "value": "{avg_confidence*100:.0f}%",
    "label": "Confidence",
    "detail": "average accuracy"
  }},
  "tertiary_stat": {{
    "value": "{total_area:,.0f}",
    "label": "Area (m²)",
    "detail": "total affected"
  }},
  "insight": "One sentence key insight about the environmental impact",
  "action": "One specific recommended action (start with a verb)"
}}"""

    return prompt


def generate_report_sync(detection_data: dict) -> dict:
    """Generate a marine debris report using Claude."""
    if not ANTHROPIC_AVAILABLE:
        return {
            "success": False,
            "error": "Anthropic package not installed. Install with: pip install anthropic",
        }
    
    if not ANTHROPIC_API_KEY:
        return {
            "success": False,
            "error": "API key not set. Update backend/config.py and restart.",
        }
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = build_report_prompt(detection_data)
    
    try:
        message = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text.strip()
        
        # Parse JSON response
        try:
            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            report_data = json.loads(response_text)
            
            return {
                "success": True,
                "report": report_data,
            }
        except json.JSONDecodeError:
            # Fallback: return raw text
            return {
                "success": True,
                "report": {
                    "severity": "MODERATE",
                    "headline": "Marine debris detected in scan area",
                    "raw_text": response_text
                },
            }
        
    except anthropic.AuthenticationError:
        return {
            "success": False,
            "error": "Invalid API key. Check backend/config.py and restart.",
        }
    except anthropic.RateLimitError:
        return {
            "success": False,
            "error": "Rate limit exceeded. Try again in a moment.",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Report generation failed: {str(e)}",
        }


async def generate_report(detection_data: dict) -> dict:
    """Generate a marine debris report using Claude."""
    return generate_report_sync(detection_data)
