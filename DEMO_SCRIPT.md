# OceanGuard AI - 2-Minute Demo Script

## Introduction (15 seconds)

"Good [morning/afternoon]. I'm presenting OceanGuard AI, an automated marine debris detection system built for the Presidential AI Challenge. Every year, 14 million tons of plastic enter our oceans, and traditional detection methods miss 99% of debris. Our solution uses deep learning and free Sentinel-2 satellite imagery to detect floating trash before it reaches coastlines."

---

## Technology Overview (20 seconds)

"Our system combines an Improved U-Net deep learning architecture with 11-band Sentinel-2 satellite data. The model was trained on the MARIDA dataset—real marine debris imagery—and processes images in under a second. We use spectral analysis across visible, near-infrared, and shortwave infrared bands to identify debris invisible to the naked eye."

---

## Live Demo (60 seconds)

**Step 1: Navigate to Analysis Page (5 seconds)**
"Let me show you how it works. Here's our analysis interface—clean, modern, and built with Next.js."

**Step 2: Upload or Select Sample (10 seconds)**
"You can upload your own Sentinel-2 GeoTIFF files, or use our sample images for quick testing. I'll select a sample image from the Pacific Ocean region."

**Step 3: Processing (10 seconds)**
"Once uploaded, the system preprocesses the image, runs it through our deep learning model, and generates a probability heatmap. You can see the processing steps here—image preprocessing, AI inference, heatmap generation, and hotspot identification."

**Step 4: Results Display (25 seconds)**
"Here are the results. The system detected [X] debris hotspots with an average confidence of [Y] percent. You can see the before-and-after comparison—the original satellite image on the left, and our AI detection overlay on the right. Each hotspot includes GPS coordinates, area estimates in square meters, and confidence scores. The top detections are ranked by confidence, helping prioritize cleanup operations."

**Step 5: Export Options (10 seconds)**
"Results can be exported in multiple formats—GeoTIFF heatmaps for GIS software, GeoJSON for mapping applications, CSV for data analysis, and PDF reports. This makes it easy to integrate with existing environmental monitoring workflows."

---

## Impact and Closing (25 seconds)

"This system addresses a critical gap in marine conservation. By detecting debris offshore using publicly available satellite data, we enable scalable, low-cost monitoring at a global scale. The technology is production-ready, optimized for Apple Silicon, and can process thousands of images per day. We're making ocean protection accessible, affordable, and actionable."

"Thank you. I'm happy to answer any questions."

---

## Key Talking Points (if time allows)

- **Speed**: Sub-second processing per image
- **Accuracy**: Up to 80% confidence on debris detections
- **Scale**: Can monitor entire ocean regions continuously
- **Cost**: Uses free Sentinel-2 data—no expensive sensors required
- **Accessibility**: Web-based interface, no specialized software needed

---

## Demo Flow Checklist

- [ ] Open browser to http://localhost:3000/analyze
- [ ] Click on a sample image (Pacific Ocean or Caribbean)
- [ ] Wait for processing to complete
- [ ] Point out the AI-generated report at the top
- [ ] Show the before/after slider comparison
- [ ] Highlight hotspot details (confidence, area, coordinates)
- [ ] Mention export options
- [ ] Close with impact statement

---

## Troubleshooting Notes

- If backend is not responding, check http://localhost:8000/health
- If frontend is not loading, verify port 3000 is available
- Sample images should load automatically
- Processing typically takes 1-3 seconds

---

*Total time: ~120 seconds (2 minutes)*
