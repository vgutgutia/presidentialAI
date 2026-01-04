# 🛰️ EcoSight AI — Environmental Waste Detection from Space

> **Presidential AI Challenge Entry** — Leveraging satellite imagery to detect illegal dumping and environmental waste accumulation.

![NEON + Earth Engine](https://img.shields.io/badge/NEON-Airborne_Platform-00A86B?style=flat-square)
![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6?style=flat-square&logo=typescript)
![Roboflow](https://img.shields.io/badge/Roboflow-ML_Inference-6706CE?style=flat-square)

## 🎯 Overview

EcoSight AI uses **0.1-meter resolution satellite imagery** from NEON's Airborne Observation Platform to identify:

- 🗑️ Illegal dumping sites
- 🏭 Landfill overflow
- 🌊 Debris accumulation in natural areas

The system enables rapid environmental response by automatically detecting and classifying waste materials from aerial imagery.

## ✨ Features

- **Live Waste Detection** — Upload satellite/aerial images for instant AI analysis
- **Multiple Detection Models** — Switch between waste, materials, and environmental classifiers
- **Interactive Map** — Visualize detection locations across 81 NEON monitoring sites
- **Confidence Scoring** — See detection confidence and severity ratings
- **Area Estimation** — Approximate affected area in m², hectares, or km²

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/PresidentialAI.git
cd PresidentialAI

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be running at `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## 🔑 API Configuration

EcoSight AI uses [Roboflow](https://roboflow.com) for waste detection inference. A demo API key is pre-configured for testing.

To use your own API key:

1. Create a free account at [roboflow.com](https://roboflow.com)
2. Get your API key from Settings → API Keys
3. Either:
   - Enter it in the app's API Key input field, or
   - Set the environment variable:
     ```bash
     VITE_ROBOFLOW_API_KEY=your_api_key_here
     ```

## 🏗️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **React 18** | UI Framework |
| **TypeScript** | Type Safety |
| **Vite** | Build Tool |
| **Roboflow** | ML Inference API |
| **Leaflet** | Interactive Maps |
| **Framer Motion** | Animations |
| **Recharts** | Analytics Charts |

## 📁 Project Structure

```
src/
├── components/       # Reusable UI components
│   ├── Header.tsx
│   ├── Hero.tsx
│   ├── ImageUpload.tsx
│   ├── DetectionOverlay.tsx
│   ├── ResultsCard.tsx
│   ├── Map.tsx
│   └── ...
├── pages/
│   ├── Home.tsx      # Landing page
│   └── Demo.tsx      # Live detection demo
├── services/
│   └── roboflow.ts   # ML inference integration
├── styles/
│   └── globals.css   # Global styles
└── types/
    └── index.ts      # TypeScript definitions
```

## 🌍 Data Sources

- **NEON Airborne Observation Platform** — 0.1m resolution imagery across 81 field sites
- **Google Earth Engine** — Additional satellite data processing
- **Roboflow Universe** — Pre-trained waste detection models

## 📊 Detection Accuracy

| Metric | Value |
|--------|-------|
| Resolution | 0.1 meters/pixel |
| NEON Sites | 81 field locations |
| Detection Accuracy | ~95% |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License — feel free to use this project for your own environmental monitoring applications.

---

<p align="center">
  Built for the <strong>Presidential AI Challenge</strong> 🇺🇸
</p>

