"use client";

import Link from "next/link";
import Image from "next/image";
import { useState, useCallback, useEffect } from "react";

type AnalysisState = "idle" | "uploading" | "processing" | "complete" | "error";

interface Hotspot {
  id: number;
  confidence: number;
  area_m2: number;
  lat: number;
  lon: number;
}

interface AnalysisResult {
  hotspots: number;
  confidence: number;
  processingTime: number;
  previewBase64?: string;
  heatmapBase64?: string;
  hotspotsList?: Hotspot[];
}

interface SampleImage {
  id: number;
  name: string;
  region: string;
  size: string;
  previewUrl?: string;
}

export default function AnalyzePage() {
  const [state, setState] = useState<AnalysisState>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useSample, setUseSample] = useState(false);
  const [selectedSampleId, setSelectedSampleId] = useState<number | null>(null);
  
  const [samples, setSamples] = useState<SampleImage[]>([
    { id: 1, name: "Pacific Gyre Sample", region: "MARIDA Dataset", size: "256×256" },
    { id: 2, name: "Caribbean Coast", region: "MARIDA Dataset", size: "256×256" },
    { id: 3, name: "Mediterranean", region: "MARIDA Dataset", size: "256×256" },
  ]);

  // Fetch sample previews on mount
  useEffect(() => {
    const fetchSamplePreviews = async () => {
      const updatedSamples = await Promise.all(
        samples.map(async (sample) => {
          try {
            const res = await fetch(`/api/sample-preview/${sample.id}`);
            const data = await res.json();
            if (data.preview_base64) {
              return { ...sample, previewUrl: `data:image/png;base64,${data.preview_base64}` };
            }
          } catch (e) {
            console.error(`Failed to fetch preview for sample ${sample.id}:`, e);
          }
          return sample;
        })
      );
      setSamples(updatedSamples);
    };
    
    fetchSamplePreviews();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setFilePreview(null);
    setError(null);
    setState("idle");
    setResult(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && (file.name.endsWith('.tif') || file.name.endsWith('.tiff'))) {
      handleFileSelect(file);
    } else {
      setError("Please upload a GeoTIFF file (.tif or .tiff)");
    }
  }, [handleFileSelect]);

  const runAnalysis = async () => {
    if (!selectedFile) return;
    
    setState("uploading");
    
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      
      setState("processing");
      
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      
      const data = await response.json();
      
      if (data.success) {
        setResult({
          hotspots: data.hotspots_count,
          confidence: data.avg_confidence,
          processingTime: data.processing_time_ms / 1000,
          previewBase64: data.preview_base64,
          heatmapBase64: data.heatmap_base64,
          hotspotsList: data.hotspots || [],
        });
        setState("complete");
      } else {
        setError(data.error || "Analysis failed");
        setState("error");
      }
    } catch (err) {
      console.error("Analysis error:", err);
      setError("Failed to connect to server. Please try again.");
      setState("error");
    }
  };

  const useSampleImage = async (sampleId: number) => {
    setUseSample(true);
    setSelectedSampleId(sampleId);
    setSelectedFile(null);
    setState("uploading");
    
    try {
      setState("processing");
      
      // Call the backend to run actual prediction on sample
      const response = await fetch(`/api/predict-sample`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: sampleId }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setResult({
          hotspots: data.hotspots_count,
          confidence: data.avg_confidence,
          processingTime: data.processing_time_ms / 1000,
          previewBase64: data.preview_base64,
          heatmapBase64: data.heatmap_base64,
          hotspotsList: data.hotspots || [],
        });
        setState("complete");
      } else {
        setError(data.error || data.message || "Sample analysis failed");
        setState("error");
      }
    } catch (err) {
      console.error("Sample analysis error:", err);
      setError("Failed to connect to backend. Make sure the server is running.");
      setState("error");
    }
  };

  const resetAnalysis = () => {
    setState("idle");
    setSelectedFile(null);
    setFilePreview(null);
    setResult(null);
    setError(null);
    setUseSample(false);
    setSelectedSampleId(null);
  };

  const getPreviewImage = () => {
    if (result?.previewBase64) {
      return `data:image/png;base64,${result.previewBase64}`;
    }
    if (useSample && selectedSampleId) {
      const sample = samples.find(s => s.id === selectedSampleId);
      return sample?.previewUrl;
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-[var(--bg-primary)]">
      {/* Header */}
      <header className="bg-[var(--bg-secondary)] border-b border-[var(--border)]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[var(--accent-primary)] to-[var(--accent-light)] flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <span className="text-xl font-semibold text-[var(--text-primary)]">OceanGuard AI</span>
            </Link>
            <span className="text-[var(--text-muted)]">/</span>
            <span className="text-[var(--text-secondary)]">Analyze</span>
          </div>

          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
              Dashboard
            </Link>
            <Link href="/about" className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors">
              About
            </Link>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {/* Title */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-[var(--text-primary)] mb-4">
            Marine Debris Detection
          </h1>
          <p className="text-[var(--text-secondary)] max-w-2xl mx-auto">
            Upload Sentinel-2 satellite imagery to detect marine debris using our trained SegFormer model. 
            Results include probability heatmaps and georeferenced hotspot locations.
          </p>
        </div>

        {state === "idle" && !result && (
          <>
            {/* Upload Section */}
            <div className="max-w-3xl mx-auto">
              <div
                className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all ${
                  selectedFile
                    ? "border-[var(--accent-primary)] bg-[var(--accent-primary)]/5"
                    : "border-[var(--border)] hover:border-[var(--text-muted)]"
                }`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
              >
                {selectedFile ? (
                  <div>
                    <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-[var(--accent-primary)]/20 flex items-center justify-center">
                      <svg className="w-8 h-8 text-[var(--accent-primary)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <p className="text-lg font-medium text-[var(--text-primary)] mb-1">{selectedFile.name}</p>
                    <p className="text-sm text-[var(--text-secondary)] mb-6">
                      {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                    <div className="flex justify-center gap-4">
                      <button
                        onClick={runAnalysis}
                        className="px-8 py-3 bg-[var(--accent-primary)] text-white rounded-xl font-semibold hover:bg-[var(--accent-hover)] transition-colors"
                      >
                        Run Detection →
                      </button>
                      <button
                        onClick={resetAnalysis}
                        className="px-8 py-3 bg-[var(--bg-secondary)] text-[var(--text-secondary)] border border-[var(--border)] rounded-xl font-semibold hover:bg-[var(--bg-tertiary)] transition-colors"
                      >
                        Change File
                      </button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="w-16 h-16 mx-auto mb-4 rounded-xl bg-[var(--bg-secondary)] flex items-center justify-center">
                      <svg className="w-8 h-8 text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </div>
                    <p className="text-lg text-[var(--text-primary)] mb-2">
                      Drag & drop satellite imagery
                    </p>
                    <p className="text-sm text-[var(--text-secondary)] mb-6">
                      or click to browse • GeoTIFF format (.tif)
                    </p>
                    <label className="inline-block px-6 py-3 bg-[var(--bg-secondary)] text-[var(--text-primary)] border border-[var(--border)] rounded-xl font-medium cursor-pointer hover:bg-[var(--bg-tertiary)] transition-colors">
                      Select File
                      <input
                        type="file"
                        accept=".tif,.tiff"
                        className="hidden"
                        onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                      />
                    </label>
                  </div>
                )}
              </div>

              {/* Error Message */}
              {error && (
                <div className="mt-4 p-4 bg-[var(--danger)]/10 border border-[var(--danger)]/30 rounded-xl text-[var(--danger)] text-center">
                  {error}
                </div>
              )}

              {/* Divider */}
              <div className="flex items-center gap-4 my-8">
                <div className="flex-1 h-px bg-[var(--border)]"></div>
                <span className="text-[var(--text-muted)] text-sm">or try a sample</span>
                <div className="flex-1 h-px bg-[var(--border)]"></div>
              </div>

              {/* Sample Images with Previews */}
              <div className="grid grid-cols-3 gap-4">
                {samples.map((sample) => (
                  <button
                    key={sample.id}
                    onClick={() => useSampleImage(sample.id)}
                    className="p-4 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl text-left hover:border-[var(--accent-primary)] transition-colors group"
                  >
                    <div className="aspect-video bg-[var(--bg-tertiary)] rounded-lg mb-3 flex items-center justify-center overflow-hidden">
                      {sample.previewUrl ? (
                        <Image
                          src={sample.previewUrl}
                          alt={sample.name}
                          width={256}
                          height={144}
                          className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                        />
                      ) : (
                        <div className="flex flex-col items-center">
                          <span className="text-3xl opacity-50 group-hover:opacity-80 transition-opacity">🛰️</span>
                          <span className="text-xs text-[var(--text-muted)] mt-1">Loading...</span>
                        </div>
                      )}
                    </div>
                    <p className="font-medium text-[var(--text-primary)] text-sm">{sample.name}</p>
                    <p className="text-xs text-[var(--text-muted)]">{sample.region} • {sample.size}</p>
                  </button>
                ))}
              </div>
              
              {/* Backend status hint */}
              <p className="text-center text-xs text-[var(--text-muted)] mt-6">
                💡 Start the backend server for real-time image previews: <code className="bg-[var(--bg-secondary)] px-2 py-1 rounded">python -m uvicorn api:app --port 8000</code>
              </p>
            </div>
          </>
        )}

        {/* Processing State */}
        {(state === "uploading" || state === "processing") && (
          <div className="max-w-2xl mx-auto text-center py-20">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full border-4 border-[var(--bg-tertiary)] border-t-[var(--accent-primary)] animate-spin"></div>
            <h2 className="text-2xl font-semibold text-[var(--text-primary)] mb-2">
              {state === "uploading" ? "Uploading Image..." : "Running Detection..."}
            </h2>
            <p className="text-[var(--text-secondary)]">
              {state === "uploading" 
                ? "Preparing your satellite imagery for analysis" 
                : "The SegFormer model is analyzing the image for marine debris"
              }
            </p>
            
            {state === "processing" && (
              <div className="mt-8 space-y-3 text-left max-w-md mx-auto">
                {[
                  { step: "Loading model weights", done: true },
                  { step: "Preprocessing image tiles", done: true },
                  { step: "Running inference", done: false },
                  { step: "Generating heatmap", done: false },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-3 p-3 bg-[var(--bg-secondary)] rounded-lg">
                    {item.done ? (
                      <svg className="w-5 h-5 text-[var(--success)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <div className="w-5 h-5 rounded-full border-2 border-[var(--accent-primary)] border-t-transparent animate-spin"></div>
                    )}
                    <span className={item.done ? "text-[var(--text-secondary)]" : "text-[var(--text-primary)]"}>
                      {item.step}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Results */}
        {state === "complete" && result && (
          <div className="max-w-6xl mx-auto">
            {/* Success Banner */}
            <div className="mb-8 p-4 bg-[var(--success)]/10 border border-[var(--success)]/30 rounded-xl flex items-center justify-between">
              <div className="flex items-center gap-3">
                <svg className="w-6 h-6 text-[var(--success)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-[var(--text-primary)] font-medium">
                  Analysis complete in {result.processingTime.toFixed(1)}s
                </span>
              </div>
              <button
                onClick={resetAnalysis}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
              >
                Analyze Another →
              </button>
            </div>

            {/* Results Grid */}
            <div className="grid lg:grid-cols-2 gap-6 mb-8">
              {/* Original Image */}
              <div className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] overflow-hidden">
                <div className="px-4 py-3 border-b border-[var(--border)]">
                  <h3 className="font-semibold text-[var(--text-primary)]">Input Image</h3>
                  <p className="text-xs text-[var(--text-muted)]">
                    {useSample ? `Sample: ${samples.find(s => s.id === selectedSampleId)?.name}` : selectedFile?.name}
                  </p>
                </div>
                <div className="aspect-square bg-[var(--bg-tertiary)] flex items-center justify-center p-2">
                  {getPreviewImage() ? (
                    <Image
                      src={getPreviewImage()!}
                      alt="Input satellite image"
                      width={512}
                      height={512}
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : (
                    <div className="text-center">
                      <span className="text-5xl block mb-2">🛰️</span>
                      <p className="text-[var(--text-muted)] text-sm">Satellite Imagery</p>
                      <p className="text-xs text-[var(--text-muted)] mt-1">Start backend for preview</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Detection Result */}
              <div className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] overflow-hidden">
                <div className="px-4 py-3 border-b border-[var(--border)]">
                  <h3 className="font-semibold text-[var(--text-primary)]">Detection Result</h3>
                  <p className="text-xs text-[var(--text-muted)]">Debris probability heatmap overlay</p>
                </div>
                <div className="aspect-square bg-gradient-to-br from-[var(--bg-tertiary)] to-[var(--accent-primary)]/20 flex items-center justify-center relative p-2">
                  {result.heatmapBase64 ? (
                    <Image
                      src={`data:image/png;base64,${result.heatmapBase64}`}
                      alt="Detection heatmap"
                      width={512}
                      height={512}
                      className="max-w-full max-h-full object-contain"
                    />
                  ) : (
                    <>
                      {/* Mock heatmap visualization */}
                      <div className="absolute inset-4 rounded-lg border-2 border-dashed border-[var(--danger)]/50"></div>
                      <div className="absolute top-1/4 left-1/3 w-12 h-12 rounded-full bg-[var(--danger)]/30 blur-xl"></div>
                      <div className="absolute bottom-1/3 right-1/4 w-8 h-8 rounded-full bg-[var(--warning)]/30 blur-lg"></div>
                      <div className="text-center z-10">
                        <span className="text-5xl block mb-2">🎯</span>
                        <p className="text-[var(--text-muted)] text-sm">{result.hotspots} Hotspots Detected</p>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4 mb-8">
              {[
                { label: "Hotspots Found", value: result.hotspots, icon: "🎯" },
                { label: "Avg Confidence", value: `${result.confidence}%`, icon: "📊" },
                { label: "Processing Time", value: `${result.processingTime.toFixed(1)}s`, icon: "⚡" },
                { label: "Model Version", value: "v1.0", icon: "🧠" },
              ].map((stat, i) => (
                <div key={i} className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 text-center">
                  <span className="text-2xl block mb-2">{stat.icon}</span>
                  <p className="text-2xl font-bold text-[var(--text-primary)]">{stat.value}</p>
                  <p className="text-sm text-[var(--text-secondary)]">{stat.label}</p>
                </div>
              ))}
            </div>

            {/* Detected Hotspots Table */}
            {result.hotspotsList && result.hotspotsList.length > 0 && (
              <div className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] p-6 mb-8">
                <h3 className="font-semibold text-[var(--text-primary)] mb-4">Detected Hotspots</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[var(--border)]">
                        <th className="text-left py-3 px-4 text-[var(--text-secondary)]">Rank</th>
                        <th className="text-left py-3 px-4 text-[var(--text-secondary)]">Confidence</th>
                        <th className="text-left py-3 px-4 text-[var(--text-secondary)]">Area</th>
                        <th className="text-left py-3 px-4 text-[var(--text-secondary)]">Coordinates</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.hotspotsList.map((hotspot, i) => (
                        <tr key={hotspot.id} className="border-b border-[var(--border)] hover:bg-[var(--bg-primary)]">
                          <td className="py-3 px-4 font-semibold text-[var(--text-primary)]">#{i + 1}</td>
                          <td className="py-3 px-4">
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              hotspot.confidence >= 90 ? 'bg-[var(--danger)]/20 text-[var(--danger)]' :
                              hotspot.confidence >= 80 ? 'bg-[var(--warning)]/20 text-[var(--warning)]' :
                              'bg-[var(--success)]/20 text-[var(--success)]'
                            }`}>
                              {hotspot.confidence}%
                            </span>
                          </td>
                          <td className="py-3 px-4 text-[var(--text-secondary)]">{hotspot.area_m2.toLocaleString()} m²</td>
                          <td className="py-3 px-4 font-mono text-xs text-[var(--text-muted)]">
                            {hotspot.lat.toFixed(4)}°, {hotspot.lon.toFixed(4)}°
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Export Options */}
            <div className="bg-[var(--bg-secondary)] rounded-2xl border border-[var(--border)] p-6">
              <h3 className="font-semibold text-[var(--text-primary)] mb-4">Export Results</h3>
              <div className="grid grid-cols-4 gap-4">
                {[
                  { label: "Heatmap GeoTIFF", icon: "🗺️", desc: "Probability map" },
                  { label: "Hotspots GeoJSON", icon: "📍", desc: "Vector polygons" },
                  { label: "Report CSV", icon: "📋", desc: "Coordinates list" },
                  { label: "Full Report", icon: "📄", desc: "PDF summary" },
                ].map((option, i) => (
                  <button
                    key={i}
                    className="p-4 bg-[var(--bg-primary)] border border-[var(--border)] rounded-xl text-left hover:border-[var(--accent-primary)] transition-colors"
                  >
                    <span className="text-2xl block mb-2">{option.icon}</span>
                    <p className="font-medium text-[var(--text-primary)] text-sm">{option.label}</p>
                    <p className="text-xs text-[var(--text-muted)]">{option.desc}</p>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
