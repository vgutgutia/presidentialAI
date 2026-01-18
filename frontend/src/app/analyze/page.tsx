"use client";

import Link from "next/link";
import Image from "next/image";
import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

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
  message?: string;
}

interface SampleImage {
  id: number;
  name: string;
  region: string;
  size: string;
  previewUrl?: string;
}

// Animated background component
function AnimatedBackground() {
  return (
    <div className="fixed inset-0 pointer-events-none">
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl" />
      <div className="absolute inset-0 grid-lines opacity-30" />
    </div>
  );
}

// Processing animation
function ProcessingAnimation() {
  return (
    <div className="relative w-32 h-32">
      {/* Outer ring */}
      <motion.div
        className="absolute inset-0 border-2 border-cyan-500/30 rounded-full"
        animate={{ rotate: 360 }}
        transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
      />
      {/* Middle ring */}
      <motion.div
        className="absolute inset-4 border-2 border-cyan-400/50 rounded-full"
        animate={{ rotate: -360 }}
        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
      />
      {/* Inner ring */}
      <motion.div
        className="absolute inset-8 border-2 border-cyan-300/70 rounded-full"
        animate={{ rotate: 360 }}
        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
      />
      {/* Center pulse */}
      <motion.div
        className="absolute inset-12 bg-cyan-400 rounded-full"
        animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      />
    </div>
  );
}

export default function AnalyzePage() {
  const [state, setState] = useState<AnalysisState>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useSample, setUseSample] = useState(false);
  const [selectedSampleId, setSelectedSampleId] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [sensitivity, setSensitivity] = useState(0.5);
  
  const [samples, setSamples] = useState<SampleImage[]>([
    { id: 1, name: "Pacific Ocean (Debris)", region: "48MYU", size: "256×256" },
    { id: 2, name: "Caribbean (Debris)", region: "48MYU", size: "256×256" },
    { id: 3, name: "Atlantic Coast", region: "19QDA", size: "256×256" },
  ]);

  // Fetch sample previews
  useEffect(() => {
    const fetchPreviews = async () => {
      const updated = await Promise.all(
        samples.map(async (sample) => {
          try {
            const res = await fetch(`/api/sample-preview/${sample.id}`);
            const data = await res.json();
            if (data.preview_base64) {
              return { ...sample, previewUrl: `data:image/png;base64,${data.preview_base64}` };
            }
          } catch (e) {
            console.error(`Preview fetch failed for sample ${sample.id}:`, e);
          }
          return sample;
        })
      );
      setSamples(updated);
    };
    fetchPreviews();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setError(null);
    setState("idle");
    setResult(null);
    setUseSample(false);
    setSelectedSampleId(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
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
      formData.append("sensitivity", sensitivity.toString());
      
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
          message: data.message,
        });
        setState("complete");
      } else {
        setError(data.error || "Analysis failed");
        setState("error");
      }
    } catch (err) {
      console.error("Analysis error:", err);
      setError("Connection failed. Is the backend running?");
      setState("error");
    }
  };

  const useSampleImage = async (sampleId: number) => {
    setUseSample(true);
    setSelectedSampleId(sampleId);
    setSelectedFile(null);
    setState("processing");
    
    try {
      const response = await fetch("/api/predict-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: sampleId, sensitivity }),
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
          message: data.message,
        });
        setState("complete");
      } else {
        setError(data.error || data.message || "Sample analysis failed");
        setState("error");
      }
    } catch (err) {
      console.error("Sample error:", err);
      setError("Backend connection failed. Start the server.");
      setState("error");
    }
  };

  const resetAnalysis = () => {
    setState("idle");
    setSelectedFile(null);
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
      return samples.find(s => s.id === selectedSampleId)?.previewUrl;
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-black relative">
      <AnimatedBackground />
      
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 nav-blur border-b border-white/5">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <span className="text-xl font-semibold">OCEANGUARD</span>
            </Link>
            <span className="text-white/20">/</span>
            <span className="text-white/60">ANALYZE</span>
          </div>

          <div className="flex items-center gap-6">
            <Link href="/dashboard" className="text-sm text-white/60 hover:text-white transition-colors">
              DASHBOARD
            </Link>
            <Link href="/about" className="text-sm text-white/60 hover:text-white transition-colors">
              ABOUT
            </Link>
          </div>
        </div>
      </nav>

      <main className="relative z-10 pt-24 pb-12 px-6">
        <div className="max-w-6xl mx-auto">
          
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <h1 className="text-4xl md:text-5xl font-bold mb-4">
              DEBRIS <span className="text-cyan-400">DETECTION</span>
            </h1>
            <p className="text-white/50 text-lg max-w-2xl mx-auto">
              Upload Sentinel-2 imagery for AI-powered marine debris analysis
            </p>
          </motion.div>

          <AnimatePresence mode="wait">
            {/* Upload State */}
            {state === "idle" && !result && (
              <motion.div
                key="upload"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="max-w-4xl mx-auto"
              >
                {/* Drop Zone */}
                <motion.div
                  className={`relative border-2 border-dashed rounded-2xl p-16 text-center transition-all duration-300 ${
                    isDragging 
                      ? "border-cyan-400 bg-cyan-400/10" 
                      : selectedFile 
                        ? "border-cyan-400/50 bg-cyan-400/5" 
                        : "border-white/10 hover:border-white/30"
                  }`}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  whileHover={{ scale: 1.01 }}
                  transition={{ type: "spring", stiffness: 300 }}
                >
                  {selectedFile ? (
                    <div>
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-cyan-400/20 to-blue-600/20 flex items-center justify-center"
                      >
                        <svg className="w-10 h-10 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </motion.div>
                      <p className="text-xl font-semibold text-white mb-1">{selectedFile.name}</p>
                      <p className="text-white/40 mb-8 font-mono">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                      <div className="flex justify-center gap-4">
                        <button onClick={runAnalysis} className="btn-primary">
                          ANALYZE IMAGE →
                        </button>
                        <button onClick={resetAnalysis} className="btn-secondary">
                          CHANGE FILE
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-white/5 flex items-center justify-center">
                        <svg className="w-10 h-10 text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                      </div>
                      <p className="text-xl text-white mb-2">
                        Drag & drop satellite imagery
                      </p>
                      <p className="text-white/40 mb-6">
                        GeoTIFF format (.tif) • Sentinel-2 recommended
                      </p>
                      <label className="btn-secondary cursor-pointer inline-block">
                        SELECT FILE
                        <input
                          type="file"
                          accept=".tif,.tiff"
                          className="hidden"
                          onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                        />
                      </label>
                    </div>
                  )}
                </motion.div>

                {/* Error Message */}
                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-center"
                    >
                      {error}
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Sensitivity Slider */}
                <div className="mt-8 p-6 glass rounded-2xl">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <h3 className="font-semibold text-white">Detection Sensitivity</h3>
                      <p className="text-sm text-white/40">Adjust threshold for hotspot detection</p>
                    </div>
                    <span className="text-2xl font-mono text-cyan-400">{Math.round(sensitivity * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={sensitivity * 100}
                    onChange={(e) => setSensitivity(Number(e.target.value) / 100)}
                    className="w-full h-2 bg-white/10 rounded-full appearance-none cursor-pointer
                      [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-5 [&::-webkit-slider-thumb]:h-5 
                      [&::-webkit-slider-thumb]:bg-cyan-400 [&::-webkit-slider-thumb]:rounded-full 
                      [&::-webkit-slider-thumb]:shadow-[0_0_20px_rgba(6,182,212,0.5)]
                      [&::-webkit-slider-thumb]:cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-white/30 mt-2">
                    <span>Fewer detections</span>
                    <span>More detections</span>
                  </div>
                </div>

                {/* Divider */}
                <div className="flex items-center gap-4 my-12">
                  <div className="flex-1 h-px bg-white/10" />
                  <span className="text-white/30 text-sm tracking-widest">OR TRY SAMPLES</span>
                  <div className="flex-1 h-px bg-white/10" />
                </div>

                {/* Sample Images */}
                <div className="grid md:grid-cols-3 gap-6">
                  {samples.map((sample, i) => (
                    <button
                      key={sample.id}
                      onClick={() => {
                        console.log("Sample clicked:", sample.id);
                        useSampleImage(sample.id);
                      }}
                      className="group relative glass rounded-2xl overflow-hidden text-left transition-all duration-500 hover:bg-white/5"
                    >
                      <div className="aspect-square bg-gradient-to-br from-cyan-900/20 to-blue-900/20 flex items-center justify-center overflow-hidden">
                        {sample.previewUrl ? (
                          <Image
                            src={sample.previewUrl}
                            alt={sample.name}
                            width={256}
                            height={256}
                            className="w-full h-full object-contain group-hover:scale-110 transition-transform duration-700"
                          />
                        ) : (
                          <div className="text-center">
                            <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-white/5 flex items-center justify-center">
                              <svg className="w-6 h-6 text-white/40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                              </svg>
                            </div>
                            <span className="text-xs text-white/30">Loading...</span>
                          </div>
                        )}
                      </div>
                      <div className="p-4 border-t border-white/5">
                        <p className="font-semibold text-white group-hover:text-cyan-400 transition-colors">{sample.name}</p>
                        <p className="text-sm text-white/40 font-mono">{sample.region} • {sample.size}</p>
                      </div>
                      {/* Hover glow */}
                      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none">
                        <div className="absolute inset-0 bg-gradient-to-t from-cyan-500/10 to-transparent" />
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Processing State */}
            {(state === "uploading" || state === "processing") && (
              <motion.div
                key="processing"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="max-w-2xl mx-auto text-center py-20"
              >
                <div className="flex justify-center mb-8">
                  <ProcessingAnimation />
                </div>
                <h2 className="text-3xl font-bold text-white mb-4">
                  {state === "uploading" ? "UPLOADING..." : "ANALYZING..."}
                </h2>
                <p className="text-white/50 mb-8">
                  {state === "uploading" 
                    ? "Preparing satellite imagery for analysis" 
                    : "SegFormer AI scanning for marine debris"
                  }
                </p>
                
                {/* Progress steps */}
                <div className="space-y-3 text-left max-w-sm mx-auto">
                  {[
                    { label: "Image preprocessing", complete: true },
                    { label: "Running AI inference", complete: state === "processing" },
                    { label: "Generating heatmap", complete: false },
                    { label: "Identifying hotspots", complete: false },
                  ].map((step, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                      className="flex items-center gap-3 p-3 glass rounded-lg"
                    >
                      {step.complete ? (
                        <svg className="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <div className="w-5 h-5 rounded-full border-2 border-white/20 border-t-cyan-400 animate-spin" />
                      )}
                      <span className={step.complete ? "text-white/50" : "text-white"}>{step.label}</span>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Results State */}
            {state === "complete" && result && (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="max-w-6xl mx-auto"
              >
                {/* Demo mode warning */}
                {result.message?.includes("DEMO") && (
                  <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-6 p-4 bg-amber-500/10 border border-amber-500/30 rounded-xl text-amber-400 text-center"
                  >
                    ⚠️ {result.message}
                  </motion.div>
                )}

                {/* Success Banner */}
                <motion.div
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  className="mb-8 p-4 glass rounded-xl flex items-center justify-between"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-green-500/20 flex items-center justify-center">
                      <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <span className="text-white font-medium">
                      Analysis complete in {result.processingTime.toFixed(2)}s
                    </span>
                  </div>
                  <button onClick={resetAnalysis} className="text-cyan-400 hover:text-cyan-300 transition-colors">
                    ANALYZE ANOTHER →
                  </button>
                </motion.div>

                {/* Image Grid */}
                <div className="grid lg:grid-cols-2 gap-6 mb-8">
                  {/* Input Image */}
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                    className="glass rounded-2xl overflow-hidden"
                  >
                    <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
                      <div>
                        <h3 className="font-semibold text-white">INPUT IMAGE</h3>
                        <p className="text-xs text-white/40 font-mono">
                          {useSample ? samples.find(s => s.id === selectedSampleId)?.name : selectedFile?.name}
                        </p>
                      </div>
                      <span className="px-2 py-1 bg-white/5 rounded text-xs text-white/60">RAW</span>
                    </div>
                    <div className="aspect-square bg-black/50 flex items-center justify-center p-4">
                      {getPreviewImage() ? (
                        <Image
                          src={getPreviewImage()!}
                          alt="Input"
                          width={512}
                          height={512}
                          className="max-w-full max-h-full object-contain rounded-lg"
                        />
                      ) : (
                        <span className="text-white/30">No preview</span>
                      )}
                    </div>
                  </motion.div>

                  {/* Detection Result */}
                  <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass rounded-2xl overflow-hidden"
                  >
                    <div className="px-4 py-3 border-b border-white/5 flex items-center justify-between">
                      <div>
                        <h3 className="font-semibold text-white">DETECTION RESULT</h3>
                        <p className="text-xs text-white/40">Debris probability heatmap</p>
                      </div>
                      <span className="px-2 py-1 bg-cyan-500/20 rounded text-xs text-cyan-400">PROCESSED</span>
                    </div>
                    <div className="aspect-square bg-black/50 flex items-center justify-center p-4 relative">
                      {result.heatmapBase64 ? (
                        <Image
                          src={`data:image/png;base64,${result.heatmapBase64}`}
                          alt="Heatmap"
                          width={512}
                          height={512}
                          className="max-w-full max-h-full object-contain rounded-lg"
                        />
                      ) : (
                        <div className="text-center">
                          <div className="text-6xl mb-4">🎯</div>
                          <p className="text-white/50">{result.hotspots} Hotspots Detected</p>
                        </div>
                      )}
                    </div>
                  </motion.div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  {[
                    { label: "HOTSPOTS", value: result.hotspots, color: "cyan" },
                    { label: "CONFIDENCE", value: `${result.confidence}%`, color: "green" },
                    { label: "PROCESS TIME", value: `${result.processingTime.toFixed(2)}s`, color: "blue" },
                    { label: "MODEL", value: "SegFormer", color: "indigo" },
                  ].map((stat, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 + i * 0.1 }}
                      className="stat-card text-center"
                    >
                      <div className={`text-3xl font-bold font-mono text-${stat.color}-400 mb-1`}>
                        {stat.value}
                      </div>
                      <div className="text-xs text-white/40 tracking-widest">{stat.label}</div>
                    </motion.div>
                  ))}
                </div>

                {/* Hotspots Table */}
                {result.hotspotsList && result.hotspotsList.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                    className="glass rounded-2xl overflow-hidden"
                  >
                    <div className="px-6 py-4 border-b border-white/5">
                      <h3 className="font-semibold text-white">DETECTED HOTSPOTS</h3>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b border-white/5 text-xs text-white/40 tracking-widest">
                            <th className="text-left py-3 px-6">RANK</th>
                            <th className="text-left py-3 px-6">CONFIDENCE</th>
                            <th className="text-left py-3 px-6">AREA</th>
                            <th className="text-left py-3 px-6">COORDINATES</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.hotspotsList.map((hotspot, i) => (
                            <motion.tr
                              key={hotspot.id}
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: 0.6 + i * 0.05 }}
                              className="border-b border-white/5 hover:bg-white/5 transition-colors"
                            >
                              <td className="py-4 px-6 font-mono text-white">#{i + 1}</td>
                              <td className="py-4 px-6">
                                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                                  hotspot.confidence >= 90 
                                    ? "bg-red-500/20 text-red-400" 
                                    : hotspot.confidence >= 80 
                                      ? "bg-amber-500/20 text-amber-400" 
                                      : "bg-green-500/20 text-green-400"
                                }`}>
                                  {hotspot.confidence}%
                                </span>
                              </td>
                              <td className="py-4 px-6 text-white/60 font-mono">
                                {hotspot.area_m2.toLocaleString()} m²
                              </td>
                              <td className="py-4 px-6 text-white/40 font-mono text-sm">
                                {hotspot.lat.toFixed(4)}°N, {hotspot.lon.toFixed(4)}°W
                              </td>
                            </motion.tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </motion.div>
                )}

                {/* Export Options */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.7 }}
                  className="mt-8 glass rounded-2xl p-6"
                >
                  <h3 className="font-semibold text-white mb-4">EXPORT RESULTS</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                      { label: "GeoTIFF", icon: "🗺️", desc: "Heatmap" },
                      { label: "GeoJSON", icon: "📍", desc: "Vectors" },
                      { label: "CSV", icon: "📋", desc: "Data" },
                      { label: "PDF", icon: "📄", desc: "Report" },
                    ].map((opt, i) => (
                      <button
                        key={i}
                        className="p-4 bg-white/5 rounded-xl text-left hover:bg-white/10 transition-colors group"
                      >
                        <span className="text-2xl block mb-2">{opt.icon}</span>
                        <p className="font-medium text-white group-hover:text-cyan-400 transition-colors">{opt.label}</p>
                        <p className="text-xs text-white/40">{opt.desc}</p>
                      </button>
                    ))}
                  </div>
                </motion.div>
              </motion.div>
            )}

            {/* Error State */}
            {state === "error" && (
              <motion.div
                key="error"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="max-w-md mx-auto text-center py-20"
              >
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-red-500/20 flex items-center justify-center">
                  <svg className="w-10 h-10 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-white mb-4">ANALYSIS FAILED</h2>
                <p className="text-white/50 mb-8">{error}</p>
                <button onClick={resetAnalysis} className="btn-primary">
                  TRY AGAIN
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
      
      {/* Version indicator */}
      <div className="fixed bottom-4 right-4 text-xs text-white/30 font-mono z-50">
        v2.3.0-balanced
      </div>
    </div>
  );
}
