"use client";

import Link from "next/link";
import { motion, useScroll, useTransform } from "framer-motion";
import { useRef, useEffect, useState } from "react";

// Particle component for background effect - uses fixed positions to avoid hydration mismatch
function Particles() {
  // Pre-computed positions to avoid SSR/client mismatch
  const particles = [
    { left: 5, delay: 0, duration: 16 }, { left: 12, delay: 3, duration: 18 },
    { left: 18, delay: 7, duration: 20 }, { left: 25, delay: 1, duration: 17 },
    { left: 32, delay: 9, duration: 22 }, { left: 38, delay: 4, duration: 19 },
    { left: 45, delay: 11, duration: 21 }, { left: 52, delay: 2, duration: 16 },
    { left: 58, delay: 8, duration: 23 }, { left: 65, delay: 5, duration: 18 },
    { left: 72, delay: 12, duration: 20 }, { left: 78, delay: 6, duration: 17 },
    { left: 85, delay: 10, duration: 22 }, { left: 92, delay: 3, duration: 19 },
    { left: 98, delay: 14, duration: 21 }, { left: 8, delay: 7, duration: 24 },
    { left: 22, delay: 0, duration: 16 }, { left: 35, delay: 13, duration: 18 },
    { left: 48, delay: 5, duration: 20 }, { left: 62, delay: 9, duration: 17 },
    { left: 75, delay: 2, duration: 22 }, { left: 88, delay: 11, duration: 19 },
    { left: 15, delay: 8, duration: 21 }, { left: 42, delay: 4, duration: 16 },
    { left: 68, delay: 12, duration: 23 }, { left: 95, delay: 6, duration: 18 },
  ];
  
  return (
    <div className="particles">
      {particles.map((p, i) => (
        <div
          key={i}
          className="particle"
          style={{
            left: `${p.left}%`,
            animationDelay: `${p.delay}s`,
            animationDuration: `${p.duration}s`,
          }}
        />
      ))}
    </div>
  );
}

// Animated counter component
function AnimatedCounter({ value, suffix = "" }: { value: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const increment = value / steps;
    let current = 0;
    
    const timer = setInterval(() => {
      current += increment;
      if (current >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, duration / steps);
    
    return () => clearInterval(timer);
  }, [value]);
  
  return <span>{count.toLocaleString()}{suffix}</span>;
}

// Navigation
function Navigation() {
  const [scrolled, setScrolled] = useState(false);
  
  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);
  
  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled ? "nav-blur border-b border-white/5" : ""
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="relative w-10 h-10">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg opacity-80 group-hover:opacity-100 transition-opacity" />
            <div className="absolute inset-0 flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
          <span className="text-xl font-semibold tracking-tight">OCEANGUARD</span>
        </Link>

        <div className="flex items-center gap-8">
          {["Dashboard", "Analyze", "About"].map((item) => (
            <Link
              key={item}
              href={`/${item.toLowerCase()}`}
              className="text-sm text-white/60 hover:text-white transition-colors relative group"
            >
              {item.toUpperCase()}
              <span className="absolute -bottom-1 left-0 w-0 h-px bg-cyan-400 group-hover:w-full transition-all duration-300" />
            </Link>
          ))}
          <Link
            href="/analyze"
            className="btn-primary text-sm py-3 px-6"
          >
            START DETECTION
          </Link>
        </div>
      </div>
    </motion.nav>
  );
}

export default function HomePage() {
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"]
  });
  
  const heroOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);
  const heroScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.8]);
  const heroY = useTransform(scrollYProgress, [0, 0.5], [0, -100]);

  return (
    <div className="min-h-screen bg-black">
      <Particles />
      <Navigation />
      
      {/* Hero Section */}
      <motion.section
        ref={heroRef}
        style={{ opacity: heroOpacity, scale: heroScale, y: heroY }}
        className="relative min-h-screen flex items-center justify-center hero-gradient grid-lines overflow-hidden"
      >
        {/* Animated gradient orbs */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-float" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-indigo-500/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "-3s" }} />
        
        <div className="relative z-10 text-center px-6 max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.2 }}
          >
            <span className="inline-block px-4 py-2 rounded-full glass text-cyan-400 text-sm font-medium mb-8 tracking-wider">
              PRESIDENTIAL AI CHALLENGE 2026
            </span>
          </motion.div>
          
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.4 }}
            className="text-6xl md:text-8xl font-bold tracking-tighter mb-6"
          >
            <span className="block text-white">PROTECTING</span>
            <span className="block text-glow bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-500 bg-clip-text text-transparent">
              OUR OCEANS
            </span>
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.6 }}
            className="text-xl md:text-2xl text-white/60 max-w-3xl mx-auto mb-12 leading-relaxed"
          >
            AI-powered satellite imagery analysis detecting marine debris 
            before it devastates our coastlines. Real-time monitoring. 
            Precision detection. Global scale.
          </motion.p>
          
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="flex flex-wrap justify-center gap-4"
          >
            <Link href="/analyze" className="btn-primary text-lg">
              LAUNCH DETECTION →
            </Link>
            <Link href="/about" className="btn-secondary text-lg">
              LEARN MORE
            </Link>
          </motion.div>
        </div>
        
        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-10 left-1/2 -translate-x-1/2"
        >
          <div className="flex flex-col items-center gap-2 text-white/40">
            <span className="text-xs tracking-widest">SCROLL</span>
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ repeat: Infinity, duration: 1.5 }}
              className="w-px h-8 bg-gradient-to-b from-white/40 to-transparent"
            />
          </div>
        </motion.div>
      </motion.section>

      {/* Stats Section */}
      <section className="relative py-32 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8 }}
            className="text-center mb-20"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-4">
              THE SCALE OF THE <span className="text-cyan-400">CRISIS</span>
            </h2>
            <p className="text-white/50 text-lg max-w-2xl mx-auto">
              Every year, millions of tons of plastic enter our oceans. 
              Traditional detection methods can&apos;t keep pace.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-4 gap-6">
            {[
              { value: 14, suffix: "M", label: "TONS OF PLASTIC", sub: "Enter oceans yearly" },
              { value: 500, suffix: "+", label: "DEAD ZONES", sub: "Globally identified" },
              { value: 80, suffix: "%", label: "DEBRIS SOURCE", sub: "From land-based activity" },
              { value: 99, suffix: "%", label: "UNDETECTED", sub: "Traditional methods miss" },
            ].map((stat, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: i * 0.1 }}
                className="stat-card text-center group"
              >
                <div className="text-5xl md:text-6xl font-bold text-cyan-400 mb-2 font-mono">
                  <AnimatedCounter value={stat.value} suffix={stat.suffix} />
                </div>
                <div className="text-white font-semibold mb-1">{stat.label}</div>
                <div className="text-white/40 text-sm">{stat.sub}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Section */}
      <section className="relative py-32 px-6 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/5 to-transparent" />
        
        <div className="max-w-7xl mx-auto relative">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <span className="text-cyan-400 text-sm tracking-widest mb-4 block">TECHNOLOGY</span>
              <h2 className="text-4xl md:text-5xl font-bold mb-6 leading-tight">
                SEGFORMER AI
                <span className="block text-white/60 text-3xl mt-2">MEETS SENTINEL-2</span>
              </h2>
              <p className="text-white/60 text-lg mb-8 leading-relaxed">
                Our system combines state-of-the-art transformer architecture 
                with free, publicly available satellite imagery. 11 spectral bands 
                reveal debris invisible to the naked eye.
              </p>
              
              <div className="space-y-4">
                {[
                  { label: "11 Spectral Bands", desc: "From visible to SWIR wavelengths" },
                  { label: "10m Resolution", desc: "Detect objects as small as a car" },
                  { label: "5-Day Revisit", desc: "Continuous global monitoring" },
                ].map((item, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.2 + i * 0.1 }}
                    className="flex items-start gap-4 p-4 glass rounded-xl"
                  >
                    <div className="w-2 h-2 rounded-full bg-cyan-400 mt-2 animate-pulse-glow" />
                    <div>
                      <div className="font-semibold text-white">{item.label}</div>
                      <div className="text-white/50 text-sm">{item.desc}</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
            
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="relative"
            >
              {/* Visualization placeholder */}
              <div className="aspect-square relative">
                <div className="absolute inset-0 glass rounded-3xl overflow-hidden">
                  {/* Animated grid visualization */}
                  <div className="absolute inset-0 grid-lines opacity-50" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="relative w-64 h-64">
                      {/* Orbiting rings */}
                      {[1, 2, 3].map((ring) => (
                        <motion.div
                          key={ring}
                          className="absolute inset-0 border border-cyan-500/30 rounded-full"
                          style={{ 
                            scale: 0.5 + ring * 0.25,
                          }}
                          animate={{ rotate: 360 }}
                          transition={{ 
                            duration: 20 + ring * 10, 
                            repeat: Infinity, 
                            ease: "linear" 
                          }}
                        >
                          <div 
                            className="absolute w-3 h-3 bg-cyan-400 rounded-full -top-1.5 left-1/2 -translate-x-1/2"
                            style={{ boxShadow: "0 0 20px rgba(6, 182, 212, 0.8)" }}
                          />
                        </motion.div>
                      ))}
                      {/* Center globe */}
                      <div className="absolute inset-1/4 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-full opacity-80 animate-pulse-glow" />
                    </div>
                  </div>
                </div>
                
                {/* Floating labels */}
                <motion.div
                  animate={{ y: [-5, 5, -5] }}
                  transition={{ repeat: Infinity, duration: 4 }}
                  className="absolute top-10 -right-4 glass px-4 py-2 rounded-lg text-sm"
                >
                  <span className="text-cyan-400">●</span> Debris Detected
                </motion.div>
                <motion.div
                  animate={{ y: [5, -5, 5] }}
                  transition={{ repeat: Infinity, duration: 4 }}
                  className="absolute bottom-20 -left-4 glass px-4 py-2 rounded-lg text-sm"
                >
                  <span className="text-green-400">●</span> Processing: 98.7%
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-32 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-6xl font-bold mb-6">
              READY TO <span className="text-cyan-400">DETECT?</span>
            </h2>
            <p className="text-white/50 text-xl mb-10 max-w-2xl mx-auto">
              Upload your satellite imagery and let our AI identify marine debris 
              hotspots in seconds. Join the fight for cleaner oceans.
            </p>
            <Link href="/analyze" className="btn-primary text-xl px-12 py-5">
              START ANALYSIS →
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-12 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className="font-semibold">OCEANGUARD AI</span>
          </div>
          <div className="text-white/40 text-sm">
            Presidential AI Challenge 2026 • Built for a cleaner future
          </div>
        </div>
      </footer>
    </div>
  );
}
