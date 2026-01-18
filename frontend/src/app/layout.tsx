import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OceanGuard AI | Marine Debris Detection",
  description: "AI-powered satellite imagery analysis for marine debris detection. Presidential AI Challenge 2026.",
  keywords: ["marine debris", "satellite imagery", "AI", "ocean conservation", "SegFormer"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
