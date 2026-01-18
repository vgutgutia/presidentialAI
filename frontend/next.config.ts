import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    // Allow data URLs (base64 images) for previews
    dangerouslyAllowSVG: true,
    contentDispositionType: 'attachment',
    contentSecurityPolicy: "default-src 'self'; script-src 'none'; sandbox;",
  },
};

export default nextConfig;
