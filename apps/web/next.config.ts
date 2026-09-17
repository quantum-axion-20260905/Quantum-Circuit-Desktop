import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // React Flow 11 emits a false-positive type-map warning under React
  // Strict Mode's development double-render. Keep the research workspace
  // console clean; production behavior is unchanged.
  reactStrictMode: false,
  output: "export",
  allowedDevOrigins: ["127.0.0.1", "localhost"]
};

export default nextConfig;

