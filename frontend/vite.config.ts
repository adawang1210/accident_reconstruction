import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The viewer reads the backend's reconstruction JSON. Two options:
//   1. Static file: drop it at public/reconstruction.json (default).
//   2. Live API: set VITE_RECONSTRUCTION_URL=/api/reconstruction?video=<name>
//      and rely on the dev proxy below to reach the FastAPI workbench.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/media": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
