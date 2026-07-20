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
  build: {
    rollupOptions: {
      output: {
        // Split the large, stable 3D libs (three + @react-three + tiles + splat,
        // ~1 MB) into their own chunk so the browser caches them across app
        // deploys instead of re-downloading the whole ~1.3 MB bundle on every
        // code change. Kept to a single 3D group (not react/vendor sub-splits)
        // to avoid circular chunk references.
        manualChunks(id) {
          if (
            id.includes("node_modules") &&
            (id.includes("three") ||
              id.includes("3d-tiles-renderer") ||
              id.includes("gaussian-splats"))
          )
            return "vendor-three";
          return undefined;
        },
      },
    },
  },
});
