import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The bundle is served by demo.py (FastAPI StaticFiles at /) on 127.0.0.1; ui/dist is gitignored.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
