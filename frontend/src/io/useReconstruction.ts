import { useEffect, useState } from "react";
import type { Reconstruction } from "../types";

// Default to a static file in public/. Switch to the live API with e.g.
// VITE_RECONSTRUCTION_URL="/api/reconstruction?video=<name>" (dev proxy in
// vite.config.ts forwards /api to the FastAPI workbench on :8000).
// `||` (not `??`) so an empty VITE_RECONSTRUCTION_URL= in .env still falls back.
const URL =
  import.meta.env.VITE_RECONSTRUCTION_URL || "/reconstruction.sample.json";

export interface LoadState {
  data: Reconstruction | null;
  error: string | null;
}

export function useReconstruction(): LoadState {
  const [state, setState] = useState<LoadState>({ data: null, error: null });

  useEffect(() => {
    let cancelled = false;
    fetch(URL)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${URL}`);
        return r.json();
      })
      .then((d: Reconstruction) => {
        if (!d.ready) throw new Error(d.reason ?? "reconstruction not ready");
        if (!cancelled) setState({ data: d, error: null });
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setState({ data: null, error: e instanceof Error ? e.message : String(e) });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
