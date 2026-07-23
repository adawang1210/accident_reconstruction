import { useCallback, useEffect, useState } from "react";
import type { Reconstruction } from "../types";
import {
  SCENES_URL,
  requestedSceneId,
  sceneUrl,
  setSceneInUrl,
  type SceneEntry,
} from "./scenes";

// Where the reconstruction comes from, in precedence order:
//   1. VITE_RECONSTRUCTION_URL — a single fixed file or the live API, e.g.
//      "/api/reconstruction?video=<name>" (vite.config.ts proxies /api to the
//      FastAPI workbench on :8000). This pins one scene and hides the picker.
//   2. public/scenes/index.json — the bundled catalogue of real pipeline runs,
//      selectable in the HUD and via `?scene=<id>`.
// `||` (not `??`) so an empty VITE_RECONSTRUCTION_URL= in .env still falls back.
const FIXED_URL = import.meta.env.VITE_RECONSTRUCTION_URL || "";

export interface LoadState {
  data: Reconstruction | null;
  error: string | null;
  /** Bundled scenes to choose between; empty when pinned to a fixed URL. */
  scenes: SceneEntry[];
  /** Currently loaded scene id, or null while loading / when pinned. */
  sceneId: string | null;
  selectScene: (id: string) => void;
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${url}`);
  return (await r.json()) as T;
}

export function useReconstruction(): LoadState {
  const [data, setData] = useState<Reconstruction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scenes, setScenes] = useState<SceneEntry[]>([]);
  const [sceneId, setSceneId] = useState<string | null>(requestedSceneId());

  // Load the catalogue once, and settle on which scene to open.
  useEffect(() => {
    if (FIXED_URL) return;
    let cancelled = false;
    fetchJson<SceneEntry[]>(SCENES_URL)
      .then((list) => {
        if (cancelled) return;
        setScenes(list);
        // Keep `?scene=` when it names a real scene, else open the catalogue's
        // first entry (sync-scenes sorts best-calibrated first).
        const wanted = requestedSceneId();
        const pick = list.find((s) => s.id === wanted) ?? list[0];
        if (pick) setSceneId(pick.id);
        else setError(`${SCENES_URL} 沒有任何場景`);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Load whichever scene is selected (or the pinned fixed URL).
  useEffect(() => {
    const entry = scenes.find((s) => s.id === sceneId);
    const url = FIXED_URL || (entry ? sceneUrl(entry) : null);
    if (!url) return;

    let cancelled = false;
    setData(null);
    setError(null);
    fetchJson<Reconstruction>(url)
      .then((d) => {
        if (cancelled) return;
        if (!d.ready) throw new Error(d.reason ?? "reconstruction not ready");
        setData(d);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [scenes, sceneId]);

  const selectScene = useCallback((id: string) => {
    setSceneId(id);
    setSceneInUrl(id);
  }, []);

  return { data, error, scenes: FIXED_URL ? [] : scenes, sceneId, selectScene };
}
