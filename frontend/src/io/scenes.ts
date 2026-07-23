// The catalogue of real pipeline runs bundled under public/scenes/, written by
// `npm run sync:scenes` (scripts/sync-scenes.mjs) from the backend's
// data/<scene>/<scene>_reconstruction.json outputs.

export interface SceneEntry {
  /** Backend scene name, e.g. "keelung_xinwu_yier". */
  id: string;
  /** Filename under public/scenes/ (may contain CJK -> encode before fetch). */
  file: string;
  vehicles: string[];
  duration_sec: number;
  origin_latlon: [number, number] | null;
  has_impact: boolean;
  /** Real-world span the GCPs cover; small (<30 m) means speeds under-read. */
  gcp_ground_span_m: number | null;
}

export const SCENES_URL = "/scenes/index.json";

export function sceneUrl(entry: SceneEntry): string {
  return `/scenes/${encodeURIComponent(entry.file)}`;
}

/** Scene requested via `?scene=<id>`, so a framing/scene is linkable. */
export function requestedSceneId(): string | null {
  return new URLSearchParams(window.location.search).get("scene");
}

/** Reflect the picked scene in the URL without adding history entries. */
export function setSceneInUrl(id: string): void {
  const url = new URL(window.location.href);
  url.searchParams.set("scene", id);
  window.history.replaceState(null, "", url);
}
