import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { DropInViewer } from "@mkkellogg/gaussian-splats-3d";

// Gaussian Splatting basemap: a photoreal capture of the real intersection.
// Set VITE_SPLAT_URL to a .ply/.splat/.ksplat/.spz (under public/, or a CORS URL).
// IMPORTANT: a splat must come from REAL multi-view photos of the place (your own
// phone/drone capture). Screenshotting Google Maps/Earth cannot exceed that blurry
// source, so it does NOT help -- see SPLAT_NOTES.md.
const URL = import.meta.env.VITE_SPLAT_URL as string | undefined;

export const HAS_SPLAT = Boolean(URL);

// A splat comes out of Structure-from-Motion at an arbitrary scale / orientation /
// origin, so aligning a real capture to our metre scene is manual: tune these env
// knobs until the splat ground + landmarks line up with the roads/impact.
// Scene frame: x = east, z = -north, y = up.
const n = (v: string | undefined, d: number) => (v ? Number(v) : d);
const e = import.meta.env;
const SCALE = n(e.VITE_SPLAT_SCALE, 1);
const ROT: [number, number, number] = [
  n(e.VITE_SPLAT_ROT_X_DEG, 0) * THREE.MathUtils.DEG2RAD,
  n(e.VITE_SPLAT_ROT_Y_DEG, 0) * THREE.MathUtils.DEG2RAD,
  n(e.VITE_SPLAT_ROT_Z_DEG, 0) * THREE.MathUtils.DEG2RAD,
];
const POS: [number, number, number] = [
  n(e.VITE_SPLAT_X, 0),
  n(e.VITE_SPLAT_Y, 0),
  n(e.VITE_SPLAT_Z, 0),
];

export function SplatScene() {
  const viewer = useMemo(() => {
    if (!URL) return null;
    // DropInViewer is a THREE.Group that self-updates via onBeforeRender, so it
    // drops straight into the R3F render loop. No COOP/COEP headers needed with
    // sharedMemoryForWorkers off.
    const v = new DropInViewer({
      gpuAcceleratedSort: true,
      sharedMemoryForWorkers: false,
    });
    v.addSplatScene(URL, { showLoadingUI: false, progressiveLoad: false }).catch(
      (err: unknown) => console.error("[splat] failed to load", URL, err),
    );
    return v;
  }, []);

  useEffect(() => () => viewer?.dispose?.(), [viewer]);

  if (!viewer) return null;
  return <primitive object={viewer} position={POS} rotation={ROT} scale={SCALE} />;
}
