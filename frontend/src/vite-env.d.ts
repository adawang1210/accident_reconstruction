/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where to fetch reconstruction.json. Defaults to the bundled sample. */
  readonly VITE_RECONSTRUCTION_URL?: string;
  /** Google Maps Platform API key (Map Tiles API) for Photorealistic 3D Tiles. */
  readonly VITE_GOOGLE_TILES_KEY?: string;
  /** Optional .glb/.gltf car model URL; falls back to a procedural car. */
  readonly VITE_CAR_MODEL_URL?: string;
  /** Gaussian Splat basemap: .ply/.splat/.ksplat/.spz URL (real capture). */
  readonly VITE_SPLAT_URL?: string;
  /** Splat alignment knobs (SfM scale/orientation/origin are arbitrary). */
  readonly VITE_SPLAT_SCALE?: string;
  readonly VITE_SPLAT_ROT_X_DEG?: string;
  readonly VITE_SPLAT_ROT_Y_DEG?: string;
  readonly VITE_SPLAT_ROT_Z_DEG?: string;
  readonly VITE_SPLAT_X?: string;
  readonly VITE_SPLAT_Y?: string;
  readonly VITE_SPLAT_Z?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// mkkellogg's library ships no types; we only use DropInViewer (a THREE.Group).
declare module "@mkkellogg/gaussian-splats-3d" {
  import type * as THREE from "three";
  export class DropInViewer extends THREE.Group {
    constructor(options?: Record<string, unknown>);
    addSplatScene(path: string, options?: Record<string, unknown>): Promise<void>;
    dispose?: () => void;
  }
}
