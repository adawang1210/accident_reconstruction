/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where to fetch reconstruction.json. Defaults to the bundled sample. */
  readonly VITE_RECONSTRUCTION_URL?: string;
  /** Google Maps Platform API key (Map Tiles API) for Photorealistic 3D Tiles. */
  readonly VITE_GOOGLE_TILES_KEY?: string;
  /** Optional .glb/.gltf car model URL; falls back to a procedural car. */
  readonly VITE_CAR_MODEL_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
