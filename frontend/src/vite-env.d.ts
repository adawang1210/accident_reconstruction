/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where to fetch reconstruction.json. Defaults to the bundled sample. */
  readonly VITE_RECONSTRUCTION_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
