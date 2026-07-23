// Copy every reconstruction.json the backend pipeline produced under ../data
// into public/scenes/, and write the index.json manifest the viewer reads.
//
//   npm run sync:scenes
//
// `data/` is gitignored and lives only in the main repo checkout, so run this
// from that checkout (not a worktree). public/scenes/ is gitignored too: these
// carry real accident GPS, which this repo deliberately keeps out of version
// control (see .gitignore next to frontend/public/reconstruction.json). Run
// this after a fresh clone to populate the viewer.

import { readdirSync, readFileSync, writeFileSync, mkdirSync, rmSync } from "node:fs";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = resolve(HERE, "../../data");
const OUT_DIR = resolve(HERE, "../public/scenes");

/** Recursively collect every `*_reconstruction.json` under `dir`. */
function findReconstructions(dir) {
  const found = [];
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return found; // missing/unreadable dir -> nothing to sync
  }
  for (const e of entries) {
    const p = join(dir, e.name);
    if (e.isDirectory()) found.push(...findReconstructions(p));
    else if (e.name.endsWith("_reconstruction.json")) found.push(p);
  }
  return found;
}

/** Filesystem/URL-safe name. CJK is kept (browsers handle it once encoded). */
function slugify(name) {
  return name.replace(/[/\\?%*:|"<>#\s]+/g, "_");
}

const files = findReconstructions(DATA_DIR);
if (files.length === 0) {
  console.error(`no *_reconstruction.json found under ${DATA_DIR}`);
  process.exit(1);
}

rmSync(OUT_DIR, { recursive: true, force: true });
mkdirSync(OUT_DIR, { recursive: true });

const manifest = [];
for (const src of files) {
  const raw = readFileSync(src, "utf8");
  const data = JSON.parse(raw);
  if (!data.ready) {
    console.warn(`skip (not ready): ${data.scene} — ${data.reason ?? ""}`);
    continue;
  }
  const file = `${slugify(data.scene)}.json`;
  writeFileSync(join(OUT_DIR, file), raw);

  // Longest track wins as the scene duration; used to sort/label the picker.
  const tracks = Object.values(data.vehicles ?? {});
  const duration = Math.max(
    0,
    ...tracks.map((v) => (v.track.at(-1)?.t_sec ?? 0) - (v.track[0]?.t_sec ?? 0)),
  );
  manifest.push({
    id: data.scene,
    file,
    vehicles: tracks.map((v) => v.name),
    duration_sec: Number(duration.toFixed(2)),
    origin_latlon: data.origin_latlon ?? null,
    has_impact: data.impact != null,
    gcp_ground_span_m: data.speed_reliability?.gcp_ground_span_m ?? null,
  });
  console.log(`+ ${data.scene} -> scenes/${file}`);
}

// Best-calibrated scene first: a wide GCP span means trustworthy speeds, so
// that is what the viewer should open on by default.
manifest.sort((a, b) => (b.gcp_ground_span_m ?? 0) - (a.gcp_ground_span_m ?? 0));
writeFileSync(join(OUT_DIR, "index.json"), JSON.stringify(manifest, null, 2) + "\n");
console.log(`\nwrote scenes/index.json (${manifest.length} scenes)`);
