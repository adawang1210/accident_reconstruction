import { useEffect, useState } from "react";

// One shared Overpass fetch for the schematic basemap: building footprints AND
// road centrelines around the scene origin. Previously buildings were fetched
// on their own and streets came only from reconstruction.json's `roads`, which
// is `{}` for most real scenes — those rendered with no streets at all. The
// scene origin is a real GPS fix, so OSM has the true road geometry for it.

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

// Drivable ways worth drawing. Footpaths/cycleways are skipped: at accident
// scale they add clutter without helping read the collision.
const HIGHWAY_RE =
  "^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service)(_link)?$";

export interface OSMNode {
  lat: number;
  lon: number;
}

export interface OSMWay {
  type: string;
  id: number;
  geometry?: OSMNode[];
  tags?: Record<string, string>;
}

export interface OSMData {
  buildings: OSMWay[];
  highways: OSMWay[];
}

/**
 * Local equirectangular projection: lat/lon -> scene metres, anchored at the
 * scene origin. Matches the reconstruction frame (`x = east`, `z = -north`) and
 * is sub-metre accurate over the few hundred metres we ever draw.
 */
export function makeProjector(lat0: number, lon0: number) {
  const mPerDegLat = 111320;
  const mPerDegLon = 111320 * Math.cos((lat0 * Math.PI) / 180);
  /** Returns three.js ground coordinates `[x, z]`. */
  return (lat: number, lon: number): [number, number] => [
    (lon - lon0) * mPerDegLon,
    -(lat - lat0) * mPerDegLat,
  ];
}

/**
 * Deterministic 0..1 hash of a way id.
 *
 * Building tint used `Math.random()`, so every reload (and every React strict-
 * mode double-render) recoloured the block — distracting when comparing frames.
 */
export function hash01(id: number): number {
  const x = Math.sin(id * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function cacheKey(lat: number, lon: number, radius: number): string {
  return `osm:${lat.toFixed(6)}:${lon.toFixed(6)}:${radius}`;
}

/**
 * Fetch buildings + roads around a point, memoised in sessionStorage.
 *
 * The public Overpass endpoint rate-limits, and the viewer re-queried it on
 * every reload and every scene switch back. Caching per (lat, lon, radius)
 * makes repeat views instant and keeps us well inside the quota.
 */
export async function fetchOsm(
  lat: number,
  lon: number,
  radius: number,
): Promise<OSMData> {
  const key = cacheKey(lat, lon, radius);
  try {
    const hit = sessionStorage.getItem(key);
    if (hit) return JSON.parse(hit) as OSMData;
  } catch {
    /* private mode / quota -> just refetch */
  }

  const area = `(around:${radius},${lat},${lon})`;
  const query =
    `[out:json][timeout:25];(` +
    `way["building"]${area};` +
    `way["highway"~"${HIGHWAY_RE}"]${area};` +
    `);out geom;`;

  const res = await fetch(OVERPASS_URL, { method: "POST", body: query });
  if (!res.ok) throw new Error(`Overpass ${res.status}`);
  const raw = (await res.json()) as { elements?: OSMWay[] };

  const data: OSMData = { buildings: [], highways: [] };
  for (const el of raw.elements ?? []) {
    if (el.type !== "way" || !el.geometry) continue;
    if (el.tags?.building) data.buildings.push(el);
    else if (el.tags?.highway) data.highways.push(el);
  }

  try {
    sessionStorage.setItem(key, JSON.stringify(data));
  } catch {
    /* payload over quota -> serve uncached */
  }
  return data;
}

export interface OsmState {
  data: OSMData | null;
  /** True once the request settled, successfully or not. */
  settled: boolean;
}

/** Load the OSM basemap data for a scene origin. Never throws. */
export function useOsm(lat: number, lon: number, radius: number): OsmState {
  const [state, setState] = useState<OsmState>({ data: null, settled: false });

  useEffect(() => {
    let cancelled = false;
    setState({ data: null, settled: false });
    fetchOsm(lat, lon, radius)
      .then((d) => {
        if (!cancelled) setState({ data: d, settled: true });
      })
      .catch(() => {
        // Offline / rate-limited -> no basemap, the ground plane still shows.
        if (!cancelled) setState({ data: null, settled: true });
      });
    return () => {
      cancelled = true;
    };
  }, [lat, lon, radius]);

  return state;
}

/** Carriageway width in metres from OSM tags, with sane per-class defaults. */
export function roadWidth(tags: Record<string, string> = {}): number {
  const explicit = parseFloat(tags.width ?? "");
  if (!Number.isNaN(explicit) && explicit > 0) return explicit;

  const lanes = parseFloat(tags.lanes ?? "");
  if (!Number.isNaN(lanes) && lanes > 0) return lanes * 3.2;

  switch (tags.highway) {
    case "motorway":
    case "trunk":
      return 14;
    case "primary":
      return 12;
    case "secondary":
      return 10;
    case "tertiary":
      return 8;
    case "service":
      return 4;
    default:
      return 6.5;
  }
}

/** Extruded height in metres from OSM tags. */
export function buildingHeight(tags: Record<string, string> = {}): number {
  const h = parseFloat(tags.height ?? "");
  if (!Number.isNaN(h) && h > 0) return h;
  const levels = parseFloat(tags["building:levels"] ?? "");
  if (!Number.isNaN(levels) && levels > 0) return levels * 3.2;
  return 12; // ~4 storeys when untagged
}
