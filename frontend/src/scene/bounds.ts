import type { Reconstruction } from "../types";

// Every framing constant used to be tuned by hand for the 60 m synthetic sample
// (camera 150 m up, fog 150->360 m, 120 m shadow catcher). Real pipeline runs
// span 5-30 m, so those numbers rendered the accident as a speck. Derive them
// from the content instead, so any scene opens correctly framed.

export interface SceneBounds {
  /** three.js centre of the action (x, y=0, z). */
  center: [number, number, number];
  /** Half-extent in metres covering every track/road/impact point. */
  radius: number;
}

// Below this the camera would clip into the cars; a 4 m crash still needs a
// readable amount of street around it.
const MIN_RADIUS = 14;

/**
 * Bounding radius of the *action* — the vehicle tracks and the impact — in
 * metres.
 *
 * Deliberately excludes `data.roads`: those centrelines run the full length of
 * each street (±100 m in keelung_xinwu_yier, against a 10 m collision), so
 * framing on them would push the camera 20x too far out and leave the crash a
 * speck. Roads are unbounded context, not subject matter.
 *
 * Uses the max half-extent (not the diagonal) so a long, thin track — the
 * common case for a straight approach — doesn't over-zoom the camera out.
 */
export function computeBounds(data: Reconstruction): SceneBounds {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;

  const add = (xM: number, zM: number) => {
    const x = xM;
    const z = -zM; // metric north -> three.js -Z (north-up when viewed top-down)
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  };

  for (const v of Object.values(data.vehicles))
    for (const s of v.track) add(s.x_m, s.z_m);
  if (data.impact) add(data.impact.x_m, data.impact.z_m);

  if (!Number.isFinite(minX)) return { center: [0, 0, 0], radius: MIN_RADIUS };

  const cx = (minX + maxX) / 2;
  const cz = (minZ + maxZ) / 2;
  const half = Math.max((maxX - minX) / 2, (maxZ - minZ) / 2);
  // Pad so the outermost vehicle isn't flush against the frame edge.
  return { center: [cx, 0, cz], radius: Math.max(half * 1.35, MIN_RADIUS) };
}

/** Camera/fog/basemap dial settings derived from the scene's size. */
export interface Framing {
  introStartY: number;
  introEndY: number;
  introOffset: number;
  fogNear: number;
  fogFar: number;
  minDistance: number;
  maxDistance: number;
  shadowScale: number;
  /** Overpass fetch radius: enough street context without hammering the API. */
  osmRadius: number;
  groundSize: number;
}

export function framingFor(radius: number): Framing {
  return {
    introStartY: radius * 5,
    introEndY: radius * 1.7,
    introOffset: radius * 1.1,
    fogNear: radius * 4,
    fogFar: radius * 11,
    minDistance: Math.max(radius * 0.4, 5),
    maxDistance: radius * 14,
    shadowScale: radius * 4,
    // Context, not framing: a 10 m collision still wants the whole junction and
    // the block around it. 70 m returned 2 buildings at keelung_xinwu_yier;
    // 140 m is the floor that reliably brings in the surrounding streets.
    osmRadius: Math.round(Math.min(Math.max(radius * 5, 140), 250)),
    groundSize: radius * 40,
  };
}
