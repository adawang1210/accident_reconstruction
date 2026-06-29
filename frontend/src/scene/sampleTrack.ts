import type { TrackSample } from "../types";

export interface Sampled {
  x: number; // metres east
  z: number; // metres north
  speed: number; // km/h (interpolated)
  /** true while t is within the track's time span (vehicle actually moving). */
  active: boolean;
}

// Position by TIME (t_sec), not by arc length: this preserves the real
// acceleration/deceleration encoded in the samples. Arc-length parametrisation
// (e.g. CatmullRomCurve3.getPointAt) would walk at constant speed and erase it.
// Outside the track span the position is clamped to the first/last sample
// (a vehicle rests at its last analysed position).
export function sampleTrack(track: TrackSample[], t: number): Sampled | null {
  const n = track.length;
  if (n === 0) return null;
  const first = track[0];
  if (t <= first.t_sec)
    return { x: first.x_m, z: first.z_m, speed: first.speed_kmh, active: false };
  const last = track[n - 1];
  if (t >= last.t_sec)
    return { x: last.x_m, z: last.z_m, speed: last.speed_kmh, active: false };

  // Binary search for the bracketing pair [lo, lo+1].
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (track[mid].t_sec <= t) lo = mid;
    else hi = mid;
  }
  const a = track[lo];
  const b = track[lo + 1];
  const span = b.t_sec - a.t_sec || 1;
  const f = (t - a.t_sec) / span;
  return {
    x: a.x_m + (b.x_m - a.x_m) * f,
    z: a.z_m + (b.z_m - a.z_m) * f,
    speed: a.speed_kmh + (b.speed_kmh - a.speed_kmh) * f,
    active: true,
  };
}

/** Last sample's t_sec, i.e. when this vehicle's analysed track ends. */
export function trackEnd(track: TrackSample[]): number {
  return track.length ? track[track.length - 1].t_sec : 0;
}

/** First sample's t_sec, i.e. when this vehicle first appears. */
export function trackStart(track: TrackSample[]): number {
  return track.length ? track[0].t_sec : 0;
}
