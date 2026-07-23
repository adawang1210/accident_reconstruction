import * as THREE from "three";

// Flat road ribbons on the XZ plane, shared by the OSM street layer and the
// reconstruction-centreline fallback.

/** A point on the ground plane: `[x, z]` in three.js metres. */
export type V2 = [number, number];

/** Stacking heights, kept apart so the layers never z-fight. */
export const Y_ROAD = 0.02;
export const Y_EDGE = 0.04;
export const Y_CENTER = 0.05;

export interface Ribbon {
  geo: THREE.BufferGeometry;
  left: V2[];
  right: V2[];
}

/**
 * Build a flat band of `width` metres along a centreline, at height `y`.
 *
 * Returns the left/right rails too, so callers can draw the edge lines exactly
 * on the asphalt boundary.
 */
export function ribbon(points: V2[], width: number, y: number): Ribbon {
  const n = points.length;
  const left: V2[] = [];
  const right: V2[] = [];
  const half = width / 2;
  const t = new THREE.Vector2();
  for (let i = 0; i < n; i++) {
    const a = points[Math.max(0, i - 1)];
    const b = points[Math.min(n - 1, i + 1)];
    t.set(b[0] - a[0], b[1] - a[1]);
    if (t.lengthSq() < 1e-9) t.set(1, 0);
    t.normalize();
    const nx = -t.y;
    const nz = t.x; // perpendicular in XZ
    left.push([points[i][0] + nx * half, points[i][1] + nz * half]);
    right.push([points[i][0] - nx * half, points[i][1] - nz * half]);
  }
  const pos: number[] = [];
  for (let i = 0; i < n - 1; i++) {
    const l0 = left[i];
    const r0 = right[i];
    const l1 = left[i + 1];
    const r1 = right[i + 1];
    // Wind left -> next-left -> right, which puts the face normal at +Y. The
    // previous order (left, right, next-left) wound the other way: the normals
    // pointed DOWN, so backface culling hid every road surface from any camera
    // above the ground, and the lighting on them was black anyway. Only the
    // edge lines showed, which is why the streets looked like bare outlines.
    pos.push(l0[0], y, l0[1], l1[0], y, l1[1], r0[0], y, r0[1]);
    pos.push(r0[0], y, r0[1], l1[0], y, l1[1], r1[0], y, r1[1]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.computeVertexNormals();
  return { geo, left, right };
}

/** Lift ground points to a `Line`-ready triple at height `y`. */
export function to3(pts: V2[], y: number): [number, number, number][] {
  return pts.map((p) => [p[0], y, p[1]]);
}
