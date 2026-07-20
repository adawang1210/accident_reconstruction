import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { Line } from "@react-three/drei";
import type { RoadPoint } from "../types";

// Clean street layer for the schematic basemap: a light ground plane plus, for
// each road centreline, an asphalt band with white edge lines and a dashed
// yellow centre line (lane markings). Metric -> three.js: x = x_m, z = -z_m.

const ROAD_WIDTH = 11; // metres (carriageway)
const Y_ROAD = 0.02;
const Y_EDGE = 0.04;
const Y_CENTER = 0.05;

type V2 = [number, number];

// Build a flat ribbon BufferGeometry of `width` along a centreline (XZ plane at
// height y), returning the left/right rails so we can draw edge lines on them.
function ribbon(points: V2[], width: number, y: number) {
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
    pos.push(l0[0], y, l0[1], r0[0], y, r0[1], l1[0], y, l1[1]);
    pos.push(r0[0], y, r0[1], r1[0], y, r1[1], l1[0], y, l1[1]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.computeVertexNormals();
  return { geo, left, right };
}

export function SchematicStreets({
  roads,
}: {
  roads: Record<string, RoadPoint[]>;
}) {
  const built = useMemo(() => {
    return Object.entries(roads)
      .filter(([, pts]) => pts.length >= 2)
      .map(([id, pts]) => {
        const center2: V2[] = pts.map((p) => [p.x_m, -p.z_m]);
        const { geo, left, right } = ribbon(center2, ROAD_WIDTH, Y_ROAD);
        const to3 = (a: V2[], y: number) =>
          a.map((p) => [p[0], y, p[1]] as [number, number, number]);
        return {
          id,
          geo,
          center: pts.map(
            (p) => [p.x_m, Y_CENTER, -p.z_m] as [number, number, number],
          ),
          leftPts: to3(left, Y_EDGE),
          rightPts: to3(right, Y_EDGE),
        };
      });
  }, [roads]);

  useEffect(
    () => () => built.forEach((b) => b.geo.dispose()),
    [built],
  );

  return (
    <>
      {/* Light "pavement" ground. Fog/camera keep the visible area scene-sized. */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
        <planeGeometry args={[600, 600]} />
        <meshStandardMaterial color="#d9dde3" roughness={1} metalness={0} />
      </mesh>

      {built.map((r) => (
        <group key={r.id}>
          <mesh geometry={r.geo} receiveShadow>
            <meshStandardMaterial color="#363b43" roughness={1} metalness={0} />
          </mesh>
          <Line points={r.leftPts} color="#eef1f4" lineWidth={2.5} />
          <Line points={r.rightPts} color="#eef1f4" lineWidth={2.5} />
          <Line
            points={r.center}
            color="#ffd34d"
            lineWidth={2.5}
            dashed
            dashSize={2.6}
            gapSize={2.4}
          />
        </group>
      ))}
    </>
  );
}
