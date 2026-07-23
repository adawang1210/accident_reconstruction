import { useEffect, useMemo } from "react";
import { Line } from "@react-three/drei";
import type { RoadPoint } from "../types";
import { Y_CENTER, Y_EDGE, Y_ROAD, ribbon, to3, type V2 } from "./ribbon";

// Fallback street layer, drawn from reconstruction.json's aligned centrelines
// when OSM is unreachable (offline / Overpass rate-limited). OsmStreets is
// preferred: it has the true junction geometry and per-road widths, while this
// only knows the 5-point centreline the pipeline snapped each vehicle to.

const ROAD_WIDTH = 11; // metres (assumed carriageway)

export function SchematicStreets({
  roads,
}: {
  roads: Record<string, RoadPoint[]>;
}) {
  const built = useMemo(
    () =>
      Object.entries(roads)
        .filter(([, pts]) => pts.length >= 2)
        .map(([id, pts]) => {
          const center: V2[] = pts.map((p) => [p.x_m, -p.z_m]);
          const { geo, left, right } = ribbon(center, ROAD_WIDTH, Y_ROAD);
          return {
            id,
            geo,
            center: to3(center, Y_CENTER),
            leftPts: to3(left, Y_EDGE),
            rightPts: to3(right, Y_EDGE),
          };
        }),
    [roads],
  );

  useEffect(() => () => built.forEach((b) => b.geo.dispose()), [built]);

  return (
    <>
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

/** The light "pavement" plane the schematic basemap sits on. */
export function SchematicGround({ size }: { size: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]} receiveShadow>
      <planeGeometry args={[size, size]} />
      <meshStandardMaterial color="#d9dde3" roughness={1} metalness={0} />
    </mesh>
  );
}
