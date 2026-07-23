import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { mergeBufferGeometries } from "three-stdlib";
import { Line } from "@react-three/drei";
import { makeProjector, roadWidth, type OSMWay } from "./osm";
import { Y_CENTER, Y_EDGE, Y_ROAD, ribbon, to3, type V2 } from "./ribbon";

// The real street network at the scene origin, straight from OpenStreetMap.
// reconstruction.json's `roads` only carries the 5-point centrelines the
// pipeline aligned to, and is empty for most scenes -- OSM gives the true
// junction geometry (all approaches, real widths) around the same GPS fix.

// Dashed centre lines only on roads big enough to be divided; a 4 m service
// lane with a lane divider looks wrong.
const DIVIDED_MIN_WIDTH = 7;

export function OsmStreets({
  ways,
  lat,
  lon,
}: {
  ways: OSMWay[];
  lat: number;
  lon: number;
}) {
  const built = useMemo(() => {
    const project = makeProjector(lat, lon);
    const surfaces: THREE.BufferGeometry[] = [];
    const edges: [number, number, number][][] = [];
    const centres: [number, number, number][][] = [];

    for (const way of ways) {
      const nodes = way.geometry;
      if (!nodes || nodes.length < 2) continue;
      const line: V2[] = nodes.map((n) => project(n.lat, n.lon));
      const width = roadWidth(way.tags);
      const { geo, left, right } = ribbon(line, width, Y_ROAD);
      surfaces.push(geo);
      edges.push(to3(left, Y_EDGE), to3(right, Y_EDGE));
      if (width >= DIVIDED_MIN_WIDTH) centres.push(to3(line, Y_CENTER));
    }

    // One merged mesh for all asphalt: a busy junction is easily 100+ ways, and
    // that many separate meshes is 100+ draw calls for a flat grey surface.
    const merged = surfaces.length ? mergeBufferGeometries(surfaces, false) : null;
    surfaces.forEach((g) => g.dispose());
    return { merged, edges, centres };
  }, [ways, lat, lon]);

  useEffect(() => () => built.merged?.dispose(), [built]);

  if (!built.merged) return null;
  return (
    <group>
      <mesh geometry={built.merged} receiveShadow>
        <meshStandardMaterial color="#363b43" roughness={1} metalness={0} />
      </mesh>
      {built.edges.map((pts, i) => (
        <Line key={`e${i}`} points={pts} color="#eef1f4" lineWidth={2} />
      ))}
      {built.centres.map((pts, i) => (
        <Line
          key={`c${i}`}
          points={pts}
          color="#ffd34d"
          lineWidth={2}
          dashed
          dashSize={2.6}
          gapSize={2.4}
        />
      ))}
    </group>
  );
}
