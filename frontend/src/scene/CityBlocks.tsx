import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { mergeBufferGeometries } from "three-stdlib";
import { buildingHeight, hash01, makeProjector, type OSMWay } from "./osm";

// Clean "schematic" basemap: real building footprints from OpenStreetMap,
// extruded to height in our metric scene frame. Sharp, free-orbitable, and free
// of the photogrammetry warping/baked-in-vehicles of Google 3D Tiles -- the
// trade-off is it's stylised, not photoreal.
//
// The Overpass request itself lives in osm.ts, shared with OsmStreets so a
// scene costs one (cached) round trip instead of one per layer.

export function CityBlocks({
  ways,
  lat,
  lon,
}: {
  ways: OSMWay[];
  lat: number;
  lon: number;
}) {
  const geom = useMemo(() => {
    const project = makeProjector(lat, lon);
    const parts: THREE.BufferGeometry[] = [];

    for (const way of ways) {
      const ring = way.geometry;
      if (!ring || ring.length < 4) continue;
      try {
        const shape = new THREE.Shape();
        ring.forEach((n, i) => {
          // Draw in (x, -z) so that after the extrusion is tipped upright the
          // footprint lands back on the scene's (x, z).
          const [x, z] = project(n.lat, n.lon);
          if (i === 0) shape.moveTo(x, -z);
          else shape.lineTo(x, -z);
        });
        const g = new THREE.ExtrudeGeometry(shape, {
          depth: buildingHeight(way.tags),
          bevelEnabled: false,
        });
        g.rotateX(-Math.PI / 2); // extrusion axis +z -> +Y (up)
        // Subtle per-building tint (cool light greys) so blocks read apart even
        // before the edge outlines. Keyed on the OSM id, so a reload or a
        // strict-mode double render reproduces the exact same colours.
        const col = new THREE.Color().setHSL(
          0.62,
          0.05,
          0.72 + hash01(way.id) * 0.13,
        );
        const vn = g.attributes.position.count;
        const colors = new Float32Array(vn * 3);
        for (let k = 0; k < vn; k++) {
          colors[k * 3] = col.r;
          colors[k * 3 + 1] = col.g;
          colors[k * 3 + 2] = col.b;
        }
        g.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
        parts.push(g);
      } catch {
        // OSM has self-intersecting/degenerate footprints that make
        // ExtrudeGeometry throw. Skip the one building, never the batch.
      }
    }
    if (parts.length === 0) return null;
    const merged = mergeBufferGeometries(parts, false);
    parts.forEach((g) => g.dispose());
    merged?.computeVertexNormals();
    return merged;
  }, [ways, lat, lon]);

  // Crisp dark outlines on every building edge -> the clean CAD/"Horizon" look.
  const edges = useMemo(
    () => (geom ? new THREE.EdgesGeometry(geom, 20) : null),
    [geom],
  );
  useEffect(() => () => geom?.dispose(), [geom]);
  useEffect(() => () => edges?.dispose(), [edges]);

  if (!geom) return null;
  return (
    <group>
      <mesh geometry={geom} castShadow receiveShadow>
        <meshStandardMaterial vertexColors roughness={0.95} metalness={0} />
      </mesh>
      {edges && (
        <lineSegments geometry={edges}>
          <lineBasicMaterial color="#3b414c" transparent opacity={0.55} />
        </lineSegments>
      )}
    </group>
  );
}
