import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { mergeBufferGeometries } from "three-stdlib";

// Clean "schematic" basemap: real building footprints from OpenStreetMap,
// extruded to height in our metric scene frame. Sharp, free-orbitable, and free
// of the photogrammetry warping/baked-in-vehicles of Google 3D Tiles -- the
// trade-off is it's stylised, not photoreal.

const OVERPASS_URL = "https://overpass-api.de/api/interpreter";

type OSMNode = { lat: number; lon: number };
type OSMElement = {
  type: string;
  geometry?: OSMNode[];
  tags?: Record<string, string>;
};

// Local equirectangular projection: lat/lon -> scene metres, anchored at the
// scene origin (origin_latlon). Matches the reconstruction frame: x = east,
// z = -north (see Roads.tsx). Accurate to well under a metre over a few hundred.
function makeProjector(lat0: number, lon0: number) {
  const mPerDegLat = 111320;
  const mPerDegLon = 111320 * Math.cos((lat0 * Math.PI) / 180);
  // Returns [east, north]; the extrude rotation below maps north -> -z.
  return (lat: number, lon: number): [number, number] => [
    (lon - lon0) * mPerDegLon,
    (lat - lat0) * mPerDegLat,
  ];
}

function buildingHeight(tags: Record<string, string> = {}): number {
  const h = parseFloat(tags.height ?? "");
  if (!Number.isNaN(h) && h > 0) return h;
  const levels = parseFloat(tags["building:levels"] ?? "");
  if (!Number.isNaN(levels) && levels > 0) return levels * 3.2;
  return 12; // ~4 storeys default when untagged
}

export function CityBlocks({
  lat,
  lon,
  radius = 80,
}: {
  lat: number;
  lon: number;
  radius?: number;
}) {
  const [geom, setGeom] = useState<THREE.BufferGeometry | null>(null);

  useEffect(() => {
    let cancelled = false;
    const project = makeProjector(lat, lon);
    const query = `[out:json][timeout:25];(way["building"](around:${radius},${lat},${lon}););out geom;`;

    fetch(OVERPASS_URL, { method: "POST", body: query })
      .then((r) => r.json())
      .then((data: { elements?: OSMElement[] }) => {
        if (cancelled) return;
        const parts: THREE.BufferGeometry[] = [];
        for (const el of data.elements ?? []) {
          const ring = el.geometry;
          if (el.type !== "way" || !ring || ring.length < 4) continue;
          try {
            const shape = new THREE.Shape();
            ring.forEach((n, i) => {
              const [east, north] = project(n.lat, n.lon);
              if (i === 0) shape.moveTo(east, north);
              else shape.lineTo(east, north);
            });
            const g = new THREE.ExtrudeGeometry(shape, {
              depth: buildingHeight(el.tags),
              bevelEnabled: false,
            });
            // Shape is (east, north) extruded along +z; rotate so the extrusion
            // becomes +Y (up) and the footprint lands at scene (east, -north).
            g.rotateX(-Math.PI / 2);
            // Subtle per-building colour variation (cool light greys) so blocks
            // read apart even before the edge outlines.
            const col = new THREE.Color().setHSL(
              0.62,
              0.05,
              0.72 + Math.random() * 0.13,
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
            // skip degenerate / self-intersecting footprints
          }
        }
        if (parts.length === 0 || cancelled) return;
        const merged = mergeBufferGeometries(parts, false);
        parts.forEach((g) => g.dispose());
        if (merged && !cancelled) {
          merged.computeVertexNormals();
          setGeom(merged);
        }
      })
      .catch(() => {
        /* network/Overpass failure -> no buildings, ground still shows */
      });

    return () => {
      cancelled = true;
    };
  }, [lat, lon, radius]);

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
