import { Line } from "@react-three/drei";
import type { RoadPoint } from "../types";

// Road centrelines from reconstruction.json. Slightly above ground (y=0.05) so
// they don't z-fight with the plane. Metric -> three.js: x = x_m, z = -z_m.
export function Roads({ roads }: { roads: Record<string, RoadPoint[]> }) {
  return (
    <>
      {Object.entries(roads).map(([id, pts]) =>
        pts.length >= 2 ? (
          <Line
            key={id}
            points={pts.map(
              (p) => [p.x_m, 0.05, -p.z_m] as [number, number, number],
            )}
            color="#ffd34d"
            lineWidth={3}
            dashed={false}
          />
        ) : null,
      )}
    </>
  );
}
