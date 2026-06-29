import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Html } from "@react-three/drei";
import * as THREE from "three";
import type { ImpactData } from "../types";
import { usePlayback } from "../playback/store";

// A pulsing ring on the ground at the impact point; brightens once the playback
// clock passes the impact time.
export function ImpactMarker({
  impact,
  impactTime,
}: {
  impact: ImpactData;
  impactTime: number | null;
}) {
  const mat = useRef<THREE.MeshBasicMaterial>(null!);

  useFrame(() => {
    if (!mat.current) return;
    const t = usePlayback.getState().currentTime;
    const reached = impactTime != null && t >= impactTime;
    mat.current.opacity = reached
      ? 0.45 + 0.45 * Math.sin(performance.now() * 0.008)
      : 0.22;
  });

  return (
    <group position={[impact.x_m, 0, -impact.z_m]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.07, 0]}>
        <ringGeometry args={[1.4, 2.1, 40]} />
        <meshBasicMaterial
          ref={mat}
          color="#ff3a3a"
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
        />
      </mesh>
      <Html position={[0, 2.2, 0]} center distanceFactor={48} occlude={false}>
        <div className="impact-label">撞擊點</div>
      </Html>
    </group>
  );
}
