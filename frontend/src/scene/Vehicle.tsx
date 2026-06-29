import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { VehicleData } from "../types";
import { usePlayback } from "../playback/store";
import { sampleTrack, trackStart } from "./sampleTrack";

const UP = new THREE.Vector3(0, 1, 0);
const HEADING_LOOKAHEAD_SEC = 0.15;

// A placeholder car (Phase 0). Swap the inner meshes for a glTF in Phase 1.
// Metric -> three.js mapping: x = x_m (east), z = -z_m (north -> -Z, north-up).
export function Vehicle({ data }: { data: VehicleData }) {
  const group = useRef<THREE.Group>(null!);
  const color = useMemo(() => {
    const [r, g, b] = data.color_rgb;
    return new THREE.Color(r / 255, g / 255, b / 255);
  }, [data.color_rgb]);
  const startT = useMemo(() => trackStart(data.track), [data.track]);
  const targetQuat = useRef(new THREE.Quaternion());

  useFrame(() => {
    const grp = group.current;
    if (!grp) return;
    const t = usePlayback.getState().currentTime;
    grp.visible = t >= startT - 1e-3;
    if (!grp.visible) return;

    const here = sampleTrack(data.track, t);
    if (!here) return;
    grp.position.set(here.x, 0, -here.z);

    // Heading from a short look-ahead along the path.
    const ahead = sampleTrack(data.track, t + HEADING_LOOKAHEAD_SEC) ?? here;
    const dx = ahead.x - here.x;
    const dz = -(ahead.z - here.z);
    if (dx * dx + dz * dz > 1e-6) {
      // Car's local forward is +Z; yaw so +Z aligns with (dx, dz).
      targetQuat.current.setFromAxisAngle(UP, Math.atan2(dx, dz));
      grp.quaternion.slerp(targetQuat.current, 0.25);
    }
  });

  return (
    <group ref={group}>
      <mesh castShadow position={[0, 0.65, 0]}>
        <boxGeometry args={[1.8, 1.3, 4.3]} />
        <meshPhysicalMaterial
          color={color}
          metalness={0.6}
          roughness={0.35}
          clearcoat={1}
          clearcoatRoughness={0.15}
        />
      </mesh>
      {/* cabin */}
      <mesh castShadow position={[0, 1.45, -0.2]}>
        <boxGeometry args={[1.6, 0.8, 2.2]} />
        <meshStandardMaterial color={color} metalness={0.3} roughness={0.5} />
      </mesh>
      {/* heading arrow (forward = +Z) */}
      <mesh position={[0, 0.65, 2.7]} rotation={[Math.PI / 2, 0, 0]}>
        <coneGeometry args={[0.35, 0.8, 12]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </group>
  );
}
