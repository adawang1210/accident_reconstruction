import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { VehicleData } from "../types";
import { usePlayback } from "../playback/store";
import { sampleTrack, trackStart } from "./sampleTrack";
import { CarModel } from "./CarModel";
import { headingLookahead, vehicleKind } from "./vehicleKind";

const UP = new THREE.Vector3(0, 1, 0);
// How fast the model swings onto a new heading, as a fraction settled per
// second. The old code slerped a fixed 0.25 per *frame*, so the turn-in rate
// silently tripled between a 30 Hz and a 120 Hz display.
const TURN_RATE = 12;

// Metric -> three.js mapping: x = x_m (east), z = -z_m (north -> -Z, north-up).
export function Vehicle({ id, data }: { id: string; data: VehicleData }) {
  const group = useRef<THREE.Group>(null!);
  const color = useMemo(() => {
    const [r, g, b] = data.color_rgb;
    return new THREE.Color(r / 255, g / 255, b / 255);
  }, [data.color_rgb]);
  const kind = useMemo(() => vehicleKind(id, data.name), [id, data.name]);
  const lookahead = useMemo(() => headingLookahead(kind), [kind]);
  const startT = useMemo(() => trackStart(data.track), [data.track]);
  const targetQuat = useRef(new THREE.Quaternion());

  useFrame((_, dt) => {
    const grp = group.current;
    if (!grp) return;
    const t = usePlayback.getState().currentTime;
    grp.visible = t >= startT - 1e-3;
    if (!grp.visible) return;

    const here = sampleTrack(data.track, t);
    if (!here) return;
    grp.position.set(here.x, 0, -here.z);

    // Heading from a short look-ahead along the path.
    const ahead = sampleTrack(data.track, t + lookahead) ?? here;
    const dx = ahead.x - here.x;
    const dz = -(ahead.z - here.z);
    if (dx * dx + dz * dz > 1e-6) {
      // Model's local forward is +Z; yaw so +Z aligns with (dx, dz).
      targetQuat.current.setFromAxisAngle(UP, Math.atan2(dx, dz));
      grp.quaternion.slerp(
        targetQuat.current,
        1 - Math.exp(-TURN_RATE * Math.min(dt, 0.1)),
      );
    }
  });

  return (
    <group ref={group}>
      <CarModel color={color} kind={kind} />
    </group>
  );
}
