import { useMemo, useRef } from "react";
import type { ElementRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import type { VehicleData } from "../types";
import { usePlayback } from "../playback/store";
import { sampleTrack } from "./sampleTrack";

// The route each vehicle actually drove, as reconstructed by the pipeline.
// Until now only the car itself moved, so the path it took vanished the moment
// it passed -- the single most useful thing to see in a crash reconstruction.
//
// Drawn as two overlaid lines: the full route ghosted in, and the portion
// already travelled at the current playback time drawn solid on top.

const Y_TRACK = 0.12; // above the road markings (Y_CENTER = 0.05)

/** Sample index whose t_sec is <= t; -1 before the track starts. */
function travelledCount(times: number[], t: number): number {
  let lo = -1;
  let hi = times.length;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid;
    else hi = mid;
  }
  return lo + 1;
}

export function TrackPath({ data }: { data: VehicleData }) {
  const color = useMemo(() => {
    const [r, g, b] = data.color_rgb;
    return new THREE.Color(r / 255, g / 255, b / 255);
  }, [data.color_rgb]);

  const points = useMemo(
    () =>
      data.track.map(
        (s) => [s.x_m, Y_TRACK, -s.z_m] as [number, number, number],
      ),
    [data.track],
  );
  const times = useMemo(() => data.track.map((s) => s.t_sec), [data.track]);

  // Pre-size the travelled line to the full track: drei's <Line> keeps a fixed
  // buffer, so we rewrite it in place each frame rather than remounting it
  // (which would rebuild the geometry ~60x a second).
  const travelled = useRef<ElementRef<typeof Line>>(null);
  const scratch = useMemo(() => new Float32Array(points.length * 3), [points]);

  useFrame(() => {
    const line = travelled.current;
    if (!line) return;
    const t = usePlayback.getState().currentTime;
    const n = travelledCount(times, t);

    if (n < 2) {
      line.visible = false;
      return;
    }
    line.visible = true;

    // Collapse the not-yet-travelled tail onto the live interpolated position,
    // so the line ends exactly at the car instead of jumping sample to sample.
    const here = sampleTrack(data.track, t);
    const tipX = here ? here.x : points[n - 1][0];
    const tipZ = here ? -here.z : points[n - 1][2];
    for (let i = 0; i < points.length; i++) {
      const p = i < n - 1 ? points[i] : ([tipX, Y_TRACK, tipZ] as const);
      scratch[i * 3] = p[0];
      scratch[i * 3 + 1] = p[1];
      scratch[i * 3 + 2] = p[2];
    }
    line.geometry.setPositions(scratch);
  });

  if (points.length < 2) return null;
  return (
    <group>
      {/* Full reconstructed route, ghosted. */}
      <Line
        points={points}
        color={color}
        lineWidth={2}
        transparent
        opacity={0.28}
        depthWrite={false}
      />
      {/* Travelled so far, solid. */}
      <Line
        ref={travelled}
        points={points}
        color={color}
        lineWidth={3.5}
        depthWrite={false}
      />
    </group>
  );
}
