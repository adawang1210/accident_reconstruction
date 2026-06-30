import { useEffect, useMemo } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, ContactShadows, Environment } from "@react-three/drei";
import * as THREE from "three";
import type { Reconstruction } from "../types";
import { usePlayback } from "../playback/store";
import { Ground } from "./Ground";
import { Roads } from "./Roads";
import { Vehicle } from "./Vehicle";
import { ImpactMarker } from "./ImpactMarker";
import { GoogleTiles, GOOGLE_TILES_KEY } from "./GoogleTiles";
import { SplatScene, HAS_SPLAT } from "./SplatScene";

// Drives the shared playback clock once per rendered frame.
function PlaybackDriver() {
  const tick = usePlayback((s) => s.tick);
  useFrame((_, delta) => tick(delta));
  return null;
}

// Centre the camera target on the action (mean of all road/track points).
function useSceneCenter(data: Reconstruction): [number, number, number] {
  return useMemo(() => {
    const xs: number[] = [];
    const zs: number[] = [];
    for (const v of Object.values(data.vehicles))
      for (const s of v.track) {
        xs.push(s.x_m);
        zs.push(-s.z_m);
      }
    if (xs.length === 0) return [0, 0, 0];
    const mean = (a: number[]) => a.reduce((p, c) => p + c, 0) / a.length;
    return [mean(xs), 0, mean(zs)];
  }, [data]);
}

export function Scene({ data }: { data: Reconstruction }) {
  const center = useSceneCenter(data);
  const impactTime =
    data.impact?.frame != null ? data.impact.frame / data.fps : null;

  // Frame the action once on load (camera offset from the scene centre).
  const camera = useThree((s) => s.camera);
  useEffect(() => {
    camera.position.set(center[0] + 35, 45, center[2] + 50);
    camera.updateProjectionMatrix();
  }, [camera, center]);

  // Basemap precedence: Gaussian Splat (real capture) > Google 3D Tiles (GPS) >
  // placeholder ground.
  const useSplat = HAS_SPLAT;
  const useTiles = !useSplat && Boolean(GOOGLE_TILES_KEY && data.origin_latlon);
  const usePlaceholder = !useSplat && !useTiles;

  return (
    <>
      {!useTiles && (
        <>
          <color attach="background" args={["#15171c"]} />
          {usePlaceholder && (
            <fog attach="fog" args={["#15171c", 120, 320]} />
          )}
        </>
      )}

      {useSplat && <SplatScene />}

      {/* HDRI image-based lighting for realistic car paint/reflections. */}
      <Environment preset="city" />

      <hemisphereLight args={["#cdd6ff", "#2a2118", 0.6]} />
      <ambientLight intensity={0.25} />
      <directionalLight
        position={[40, 70, 20]}
        intensity={2.2}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-80}
        shadow-camera-right={80}
        shadow-camera-top={80}
        shadow-camera-bottom={-80}
        shadow-bias={-0.0002}
      />

      <PlaybackDriver />

      {useTiles && data.origin_latlon && (
        <GoogleTiles
          lat={data.origin_latlon[0]}
          lon={data.origin_latlon[1]}
        />
      )}

      <group>
        {usePlaceholder && <Ground />}
        <Roads roads={data.roads} />
        {Object.entries(data.vehicles).map(([id, v]) => (
          <Vehicle key={id} data={v} />
        ))}
        {data.impact && (
          <ImpactMarker impact={data.impact} impactTime={impactTime} />
        )}
        <ContactShadows
          position={[center[0], 0.02, center[2]]}
          scale={120}
          resolution={1024}
          blur={2.2}
          opacity={0.5}
          far={12}
        />
      </group>

      <OrbitControls
        target={new THREE.Vector3(...center)}
        maxPolarAngle={Math.PI / 2.05}
        minDistance={8}
        maxDistance={260}
        makeDefault
      />
    </>
  );
}
