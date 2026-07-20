import { useEffect, useMemo, useRef, useState } from "react";
import type { ElementRef } from "react";
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
import { CityBlocks } from "./CityBlocks";
import { SchematicStreets } from "./SchematicStreets";
import { SplatScene, HAS_SPLAT } from "./SplatScene";

// Which basemap to render under the reconstruction. "schematic" (default) =
// clean OSM extruded buildings (sharp, free-orbit). "tiles" = Google photoreal
// 3D Tiles. A real Gaussian splat (VITE_SPLAT_URL) overrides both. Set via
// VITE_BASEMAP.
export const BASEMAP =
  (import.meta.env.VITE_BASEMAP as string | undefined) ?? "schematic";

// Drives the shared playback clock once per rendered frame.
function PlaybackDriver() {
  const tick = usePlayback((s) => s.tick);
  useFrame((_, delta) => tick(delta));
  return null;
}

// Duration of the opening fly-down (top-down over the action -> ground level).
const INTRO_SECONDS = 3.5;

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

  // The Google tiles place the real street at its true ellipsoid height (tens of
  // metres up); GroundProbe measures it so we can lift the whole scene onto it.
  const [groundY, setGroundY] = useState(0);
  // True once the photoreal tiles actually snap a surface under the action.
  // Until then we keep the fallback grid so the scene is never empty.
  const [tilesReady, setTilesReady] = useState(false);
  // Set if the tiles give up (auth/stream failure) -- we then drop them and
  // keep the grid so the reconstruction stays visible and stops crashing.
  const [tilesFailed, setTilesFailed] = useState(false);
  const appliedGroundY = useRef(0);

  const camera = useThree((s) => s.camera);
  const controlsRef = useRef<ElementRef<typeof OrbitControls>>(null);
  const introElapsed = useRef(0);
  const introDone = useRef(false);

  // Basemap precedence: real Gaussian Splat capture (VITE_SPLAT_URL) > Google
  // photoreal 3D Tiles (only when explicitly selected + available) > clean
  // schematic OSM basemap > placeholder grid.
  const useSplat = HAS_SPLAT;
  const useTiles =
    !useSplat &&
    BASEMAP === "tiles" &&
    Boolean(GOOGLE_TILES_KEY && data.origin_latlon);
  const useSchematic = !useSplat && !useTiles && Boolean(data.origin_latlon);

  // Opening move: a high near-top-down look at the action easing down into a
  // low cinematic shot. Both ends keep the cars in frame, so the scene is never
  // staring into empty space while tiles stream (or fail to).
  useFrame((_, dt) => {
    if (introDone.current) return;
    const controls = controlsRef.current;
    if (controls) controls.enabled = false; // don't fight the user mid-intro

    introElapsed.current += dt;
    const raw = Math.min(introElapsed.current / INTRO_SECONDS, 1);
    const t = raw * raw * (3 - 2 * raw); // smoothstep ease

    const cx = center[0];
    const cz = center[2];
    const gy = groundY; // follow the street height as the snap refines it
    // start high top-down (shows the layout) -> ease into a three-quarter
    // overview framed tight on the accident scene + immediate buildings.
    camera.position.set(cx + 12 * t, gy + 150 - 95 * t, cz + 22 * t);
    camera.lookAt(cx, gy, cz);
    camera.updateProjectionMatrix();
    appliedGroundY.current = gy;
    if (controls) {
      controls.target.set(cx, gy, cz);
      controls.update();
    }

    if (raw >= 1) {
      introDone.current = true;
      if (controls) controls.enabled = true;
    }
  });

  // After the intro, if the probed ground height refines further, ride the
  // camera + target up with it so the framing stays glued to the street.
  useEffect(() => {
    if (!introDone.current) return;
    const delta = groundY - appliedGroundY.current;
    if (delta === 0) return;
    appliedGroundY.current = groundY;
    camera.position.y += delta;
    camera.updateProjectionMatrix();
    if (controlsRef.current) {
      controlsRef.current.target.y += delta;
      controlsRef.current.update();
    }
  }, [camera, groundY]);

  // OrbitControls target sits on the real street.
  const target = useMemo(
    () => new THREE.Vector3(center[0], groundY, center[2]),
    [center, groundY],
  );

  return (
    <>
      {/* Always paint a background so the scene is never a black void, even if
          the photoreal tiles fail to authenticate/stream. */}
      <color attach="background" args={[useSchematic ? "#e9edf3" : "#10131a"]} />
      {/* Fade the basemap into the background so the (intentionally limited)
          range has a soft edge instead of a hard clip at the far plane. A real
          splat carries its own depth, so it is left un-fogged. */}
      {!useSplat && (
        <fog
          attach="fog"
          args={
            useTiles
              ? ["#10131a", 600, 1600]
              : useSchematic
                ? ["#e9edf3", 150, 360]
                : ["#15171c", 120, 320]
          }
        />
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

      {useSchematic && data.origin_latlon && (
        <CityBlocks lat={data.origin_latlon[0]} lon={data.origin_latlon[1]} />
      )}
      {useSchematic && <SchematicStreets roads={data.roads} />}

      {useTiles && !tilesFailed && data.origin_latlon && (
        <GoogleTiles
          lat={data.origin_latlon[0]}
          lon={data.origin_latlon[1]}
          onGround={(y) => setGroundY(y)}
          onLoaded={() => setTilesReady(true)}
          onUnavailable={() => {
            setTilesFailed(true);
            // Undo any (possibly bad) snap so the fallback grid sits at y=0.
            setGroundY(0);
            setTilesReady(false);
          }}
        />
      )}

      <group position={[0, groundY, 0]}>
        {!useSchematic && !useSplat && !tilesReady && <Ground />}
        {!useSchematic && <Roads roads={data.roads} />}
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
        ref={controlsRef}
        target={target}
        maxPolarAngle={Math.PI / 2.05}
        minDistance={8}
        maxDistance={260}
        makeDefault
      />
    </>
  );
}
