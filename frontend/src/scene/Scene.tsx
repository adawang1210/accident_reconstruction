import { useEffect, useMemo, useRef, useState } from "react";
import type { ElementRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, ContactShadows } from "@react-three/drei";
import * as THREE from "three";
import type { Reconstruction, RoadPoint } from "../types";
import { usePlayback } from "../playback/store";
import { Ground } from "./Ground";
import { Roads } from "./Roads";
import { Vehicle } from "./Vehicle";
import { TrackPath } from "./TrackPath";
import { ImpactMarker } from "./ImpactMarker";
import { GoogleTiles, GOOGLE_TILES_KEY } from "./GoogleTiles";
import { CityBlocks } from "./CityBlocks";
import { OsmStreets } from "./OsmStreets";
import { SchematicStreets, SchematicGround } from "./SchematicStreets";
import { SplatScene, HAS_SPLAT } from "./SplatScene";
import { SafeEnvironment } from "./SafeEnvironment";
import { computeBounds, framingFor } from "./bounds";
import { useOsm } from "./osm";

// Which basemap to render under the reconstruction. "schematic" (default) =
// clean OSM extruded buildings + real OSM streets (sharp, free-orbit).
// "tiles" = Google photoreal 3D Tiles. A real Gaussian splat (VITE_SPLAT_URL)
// overrides both. Set via VITE_BASEMAP.
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

/** OSM basemap layers, split out so the fetch hook can run unconditionally. */
function SchematicBasemap({
  lat,
  lon,
  radius,
  roads,
}: {
  lat: number;
  lon: number;
  radius: number;
  roads: Record<string, RoadPoint[]>;
}) {
  const { data, settled } = useOsm(lat, lon, radius);
  const highways = data?.highways ?? [];
  // Only fall back to the pipeline's own centrelines once OSM has definitively
  // come back without streets -- otherwise both layers would flash on load.
  const useFallbackStreets = settled && highways.length === 0;

  return (
    <>
      {data && <CityBlocks ways={data.buildings} lat={lat} lon={lon} />}
      {highways.length > 0 && <OsmStreets ways={highways} lat={lat} lon={lon} />}
      {useFallbackStreets && <SchematicStreets roads={roads} />}
    </>
  );
}

export function Scene({ data }: { data: Reconstruction }) {
  const bounds = useMemo(() => computeBounds(data), [data]);
  const framing = useMemo(() => framingFor(bounds.radius), [bounds.radius]);
  const center = bounds.center;
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

  // The depth backdrop is a FRONTAL point cloud lying along -z from the origin
  // (the CCTV's own view direction), not a top-down map, so the trajectory
  // fly-down would stare past it into empty space. Look INTO the cloud instead.
  const SPLAT_TARGET: [number, number, number] = [0, 2, -20];
  const SPLAT_EYE: [number, number, number] = [0, 6, 16];

  // Replay the intro whenever the scene changes, so switching scenes reframes
  // on the new action instead of leaving the camera over the old one.
  useEffect(() => {
    introElapsed.current = 0;
    introDone.current = false;
  }, [data]);

  // Opening move: a high near-top-down look at the action easing down into a
  // low cinematic shot. Both ends keep the cars in frame, so the scene is never
  // staring into empty space while tiles stream (or fail to). Every distance
  // comes from the scene's own size (bounds.ts) -- the old fixed 150 m start
  // height framed the 60 m synthetic sample and left the real 5-30 m scenes as
  // a speck. Skipped for a splat backdrop, which gets a fixed frontal camera.
  useFrame((_, dt) => {
    if (introDone.current) return;
    const controls = controlsRef.current;

    if (useSplat) {
      camera.position.set(...SPLAT_EYE);
      camera.lookAt(...SPLAT_TARGET);
      camera.updateProjectionMatrix();
      if (controls) {
        controls.target.set(...SPLAT_TARGET);
        controls.enabled = true;
        controls.update();
      }
      introDone.current = true;
      return;
    }

    if (controls) controls.enabled = false; // don't fight the user mid-intro

    introElapsed.current += dt;
    const raw = Math.min(introElapsed.current / INTRO_SECONDS, 1);
    const t = raw * raw * (3 - 2 * raw); // smoothstep ease

    const cx = center[0];
    const cz = center[2];
    const gy = groundY; // follow the street height as the snap refines it
    // start high top-down (shows the layout) -> ease into a three-quarter
    // overview framed tight on the accident scene + immediate buildings.
    const { introStartY, introEndY, introOffset } = framing;
    camera.position.set(
      cx + introOffset * 0.55 * t,
      gy + introStartY - (introStartY - introEndY) * t,
      cz + introOffset * t,
    );
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

  // OrbitControls target sits on the real street -- or, for a splat backdrop,
  // inside the point cloud so orbiting pivots around the intersection.
  const target = useMemo(
    () =>
      useSplat
        ? new THREE.Vector3(...SPLAT_TARGET)
        : new THREE.Vector3(center[0], groundY, center[2]),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [center, groundY, useSplat],
  );

  return (
    <>
      {/* Always paint a background so the scene is never a black void, even if
          the photoreal tiles fail to authenticate/stream. */}
      <color attach="background" args={[useSchematic ? "#e9edf3" : "#10131a"]} />
      {/* Fade the basemap into the background so the (intentionally limited)
          range has a soft edge instead of a hard clip at the far plane. A real
          splat carries its own depth, so it is left un-fogged. Tiles stream
          city-wide context, so they get a much longer range than the
          scene-sized schematic basemap. */}
      {!useSplat && (
        <fog
          attach="fog"
          args={
            useTiles
              ? ["#10131a", 600, 1600]
              : useSchematic
                ? ["#e9edf3", framing.fogNear, framing.fogFar]
                : ["#15171c", framing.fogNear, framing.fogFar]
          }
        />
      )}

      {useSplat && <SplatScene />}

      {/* HDRI image-based lighting for realistic car paint/reflections, with a
          plain-light fallback if the HDRI CDN can't be reached. */}
      <SafeEnvironment />

      <hemisphereLight args={["#cdd6ff", "#2a2118", 0.6]} />
      <ambientLight intensity={0.25} />
      {/* Sun + shadow frustum both sized and centred on the action. The frustum
          used to be a fixed 80 m box at the world origin, so a scene offset
          from the origin had its shadows clipped away. */}
      <directionalLight
        position={[
          center[0] + bounds.radius,
          bounds.radius * 3.5,
          center[2] + bounds.radius * 0.6,
        ]}
        intensity={2.2}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-bounds.radius * 3}
        shadow-camera-right={bounds.radius * 3}
        shadow-camera-top={bounds.radius * 3}
        shadow-camera-bottom={-bounds.radius * 3}
        shadow-bias={-0.0002}
      />

      <PlaybackDriver />

      {useSchematic && data.origin_latlon && (
        <>
          <SchematicGround size={framing.groundSize} />
          <SchematicBasemap
            lat={data.origin_latlon[0]}
            lon={data.origin_latlon[1]}
            radius={framing.osmRadius}
            roads={data.roads}
          />
        </>
      )}

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
          <TrackPath key={`path-${id}`} data={v} />
        ))}
        {Object.entries(data.vehicles).map(([id, v]) => (
          <Vehicle key={id} id={id} data={v} />
        ))}
        {data.impact && (
          <ImpactMarker impact={data.impact} impactTime={impactTime} />
        )}
        <ContactShadows
          position={[center[0], 0.02, center[2]]}
          scale={framing.shadowScale}
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
        minDistance={framing.minDistance}
        maxDistance={framing.maxDistance}
        makeDefault
      />
    </>
  );
}
