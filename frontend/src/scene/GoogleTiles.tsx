import { useContext, useEffect, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import {
  TilesPlugin,
  TilesRenderer,
  TilesRendererContext,
} from "3d-tiles-renderer/r3f";
import {
  GoogleCloudAuthPlugin,
  ReorientationPlugin,
  TilesFadePlugin,
} from "3d-tiles-renderer/plugins";

// Google Photorealistic 3D Tiles = the real intersection's photoreal 3D, from
// just GPS. Requires a Google Maps Platform API key (Map Tiles API enabled,
// billing on) supplied via VITE_GOOGLE_TILES_KEY.
export const GOOGLE_TILES_KEY = import.meta.env.VITE_GOOGLE_TILES_KEY as
  | string
  | undefined;

const DEG2RAD = Math.PI / 180;

// ReorientationPlugin re-centres the global tileset so the (lat, lon, height=0)
// surface point sits at the local origin. But "height = 0" is the *ellipsoid*,
// while Taipei's real ground is tens of metres higher (terrain + geoid), so the
// photoreal street ends up well above our origin and the cars sit buried
// underneath it. GroundProbe measures the true surface height under the origin
// by raycasting straight down onto the loaded tiles, so the scene content can be
// lifted onto the real street.
function GroundProbe({ onGround }: { onGround: (y: number) => void }) {
  const tiles = useContext(TilesRendererContext);
  const raycaster = useRef(new THREE.Raycaster());
  // Start high above the origin and look straight down.
  const from = useRef(new THREE.Vector3(0, 1e5, 0));
  const down = useRef(new THREE.Vector3(0, -1, 0));
  const last = useRef<number | null>(null);
  const stable = useRef(0);
  const done = useRef(false);
  const elapsed = useRef(0);

  useFrame((_, delta) => {
    if (done.current || !tiles?.group) return;
    // Throttle to a few probes per second; the cast is cheap but pointless every
    // frame.
    elapsed.current += delta;
    if (elapsed.current < 0.3) return;
    elapsed.current = 0;

    raycaster.current.set(from.current, down.current);
    const hits = raycaster.current.intersectObject(tiles.group, true);
    if (hits.length === 0) return;

    const y = hits[0].point.y;
    // Reject implausible hits: after reorientation the real ground sits within a
    // few hundred metres of the origin (terrain + geoid). Half-loaded or coarse
    // tiles can report wildly off-axis surfaces (tens of km), which would shove
    // the whole scene into the void -- ignore those.
    if (Math.abs(y) > 1000) return;
    if (last.current === null || Math.abs(y - last.current) > 0.1) {
      last.current = y;
      stable.current = 0;
      onGround(y);
    } else if (++stable.current > 10) {
      // Surface height has held steady across higher-LOD refinements: lock it in
      // and stop probing.
      done.current = true;
    }
  });

  return null;
}

export function GoogleTiles({
  lat,
  lon,
  onGround,
  onUnavailable,
  onLoaded,
}: {
  lat: number;
  lon: number;
  onGround?: (y: number) => void;
  onUnavailable?: () => void;
  onLoaded?: () => void;
}) {
  const healthy = useRef(false);
  const onUnavailableRef = useRef(onUnavailable);
  onUnavailableRef.current = onUnavailable;
  const onLoadedRef = useRef(onLoaded);
  onLoadedRef.current = onLoaded;
  useEffect(() => {
    // If nothing has loaded after a while, fall back to the grid rather than
    // churn forever (which spams errors and can lose the WebGL context).
    const id = window.setTimeout(() => {
      if (!healthy.current) onUnavailableRef.current?.();
    }, 12000);
    return () => window.clearTimeout(id);
  }, []);

  if (!GOOGLE_TILES_KEY) return null;
  return (
    <TilesRenderer
      // Refine to finer tiles than Google's default (errorTarget 20) for a
      // sharper street. Lower = sharper but more tile requests/bandwidth.
      errorTarget={6}
      onLoadModel={() => {
        if (!healthy.current) {
          healthy.current = true; // first real tile -> tiles are working
          onLoadedRef.current?.();
        }
      }}
    >
      <TilesPlugin
        plugin={GoogleCloudAuthPlugin}
        // autoRefreshToken: re-fetch the Google session token and retry on a
        // 4xx, instead of failing every tile that raced ahead of the token.
        // useRecommendedSettings:false so the plugin doesn't force errorTarget
        // back to 20 (we set our own, sharper, value on the renderer above).
        args={[
          {
            apiToken: GOOGLE_TILES_KEY,
            autoRefreshToken: true,
            useRecommendedSettings: false,
          },
        ]}
      />
      <TilesPlugin
        plugin={ReorientationPlugin}
        args={[{ lat: lat * DEG2RAD, lon: lon * DEG2RAD, height: 0 }]}
      />
      <TilesPlugin plugin={TilesFadePlugin} />
      {onGround && <GroundProbe onGround={onGround} />}
    </TilesRenderer>
  );
}
