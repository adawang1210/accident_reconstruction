import { TilesPlugin, TilesRenderer } from "3d-tiles-renderer/r3f";
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

// ReorientationPlugin re-centres the global tileset so the given lat/lon surface
// point sits at the local origin with up = +Y -- which is exactly our scene's
// frame (origin = origin_latlon, x = east, z = -north, y = up). So the roads,
// vehicles and impact marker (already in metres) land on the real ground with no
// extra transform; the existing camera/controls stay as-is.
export function GoogleTiles({ lat, lon }: { lat: number; lon: number }) {
  if (!GOOGLE_TILES_KEY) return null;
  return (
    <TilesRenderer>
      <TilesPlugin
        plugin={GoogleCloudAuthPlugin}
        args={[{ apiToken: GOOGLE_TILES_KEY }]}
      />
      <TilesPlugin
        plugin={ReorientationPlugin}
        args={[{ lat: lat * DEG2RAD, lon: lon * DEG2RAD, height: 0 }]}
      />
      <TilesPlugin plugin={TilesFadePlugin} />
    </TilesRenderer>
  );
}
