import { useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { Scene } from "./scene/Scene";
import { Timeline } from "./ui/Timeline";
import { useReconstruction } from "./io/useReconstruction";
import { usePlayback } from "./playback/store";
import { trackEnd, trackStart } from "./scene/sampleTrack";
import { HAS_SPLAT } from "./scene/SplatScene";
import { GOOGLE_TILES_KEY } from "./scene/GoogleTiles";
import { BASEMAP } from "./scene/Scene";

export function App() {
  const { data, error, scenes, sceneId, selectScene } = useReconstruction();
  const setDuration = usePlayback((s) => s.setDuration);
  const setTime = usePlayback((s) => s.setTime);

  // Seek to the first frame any vehicle was actually tracked. The tracks don't
  // start at t=0 -- keelung_xinwu_yier's first sample is frame 120 (4.1 s) --
  // so starting the clock at zero opened on an empty street and made the
  // reconstruction look broken until you scrubbed past the dead air.
  useEffect(() => {
    if (!data) return;
    let duration = 0;
    let firstSeen = Infinity;
    for (const v of Object.values(data.vehicles)) {
      duration = Math.max(duration, trackEnd(v.track));
      firstSeen = Math.min(firstSeen, trackStart(v.track));
    }
    setDuration(duration);
    setTime(Number.isFinite(firstSeen) ? firstSeen : 0);
  }, [data, setDuration, setTime]);

  if (error)
    return (
      <div className="fatal">
        無法載入重建結果：
        <br />
        {error}
        <br />
        <br />
        請先跑後端 pipeline，再執行 <code>npm run sync:scenes</code> 把{" "}
        <code>data/&lt;場景&gt;/…_reconstruction.json</code> 複製進{" "}
        <code>public/scenes/</code>，或設定 <code>VITE_RECONSTRUCTION_URL</code>
        （見 README）。
      </div>
    );
  if (!data) return <div className="fatal">載入中…</div>;

  const span = data.speed_reliability?.gcp_ground_span_m;
  // Only Google 3D Tiles (ECEF scale) needs the huge far plane + log depth
  // buffer; splat and the schematic OSM basemap are local-metre scenes. Must
  // mirror Scene.tsx's tiles gate (splat overrides, and BASEMAP must be "tiles").
  const useTiles =
    !HAS_SPLAT &&
    BASEMAP === "tiles" &&
    Boolean(GOOGLE_TILES_KEY && data.origin_latlon);
  return (
    <>
      <div className="hud">
        {scenes.length > 1 ? (
          <label className="scene-pick">
            <b>場景</b>{" "}
            <select
              value={sceneId ?? ""}
              onChange={(e) => selectScene(e.target.value)}
            >
              {scenes.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.id}（{s.duration_sec.toFixed(1)} s）
                </option>
              ))}
            </select>
          </label>
        ) : (
          <div>
            <b>場景</b> {data.scene}
          </div>
        )}
        <div>
          <b>車輛</b>{" "}
          {Object.values(data.vehicles)
            .map((v) => v.name)
            .join("、")}
        </div>
        <div>
          <b>fps</b> {data.fps} ｜ <b>撞擊幀</b> {data.impact_frame ?? "—"}
        </div>
        {data.origin_latlon && (
          <div className="dim">
            <b>原點</b> {data.origin_latlon[0].toFixed(6)},{" "}
            {data.origin_latlon[1].toFixed(6)}
          </div>
        )}
        {span != null && span < 30 && (
          <div className="warn">
            ⚠ GCP 涵蓋僅 ~{span.toFixed(0)} m，車速為估計值（可能偏低）
          </div>
        )}
      </div>
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{
          position: [40, 45, 60],
          fov: 50,
          near: 0.5,
          far: useTiles ? 5_000_000 : 5000,
        }}
        gl={{
          antialias: true,
          // Only Google 3D Tiles needs the huge ECEF depth range; it breaks the
          // Gaussian-splat shader, so enable it solely for tiles.
          logarithmicDepthBuffer: useTiles,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.0,
        }}
      >
        <Scene data={data} />
      </Canvas>
      <Timeline />
    </>
  );
}
