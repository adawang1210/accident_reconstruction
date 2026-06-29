import { useEffect } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { Scene } from "./scene/Scene";
import { Timeline } from "./ui/Timeline";
import { useReconstruction } from "./io/useReconstruction";
import { usePlayback } from "./playback/store";
import { trackEnd } from "./scene/sampleTrack";

export function App() {
  const { data, error } = useReconstruction();
  const setDuration = usePlayback((s) => s.setDuration);
  const setTime = usePlayback((s) => s.setTime);

  useEffect(() => {
    if (!data) return;
    let duration = 0;
    for (const v of Object.values(data.vehicles))
      duration = Math.max(duration, trackEnd(v.track));
    setDuration(duration);
    setTime(0);
  }, [data, setDuration, setTime]);

  if (error)
    return (
      <div className="fatal">
        無法載入 reconstruction.json：
        <br />
        {error}
        <br />
        <br />
        請先跑後端 pipeline 產生，或把檔放到 <code>public/reconstruction.json</code>
        ，或設定 <code>VITE_RECONSTRUCTION_URL</code>（見 README）。
      </div>
    );
  if (!data) return <div className="fatal">載入中…</div>;

  const span = data.speed_reliability?.gcp_ground_span_m;
  return (
    <>
      <div className="hud">
        <div>
          <b>場景</b> {data.scene}
        </div>
        <div>
          <b>車輛</b>{" "}
          {Object.values(data.vehicles)
            .map((v) => v.name)
            .join("、")}
        </div>
        <div>
          <b>fps</b> {data.fps} ｜ <b>撞擊幀</b> {data.impact_frame ?? "—"}
        </div>
        {span != null && span < 30 && (
          <div className="warn">
            ⚠ GCP 涵蓋僅 ~{span.toFixed(0)} m，車速為估計值（可能偏低）
          </div>
        )}
      </div>
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [40, 45, 60], fov: 50, near: 0.5, far: 5_000_000 }}
        gl={{
          antialias: true,
          logarithmicDepthBuffer: true,
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
