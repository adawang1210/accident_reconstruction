import { usePlayback } from "../playback/store";

const RATES = [0.25, 0.5, 1] as const;
const fmt = (t: number) => `${t.toFixed(2)}s`;

export function Timeline() {
  const currentTime = usePlayback((s) => s.currentTime);
  const duration = usePlayback((s) => s.duration);
  const playing = usePlayback((s) => s.playing);
  const rate = usePlayback((s) => s.rate);
  const setTime = usePlayback((s) => s.setTime);
  const toggle = usePlayback((s) => s.toggle);
  const setRate = usePlayback((s) => s.setRate);

  return (
    <div className="timeline">
      <button onClick={toggle} title={playing ? "暫停" : "播放"}>
        {playing ? "⏸" : "▶"}
      </button>
      <input
        type="range"
        min={0}
        max={duration || 1}
        step={0.01}
        value={currentTime}
        onChange={(e) => setTime(parseFloat(e.target.value))}
      />
      <span className="time">
        {fmt(currentTime)} / {fmt(duration)}
      </span>
      {RATES.map((r) => (
        <button
          key={r}
          className={rate === r ? "active" : ""}
          onClick={() => setRate(r)}
        >
          {r}×
        </button>
      ))}
    </div>
  );
}
