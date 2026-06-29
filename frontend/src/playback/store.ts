import { create } from "zustand";

// One shared clock for every vehicle, so they stay in sync automatically.
// currentTime is in SECONDS (matches each track sample's t_sec).
interface PlaybackState {
  currentTime: number;
  duration: number;
  playing: boolean;
  rate: number;
  setTime: (t: number) => void;
  setDuration: (d: number) => void;
  play: () => void;
  pause: () => void;
  toggle: () => void;
  setRate: (r: number) => void;
  /** Advance the clock by a real-time delta (seconds); pauses at the end. */
  tick: (delta: number) => void;
}

const clamp = (t: number, max: number) => Math.max(0, Math.min(t, max));

export const usePlayback = create<PlaybackState>((set, get) => ({
  currentTime: 0,
  duration: 0,
  playing: false,
  rate: 1,
  setTime: (t) => set({ currentTime: clamp(t, get().duration) }),
  setDuration: (d) => set({ duration: d }),
  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),
  toggle: () => set((s) => ({ playing: !s.playing })),
  setRate: (r) => set({ rate: r }),
  tick: (delta) => {
    const s = get();
    if (!s.playing || s.duration <= 0) return;
    let t = s.currentTime + delta * s.rate;
    if (t >= s.duration) {
      set({ currentTime: s.duration, playing: false });
      return;
    }
    set({ currentTime: t });
  },
}));
