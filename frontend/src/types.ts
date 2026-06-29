// Mirrors the backend's reconstruction.json (see docs/frontend_api.md).
// Coordinates share one local east/north metre plane: x_m = east, z_m = north.

export interface TrackSample {
  frame: number;
  t_sec: number;
  x_m: number;
  z_m: number;
  lat: number;
  lon: number;
  speed_kmh: number;
  is_impact: boolean;
}

export interface VehicleData {
  name: string;
  color_rgb: [number, number, number];
  road: string;
  track: TrackSample[];
}

export interface RoadPoint {
  x_m: number;
  z_m: number;
  lat: number;
  lon: number;
}

export interface ImpactData {
  frame: number | null;
  lat: number;
  lon: number;
  x_m: number;
  z_m: number;
}

export interface Reconstruction {
  scene: string;
  ready: boolean;
  reason?: string;
  fps: number;
  impact_frame: number | null;
  axes: string;
  origin_latlon: [number, number] | null;
  impact: ImpactData | null;
  vehicles: Record<string, VehicleData>;
  roads: Record<string, RoadPoint[]>;
  speed_reliability: { gcp_ground_span_m: number | null };
}
