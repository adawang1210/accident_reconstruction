import { Grid } from "@react-three/drei";

// Phase 0 placeholder ground. In Phase 2 this is replaced by Google
// Photorealistic 3D Tiles (3d-tiles-renderer) aligned to origin_latlon; the
// vehicles/roads/impact stay exactly as-is inside the same metre frame.
export function Ground() {
  return (
    <>
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[400, 400]} />
        <meshStandardMaterial color="#23262c" roughness={1} metalness={0} />
      </mesh>
      <Grid
        args={[400, 400]}
        position={[0, 0.01, 0]}
        cellSize={2}
        cellThickness={0.6}
        cellColor="#343843"
        sectionSize={20}
        sectionThickness={1}
        sectionColor="#474c58"
        fadeDistance={180}
        fadeStrength={1}
        infiniteGrid
      />
    </>
  );
}
