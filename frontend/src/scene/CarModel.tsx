import { Suspense, useMemo } from "react";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import type { VehicleKind } from "./vehicleKind";

// Drop in a real car: set VITE_CAR_MODEL_URL to a .glb/.gltf (e.g. a CC0 model
// in public/models/). When unset, a clean procedural car is used as a stand-in.
const CAR_URL = import.meta.env.VITE_CAR_MODEL_URL as string | undefined;

function ProceduralCar({ color }: { color: THREE.Color }) {
  // Forward is +Z (matches Vehicle heading). Rough but PBR-shaded so it reads as
  // a car under HDRI + tone mapping rather than a box.
  return (
    <group>
      {/* lower body */}
      <mesh castShadow position={[0, 0.5, 0]}>
        <boxGeometry args={[1.9, 0.55, 4.4]} />
        <meshPhysicalMaterial
          color={color}
          metalness={0.7}
          roughness={0.3}
          clearcoat={1}
          clearcoatRoughness={0.1}
        />
      </mesh>
      {/* hood / trunk shoulders */}
      <mesh castShadow position={[0, 0.85, 0.2]}>
        <boxGeometry args={[1.78, 0.4, 3.4]} />
        <meshPhysicalMaterial
          color={color}
          metalness={0.7}
          roughness={0.3}
          clearcoat={1}
        />
      </mesh>
      {/* greenhouse / cabin (tinted glass) */}
      <mesh castShadow position={[0, 1.22, -0.25]}>
        <boxGeometry args={[1.5, 0.55, 1.9]} />
        <meshPhysicalMaterial
          color="#101418"
          metalness={0.2}
          roughness={0.05}
          transmission={0.6}
          transparent
          opacity={0.85}
        />
      </mesh>
      {/* wheels (axis along X) */}
      {(
        [
          [0.95, 0.34, 1.45],
          [-0.95, 0.34, 1.45],
          [0.95, 0.34, -1.45],
          [-0.95, 0.34, -1.45],
        ] as const
      ).map(([x, y, z], i) => (
        <mesh key={i} position={[x, y, z]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.34, 0.34, 0.28, 24]} />
          <meshStandardMaterial color="#0b0d10" roughness={0.85} metalness={0.1} />
        </mesh>
      ))}
      {/* headlights (front, +Z) */}
      {([0.6, -0.6] as const).map((x) => (
        <mesh key={x} position={[x, 0.6, 2.18]}>
          <boxGeometry args={[0.45, 0.18, 0.08]} />
          <meshStandardMaterial
            color="#ffffff"
            emissive="#fff7e0"
            emissiveIntensity={1.4}
          />
        </mesh>
      ))}
      {/* taillights (rear, -Z) */}
      {([0.62, -0.62] as const).map((x) => (
        <mesh key={x} position={[x, 0.62, -2.2]}>
          <boxGeometry args={[0.5, 0.16, 0.08]} />
          <meshStandardMaterial
            color="#ff2a2a"
            emissive="#ff1a1a"
            emissiveIntensity={1.6}
          />
        </mesh>
      ))}
    </group>
  );
}

function LoadedCar({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  // Clone so several vehicles can share one loaded model independently.
  const model = useMemo(() => {
    const c = scene.clone(true);
    c.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) o.castShadow = true;
    });
    return c;
  }, [scene]);
  return <primitive object={model} />;
}

// A scooter/motorbike: ~1.9 m long, ~0.7 m wide. Drawing one as a 4.4 m sedan
// (as every tracked object used to be) badly misreads the gap in a near-miss.
function ProceduralMotorcycle({ color }: { color: THREE.Color }) {
  return (
    <group>
      {/* body / seat */}
      <mesh castShadow position={[0, 0.62, -0.1]}>
        <boxGeometry args={[0.42, 0.34, 1.25]} />
        <meshPhysicalMaterial
          color={color}
          metalness={0.6}
          roughness={0.35}
          clearcoat={0.8}
        />
      </mesh>
      {/* front fairing */}
      <mesh castShadow position={[0, 0.85, 0.62]}>
        <boxGeometry args={[0.38, 0.5, 0.3]} />
        <meshPhysicalMaterial color={color} metalness={0.6} roughness={0.35} />
      </mesh>
      {/* handlebars */}
      <mesh castShadow position={[0, 1.05, 0.55]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.035, 0.035, 0.66, 12]} />
        <meshStandardMaterial color="#20242a" roughness={0.6} metalness={0.4} />
      </mesh>
      {/* rider */}
      <mesh castShadow position={[0, 1.15, -0.15]}>
        <capsuleGeometry args={[0.21, 0.5, 6, 12]} />
        <meshStandardMaterial color="#2f3540" roughness={0.85} />
      </mesh>
      {/* wheels (axis along X) */}
      {([0.72, -0.72] as const).map((z) => (
        <mesh key={z} position={[0, 0.31, z]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.31, 0.31, 0.13, 20]} />
          <meshStandardMaterial color="#0b0d10" roughness={0.85} metalness={0.1} />
        </mesh>
      ))}
      {/* headlight (front, +Z) */}
      <mesh position={[0, 0.88, 0.78]}>
        <boxGeometry args={[0.22, 0.14, 0.06]} />
        <meshStandardMaterial
          color="#ffffff"
          emissive="#fff7e0"
          emissiveIntensity={1.4}
        />
      </mesh>
    </group>
  );
}

// A pedestrian: roughly human-sized so the scale of a near-miss reads true.
function ProceduralPerson({ color }: { color: THREE.Color }) {
  return (
    <group>
      <mesh castShadow position={[0, 1.0, 0]}>
        <capsuleGeometry args={[0.24, 0.72, 6, 14]} />
        <meshStandardMaterial color={color} roughness={0.9} />
      </mesh>
      <mesh castShadow position={[0, 1.62, 0]}>
        <sphereGeometry args={[0.13, 16, 12]} />
        <meshStandardMaterial color="#d9b48f" roughness={0.9} />
      </mesh>
    </group>
  );
}

/**
 * The 3D stand-in for one tracked object.
 *
 * `VITE_CAR_MODEL_URL` only replaces the *car*; bikes and pedestrians keep the
 * procedural models, since a car glTF at their position would be worse than the
 * rough shape.
 */
export function CarModel({
  color,
  kind = "car",
}: {
  color: THREE.Color;
  kind?: VehicleKind;
}) {
  if (kind === "motorcycle") return <ProceduralMotorcycle color={color} />;
  if (kind === "person") return <ProceduralPerson color={color} />;
  if (CAR_URL)
    return (
      <Suspense fallback={<ProceduralCar color={color} />}>
        <LoadedCar url={CAR_URL} />
      </Suspense>
    );
  return <ProceduralCar color={color} />;
}

if (CAR_URL) useGLTF.preload(CAR_URL);
