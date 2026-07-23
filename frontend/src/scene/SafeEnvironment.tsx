import { Component, Suspense } from "react";
import type { ReactNode } from "react";
import { Environment } from "@react-three/drei";

// drei's <Environment preset="city"> streams an HDRI from a public CDN. When
// that request fails (offline, firewall, CDN outage) the loader error escapes
// React's error boundary into R3F's own Canvas boundary, which unmounts and
// remounts the whole <Canvas> -- forever. The observed damage: repeated
// "WebGLRenderer: Context Lost", and Scene remounting every frame so the intro
// camera move restarted continuously and never flew down.
//
// So the HDRI is opt-in. The default is a plain analytic light rig, which needs
// no network and suits the matte schematic/CAD basemap anyway. Set
// VITE_HDRI=city (or any drei preset) to get image-based lighting back for the
// photoreal tiles basemap, where car paint reflections actually pay off.
const HDRI_PRESET = import.meta.env.VITE_HDRI as string | undefined;

class ErrorBoundary extends Component<
  { children: ReactNode; fallback: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

/** Network-free fill lighting; stands in for the HDRI. */
function FallbackLighting() {
  return (
    <>
      <hemisphereLight args={["#dce6ff", "#4a4438", 0.75]} />
      <ambientLight intensity={0.35} />
    </>
  );
}

export function SafeEnvironment() {
  if (!HDRI_PRESET) return <FallbackLighting />;
  return (
    <ErrorBoundary fallback={<FallbackLighting />}>
      <Suspense fallback={<FallbackLighting />}>
        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
        <Environment preset={HDRI_PRESET as never} />
      </Suspense>
    </ErrorBoundary>
  );
}
