// What a tracked object actually is, so it isn't all drawn as a 4.4 m sedan.
// The backend labels vehicles inconsistently across scenes -- SAM2's English
// class names ("motorbike", "person") in some, Chinese display names ("機車")
// in others -- so match both the id and the name.

export type VehicleKind = "car" | "motorcycle" | "person";

const MOTORCYCLE = /motorcycle|motorbike|scooter|機車|摩托/i;
const PERSON = /person|pedestrian|行人|人/i;

/** Classify from the vehicle's backend id and display name. */
export function vehicleKind(id: string, name: string): VehicleKind {
  const s = `${id} ${name}`;
  if (MOTORCYCLE.test(s)) return "motorcycle";
  if (PERSON.test(s)) return "person";
  return "car";
}

/** Look-ahead used to derive heading, in seconds. Slower things turn tighter. */
export function headingLookahead(kind: VehicleKind): number {
  return kind === "person" ? 0.3 : 0.15;
}
