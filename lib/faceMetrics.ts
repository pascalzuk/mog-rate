import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { analyzeSkin, type SkinReport } from '@/lib/skinAnalysis';
import { flipSymmetry, type FlipReport } from '@/lib/symmetry';

export type MetricKey =
  | 'symmetry'
  | 'jawline'
  | 'skin'
  | 'puffiness'
  | 'thirds'
  | 'midface'
  | 'canthal'
  | 'eyes'
  | 'fwhr'
  | 'lips';

export type Metric = {
  key: MetricKey;
  label: string;
  /** 1.0 - 9.6, shown in the breakdown. Not padded: 3.4 means 3.4. */
  rating: number;
  /** raw 0..1 quality */
  score: number;
};

export type Pose = { yaw: number; roll: number; pitch: number; ok: boolean };

export type Flag = {
  key: string;
  label: string;
  detail: string;
  severity: 1 | 2 | 3;
  kind: 'face' | 'capture';
};

export type SkinSummary = {
  clarity: number;
  texture: number;
  underEyes: number;
  blemishShare: number;
  spots: number;
  measured: boolean;
  regions: Record<string, number>;
};

export type RawMeasures = {
  symmetryDev: number;
  flipDifference: number;
  jawRatio: number;
  gonion: number;
  chinProjection: number | null;
  thirdsRatio: number;
  midfaceRatio: number;
  tilt: number;
  aperture: number;
  fwhr: number;
  lipRatio: number;
  fullness: number;
};

export type ScanResult = {
  score: number;
  tier: string;
  tierNote: string;
  verdict: string;
  metrics: Metric[];
  flags: Flag[];
  pose: Pose;
  confidence: number;
  skin: SkinSummary;
  raw: RawMeasures;
  /** frames that contributed to the final frontal read */
  frames: number;
  /** guided views captured before the read */
  views: number;
};

type Pt = { x: number; y: number };
type Pt3 = { x: number; y: number; z: number };

const L = {
  lipTop: 0,
  noseTip: 1,
  subnasale: 2,
  glabella: 9,
  forehead: 10,
  upperInner: 13,
  lowerInner: 14,
  lipBottom: 17,
  eyeOuterR: 33,
  mouthR: 61,
  browR: 105,
  noseR: 129,
  eyeInnerR: 133,
  lidBottomR: 145,
  chin: 152,
  lidTopR: 159,
  jawR: 172,
  cheekR: 234,
  eyeOuterL: 263,
  mouthL: 291,
  browL: 334,
  noseL: 358,
  eyeInnerL: 362,
  lidBottomL: 374,
  lidTopL: 386,
  jawL: 397,
  cheekL: 454,
  irisR: 468,
  irisL: 473,
} as const;

/**
 * Weights follow how much of the face a measurement actually describes, and how
 * much evidence sits behind the ideal. Jaw width from a flat photo is the least
 * reliable number here (photo bigonial width barely agrees with callipers), so
 * it is not allowed to dominate.
 */
const WEIGHTS: Record<MetricKey, number> = {
  symmetry: 0.15,
  jawline: 0.14,
  skin: 0.17,
  puffiness: 0.07,
  thirds: 0.08,
  midface: 0.09,
  canthal: 0.08,
  eyes: 0.07,
  fwhr: 0.07,
  lips: 0.08,
};

const LABELS: Record<MetricKey, string> = {
  symmetry: 'Symmetry',
  jawline: 'Jawline',
  skin: 'Skin',
  puffiness: 'Water retention',
  thirds: 'Proportions',
  midface: 'Midface',
  canthal: 'Canthal tilt',
  eyes: 'Eye area',
  fwhr: 'Face width',
  lips: 'Lips',
};

const ORDER: MetricKey[] = [
  'symmetry',
  'jawline',
  'skin',
  'puffiness',
  'thirds',
  'midface',
  'canthal',
  'eyes',
  'fwhr',
  'lips',
];

const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a: Pt, b: Pt): Pt => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const deg = (rad: number) => (rad * 180) / Math.PI;
const round1 = (v: number) => Math.round(v * 10) / 10;
const median = (values: number[]) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};

const dist3 = (a: Pt3, b: Pt3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const depthMid = (a: Pt3, b: Pt3): Pt3 => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });

function band(value: number, lo: number, hi: number, soft: number) {
  if (!Number.isFinite(value)) return 0;
  if (value >= lo && value <= hi) return 1;
  const d = value < lo ? lo - value : value - hi;
  return clamp01(1 - d / soft);
}

function angleAt(a: Pt, b: Pt, c: Pt) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const denom = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y) || 1;
  const cos = (v1.x * v2.x + v1.y * v2.y) / denom;
  return deg(Math.acos(Math.min(1, Math.max(-1, cos))));
}

const PAIRS: Array<[number, number, number]> = [
  [33, 263, 1.2],
  [133, 362, 1.1],
  [129, 358, 0.8],
  [105, 334, 0.9],
  [234, 454, 1.0],
  [172, 397, 0.9],
  // What people actually notice: one eyelid sitting lower, one brow riding
  // higher, the outer brow line. These were missing, which let faces that read
  // clearly uneven in a photo score as even.
  [159, 386, 1.2],
  [145, 374, 1.1],
  [109, 338, 0.6],
];

/** The nose tip is checked against the axis rather than paired: a crooked nose
 *  is one of the most visible asymmetries and has no second landmark. */
const NOSE_AXIS_WEIGHT = 2;

const PAIR_WEIGHT = PAIRS.reduce((sum, [, , w]) => sum + w, 0);

export type HeadPose = { yaw: number; roll: number; pitch: number };

/**
 * Fast pose estimate, used both by the guided capture and by the measurement
 * pass. Yaw comes from solving the head turn that best mirrors the paired
 * landmarks in depth; pitch comes from how far the chin sits in front of the
 * forehead, which is what tipping the head actually changes.
 */
export function estimatePose(landmarks: NormalizedLandmark[], aspect = 1): HeadPose {
  if (!landmarks || landmarks.length < 468) return { yaw: 0, roll: 0, pitch: 0 };
  const lm = landmarks;
  const eyeR = mid(lm[L.eyeOuterR], lm[L.eyeInnerR]);
  const eyeL = mid(lm[L.eyeInnerL], lm[L.eyeOuterL]);
  const roll = Math.atan2(eyeL.y - eyeR.y, (eyeL.x - eyeR.x) * aspect || 1e-6);

  const eyeMidX = (lm[L.eyeOuterR].x + lm[L.eyeOuterL].x) / 2;
  const faceW = Math.max(0.02, Math.hypot(lm[L.cheekR].x - lm[L.cheekL].x, lm[L.cheekR].y - lm[L.cheekL].y));

  const mirrorError = (theta: number) => {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    let sum = 0;
    for (const [a, b, w] of PAIRS) {
      const xa = eyeMidX + (lm[a].x - eyeMidX) * c + lm[a].z * s;
      const xb = eyeMidX + (lm[b].x - eyeMidX) * c + lm[b].z * s;
      sum += Math.abs(2 * eyeMidX - xa - xb) * w;
    }
    return sum / (PAIR_WEIGHT * faceW);
  };

  let yaw = 0;
  let best = Number.POSITIVE_INFINITY;
  for (let d = -75; d <= 75; d += 1.5) {
    const theta = (d * Math.PI) / 180;
    const err = mirrorError(theta);
    if (err < best) {
      best = err;
      yaw = theta;
    }
  }

  const dy = lm[L.chin].y - lm[L.forehead].y;
  const dz = lm[L.forehead].z - lm[L.chin].z;
  const pitch = Math.atan2(dz * aspect, Math.max(0.04, dy));

  return { yaw: deg(yaw), roll: deg(roll), pitch: deg(pitch) };
}

type Measures = {
  scores: Record<MetricKey, number>;
  skin: SkinReport | null;
  flip: FlipReport | null;
  pose: HeadPose;
  symmetryDev: number;
  jawRatio: number;
  gonion: number;
  chinProjection: number | null;
  thirdsRatio: number;
  midfaceRatio: number;
  tilt: number;
  aperture: number;
  fwhr: number;
  fullness: number;
  lipRatio: number;
};

export type Frame = {
  landmarks: NormalizedLandmark[];
  width: number;
  height: number;
  pixels?: ImageData;
  /** a turned view, used for chin projection */
  profile?: { landmarks: NormalizedLandmark[]; width: number; height: number };
};

function chinProjectionFrom(profile: { landmarks: NormalizedLandmark[] }): number | null {
  const lm = profile.landmarks;
  if (!lm || lm.length < 468) return null;
  const nose = lm[L.noseTip];
  const sub = lm[L.subnasale];
  const chin = lm[L.chin];
  const glabella = lm[L.glabella];
  const faceHeight = Math.max(0.05, chin.y - glabella.y);
  const forward = Math.sign(nose.x - sub.x) || 1;
  const projection = ((chin.x - sub.x) * forward) / faceHeight;
  return Math.abs(projection) < 0.3 ? projection : null;
}

export function symmetryScore(deviation: number, flipDifference: number | null): number {
  // Landmarks are the reliable signal. A perfectly even face lands near 0.010;
  // 0.035 is noticeable in person; 0.060 reads as clearly uneven.
  const geom = clamp01(1 - Math.max(0, deviation - 0.01) / 0.05);
  if (flipDifference === null) return geom;
  // The pixel flip only corroborates: at this crop size even a mirrored control
  // face shows about 0.40 of difference from resampling, so 0.40 is the floor.
  const flip = clamp01(1 - Math.max(0, flipDifference - 0.4) / 0.45);
  return 0.7 * geom + 0.3 * flip;
}

export function measure(frame: Frame): Measures {
  const { landmarks, width, height } = frame;
  if (!landmarks || landmarks.length < 468) throw new Error('Face not fully in frame');

  const aspect = width / height;
  const rawPoints: Pt3[] = landmarks.map((l) => ({ x: l.x * width, y: l.y * height, z: l.z * width }));
  const pose = estimatePose(landmarks, aspect);
  const rollRad = (-pose.roll * Math.PI) / 180;
  // estimatePose returns the corrective yaw: the rotation that makes mirrored
  // pairs line up. Applying its inverse here would double the head turn and
  // report a symmetric face as crooked.
  const yawRad = (pose.yaw * Math.PI) / 180;

  const eyeAnchorR = mid(rawPoints[L.eyeOuterR], rawPoints[L.eyeInnerR]);
  const eyeAnchorL = mid(rawPoints[L.eyeInnerL], rawPoints[L.eyeOuterL]);
  const pivot = mid(eyeAnchorR, eyeAnchorL);
  const cosRoll = Math.cos(rollRad);
  const sinRoll = Math.sin(rollRad);
  const rolled: Pt3[] = rawPoints.map(({ x, y, z }) => {
    const dx = x - pivot.x;
    const dy = y - pivot.y;
    return { x: pivot.x + dx * cosRoll - dy * sinRoll, y: pivot.y + dx * sinRoll + dy * cosRoll, z };
  });

  const eyeMidX = (rolled[L.eyeOuterR].x + rolled[L.eyeOuterL].x) / 2;
  const rawFaceWidth = dist(rolled[L.cheekR], rolled[L.cheekL]) || 1;
  const cosYaw = Math.cos(yawRad);
  const sinYaw = Math.sin(yawRad);
  const P: Pt[] = rolled.map(({ x, y, z }) => ({
    x: eyeMidX + (x - eyeMidX) * cosYaw + z * sinYaw,
    y,
  }));
  const p = (i: number) => P[i] ?? P[0];

  const faceWidth = dist(p(L.cheekR), p(L.cheekL)) || 1;
  const eyeCenterR = mid(p(L.eyeOuterR), p(L.eyeInnerR));
  const eyeCenterL = mid(p(L.eyeInnerL), p(L.eyeOuterL));

  // Geometric asymmetry: mirrored pairs against the mirror axis, weighted, with
  // the mouth left out so a smile is not scored as a crooked face.
  //
  // The axis is fitted from the pairs themselves rather than assumed to be the
  // eye midpoint. A few pixels of error in that assumption used to shift every
  // pair the same way and read as 5% asymmetry on a perfectly even face; a
  // best-fit axis makes the measure read only the shape difference.
  const mirroredDeviation = (() => {
    let axis = 0;
    let total = 0;
    for (const [a, b, w] of PAIRS) {
      axis += ((p(a).x + p(b).x) / 2) * w;
      total += w;
    }
    axis /= total;
    let sum = Math.abs(p(L.noseTip).x - axis) * NOSE_AXIS_WEIGHT;
    for (const [a, b, w] of PAIRS) {
      sum += Math.abs(2 * axis - (p(a).x + p(b).x)) * w;
    }
    return sum / ((PAIR_WEIGHT + NOSE_AXIS_WEIGHT) * rawFaceWidth);
  })();
  let devY = 0;
  for (const [a, b, w] of PAIRS) devY += Math.abs(p(a).y - p(b).y) * w;
  devY /= PAIR_WEIGHT * faceWidth;

  if (typeof window !== 'undefined' && (window as unknown as { __MOG_DEBUG?: boolean }).__MOG_DEBUG) {
    (window as unknown as { __mogSym?: unknown }).__mogSym = {
      devX: +mirroredDeviation.toFixed(4),
      devY: +devY.toFixed(4),
      yaw: +pose.yaw.toFixed(1),
    };
  }

  const symmetryDev = mirroredDeviation + devY * 0.5;

  const flip =
    frame.pixels && frame.pixels.width > 0
      ? flipSymmetry(frame.pixels, landmarks, width, height)
      : null;
  const symmetry = symmetryScore(symmetryDev, flip?.difference ?? null);

  const jawRatio = dist(p(L.jawR), p(L.jawL)) / faceWidth;
  const gonion =
    (angleAt(p(L.cheekR), p(L.jawR), p(L.chin)) + angleAt(p(L.cheekL), p(L.jawL), p(L.chin))) / 2;
  const chinProjection = frame.profile ? chinProjectionFrom(frame.profile) : null;

  const jawWidthBand = band(jawRatio, 0.84, 0.94, 0.1);
  const gonionBand = band(gonion, 118, 132, 16);
  const chinBand = chinProjection === null ? null : band(chinProjection, -0.02, 0.03, 0.05);
  const jawline = chinBand === null ? 0.6 * jawWidthBand + 0.4 * gonionBand : 0.45 * jawWidthBand + 0.3 * gonionBand + 0.25 * chinBand;

  const middleThird = Math.max(1, dist3(rawPoints[L.subnasale], rawPoints[L.glabella]));
  const lowerThird = Math.max(1, dist3(rawPoints[L.chin], rawPoints[L.subnasale]));
  const thirdsRatio = lowerThird / middleThird;
  const thirds = band(thirdsRatio, 0.98, 1.2, 0.16);

  const ipd3 =
    rawPoints.length > 473
      ? dist3(rawPoints[L.irisR], rawPoints[L.irisL])
      : dist(eyeCenterR, eyeCenterL);
  const midfaceHeight = Math.max(
    1,
    dist3(rawPoints[L.upperInner], depthMid(rawPoints[L.eyeOuterR], rawPoints[L.eyeOuterL])),
  );
  const midfaceRatio = ipd3 / midfaceHeight;
  const midface = band(midfaceRatio, 1.0, 1.12, 0.18);

  const tiltR = deg(Math.atan2(p(L.eyeInnerR).y - p(L.eyeOuterR).y, Math.abs(p(L.eyeInnerR).x - p(L.eyeOuterR).x)));
  const tiltL = deg(Math.atan2(p(L.eyeInnerL).y - p(L.eyeOuterL).y, Math.abs(p(L.eyeInnerL).x - p(L.eyeOuterL).x)));
  const tilt = (tiltR + tiltL) / 2;
  const canthal = band(tilt, 2.5, 6.5, 7);

  const eyeWR = dist(p(L.eyeOuterR), p(L.eyeInnerR)) || 1;
  const eyeWL = dist(p(L.eyeInnerL), p(L.eyeOuterL)) || 1;
  const aperture =
    (Math.abs(p(L.lidBottomR).y - p(L.lidTopR).y) / eyeWR +
      Math.abs(p(L.lidBottomL).y - p(L.lidTopL).y) / eyeWL) /
    2;
  const innerGap = dist(p(L.eyeInnerR), p(L.eyeInnerL)) / ((eyeWR + eyeWL) / 2);
  const eyes = 0.7 * band(aperture, 0.26, 0.35, 0.12) + 0.3 * band(innerGap, 0.95, 1.1, 0.18);

  const fwhr =
    dist3(rawPoints[L.cheekR], rawPoints[L.cheekL]) /
    Math.max(1, dist3(rawPoints[L.upperInner], depthMid(rawPoints[L.browR], rawPoints[L.browL])));
  const fwhrScore = band(fwhr, 1.78, 2.02, 0.22);

  // Lower-face fullness: jaw width against the length of the face. A puffy or
  // water-retained lower face pushes this up; it is a soft signal, not a body
  // composition measurement.
  const faceLength = Math.max(1, dist3(rawPoints[L.chin], depthMid(rawPoints[L.browR], rawPoints[L.browL])));
  const fullness = dist3(rawPoints[L.jawR], rawPoints[L.jawL]) / faceLength;
  const puffiness = band(fullness, 0.78, 0.9, 0.11);

  const lipRatio =
    Math.max(0, p(L.upperInner).y - p(L.lipTop).y) / Math.max(1, p(L.lipBottom).y - p(L.lowerInner).y);
  const mouthNose = dist(p(L.mouthR), p(L.mouthL)) / Math.max(1, dist(p(L.noseR), p(L.noseL)));
  const lips = 0.55 * band(lipRatio, 0.45, 0.62, 0.16) + 0.45 * band(mouthNose, 1.45, 1.62, 0.2);

  const skin = frame.pixels ? analyzeSkin(frame.pixels, landmarks, width, height) : null;
  const skinScore = skin ? clamp01(0.72 * skin.clarity + 0.18 * skin.texture + 0.1 * skin.underEyes) : 0;

  return {
    scores: {
      symmetry,
      jawline,
      skin: skinScore,
      puffiness,
      thirds,
      midface,
      canthal,
      eyes,
      fwhr: fwhrScore,
      lips,
    },
    skin,
    flip,
    pose,
    symmetryDev,
    jawRatio,
    gonion,
    chinProjection,
    thirdsRatio,
    midfaceRatio,
    tilt,
    aperture,
    fwhr,
    fullness,
    lipRatio,
  };
}

function tierFor(score: number) {
  if (score >= 9) return { tier: 'CERTIFIED MOGGER', note: 'Top of the curve. Nothing on the sheet needs fixing.' };
  if (score >= 8) return { tier: 'CHAD ENERGY', note: 'High-tier face. Go outside immediately.' };
  if (score >= 7) return { tier: 'LOOKSMAXXED', note: 'Mogging most rooms you walk into.' };
  if (score >= 6) return { tier: 'ABOVE MID', note: 'Solid base. A couple of fixes and you climb.' };
  if (score >= 5) return { tier: 'MID', note: 'Perfectly average. Uncomfortable, but fixable.' };
  if (score >= 4) return { tier: 'BELOW PAR', note: 'Several measurements are costing you real points.' };
  if (score >= 3) return { tier: 'ROUGH', note: 'Most categories are working against you.' };
  return { tier: 'BRUTAL', note: 'Fix the deductions below before anything else.' };
}

const TOP_LINES: Record<MetricKey, string> = {
  symmetry: 'Symmetry is carrying the whole operation.',
  jawline: 'The jawline is doing the heavy lifting.',
  skin: 'Skin is clear, which is rarer than people think.',
  puffiness: 'The lower face is lean.',
  thirds: 'Facial thirds are textbook.',
  midface: 'Compact midface. That goes a long way.',
  canthal: 'Positive canthal tilt, working in your favour.',
  eyes: 'The eye area is the strongest feature here.',
  fwhr: 'Face width reads strong in person.',
  lips: 'Lip proportions are dialled in.',
};

function regionName(key: string) {
  if (key === 'forehead') return 'forehead';
  if (key === 'left') return 'left cheek';
  if (key === 'right') return 'right cheek';
  if (key === 'chin') return 'chin';
  return 'face';
}

function buildFlags(m: Measures, skinMeasured: boolean): Flag[] {
  const flags: Flag[] = [];
  const s = m.scores;

  if (s.symmetry < 0.8) {
    flags.push({
      key: 'symmetry',
      label: 'Asymmetry',
      detail: m.flip
        ? `${Math.round((1 - m.flip.score) * 100)}% mismatch between your left and right halves`
        : `${(m.symmetryDev * 100).toFixed(1)}% offset between mirrored features`,
      severity: s.symmetry < 0.45 ? 3 : s.symmetry < 0.62 ? 2 : 1,
      kind: 'face',
    });
  }

  if (m.skin && skinMeasured) {
    if (m.skin.clarity < 0.85) {
      const spots = m.skin.spots;
      const ranked = Object.entries(m.skin.regions).sort((a, b) => b[1] - a[1]);
      const where = ranked[0] && ranked[0][1] > 0 ? regionName(ranked[0][0]) : 'face';
      flags.push({
        key: 'skin',
        label: m.skin.clarity < 0.45 ? 'Breakouts' : 'Blemishes',
        detail: spots
          ? `${spots} ${spots === 1 ? 'spot' : 'spots'}, worst on the ${where}`
          : `${Math.round(m.skin.blemishShare * 100)}% of skin affected`,
        severity:
          m.skin.spots >= 10 || m.skin.clarity < 0.3
            ? 3
            : m.skin.spots >= 4 || m.skin.clarity < 0.55
              ? 2
              : 1,
        kind: 'face',
      });
    }
    if (m.skin.texture < 0.55) {
      flags.push({
        key: 'texture',
        label: 'Skin texture',
        detail: 'Uneven surface across the cheeks',
        severity: m.skin.texture < 0.4 ? 2 : 1,
        kind: 'face',
      });
    }
    if (m.skin.underEyes < 0.42) {
      flags.push({
        key: 'under-eyes',
        label: 'Under-eye shadow',
        detail: 'Lower lids read darker than the cheeks',
        severity: m.skin.underEyes < 0.3 ? 2 : 1,
        kind: 'face',
      });
    }
  }

  if (s.puffiness < 0.6) {
    flags.push({
      key: 'puffiness',
      label: 'Bloat',
      detail: `Lower face is ${(m.fullness * 100).toFixed(0)}% of face length, which reads puffy. Salt, sleep or body fat all move this.`,
      severity: s.puffiness < 0.4 ? 2 : 1,
      kind: 'face',
    });
  }

  if (s.jawline < 0.65) {
    flags.push({
      key: 'jawline',
      label: 'Jaw definition',
      detail:
        m.chinProjection !== null
          ? `Gonial angle ${m.gonion.toFixed(0)}°, chin projection ${(m.chinProjection * 100).toFixed(1)}`
          : `Gonial angle ${m.gonion.toFixed(0)}°, jaw width ${(m.jawRatio * 100).toFixed(0)}% of cheekbones`,
      severity: s.jawline < 0.45 ? 3 : 1,
      kind: 'face',
    });
  }
  if (s.midface < 0.62) {
    flags.push({
      key: 'midface',
      label: 'Midface',
      detail: `Midface ratio ${m.midfaceRatio.toFixed(2)}, outside the compact 1.0 to 1.12 band`,
      severity: 1,
      kind: 'face',
    });
  }
  if (s.canthal < 0.55) {
    flags.push({
      key: 'canthal',
      label: 'Canthal tilt',
      detail: `${m.tilt.toFixed(1)}° average, neutral to negative`,
      severity: 1,
      kind: 'face',
    });
  }
  if (s.thirds < 0.7) {
    flags.push({
      key: 'thirds',
      label: 'Proportions',
      detail: `Lower third is ${(m.thirdsRatio * 100).toFixed(0)}% of the middle third`,
      severity: 1,
      kind: 'face',
    });
  }

  if (Math.abs(m.pose.yaw) > 12 || Math.abs(m.pose.roll) > 12 || Math.abs(m.pose.pitch) > 12) {
    flags.push({
      key: 'pose',
      label: 'Head angle',
      detail: `Yaw ${m.pose.yaw.toFixed(0)}°, pitch ${m.pose.pitch.toFixed(0)}°. Corrected, but not perfect.`,
      severity: 1,
      kind: 'capture',
    });
  }
  if (m.skin && m.skin.lighting < 0.4) {
    flags.push({
      key: 'lighting',
      label: 'Weak lighting',
      detail: 'Skin readings are less reliable in this light',
      severity: 1,
      kind: 'capture',
    });
  }

  return flags.sort((a, b) => b.severity - a.severity);
}

function grade(m: Measures, frames: number, views: number): ScanResult {
  const skinMeasured = !!m.skin && m.skin.samples > 500;
  const keys = ORDER.filter((k) => k !== 'skin' || skinMeasured);
  const totalWeight = keys.reduce((sum, k) => sum + WEIGHTS[k], 0);
  const composite = keys.reduce((sum, k) => sum + m.scores[k] * WEIGHTS[k], 0) / (totalWeight || 1);

  let score = 1.05 + 8.85 * Math.pow(clamp01(composite), 1.6);

  // Hard gates. Bone structure does not outrank visible breakouts or a clearly
  // asymmetric face, so those cap the ceiling instead of averaging away.
  // Skin is the first thing anyone sees, so a breakout caps the ceiling instead
  // of quietly averaging against eight good ratios.
  if (skinMeasured) score = Math.min(score, 9.6 - (1 - m.scores.skin) * 10);
  score = Math.min(score, 9.8 - (1 - m.scores.symmetry) * 3.6);
  score = Math.min(score, 9.8 - (1 - m.scores.jawline) * 1.4);

  const pose = {
    yaw: m.pose.yaw,
    roll: m.pose.roll,
    pitch: m.pose.pitch,
    ok: Math.abs(m.pose.yaw) < 12 && Math.abs(m.pose.roll) < 12 && Math.abs(m.pose.pitch) < 12,
  };

  const confidence = Math.min(
    0.98,
    clamp01(
      0.5 +
        (pose.ok ? 0.18 : 0) +
        (skinMeasured ? 0.14 : 0) +
        (views >= 4 ? 0.1 : 0) +
        (frames >= 5 ? 0.06 : 0),
    ),
  );

  score = round1(Math.max(1, Math.min(9.9, score)));
  const { tier, note } = tierFor(score);

  const metrics: Metric[] = ORDER.map((key) => ({
    key,
    label: LABELS[key],
    rating: skinMeasured || key !== 'skin' ? Math.max(1, round1(1 + m.scores[key] * 8.6)) : 0,
    score: m.scores[key],
  })).filter((metric) => metric.rating > 0);

  const ranked = [...metrics].sort((a, b) => b.score - a.score);
  const top = ranked[0];
  const weak = ranked[ranked.length - 1];
  const flags = buildFlags(m, skinMeasured);
  const faceFlags = flags.filter((f) => f.kind === 'face');
  const worst = faceFlags[0];

  let verdict: string;
  if (worst && worst.severity === 3) {
    verdict = `${TOP_LINES[top.key]} Biggest deduction: ${worst.label.toLowerCase()}.`;
  } else if (weak.score < 0.72) {
    verdict = `${TOP_LINES[top.key]} The weak link is ${weak.label.toLowerCase()}.`;
  } else if (!faceFlags.length) {
    verdict = `${TOP_LINES[top.key]} Nothing on the sheet is dragging you down.`;
  } else {
    verdict = `${TOP_LINES[top.key]} Only small deductions left.`;
  }

  return {
    score,
    tier,
    tierNote: note,
    verdict,
    metrics,
    flags,
    pose,
    confidence: Math.round(confidence * 100) / 100,
    skin: {
      clarity: m.skin?.clarity ?? 0.55,
      texture: m.skin?.texture ?? 0.6,
      underEyes: m.skin?.underEyes ?? 0.7,
      blemishShare: m.skin?.blemishShare ?? 0,
      spots: m.skin?.spots ?? 0,
      measured: skinMeasured,
      regions: (m.skin?.regions ?? {}) as Record<string, number>,
    },
    raw: {
      symmetryDev: m.symmetryDev,
      flipDifference: m.flip?.difference ?? 0,
      jawRatio: m.jawRatio,
      gonion: m.gonion,
      chinProjection: m.chinProjection,
      thirdsRatio: m.thirdsRatio,
      midfaceRatio: m.midfaceRatio,
      tilt: m.tilt,
      aperture: m.aperture,
      fwhr: m.fwhr,
      lipRatio: m.lipRatio,
      fullness: m.fullness,
    },
    frames,
    views,
  };
}

export function analyzeFace(frame: Frame, views = 1): ScanResult {
  return grade(measure(frame), 1, views);
}

/**
 * Live scan: several frontal frames are measured independently and the median of
 * every metric is used, so a blink or a bad tracking tick cannot swing the read.
 */
export function mergeScans(results: ScanResult[], views = 1): ScanResult {
  if (!results.length) throw new Error('No frames captured');
  if (results.length === 1) return { ...results[0], views };

  const scores = {} as Record<MetricKey, number>;
  for (const key of ORDER) {
    const values = results.map((r) => r.metrics.find((m) => m.key === key)?.score).filter((v): v is number => typeof v === 'number');
    scores[key] = values.length ? median(values) : 0;
  }

  const withSkin = results.filter((r) => r.skin.measured);
  const skin: SkinReport | null = withSkin.length
    ? {
        clarity: median(withSkin.map((r) => r.skin.clarity)),
        texture: median(withSkin.map((r) => r.skin.texture)),
        underEyes: median(withSkin.map((r) => r.skin.underEyes)),
        blemishShare: median(withSkin.map((r) => r.skin.blemishShare)),
        spots: Math.round(median(withSkin.map((r) => r.skin.spots))),
        regions: withSkin[0].skin.regions as SkinReport['regions'],
        lighting: 0.8,
        samples: 1000,
        notes: [],
      }
    : null;

  const bestPose = results.reduce((best, r) =>
    Math.abs(r.pose.yaw) + Math.abs(r.pose.pitch) < Math.abs(best.pose.yaw) + Math.abs(best.pose.pitch) ? r : best,
  );
  const rawMedian = (pick: (r: RawMeasures) => number) => median(results.map((r) => pick(r.raw)));
  const flipDifferences = results.map((r) => r.raw.flipDifference).filter((v) => v > 0);
  const flipDifference = flipDifferences.length ? median(flipDifferences) : 0;
  const symmetryDev = rawMedian((r) => r.symmetryDev);
  // Recomputed from the medians so the breakdown always matches the flip number
  // the screen reports, instead of a median of already-clamped scores.
  scores.symmetry = symmetryScore(symmetryDev, flipDifference || null);

  const merged: Measures = {
    scores,
    skin,
    flip: flipDifference
      ? { score: clamp01(1 - Math.max(0, flipDifference - 0.15) / 0.95), difference: flipDifference, pairs: 0 }
      : null,
    pose: bestPose.pose,
    symmetryDev,
    jawRatio: rawMedian((r) => r.jawRatio),
    gonion: rawMedian((r) => r.gonion),
    chinProjection: results.find((r) => r.raw.chinProjection !== null)?.raw.chinProjection ?? null,
    thirdsRatio: rawMedian((r) => r.thirdsRatio),
    midfaceRatio: rawMedian((r) => r.midfaceRatio),
    tilt: rawMedian((r) => r.tilt),
    aperture: rawMedian((r) => r.aperture),
    fwhr: rawMedian((r) => r.fwhr),
    fullness: rawMedian((r) => r.fullness),
    lipRatio: rawMedian((r) => r.lipRatio),
  };

  return grade(merged, results.length, views);
}
