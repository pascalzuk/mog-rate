import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

export type MetricKey =
  | 'symmetry'
  | 'jawline'
  | 'thirds'
  | 'midface'
  | 'canthal'
  | 'eyes'
  | 'fwhr'
  | 'lips';

export type Metric = {
  key: MetricKey;
  label: string;
  /** 3.0 - 9.8, what gets shown next to the bar */
  rating: number;
  /** raw 0..1 score, used for the verdict */
  score: number;
};

export type Pose = {
  yaw: number;
  roll: number;
  ok: boolean;
};

export type ScanResult = {
  score: number;
  tier: string;
  tierNote: string;
  verdict: string;
  metrics: Metric[];
  pose: Pose;
};

type Pt = { x: number; y: number };
type Pt3 = { x: number; y: number; z: number };

/**
 * Canonical MediaPipe Face Mesh landmark indices.
 * 468-477 are iris points, only present when refine landmarks are on (they are).
 */
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

const WEIGHTS: Record<MetricKey, number> = {
  symmetry: 0.17,
  jawline: 0.19,
  midface: 0.12,
  canthal: 0.12,
  thirds: 0.1,
  eyes: 0.09,
  fwhr: 0.1,
  lips: 0.11,
};

const LABELS: Record<MetricKey, string> = {
  symmetry: 'Symmetry',
  jawline: 'Jawline',
  midface: 'Midface',
  canthal: 'Canthal tilt',
  thirds: 'Proportions',
  eyes: 'Eye area',
  fwhr: 'Face width',
  lips: 'Lips',
};

const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a: Pt, b: Pt): Pt => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
const deg = (rad: number) => (rad * 180) / Math.PI;

/** 1 inside the ideal window, falling off linearly outside it. */
function band(value: number, lo: number, hi: number, soft: number) {
  if (value >= lo && value <= hi) return 1;
  const d = value < lo ? lo - value : value - hi;
  return clamp01(1 - d / soft);
}

/** Interior angle ABC, in degrees. */
function angleAt(a: Pt, b: Pt, c: Pt) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const denom = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y) || 1;
  const cos = (v1.x * v2.x + v1.y * v2.y) / denom;
  return deg(Math.acos(Math.min(1, Math.max(-1, cos))));
}

function tierFor(score: number) {
  if (score >= 9) return { tier: 'CERTIFIED MOGGER', note: "It's over for everyone else." };
  if (score >= 8) return { tier: 'CHAD ENERGY', note: 'Go outside. Immediately.' };
  if (score >= 7) return { tier: 'LOOKSMAXXED', note: 'Mogging most rooms you walk into.' };
  if (score >= 6) return { tier: 'ABOVE MID', note: 'Solid base. Small tweaks and you climb.' };
  if (score >= 5) return { tier: 'MID', note: 'Perfectly average. Brutal, but fixable.' };
  if (score >= 4) return { tier: 'OFF DAY', note: 'Lighting? Angle? Sleep? Something is off.' };
  return { tier: 'BRUTAL', note: "The camera did not do you any favours." };
}

const TOP_LINES: Record<MetricKey, string> = {
  symmetry: 'Your symmetry is carrying the whole operation.',
  jawline: 'That jawline is doing the heavy lifting.',
  thirds: 'Textbook facial thirds — nothing out of place.',
  midface: 'Compact midface. Rare W.',
  canthal: 'Positive canthal tilt — you know what you are doing.',
  eyes: 'The eye area is elite.',
  fwhr: 'Wide face, strong presence.',
  lips: 'Lip proportions are dialled in.',
};

const WEAK_LINES: Record<MetricKey, string> = {
  symmetry: 'One side of your face is doing more work than the other.',
  jawline: 'Jaw definition is the weak link.',
  thirds: 'Your thirds are slightly out of balance.',
  midface: 'Midface is running long.',
  canthal: 'Neutral canthal tilt — nothing to write home about.',
  eyes: 'The eye area is costing you points.',
  fwhr: 'Face reads narrow for the frame.',
  lips: 'Lip ratio is a small deduction.',
};

export function analyzeFace(
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
): ScanResult {
  if (!landmarks || landmarks.length < 468) {
    throw new Error('Face not fully in frame');
  }

  const rawPoints: Pt3[] = landmarks.map((l) => ({ x: l.x * width, y: l.y * height, z: l.z * width }));

  // --- Pose normalisation -------------------------------------------------
  // A selfie is never perfectly square to the lens. First level the eye line
  // (roll), then solve for the head turn (yaw) that best lines up mirrored
  // landmarks using each point's depth, and undo it. Measurements are then
  // taken from a virtual head-on view, so an angled shot doesn't tank the score.
  const eyeAnchorR = mid(rawPoints[L.eyeOuterR], rawPoints[L.eyeInnerR]);
  const eyeAnchorL = mid(rawPoints[L.eyeInnerL], rawPoints[L.eyeOuterL]);
  const rollRad = Math.atan2(eyeAnchorL.y - eyeAnchorR.y, eyeAnchorL.x - eyeAnchorR.x);
  const pivot = mid(eyeAnchorR, eyeAnchorL);
  const cosRoll = Math.cos(-rollRad);
  const sinRoll = Math.sin(-rollRad);
  const rolled: Pt3[] = rawPoints.map(({ x, y, z }) => {
    const dx = x - pivot.x;
    const dy = y - pivot.y;
    return { x: pivot.x + dx * cosRoll - dy * sinRoll, y: pivot.y + dx * sinRoll + dy * cosRoll, z };
  });

  const symmetryPairs: Array<[number, number]> = [
    [33, 263],
    [133, 362],
    [61, 291],
    [172, 397],
    [234, 454],
    [105, 334],
    [129, 358],
  ];

  const eyeMidX = (rolled[L.eyeOuterR].x + rolled[L.eyeOuterL].x) / 2;
  const rawFaceWidth = dist(rolled[L.cheekR], rolled[L.cheekL]) || 1;
  const mirroredError = (theta: number) => {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    let sum = 0;
    for (const [a, b] of symmetryPairs) {
      const xa = eyeMidX + (rolled[a].x - eyeMidX) * c + rolled[a].z * s;
      const xb = eyeMidX + (rolled[b].x - eyeMidX) * c + rolled[b].z * s;
      sum += Math.abs(2 * eyeMidX - xa - xb);
    }
    return sum / (symmetryPairs.length * rawFaceWidth);
  };

  let yawRad = 0;
  let mirroredDeviation = Number.POSITIVE_INFINITY;
  for (let d = -35; d <= 35; d += 1) {
    const theta = (d * Math.PI) / 180;
    const err = mirroredError(theta);
    if (err < mirroredDeviation) {
      mirroredDeviation = err;
      yawRad = theta;
    }
  }

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
  const eyeLineY = (eyeCenterR.y + eyeCenterL.y) / 2;
  const ipd = P.length > 473 ? dist(p(L.irisR), p(L.irisL)) : dist(eyeCenterR, eyeCenterL);
  const browY = (p(L.browR).y + p(L.browL).y) / 2;
  const lipY = p(L.upperInner).y;

  // --- Symmetry: how far mirrored landmark pairs sit from the eye midline ---
  let devY = 0;
  for (const [a, b] of symmetryPairs) {
    devY += Math.abs(p(a).y - p(b).y);
  }
  devY /= symmetryPairs.length * faceWidth;
  const symmetry = clamp01(1 - mirroredDeviation * 10 - devY * 5);

  // --- Pose, used to warn about a turned / tilted head ---
  const yaw = Math.abs(deg(yawRad));
  const roll = Math.abs(deg(rollRad));

  // --- Jawline: width relative to cheekbones + gonial angle sharpness ---
  const jawRatio = dist(p(L.jawR), p(L.jawL)) / faceWidth;
  const gonion = (angleAt(p(L.cheekR), p(L.jawR), p(L.chin)) + angleAt(p(L.cheekL), p(L.jawL), p(L.chin))) / 2;
  const jawline = 0.55 * band(jawRatio, 0.72, 0.88, 0.16) + 0.45 * band(gonion, 118, 142, 26);

  // --- Proportions: middle third (brow to nose base) vs lower third (nose base to chin).
  // Deliberately ignores the hairline, which is covered by hair on most people. ---
  const middleThird = Math.max(1, p(L.subnasale).y - p(L.glabella).y);
  const lowerThird = Math.max(1, p(L.chin).y - p(L.subnasale).y);
  const thirds = band(lowerThird / middleThird, 0.95, 1.16, 0.22);

  // --- Midface ratio: IPD over pupil-to-upper-lip height ---
  const midfaceRatio = ipd / Math.max(1, lipY - eyeLineY);
  const midface = band(midfaceRatio, 1.0, 1.18, 0.3);

  // --- Canthal tilt: positive means the outer corner sits higher ---
  const tiltR = deg(Math.atan2(p(L.eyeInnerR).y - p(L.eyeOuterR).y, Math.abs(p(L.eyeInnerR).x - p(L.eyeOuterR).x)));
  const tiltL = deg(Math.atan2(p(L.eyeInnerL).y - p(L.eyeOuterL).y, Math.abs(p(L.eyeInnerL).x - p(L.eyeOuterL).x)));
  const tilt = (tiltR + tiltL) / 2;
  const canthal = band(tilt, 2, 9, 11);

  // --- Eye area: how open the eye is relative to its width ---
  const eyeWR = dist(p(L.eyeOuterR), p(L.eyeInnerR)) || 1;
  const eyeWL = dist(p(L.eyeInnerL), p(L.eyeOuterL)) || 1;
  const aperture =
    (Math.abs(p(L.lidBottomR).y - p(L.lidTopR).y) / eyeWR +
      Math.abs(p(L.lidBottomL).y - p(L.lidTopL).y) / eyeWL) /
    2;
  const eyes = band(aperture, 0.25, 0.38, 0.16);

  // --- Facial width-to-height ---
  const fwhr = faceWidth / Math.max(1, lipY - browY);
  const fwhrScore = band(fwhr, 1.55, 1.85, 0.32);

  // --- Lips: upper/lower vermilion + mouth-to-nose width ---
  const lipRatio = Math.max(0, p(L.upperInner).y - p(L.lipTop).y) / Math.max(1, p(L.lipBottom).y - p(L.lowerInner).y);
  const mouthNose = dist(p(L.mouthR), p(L.mouthL)) / Math.max(1, dist(p(L.noseR), p(L.noseL)));
  const lips = 0.55 * band(lipRatio, 0.5, 0.75, 0.22) + 0.45 * band(mouthNose, 1.42, 1.72, 0.3);

  const scores: Record<MetricKey, number> = {
    symmetry,
    jawline,
    midface,
    canthal,
    thirds,
    eyes,
    fwhr: fwhrScore,
    lips,
  };

  const composite = (Object.keys(WEIGHTS) as MetricKey[]).reduce((sum, key) => sum + scores[key] * WEIGHTS[key], 0);
  const score = Math.round(Math.min(9.9, Math.max(1.2, 2.2 + 7.8 * Math.pow(composite, 2.2))) * 10) / 10;

  const order: MetricKey[] = ['symmetry', 'jawline', 'midface', 'canthal', 'thirds', 'eyes', 'fwhr', 'lips'];
  const metrics: Metric[] = order.map((key) => ({
    key,
    label: LABELS[key],
    rating: Math.round((3 + clamp01(scores[key]) * 6.8) * 10) / 10,
    score: scores[key],
  }));

  const ranked = [...metrics].sort((a, b) => b.score - a.score);
  const top = ranked[0];
  const weak = ranked[ranked.length - 1];

  const { tier, note } = tierFor(score);
  const verdict =
    score >= 8 && weak.score > 0.72
      ? `${TOP_LINES[top.key]} Nothing here needs fixing.`
      : weak.score < 0.8
        ? `${TOP_LINES[top.key]} ${WEAK_LINES[weak.key]}`
        : TOP_LINES[top.key];

  return {
    score,
    tier,
    tierNote: note,
    verdict,
    metrics,
    pose: { yaw, roll, ok: yaw < 13 && roll < 16 },
  };
}
