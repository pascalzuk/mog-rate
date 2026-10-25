import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { analyzeSkin, type SkinReport } from '@/lib/skinAnalysis';

export type MetricKey =
  | 'symmetry'
  | 'jawline'
  | 'skin'
  | 'thirds'
  | 'midface'
  | 'canthal'
  | 'eyes'
  | 'fwhr'
  | 'lips';

export type Metric = {
  key: MetricKey;
  label: string;
  /** 1.0 - 9.8, shown in the breakdown. Not padded: 3.4 means 3.4. */
  rating: number;
  /** raw 0..1 quality used internally */
  score: number;
};

export type Pose = {
  yaw: number;
  roll: number;
  ok: boolean;
};

export type Flag = {
  key: string;
  label: string;
  detail: string;
  /** 1 mild, 2 noticeable, 3 dominant */
  severity: 1 | 2 | 3;
  kind: 'face' | 'capture';
};

export type SkinSummary = {
  clarity: number;
  texture: number;
  underEyes: number;
  blemishShare: number;
  measured: boolean;
  regions: Record<string, number>;
};

export type RawMeasures = {
  symmetryDev: number;
  jawRatio: number;
  gonion: number;
  thirdsRatio: number;
  midfaceRatio: number;
  tilt: number;
  aperture: number;
  fwhr: number;
  lipRatio: number;
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
  /** measured values the copy quotes, kept so merges stay truthful */
  raw: RawMeasures;
  /** how many frames went into this read */
  frames: number;
};

type Pt = { x: number; y: number };
type Pt3 = { x: number; y: number; z: number };

/**
 * Canonical MediaPipe Face Mesh landmark indices.
 * 468-477 are iris points, present because refineLandmarks is on.
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
  symmetry: 0.15,
  jawline: 0.15,
  skin: 0.17,
  thirds: 0.09,
  midface: 0.1,
  canthal: 0.09,
  eyes: 0.08,
  fwhr: 0.08,
  lips: 0.09,
};

const LABELS: Record<MetricKey, string> = {
  symmetry: 'Symmetry',
  jawline: 'Jawline',
  skin: 'Skin',
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
  'thirds',
  'midface',
  'canthal',
  'eyes',
  'fwhr',
  'lips',
];

const dist = (a: Pt, b: Pt) => Math.hypot(a.x - b.x, a.y - b.y);
/**
 * 3D distance, using MediaPipe's relative depth. Vertical spans measured in 2D
 * shrink when the head pitches, which used to make a normal face read as a long
 * lower third. Including depth removes that foreshortening.
 */
const dist3 = (a: Pt3, b: Pt3) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const depthMid = (a: Pt3, b: Pt3): Pt3 => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 });
const mid = (a: Pt, b: Pt): Pt => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
const deg = (rad: number) => (rad * 180) / Math.PI;
const round1 = (v: number) => Math.round(v * 10) / 10;
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
};

/** 1 inside the ideal window, falling off linearly outside it. */
function band(value: number, lo: number, hi: number, soft: number) {
  if (!Number.isFinite(value)) return 0;
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

type Measures = {
  scores: Record<MetricKey, number>;
  skin: SkinReport | null;
  symmetryDev: number;
  jawRatio: number;
  gonion: number;
  thirdsRatio: number;
  midfaceRatio: number;
  tilt: number;
  aperture: number;
  fwhr: number;
  lipRatio: number;
  yaw: number;
  roll: number;
};

export type Frame = {
  landmarks: NormalizedLandmark[];
  width: number;
  height: number;
  pixels?: ImageData;
};

/**
 * Pose normalisation + measurement.
 *
 * A selfie is never square to the lens. First level the eye line (roll), then
 * solve for the head turn (yaw) that best lines up mirrored landmark pairs using
 * each point's depth, and undo it. Every measurement is then taken from a
 * virtual head-on view, so an angled shot is not scored as an asymmetric face.
 * The angle itself is reported separately so the UI can warn instead of lying.
 */
export function measure(frame: Frame): Measures {
  const { landmarks, width, height } = frame;
  if (!landmarks || landmarks.length < 468) throw new Error('Face not fully in frame');

  const rawPoints: Pt3[] = landmarks.map((l) => ({ x: l.x * width, y: l.y * height, z: l.z * width }));

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

  // Weighted pairs, and the mouth is deliberately left out: a smile pulls the
  // corners apart and would be scored as a crooked face.
  const pairs: Array<[number, number, number]> = [
    [33, 263, 1.2],
    [133, 362, 1.1],
    [129, 358, 0.8],
    [105, 334, 0.9],
    [234, 454, 1.0],
    [172, 397, 0.9],
  ];
  const pairWeight = pairs.reduce((sum, [, , w]) => sum + w, 0);

  const eyeMidX = (rolled[L.eyeOuterR].x + rolled[L.eyeOuterL].x) / 2;
  const rawFaceWidth = dist(rolled[L.cheekR], rolled[L.cheekL]) || 1;

  const mirrorError = (theta: number) => {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    let sum = 0;
    for (const [a, b, w] of pairs) {
      const xa = eyeMidX + (rolled[a].x - eyeMidX) * c + rolled[a].z * s;
      const xb = eyeMidX + (rolled[b].x - eyeMidX) * c + rolled[b].z * s;
      sum += Math.abs(2 * eyeMidX - xa - xb) * w;
    }
    return sum / (pairWeight * rawFaceWidth);
  };

  let yawRad = 0;
  let mirroredDeviation = Number.POSITIVE_INFINITY;
  for (let d = -35; d <= 35; d += 1) {
    const theta = (d * Math.PI) / 180;
    const err = mirrorError(theta);
    if (err < mirroredDeviation) {
      mirroredDeviation = err;
      yawRad = theta;
    }
  }

  const cosYaw = Math.cos(yawRad);
  const sinYaw = Math.sin(yawRad);
  const P: Pt[] = rolled.map(({ x, y, z }) => ({ x: eyeMidX + (x - eyeMidX) * cosYaw + z * sinYaw, y }));
  const p = (i: number) => P[i] ?? P[0];

  const faceWidth = dist(p(L.cheekR), p(L.cheekL)) || 1;
  const eyeCenterR = mid(p(L.eyeOuterR), p(L.eyeInnerR));
  const eyeCenterL = mid(p(L.eyeInnerL), p(L.eyeOuterL));

  // Vertical mismatch between mirrored pairs: a dropped brow or a lower jaw.
  let devY = 0;
  for (const [a, b, w] of pairs) devY += Math.abs(p(a).y - p(b).y) * w;
  devY /= pairWeight * faceWidth;

  const symmetryDev = mirroredDeviation + devY * 0.55;
  // 1.2% is inside normal noise, 3% starts to show, 7% is clearly visible.
  const symmetry = clamp01(1 - Math.max(0, symmetryDev - 0.012) / 0.06);

  const jawRatio = dist(p(L.jawR), p(L.jawL)) / faceWidth;
  const gonion =
    (angleAt(p(L.cheekR), p(L.jawR), p(L.chin)) + angleAt(p(L.cheekL), p(L.jawL), p(L.chin))) / 2;
  const jawline = 0.55 * band(jawRatio, 0.74, 0.85, 0.11) + 0.45 * band(gonion, 120, 134, 18);

  // Middle third (brow to nose base) vs lower third (nose base to chin).
  const middleThird = Math.max(1, dist3(rawPoints[L.subnasale], rawPoints[L.glabella]));
  const lowerThird = Math.max(1, dist3(rawPoints[L.chin], rawPoints[L.subnasale]));
  const thirdsRatio = lowerThird / middleThird;
  // The lower third normally runs slightly longer than the middle, so the ideal
  // window sits above 1.0 rather than at a literal one-third split.
  const thirds = band(thirdsRatio, 0.98, 1.2, 0.16);

  const ipd3 =
    rawPoints.length > 473
      ? dist3(rawPoints[L.irisR], rawPoints[L.irisL])
      : dist(eyeCenterR, eyeCenterL);
  const midfaceHeight = Math.max(1, dist3(rawPoints[L.upperInner], depthMid(rawPoints[L.eyeOuterR], rawPoints[L.eyeOuterL])));
  const midfaceRatio = ipd3 / midfaceHeight;
  const midface = band(midfaceRatio, 1.02, 1.14, 0.22);

  const tiltR = deg(Math.atan2(p(L.eyeInnerR).y - p(L.eyeOuterR).y, Math.abs(p(L.eyeInnerR).x - p(L.eyeOuterR).x)));
  const tiltL = deg(Math.atan2(p(L.eyeInnerL).y - p(L.eyeOuterL).y, Math.abs(p(L.eyeInnerL).x - p(L.eyeOuterL).x)));
  const tilt = (tiltR + tiltL) / 2;
  const canthal = band(tilt, 3, 8, 8);

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
  const fwhrScore = band(fwhr, 1.6, 1.78, 0.25);

  const lipRatio =
    Math.max(0, p(L.upperInner).y - p(L.lipTop).y) / Math.max(1, p(L.lipBottom).y - p(L.lowerInner).y);
  const mouthNose = dist(p(L.mouthR), p(L.mouthL)) / Math.max(1, dist(p(L.noseR), p(L.noseL)));
  const lips = 0.55 * band(lipRatio, 0.52, 0.7, 0.18) + 0.45 * band(mouthNose, 1.42, 1.62, 0.24);

  const skin = frame.pixels ? analyzeSkin(frame.pixels, landmarks, width, height) : null;
  const skinScore = skin ? clamp01(0.7 * skin.clarity + 0.2 * skin.texture + 0.1 * skin.underEyes) : 0;

  return {
    scores: {
      symmetry,
      jawline,
      skin: skinScore,
      thirds,
      midface,
      canthal,
      eyes,
      fwhr: fwhrScore,
      lips,
    },
    skin,
    symmetryDev,
    jawRatio,
    gonion,
    thirdsRatio,
    midfaceRatio,
    tilt,
    aperture,
    fwhr,
    lipRatio,
    yaw: Math.abs(deg(yawRad)),
    roll: Math.abs(deg(rollRad)),
  };
}

function tierFor(score: number) {
  if (score >= 9) return { tier: 'CERTIFIED MOGGER', note: 'Top of the curve. Nothing here needs fixing.' };
  if (score >= 8) return { tier: 'CHAD ENERGY', note: 'High-tier face. Go outside immediately.' };
  if (score >= 7) return { tier: 'LOOKSMAXXED', note: 'Mogging most rooms you walk into.' };
  if (score >= 6) return { tier: 'ABOVE MID', note: 'Solid base. A couple of fixes and you climb.' };
  if (score >= 5) return { tier: 'MID', note: 'Perfectly average. Uncomfortable, but fixable.' };
  if (score >= 4) return { tier: 'BELOW PAR', note: 'The measurements are doing you no favours.' };
  if (score >= 3) return { tier: 'ROUGH', note: 'Several categories are costing you real points.' };
  return { tier: 'BRUTAL', note: 'The scan found little to work with. Fix the flags below.' };
}

const TOP_LINES: Record<MetricKey, string> = {
  symmetry: 'Symmetry is carrying the whole operation.',
  jawline: 'The jawline is doing the heavy lifting.',
  skin: 'Skin is clear, which is rarer than people think.',
  thirds: 'Facial thirds are textbook.',
  midface: 'Compact midface. That goes a long way.',
  canthal: 'Positive canthal tilt, working in your favour.',
  eyes: 'The eye area is the strongest feature here.',
  fwhr: 'Wide face reads strong in person.',
  lips: 'Lip proportions are dialled in.',
};

function buildFlags(m: Measures, pose: Pose, skinMeasured: boolean): Flag[] {
  const flags: Flag[] = [];
  const s = m.scores;

  if (s.symmetry < 0.78) {
    flags.push({
      key: 'symmetry',
      label: 'Asymmetry',
      detail: `${(m.symmetryDev * 100).toFixed(1)}% offset between mirrored features`,
      severity: s.symmetry < 0.45 ? 3 : s.symmetry < 0.62 ? 2 : 1,
      kind: 'face',
    });
  }

  if (m.skin && skinMeasured) {
    const share = Math.round(m.skin.blemishShare * 100);
    if (m.skin.clarity < 0.82) {
      const ranked = Object.entries(m.skin.regions).sort((a, b) => b[1] - a[1]);
      const where = ranked[0] && ranked[0][1] > 0.02 ? regionName(ranked[0][0]) : 'scattered';
      flags.push({
        key: 'skin',
        label: m.skin.clarity < 0.45 ? 'Breakouts' : 'Blemishes',
        detail: `${share}% of the face flagged, mostly ${where}`,
        severity: m.skin.clarity < 0.4 ? 3 : m.skin.clarity < 0.65 ? 2 : 1,
        kind: 'face',
      });
    }
    if (m.skin.texture < 0.55) {
      flags.push({
        key: 'texture',
        label: 'Skin texture',
        detail: 'Uneven surface across the cheeks and forehead',
        severity: m.skin.texture < 0.4 ? 2 : 1,
        kind: 'face',
      });
    }
    if (m.skin.underEyes < 0.45) {
      flags.push({
        key: 'under-eyes',
        label: 'Under-eye shadow',
        detail: 'Lower lids are clearly darker than the cheeks',
        severity: m.skin.underEyes < 0.35 ? 2 : 1,
        kind: 'face',
      });
    }
  }

  if (s.jawline < 0.62) {
    flags.push({
      key: 'jawline',
      label: 'Jaw definition',
      detail: `Gonial angle ${m.gonion.toFixed(0)}°, jaw width ${(m.jawRatio * 100).toFixed(0)}% of cheekbones`,
      severity: s.jawline < 0.42 ? 3 : 1,
      kind: 'face',
    });
  }
  if (s.midface < 0.6) {
    flags.push({
      key: 'midface',
      label: 'Midface',
      detail: `Midface ratio ${m.midfaceRatio.toFixed(2)}, longer than the ideal window`,
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
  if (s.thirds < 0.65) {
    flags.push({
      key: 'thirds',
      label: 'Proportions',
      detail: `Lower third is ${(m.thirdsRatio * 100).toFixed(0)}% of the middle third`,
      severity: 1,
      kind: 'face',
    });
  }

  if (!pose.ok) {
    flags.push({
      key: 'pose',
      label: 'Head angle',
      detail: `Turned ${pose.yaw.toFixed(0)}° and tilted ${pose.roll.toFixed(0)}°, geometry is corrected but not perfect`,
      severity: 1,
      kind: 'capture',
    });
  }
  if (m.skin && m.skin.lighting < 0.45) {
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

function regionName(key: string) {
  if (key === 'forehead') return 'on the forehead';
  if (key === 'left') return 'on the left cheek';
  if (key === 'right') return 'on the right cheek';
  if (key === 'chin') return 'around the chin';
  return 'across the face';
}

function grade(m: Measures, pose: Pose, frames: number): ScanResult {
  const skinMeasured = !!m.skin && m.skin.samples > 400;
  const keys = ORDER.filter((k) => k !== 'skin' || skinMeasured);
  const totalWeight = keys.reduce((sum, k) => sum + WEIGHTS[k], 0);
  const composite = keys.reduce((sum, k) => sum + m.scores[k] * WEIGHTS[k], 0) / (totalWeight || 1);

  // Honest curve: a 0.85 composite is a genuine 8, a 0.5 is a genuine 4.
  let score = 1.1 + 8.8 * Math.pow(clamp01(composite), 1.55);

  // Hard gates. No amount of bone structure outranks visible breakouts or a
  // clearly asymmetric face, so those two cap the ceiling instead of averaging in.
  if (skinMeasured) score = Math.min(score, 9.85 - (1 - m.scores.skin) * 4.3);
  score = Math.min(score, 9.85 - (1 - m.scores.symmetry) * 3.4);
  score = Math.min(score, 9.85 - (1 - m.scores.jawline) * 1.6);

  const confidence = Math.min(
    0.97,
    clamp01(
      0.5 +
        (pose.ok ? 0.22 : 0) +
        (skinMeasured ? 0.15 : 0) +
        (m.skin ? (m.skin.lighting - 0.5) * 0.3 : 0) +
        (frames >= 5 ? 0.08 : 0),
    ),
  );

  score = round1(Math.max(1, Math.min(9.9, score)));
  const { tier, note } = tierFor(score);

  const metrics: Metric[] = ORDER.map((key) => ({
    key,
    label: LABELS[key],
    rating: skinMeasured || key !== 'skin' ? Math.max(1, round1(1 + m.scores[key] * 8.5)) : 0,
    score: m.scores[key],
  })).filter((metric) => metric.rating > 0);

  const ranked = [...metrics].sort((a, b) => b.score - a.score);
  const top = ranked[0];
  const weak = ranked[ranked.length - 1];

  const flags = buildFlags(m, pose, skinMeasured);

  const worst = flags.find((f) => f.kind === 'face');
  const faceBad = flags.filter((f) => f.kind === 'face').length;
  let verdict: string;
  if (worst && worst.severity === 3) {
    verdict = `${TOP_LINES[top.key]} Biggest deduction: ${worst.label.toLowerCase()}.`;
  } else if (weak.score < 0.72) {
    verdict = `${TOP_LINES[top.key]} The weak link is ${weak.label.toLowerCase()}.`;
  } else if (faceBad === 0) {
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
      clarity: m.skin?.clarity ?? 0.6,
      texture: m.skin?.texture ?? 0.6,
      underEyes: m.skin?.underEyes ?? 0.7,
      blemishShare: m.skin?.blemishShare ?? 0,
      measured: skinMeasured,
      regions: (m.skin?.regions ?? {}) as Record<string, number>,
    },
    raw: {
      symmetryDev: m.symmetryDev,
      jawRatio: m.jawRatio,
      gonion: m.gonion,
      thirdsRatio: m.thirdsRatio,
      midfaceRatio: m.midfaceRatio,
      tilt: m.tilt,
      aperture: m.aperture,
      fwhr: m.fwhr,
      lipRatio: m.lipRatio,
    },
    frames,
  };
}

export function analyzeFace(frame: Frame): ScanResult {
  const m = measure(frame);
  const pose: Pose = { yaw: m.yaw, roll: m.roll, ok: m.yaw < 14 && m.roll < 18 };
  return grade(m, pose, 1);
}

/**
 * Live scan: the camera samples several frames while you hold the pose. Each
 * measurement is taken independently, then the median is used, which throws out
 * a blink, a smeared frame or one bad tracking tick.
 */
export function mergeScans(results: ScanResult[]): ScanResult {
  if (!results.length) throw new Error('No frames captured');
  if (results.length === 1) return results[0];

  const measured = results.filter((r) => r.metrics.length > 0);
  const keys = ORDER.filter((key) => measured.every((r) => r.metrics.some((m) => m.key === key)));

  const scores = {} as Record<MetricKey, number>;
  for (const key of keys) {
    scores[key] = median(measured.map((r) => r.metrics.find((m) => m.key === key)!.score));
  }
  for (const key of ORDER) if (scores[key] === undefined) scores[key] = 0;

  const skinReports = measured.map((r) => r.skin).filter((s) => s.measured);
  const skin: SkinReport | null = skinReports.length
    ? {
        clarity: median(skinReports.map((s) => s.clarity)),
        texture: median(skinReports.map((s) => s.texture)),
        underEyes: median(skinReports.map((s) => s.underEyes)),
        blemishShare: median(skinReports.map((s) => s.blemishShare)),
        clusters: 0,
        regions: skinReports[0].regions as SkinReport['regions'],
        lighting: 0.75,
        samples: 1000,
        notes: [],
      }
    : null;

  const bestPose = measured.reduce((best, r) =>
    r.pose.yaw + r.pose.roll < best.pose.yaw + best.pose.roll ? r : best,
  );
  const rawMedian = (pick: (r: RawMeasures) => number) => median(measured.map((r) => pick(r.raw)));

  const merged: Measures = {
    scores,
    skin,
    symmetryDev: rawMedian((r) => r.symmetryDev),
    jawRatio: rawMedian((r) => r.jawRatio),
    gonion: rawMedian((r) => r.gonion),
    thirdsRatio: rawMedian((r) => r.thirdsRatio),
    midfaceRatio: rawMedian((r) => r.midfaceRatio),
    tilt: rawMedian((r) => r.tilt),
    aperture: rawMedian((r) => r.aperture),
    fwhr: rawMedian((r) => r.fwhr),
    lipRatio: rawMedian((r) => r.lipRatio),
    yaw: bestPose.pose.yaw,
    roll: bestPose.pose.roll,
  };

  return grade(merged, bestPose.pose, measured.length);
}
