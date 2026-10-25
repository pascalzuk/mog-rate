import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

/**
 * Skin is measured from pixels, not landmarks: landmarks can tell you where the
 * cheek is, but not whether there is a breakout on it. This module samples the
 * actual frame inside the face oval, subtracts the local skin tone, and looks
 * for clusters of pixels that are both redder and darker than their neighbours.
 * Freckles, moles, stubble and shadows are all real signals, so the thresholds
 * below are deliberately conservative: a flagged cluster has to survive a coarse
 * grid test, which filters out single-pixel noise and JPEG speckle.
 */

export type SkinRegion = 'forehead' | 'left' | 'right' | 'chin';

export type SkinReport = {
  /** 0..1, 1 = clear skin */
  clarity: number;
  /** 0..1, 1 = smooth surface */
  texture: number;
  /** 0..1, 1 = no dark circles */
  underEyes: number;
  /** share of measured skin cells that flagged as blemished */
  blemishShare: number;
  /** number of flagged cells */
  clusters: number;
  /** share of flagged cells by area of the face */
  regions: Record<SkinRegion, number>;
  /** 0..1 capture quality, low means dark / blown out / too small */
  lighting: number;
  /** usable skin pixels */
  samples: number;
  notes: string[];
};

const OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
  176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

type P = { x: number; y: number };
type Box = { x0: number; y0: number; x1: number; y1: number };

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

function toPts(landmarks: NormalizedLandmark[], w: number, h: number): P[] {
  return landmarks.map((l) => ({ x: l.x * w, y: l.y * h }));
}

function bbox(points: P[]): Box {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const p of points) {
    if (p.x < x0) x0 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.x > x1) x1 = p.x;
    if (p.y > y1) y1 = p.y;
  }
  return { x0, y0, x1, y1 };
}

function pad(box: Box, fx: number, top: number, bottom: number): Box {
  const w = Math.max(1, box.x1 - box.x0);
  const h = Math.max(1, box.y1 - box.y0);
  return {
    x0: box.x0 - w * fx,
    y0: box.y0 - h * top,
    x1: box.x1 + w * fx,
    y1: box.y1 + h * bottom,
  };
}

/** Scanline fill of the face oval into a 1-byte mask. */
function fillPolygon(mask: Uint8Array, w: number, h: number, poly: P[]) {
  const xs: number[] = [];
  for (let y = 0; y < h; y++) {
    xs.length = 0;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) {
        xs.push(a.x + ((y - a.y) / (b.y - a.y)) * (b.x - a.x));
      }
    }
    xs.sort((p, q) => p - q);
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const x0 = Math.max(0, Math.ceil(xs[i]));
      const x1 = Math.min(w - 1, Math.floor(xs[i + 1]));
      for (let x = x0; x <= x1; x++) mask[y * w + x] = 1;
    }
  }
}

function carve(mask: Uint8Array, w: number, h: number, box: Box) {
  const x0 = Math.max(0, Math.floor(box.x0));
  const x1 = Math.min(w - 1, Math.ceil(box.x1));
  const y0 = Math.max(0, Math.floor(box.y0));
  const y1 = Math.min(h - 1, Math.ceil(box.y1));
  for (let y = y0; y <= y1; y++) {
    const row = y * w;
    for (let x = x0; x <= x1; x++) mask[row + x] = 0;
  }
}

/** Separable box blur, clamped at the edges. Approximates the local skin tone. */
function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  const win = r * 2 + 1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sum = 0;
    for (let x = -r; x <= r; x++) sum += src[row + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = sum / win;
      sum += src[row + Math.min(w - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = -r; y <= r; y++) sum += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = sum / win;
      sum += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

function emptyReport(lighting = 0): SkinReport {
  return {
    clarity: 0.6,
    texture: 0.6,
    underEyes: 0.7,
    blemishShare: 0,
    clusters: 0,
    regions: { forehead: 0, left: 0, right: 0, chin: 0 },
    lighting,
    samples: 0,
    notes: [],
  };
}

export function analyzeSkin(
  pixels: ImageData,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
): SkinReport {
  const W = pixels.width;
  const H = pixels.height;
  const data = pixels.data;
  if (!landmarks || landmarks.length < 468 || W < 32 || H < 32) return emptyReport();

  const pts = toPts(landmarks, W, H);
  const p = (i: number) => pts[i] ?? pts[0];

  const faceW = Math.max(12, Math.hypot(p(234).x - p(454).x, p(234).y - p(454).y));
  const oval = OVAL.map((i) => p(i));
  // Pull the sampling area 6% inside the face outline: the hairline and jaw
  // edges carry colour fringing that reads as blemishes if you sample them.
  const cx = oval.reduce((s, q) => s + q.x, 0) / oval.length;
  const cy = oval.reduce((s, q) => s + q.y, 0) / oval.length;
  const mask = new Uint8Array(W * H);
  fillPolygon(
    mask,
    W,
    H,
    oval.map((q) => ({ x: cx + (q.x - cx) * 0.94, y: cy + (q.y - cy) * 0.94 })),
  );

  // Never measure eyes, brows, lips or nostrils: they are dark by design and
  // would be read as a breakout on almost every face.
  const eyeR = bbox([p(33), p(133), p(159), p(145), p(130)]);
  const eyeL = bbox([p(362), p(263), p(386), p(374), p(359)]);
  carve(mask, W, H, pad(eyeR, 0.16, 1.05, 0.3));
  carve(mask, W, H, pad(eyeL, 0.16, 1.05, 0.3));
  carve(mask, W, H, pad(bbox([p(61), p(291), p(0), p(17), p(13), p(14)]), 0.22, 0.35, 0.35));
  carve(mask, W, H, pad(bbox([p(129), p(358), p(2)]), 0.1, 0.22, 0.16));

  const lum = new Float32Array(W * H);
  const red = new Float32Array(W * H);
  for (let i = 0, j = 0; i < W * H; i++, j += 4) {
    const r = data[j];
    const g = data[j + 1];
    const b = data[j + 2];
    lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    red[i] = r - (g + b) / 2;
  }

  const radius = Math.max(2, Math.round(faceW * 0.032));
  const blurLum = boxBlur(lum, W, H, radius);
  const blurRed = boxBlur(red, W, H, radius);

  const cell = Math.max(4, Math.round(faceW / 24));
  const cols = Math.ceil(W / cell);
  const rows = Math.ceil(H / cell);
  const skinCells = new Int32Array(cols * rows);
  const flagCells = new Int32Array(cols * rows);

  const browY = (p(9).y + p(105).y + p(334).y) / 3;
  const mouthY = p(13).y;
  const midX = (p(234).x + p(454).x) / 2;

  let samples = 0;
  let lumSum = 0;
  let gradSum = 0;
  let blown = 0;

  for (let y = 1; y < H - 1; y++) {
    const row = y * W;
    for (let x = 1; x < W - 1; x++) {
      const i = row + x;
      if (!mask[i]) continue;
      const l = lum[i];
      if (l > 250) {
        blown++;
        continue;
      }
      if (l < 20) continue;

      samples++;
      lumSum += l;
      gradSum += Math.abs(lum[i + 1] - lum[i - 1]) + Math.abs(lum[i + W] - lum[i - W]);

      const redRes = red[i] - blurRed[i];
      const darkRes = blurLum[i] - l;
      const flagged = (redRes > 7 && darkRes > 5) || (redRes > 15 && darkRes > 1.5) || (darkRes > 22 && redRes > 3);

      const ci = ((y / cell) | 0) * cols + ((x / cell) | 0);
      skinCells[ci]++;
      if (flagged) flagCells[ci]++;
    }
  }

  if (samples < 400) return emptyReport();

  let totalCells = 0;
  let flaggedCells = 0;
  const regionTotal = [0, 0, 0, 0];
  const regionFlagged = [0, 0, 0, 0];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const ci = y * cols + x;
      if (skinCells[ci] < 10) continue;
      totalCells++;
      const band = y * cell < browY ? 0 : y * cell < mouthY ? (x * cell < midX ? 1 : 2) : 3;
      regionTotal[band]++;
      const share = flagCells[ci] / skinCells[ci];
      if (share < 0.24) continue;
      flaggedCells++;
      regionFlagged[band]++;
    }
  }
  if (!totalCells) return emptyReport();

  const blemishShare = flaggedCells / totalCells;
  const regions: Record<SkinRegion, number> = {
    forehead: regionTotal[0] ? regionFlagged[0] / regionTotal[0] : 0,
    left: regionTotal[1] ? regionFlagged[1] / regionTotal[1] : 0,
    right: regionTotal[2] ? regionFlagged[2] / regionTotal[2] : 0,
    chin: regionTotal[3] ? regionFlagged[3] / regionTotal[3] : 0,
  };

  const meanLum = lumSum / samples;
  const roughness = gradSum / samples / Math.max(28, meanLum);
  const blownShare = blown / Math.max(1, samples + blown);
  const lighting = clamp01(1 - Math.abs(meanLum - 142) / 125 - blownShare * 2.2);

  // Calibrated against a clear face (about 1% noise at the hairline) and
  // progressively covered faces: 3% is mild, 8% is moderate, 15% is severe.
  const excess = Math.max(0, blemishShare - 0.006);
  // Floor at 8%: severe acne should bottom out, not read as "no skin at all".
  const clarity = Math.max(0.08, clamp01(1 - Math.pow(excess / 0.1, 0.85)));

  // Laplacian-free roughness: mean neighbour delta over skin luminance.
  const texture = clamp01(1 - (roughness - 0.052) / 0.085);

  const eyeH = Math.max(2, (eyeR.y1 - eyeR.y0 + eyeL.y1 - eyeL.y0) / 2);
  const patchMean = (cx: number, cy: number, hw: number, hh: number): number | null => {
    let sum = 0;
    let n = 0;
    const x0 = Math.max(1, Math.round(cx - hw));
    const x1 = Math.min(W - 2, Math.round(cx + hw));
    const y0 = Math.max(1, Math.round(cy - hh));
    const y1 = Math.min(H - 2, Math.round(cy + hh));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * W + x;
        if (!mask[i] || lum[i] < 20 || lum[i] > 248) continue;
        sum += lum[i];
        n++;
      }
    }
    return n > 12 ? sum / n : null;
  };

  const underRatios: number[] = [];
  for (const eye of [eyeR, eyeL]) {
    const cx = (eye.x0 + eye.x1) / 2;
    const under = patchMean(cx, eye.y1 + eyeH * 0.55, (eye.x1 - eye.x0) * 0.55, eyeH * 0.45);
    const cheek = patchMean(cx, eye.y1 + eyeH * 3.1, (eye.x1 - eye.x0) * 0.55, eyeH * 0.9);
    if (under && cheek && cheek > 1) underRatios.push(under / cheek);
  }
  const underRatio = underRatios.length ? underRatios.reduce((a, b) => a + b, 0) / underRatios.length : 0.9;
  const underEyes = clamp01((underRatio - 0.52) / 0.42);

  const notes: string[] = [];
  if (blemishShare > 0.16) notes.push('Visible breakout clusters');
  else if (blemishShare > 0.06) notes.push('Minor blemishes');
  if (texture < 0.55) notes.push('Textured skin surface');
  if (underEyes < 0.55) notes.push('Dark under-eye circles');
  if (lighting < 0.45) notes.push('Lighting makes skin hard to read');

  return {
    clarity,
    texture,
    underEyes,
    blemishShare,
    clusters: flaggedCells,
    regions,
    lighting,
    samples,
    notes,
  };
}
