import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

/**
 * Symmetry by flipping.
 *
 * The face is warped into a canonical frame using the eye line (roll, scale and
 * position removed), mirrored, and compared with itself. Two things matter for
 * the result to mean anything:
 *
 * 1. The comparison only runs inside the face oval. Hair and background edges
 *    have huge contrast, and a one-pixel landmark slip there would look like a
 *    facial asymmetry.
 * 2. The crop is downsampled and blurred. At full resolution, sharp edges are
 *    alignment noise; at this scale the difference is structure: a dropped brow,
 *    a wider jaw on one side, one-sided breakouts.
 */

export type FlipReport = {
  /** 0..1, 1 = the two halves match closely */
  score: number;
  /** mean mirrored difference in contrast units */
  difference: number;
  pairs: number;
};

const CW = 120;
const CH = 152;

const OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
  176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

function bilinear(data: Uint8ClampedArray, w: number, h: number, x: number, y: number): number {
  if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) return -1;
  const x0 = x | 0;
  const y0 = y | 0;
  const fx = x - x0;
  const fy = y - y0;
  const i = (y0 * w + x0) * 4;
  const i2 = i + w * 4;
  const g00 = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  const g10 = 0.299 * data[i + 4] + 0.587 * data[i + 5] + 0.114 * data[i + 6];
  const g01 = 0.299 * data[i2] + 0.587 * data[i2 + 1] + 0.114 * data[i2 + 2];
  const g11 = 0.299 * data[i2 + 4] + 0.587 * data[i2 + 5] + 0.114 * data[i2 + 6];
  return (g00 * (1 - fx) + g10 * fx) * (1 - fy) + (g01 * (1 - fx) + g11 * fx) * fy;
}

/** 3x3 box blur twice: removes the pixel-level noise that alignment creates. */
function smooth(src: Float32Array, w: number, h: number): Float32Array {
  let a = src;
  for (let pass = 0; pass < 2; pass++) {
    const out = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        let sum = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) sum += a[(y + dy) * w + x + dx];
        }
        out[y * w + x] = sum / 9;
      }
    }
    a = out;
  }
  return a;
}

export function flipSymmetry(
  pixels: ImageData,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
): FlipReport {
  const W = pixels.width;
  const H = pixels.height;
  if (!landmarks || landmarks.length < 468) return { score: 0.5, difference: 1, pairs: 0 };

  const lm = landmarks;
  const p = (i: number) => ({ x: lm[i].x * W, y: lm[i].y * H });
  const eyeR = { x: (p(33).x + p(133).x) / 2, y: (p(33).y + p(133).y) / 2 };
  const eyeL = { x: (p(362).x + p(263).x) / 2, y: (p(362).y + p(263).y) / 2 };
  const dx = eyeL.x - eyeR.x;
  const dy = eyeL.y - eyeR.y;
  const d = Math.hypot(dx, dy);
  if (d < 8) return { score: 0.5, difference: 1, pairs: 0 };

  const rot = Math.atan2(dy, dx);
  const cos = Math.cos(rot);
  const sin = Math.sin(rot);
  const scale = d / (0.4 * CW);
  const toCanonical = (x: number, y: number) => {
    const rx = (x - eyeR.x) / scale;
    const ry = (y - eyeR.y) / scale;
    return { x: 0.3 * CW + rx * cos + ry * sin, y: 0.42 * CH - rx * sin + ry * cos };
  };

  const crop = new Float32Array(CW * CH);
  const filled = new Uint8Array(CW * CH);
  for (let cy = 0; cy < CH; cy++) {
    for (let cx = 0; cx < CW; cx++) {
      const ox = (cx - 0.3 * CW) * scale;
      const oy = (cy - 0.42 * CH) * scale;
      const sx = eyeR.x + ox * cos - oy * sin;
      const sy = eyeR.y + ox * sin + oy * cos;
      const g = bilinear(pixels.data, W, H, sx, sy);
      if (g < 0) continue;
      crop[cy * CW + cx] = g;
      filled[cy * CW + cx] = 1;
    }
  }

  // Oval mask in canonical space, pulled in slightly so the hairline is out.
  const poly = OVAL.map((i) => toCanonical(p(i).x, p(i).y));
  const ox = poly.reduce((s, q) => s + q.x, 0) / poly.length;
  const oy = poly.reduce((s, q) => s + q.y, 0) / poly.length;
  const mask = new Uint8Array(CW * CH);
  const xs: number[] = [];
  for (let y = 0; y < CH; y++) {
    xs.length = 0;
    for (let i = 0; i < poly.length; i++) {
      const a = { x: ox + (poly[i].x - ox) * 0.94, y: oy + (poly[i].y - oy) * 0.94 };
      const b = {
        x: ox + (poly[(i + 1) % poly.length].x - ox) * 0.94,
        y: oy + (poly[(i + 1) % poly.length].y - oy) * 0.94,
      };
      if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) {
        xs.push(a.x + ((y - a.y) / (b.y - a.y)) * (b.x - a.x));
      }
    }
    xs.sort((m, n) => m - n);
    for (let i = 0; i + 1 < xs.length; i += 2) {
      const x0 = Math.max(1, Math.ceil(xs[i]));
      const x1 = Math.min(CW - 2, Math.floor(xs[i + 1]));
      for (let x = x0; x <= x1; x++) mask[y * CW + x] = 1;
    }
  }

  const blurred = smooth(crop, CW, CH);
  let sum = 0;
  let count = 0;
  for (let i = 0; i < blurred.length; i++) {
    if (!mask[i] || !filled[i]) continue;
    sum += blurred[i];
    count++;
  }
  if (count < 400) return { score: 0.5, difference: 1, pairs: 0 };
  const mean = sum / count;
  let varSum = 0;
  for (let i = 0; i < blurred.length; i++) {
    if (!mask[i] || !filled[i]) continue;
    const d0 = blurred[i] - mean;
    varSum += d0 * d0;
  }
  const std = Math.sqrt(varSum / count) || 1;

  // The true mirror axis is never exactly the eye midpoint, so search a few
  // pixels around it and keep the best fit. What survives is real asymmetry, not
  // a half-pixel landmark bias.
  // Skip the top of the forehead: a fringe of hair lands there and is never
  // mirrored, so it would count as bone structure.
  const yStart = Math.round(CH * 0.36);
  const yEnd = Math.round(CH * 0.92);
  const xEnd = Math.round(CW * 0.9);

  let bestDiff = Number.POSITIVE_INFINITY;
  let bestN = 0;
  for (let axis = -4; axis <= 4; axis++) {
    let diffSum = 0;
    let diffN = 0;
    let sumL = 0;
    let sumR = 0;
    let countLR = 0;
    for (let y = yStart; y < yEnd; y++) {
      for (let x = Math.round(CW * 0.56); x < xEnd; x++) {
        const mirror = CW - 1 - x + axis;
        if (mirror < 1 || mirror >= CW - 1) continue;
        if (!mask[y * CW + x] || !mask[y * CW + mirror]) continue;
        if (!filled[y * CW + x] || !filled[y * CW + mirror]) continue;
        sumL += blurred[y * CW + x];
        sumR += blurred[y * CW + mirror];
        countLR++;
      }
    }
    if (!countLR) continue;
    const meanL = sumL / countLR;
    const meanR = sumR / countLR;
    for (let y = yStart; y < yEnd; y++) {
      for (let x = Math.round(CW * 0.56); x < xEnd; x++) {
        const mirror = CW - 1 - x + axis;
        if (mirror < 1 || mirror >= CW - 1) continue;
        if (!mask[y * CW + x] || !mask[y * CW + mirror]) continue;
        if (!filled[y * CW + x] || !filled[y * CW + mirror]) continue;
        // Each half is centred on its own mean, so one side being lit brighter
        // does not register as an asymmetry.
        const a = blurred[y * CW + x] - meanL;
        const b = blurred[y * CW + mirror] - meanR;
        diffSum += Math.abs(a - b) / std;
        diffN++;
      }
    }
    if (diffN && diffN > bestN * 0.8) {
      const d = diffSum / diffN;
      if (d < bestDiff) {
        bestDiff = d;
        bestN = diffN;
      }
    }
  }
  if (!bestN) return { score: 0.5, difference: 1, pairs: 0 };

  const difference = bestDiff;
  // Measured on a mirrored control face: 0.12. A normal face at this scale lands
  // around 0.7 to 1.0 once hair, expression and pose are included; 1.4 reads as
  // clearly uneven in person.
  const score = clamp01(1 - Math.max(0, difference - 0.3) / 1.15);
  return { score, difference, pairs: bestN };
}
