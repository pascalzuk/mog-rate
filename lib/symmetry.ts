import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

/**
 * Symmetry by flipping the face and subtracting it from itself.
 *
 * A luminance difference is the obvious approach and a bad one: skin tone is
 * smooth, so most of the face cancels no matter how lopsided the bones are, and
 * whatever is left is dominated by a light coming from one side. What actually
 * differs on an asymmetric face is where the *structure* is: the jaw outline,
 * the brow line, the lid crease, the mouth corner.
 *
 * So the comparison runs on edge magnitude, not brightness:
 *
 * 1. Warp into a canonical frame (roll, scale and head turn removed) so a
 *    mirrored copy can be compared pixel for pixel.
 * 2. Compare Sobel edges, each half scaled to the other's edge energy so a
 *    light from the left cannot look like a wider left cheek.
 * 3. Fit the mirror axis from the data instead of assuming it sits at the eye
 *    midpoint, then take a weighted RMS: mismatched jaw and eye outlines count
 *    for more than a flat forehead, because that is what reads as uneven.
 * 4. Add a small tonal term for one-sided breakouts or shadow, which have no
 *    edge to speak of.
 */

export type FlipReport = {
  /** 0..1, 1 = the halves match */
  score: number;
  /** structural mismatch, edge units. ~0.10 even face, 0.5 clearly uneven */
  difference: number;
  /** tonal mismatch, std units */
  tone: number;
  /** pixels compared */
  pairs: number;
};

const CW = 160;
const CH = 200;
// The face's own midline has to sit on the mirror axis afterwards. It used to be
// placed at 30% of the crop while the comparison mirrored about 50%, so the test
// was comparing the face with a shifted copy of itself and reported a constant
// "asymmetry" no matter what the face looked like.
const CX = CW / 2;
const CY = CH * 0.42;

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

/** 3x3 box blur, run twice to take the resampling noise out. */
function smooth(src: Float32Array, w: number, h: number, passes = 2): Float32Array {
  let a = src;
  for (let pass = 0; pass < passes; pass++) {
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
  yawDeg = 0,
): FlipReport {
  const W = pixels.width;
  const H = pixels.height;
  if (!landmarks || landmarks.length < 468) return { score: 0.5, difference: 0.5, tone: 0.5, pairs: 0 };

  const lm = landmarks;
  const p = (i: number) => ({ x: lm[i].x * W, y: lm[i].y * H });
  const eyeR = { x: (p(33).x + p(133).x) / 2, y: (p(33).y + p(133).y) / 2 };
  const eyeL = { x: (p(362).x + p(263).x) / 2, y: (p(362).y + p(263).y) / 2 };
  const dx = eyeL.x - eyeR.x;
  const dy = eyeL.y - eyeR.y;
  const d = Math.hypot(dx, dy);
  if (d < 8) return { score: 0.5, difference: 0.5, tone: 0.5, pairs: 0 };

  // The canonical frame is centred on the midpoint between the eyes, which is
  // where the mirror axis lives. Anchoring on one eye (as this did) puts the
  // face's midline half an eye-distance away from the comparison axis, so the
  // halves scored against each other were never the two halves of the face.
  const eyeMid = { x: (eyeR.x + eyeL.x) / 2, y: (eyeR.y + eyeL.y) / 2 };

  const roll = Math.atan2(dy, dx);
  const cos = Math.cos(roll);
  const sin = Math.sin(roll);
  const scale = d / (0.4 * CW);
  // A turned head compresses the face horizontally by cos(yaw). Undo it, or the
  // near side of every jaw would read as a narrower jaw.
  const cosYaw = Math.max(0.55, Math.cos((Math.max(-60, Math.min(60, yawDeg)) * Math.PI) / 180));

  const toCanonical = (x: number, y: number) => {
    const rx = (x - eyeMid.x) / scale;
    const ry = (y - eyeMid.y) / scale;
    return {
      x: CX + (rx * cos + ry * sin) / cosYaw,
      y: CY - rx * sin + ry * cos,
    };
  };

  const crop = new Float32Array(CW * CH);
  const filled = new Uint8Array(CW * CH);
  for (let cy = 0; cy < CH; cy++) {
    for (let cx = 0; cx < CW; cx++) {
      const ox = (cx - CX) * scale;
      const oy = (cy - CY) * scale;
      const ux = ox * cosYaw;
      const sx = eyeMid.x + ux * cos - oy * sin;
      const sy = eyeMid.y + ux * sin + oy * cos;
      const g = bilinear(pixels.data, W, H, sx, sy);
      if (g < 0) continue;
      crop[cy * CW + cx] = g;
      filled[cy * CW + cx] = 1;
    }
  }

  const blurLum = smooth(crop, CW, CH, 2);

  // Sobel on the smoothed luminance.
  const edge = new Float32Array(CW * CH);
  for (let y = 1; y < CH - 1; y++) {
    for (let x = 1; x < CW - 1; x++) {
      const i = y * CW + x;
      const tl = blurLum[i - CW - 1];
      const t = blurLum[i - CW];
      const tr = blurLum[i - CW + 1];
      const l = blurLum[i - 1];
      const r = blurLum[i + 1];
      const bl = blurLum[i + CW - 1];
      const b = blurLum[i + CW];
      const br = blurLum[i + CW + 1];
      const gx = tr + 2 * r + br - (tl + 2 * l + bl);
      const gy = bl + 2 * b + br - (tl + 2 * t + tr);
      edge[i] = Math.hypot(gx, gy) / 8;
    }
  }
  const blurEdge = smooth(edge, CW, CH, 1);

  // Oval mask, pulled in so the hairline and background never enter the test.
  const poly = OVAL.map((i) => toCanonical(p(i).x, p(i).y));
  const ox = poly.reduce((s, q) => s + q.x, 0) / poly.length;
  const oy = poly.reduce((s, q) => s + q.y, 0) / poly.length;
  const mask = new Uint8Array(CW * CH);
  const xs: number[] = [];
  for (let y = 0; y < CH; y++) {
    xs.length = 0;
    for (let i = 0; i < poly.length; i++) {
      // Keep almost all of the outline sideways: the jaw and cheek contour is
      // exactly where asymmetry shows, and trimming it out made the test blind
      // to a narrower jaw. Vertical is trimmed harder to stay off the hairline.
      const a = { x: ox + (poly[i].x - ox) * 0.97, y: oy + (poly[i].y - oy) * 0.84 };
      const b = {
        x: ox + (poly[(i + 1) % poly.length].x - ox) * 0.97,
        y: oy + (poly[(i + 1) % poly.length].y - oy) * 0.84,
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

  // Feature bands in canonical space, so the eyes and mouth outweigh a flat
  // forehead no matter where they landed in the frame.
  const cBrow = toCanonical((p(105).x + p(334).x) / 2, (p(105).y + p(334).y) / 2);
  const cEye = toCanonical((p(33).x + p(263).x) / 2, (p(33).y + p(263).y) / 2);
  const cMouth = toCanonical((p(61).x + p(291).x) / 2, (p(61).y + p(291).y) / 2);
  const cChin = toCanonical(p(152).x, p(152).y);
  const eyeH = Math.max(3, cMouth.y - cEye.y);
  const weight = new Float32Array(CW * CH);
  for (let y = 0; y < CH; y++) {
    for (let x = 0; x < CW; x++) {
      let w = 1;
      if (y > cBrow.y - 0.35 * eyeH && y < cEye.y + 0.45 * eyeH) w = 1.7;
      else if (y > cMouth.y - 0.25 * eyeH && y < cChin.y) w = 1.35;
      weight[y * CW + x] = w;
    }
  }

  // Only the right half is scored; the left comes in through the mirror.
  const xStart = Math.round(CW * 0.53);
  const xEnd = Math.round(CW * 0.9);
  // Start just above the brows. Reaching higher drags the fringe into a test
  // that is supposed to be about bone structure, and hair never mirrors.
  const yStart = Math.max(2, Math.round(cBrow.y - 0.18 * eyeH));
  const yEnd = Math.min(CH - 2, Math.round(cChin.y));

  let bestDiff = Number.POSITIVE_INFINITY;
  let bestPairs = 0;
  for (let axis = -7; axis <= 7; axis++) {
    let sumWL = 0;
    let sumWR = 0;
    let n = 0;
    for (let y = yStart; y < yEnd; y++) {
      for (let x = xStart; x < xEnd; x++) {
        const mx = CW - 1 - x + axis;
        if (mx < 1 || mx >= CW - 1) continue;
        const i = y * CW + x;
        const j = y * CW + mx;
        if (!mask[i] || !mask[j] || !filled[i] || !filled[j]) continue;
        const w = weight[i];
        sumWL += blurEdge[i] * w;
        sumWR += blurEdge[j] * w;
        n++;
      }
    }
    if (n < CW * 8) continue;
    const meanL = sumWL / n;
    const meanR = sumWR / n;
    if (meanL < 0.05 || meanR < 0.05) continue;
    // Scale to equal edge energy first: lighting varies, bone structure does not.
    const k = meanL / meanR;
    let acc = 0;
    let wsum = 0;
    for (let y = yStart; y < yEnd; y++) {
      for (let x = xStart; x < xEnd; x++) {
        const mx = CW - 1 - x + axis;
        if (mx < 1 || mx >= CW - 1) continue;
        const i = y * CW + x;
        const j = y * CW + mx;
        if (!mask[i] || !mask[j] || !filled[i] || !filled[j]) continue;
        const w = weight[i];
        const diff = blurEdge[i] - blurEdge[j] * k;
        acc += diff * diff * w;
        wsum += w;
      }
    }
    if (!wsum) continue;
    const diff = Math.sqrt(acc / wsum) / meanL;
    // Prefer axes that actually compare a decent share of the face.
    const scoreDiff = diff * (1 + Math.abs(axis) * 0.004);
    if (scoreDiff < bestDiff) {
      bestDiff = scoreDiff;
      bestPairs = n;
    }
  }
  if (!bestPairs) return { score: 0.5, difference: 0.5, tone: 0.5, pairs: 0 };

  // Tonal term: one-sided breakouts or shadow, which have no edge to compare.
  let toneSum = 0;
  let toneCount = 0;
  for (let y = yStart; y < yEnd; y++) {
    for (let x = xStart; x < xEnd; x++) {
      const mx = CW - 1 - x;
      const i = y * CW + x;
      const j = y * CW + mx;
      if (!mask[i] || !mask[j] || !filled[i] || !filled[j]) continue;
      toneSum += Math.abs(blurLum[i] - blurLum[j]);
      toneCount++;
    }
  }
  let tone = 0;
  if (toneCount > 200) {
    let mean = 0;
    for (let y = 0; y < CH; y++) for (let x = 0; x < CW; x++) if (mask[y * CW + x]) mean += blurLum[y * CW + x];
    const nMask = mask.reduce((a, b) => a + b, 0) || 1;
    mean /= nMask;
    let varSum = 0;
    for (let y = 0; y < CH; y++) {
      for (let x = 0; x < CW; x++) {
        const i = y * CW + x;
        if (!mask[i]) continue;
        const d0 = blurLum[i] - mean;
        varSum += d0 * d0;
      }
    }
    const std = Math.sqrt(varSum / nMask) || 1;
    tone = toneSum / toneCount / std;
  }

  const difference = 0.82 * bestDiff + 0.18 * tone;

  // 0.18 is the floor measured by mirroring a face onto itself; 0.93 is a face
  // with one side visibly out of line. The mapping is deliberately not a
  // straight line: small differences are normal, large ones are not.
  const score = clamp01(1 - Math.max(0, difference - 0.18) / 0.75);
  return { score, difference, tone, pairs: bestPairs };
}
