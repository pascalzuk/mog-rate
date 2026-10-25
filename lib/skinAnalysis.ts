import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

/**
 * Skin is measured from pixels. Landmarks locate the face; pixels decide whether
 * there is a breakout on it.
 *
 * Two scales are used on purpose. A five-pixel pimple disappears under a blur
 * sized to the whole cheek, while a broad inflamed patch disappears under a
 * five-pixel one. So the sampler keeps a fine baseline (for papules) and a coarse
 * baseline (for diffuse redness), then groups flagged pixels into blobs and only
 * counts blobs large enough to be a real spot. The thresholds adapt to the
 * image's own noise floor, because a phone frame and a recompressed photo do not
 * have the same grain.
 */

export type SkinRegion = 'forehead' | 'left' | 'right' | 'chin';

export type SkinReport = {
  /** 0..1, 1 = clear */
  clarity: number;
  /** 0..1, 1 = smooth */
  texture: number;
  /** 0..1, 1 = no dark circles */
  underEyes: number;
  /** share of measured skin pixels inside a counted spot */
  blemishShare: number;
  /** number of distinct spot clusters */
  spots: number;
  regions: Record<SkinRegion, number>;
  lighting: number;
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

/** Separable box blur, clamped at the edges. Approximates local skin tone. */
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
    clarity: 0.55,
    texture: 0.6,
    underEyes: 0.7,
    blemishShare: 0,
    spots: 0,
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
  if (!landmarks || landmarks.length < 468 || W < 48 || H < 48) return emptyReport();

  const pts: P[] = landmarks.map((l) => ({ x: l.x * W, y: l.y * H }));
  const p = (i: number) => pts[i] ?? pts[0];

  const faceW = Math.max(16, Math.hypot(p(234).x - p(454).x, p(234).y - p(454).y));
  const oval = OVAL.map((i) => p(i));
  // Stay 5% inside the outline: the hairline and jaw edge carry colour fringing.
  const cx = oval.reduce((s, q) => s + q.x, 0) / oval.length;
  const cy = oval.reduce((s, q) => s + q.y, 0) / oval.length;
  const mask = new Uint8Array(W * H);
  fillPolygon(
    mask,
    W,
    H,
    oval.map((q) => ({ x: cx + (q.x - cx) * 0.95, y: cy + (q.y - cy) * 0.95 })),
  );

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

  const rFine = Math.max(2, Math.round(faceW * 0.014));
  const rCoarse = Math.max(4, Math.round(faceW * 0.05));
  const fineLum = boxBlur(lum, W, H, rFine);
  const fineRed = boxBlur(red, W, H, rFine);
  const coarseLum = boxBlur(lum, W, H, rCoarse);
  const coarseRed = boxBlur(red, W, H, rCoarse);

  // Noise floor from the image itself, so grain and compression do not read as
  // blemishes and genuinely sharp photos are not over-smoothed.
  let noiseSum = 0;
  let noiseN = 0;
  let lumSum = 0;
  let samples = 0;
  for (let i = 0; i < W * H; i++) {
    if (!mask[i]) continue;
    if (lum[i] < 22 || lum[i] > 250) continue;
    samples++;
    lumSum += lum[i];
    noiseSum += Math.abs(lum[i] - fineLum[i]);
    noiseN++;
  }
  if (samples < 500) return emptyReport();
  const noise = noiseN ? noiseSum / noiseN : 2;
  const th = Math.max(4.5, noise * 1.9);

  const flags = new Uint8Array(W * H);
  let flaggedPixels = 0;
  for (let i = 0; i < W * H; i++) {
    if (!mask[i]) continue;
    if (lum[i] < 22 || lum[i] > 250) continue;
    const fineRedRes = red[i] - fineRed[i];
    const fineDarkRes = fineLum[i] - lum[i];
    const coarseRedRes = red[i] - coarseRed[i];
    const coarseDarkRes = coarseLum[i] - lum[i];
    // A spot has to be both redder and darker than the skin around it. Trying a
    // darkness-only path as well turned shadows and stubble on a clear face into
    // 25 "spots", so redness stays a required signal.
    const fineSpot = fineRedRes > th && fineDarkRes > th * 0.55;
    const broadSpot = coarseRedRes > th * 1.4 && coarseDarkRes > th * 0.9;
    if (fineSpot || broadSpot) {
      flags[i] = 1;
      flaggedPixels++;
    }
  }

  // Blob pass: only contiguous clusters count as spots.
  const minBlob = Math.max(9, Math.round(faceW * faceW * 0.00026));
  const visited = new Uint8Array(W * H);
  const stack: number[] = [];
  let spots = 0;
  let spotPixels = 0;
  const regionSpots: number[] = [0, 0, 0, 0];
  const regionSkin = [0, 0, 0, 0];
  const browY = p(9).y;
  const mouthY = p(13).y;
  const midX = (p(234).x + p(454).x) / 2;
  const bandOf = (y: number, x: number) => (y < browY ? 0 : y < mouthY ? (x < midX ? 1 : 2) : 3);

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (!mask[i]) continue;
      regionSkin[bandOf(y, x)]++;
    }
  }

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const start = y * W + x;
      if (!flags[start] || visited[start]) continue;
      let area = 0;
      let sumX = 0;
      let sumY = 0;
      stack.length = 0;
      stack.push(start);
      visited[start] = 1;
      while (stack.length) {
        const i = stack.pop()!;
        const iy = (i / W) | 0;
        const ix = i - iy * W;
        area++;
        sumX += ix;
        sumY += iy;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = ix + dx;
            const ny = iy + dy;
            if (nx < 1 || ny < 1 || nx >= W - 1 || ny >= H - 1) continue;
            const ni = ny * W + nx;
            if (!flags[ni] || visited[ni]) continue;
            visited[ni] = 1;
            stack.push(ni);
          }
        }
      }
      if (area < minBlob) continue;
      spots++;
      spotPixels += area;
      regionSpots[bandOf(sumY / area, sumX / area)]++;
    }
  }

  const skinPixels = Math.max(1, regionSkin.reduce((a, b) => a + b, 0));
  const blemishShare = spotPixels / skinPixels;
  const regions: Record<SkinRegion, number> = {
    forehead: regionSpots[0],
    left: regionSpots[1],
    right: regionSpots[2],
    chin: regionSpots[3],
  };

  // A spot cluster and a covered area are both real deductions. Twelve small
  // pimples is not a clear face just because they cover 2% of the skin.
  // Coverage is the honest signal; counting is a smaller correction so that a
  // few stray blobs on an otherwise clear face do not read as a breakout.
  const areaTerm = blemishShare / 0.08;
  const countTerm = Math.min(1, spots / 24);
  const severity = clamp01(areaTerm * 0.8 + countTerm * 0.2);
  const coverageCap = blemishShare > 0.04 ? 0.4 : blemishShare > 0.02 ? 0.6 : 0.98;
  const clarity = Math.max(0.04, Math.min(coverageCap, clamp01(1 - Math.pow(severity, 0.9))));

  const meanLum = lumSum / samples;
  // High-frequency residual is the honest texture signal: acne, scarring and
  // dry patches all raise it, while a smooth cheek keeps it near the noise floor.
  // The old gradient ratio saturated at 100% on almost every face.
  const texture = Math.min(0.97, clamp01(1 - Math.max(0, noise - 2) / 1.7));

  const eyeH = Math.max(2, (eyeR.y1 - eyeR.y0 + eyeL.y1 - eyeL.y0) / 2);
  const patchMean = (px: number, py: number, hw: number, hh: number): number | null => {
    let sum = 0;
    let n = 0;
    const x0 = Math.max(1, Math.round(px - hw));
    const x1 = Math.min(W - 2, Math.round(px + hw));
    const y0 = Math.max(1, Math.round(py - hh));
    const y1 = Math.min(H - 2, Math.round(py + hh));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * W + x;
        if (!mask[i] || lum[i] < 22 || lum[i] > 248) continue;
        sum += lum[i];
        n++;
      }
    }
    return n > 12 ? sum / n : null;
  };

  const ratios: number[] = [];
  for (const eye of [eyeR, eyeL]) {
    const ex = (eye.x0 + eye.x1) / 2;
    const under = patchMean(ex, eye.y1 + eyeH * 0.5, (eye.x1 - eye.x0) * 0.55, eyeH * 0.4);
    const cheek = patchMean(ex, eye.y1 + eyeH * 3.1, (eye.x1 - eye.x0) * 0.6, eyeH * 0.9);
    if (under && cheek && cheek > 1) ratios.push(under / cheek);
  }
  const underRatio = ratios.length ? ratios.reduce((a, b) => a + b, 0) / ratios.length : 0.9;
  const underEyes = clamp01((underRatio - 0.52) / 0.42);

  const notes: string[] = [];
  if (spots >= 3) notes.push(`${spots} visible spots`);
  if (clarity < 0.45) notes.push('Active breakouts');
  if (texture < 0.5) notes.push('Textured surface');
  if (underEyes < 0.42) notes.push('Under-eye shadows');

  return {
    clarity,
    texture,
    underEyes,
    blemishShare,
    spots,
    regions,
    lighting: clamp01(1 - Math.abs(meanLum - 142) / 130),
    samples,
    notes,
  };
}
