import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

/**
 * Skin is measured from pixels. Landmarks locate the face; pixels decide whether
 * there is a breakout on it.
 *
 * Three things matter for this to be honest on a real phone photo:
 *
 * 1. **Chroma, not brightness.** Redness is measured as `(r - (g+b)/2) / (r+g+b)`,
 *    which does not move when the whole frame gets brighter or darker. A pimple
 *    is redder than the skin around it; a shadow is only darker.
 * 2. **Two scales.** A small papule disappears under a blur sized to the cheek,
 *    a diffuse inflamed patch disappears under a small one.
 * 3. **A robust noise floor.** The threshold used to adapts to the image's own
 *    grain, but it is estimated with a *median*, not a mean. A face covered in
 *    spots drags a mean up to its own level and hides itself.
 *
 * Under-eye darkness is compared with a local baseline (a blur wide enough to
 * smooth over the circle but narrow enough to follow the lighting), so a light
 * from above and a general shadow are not scored as dark circles.
 */

export type SkinRegion = 'forehead' | 'left' | 'right' | 'chin';

export type SpotMark = { x: number; y: number; r: number };
export type UnderEyeMark = { x: number; y: number; rx: number; ry: number; darkness: number };

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
  /** normalised marks for the face map */
  marks: { spots: SpotMark[]; underEyes: UnderEyeMark[] };
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

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = values.sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
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
    marks: { spots: [], underEyes: [] },
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
  // Stay a little inside the outline: the hairline and jaw edge carry colour
  // fringing, and a fringe of hair samples as one huge "blemish".
  const cx = oval.reduce((s, q) => s + q.x, 0) / oval.length;
  const cy = oval.reduce((s, q) => s + q.y, 0) / oval.length;
  const mask = new Uint8Array(W * H);
  fillPolygon(
    mask,
    W,
    H,
    oval.map((q) => ({ x: cx + (q.x - cx) * 0.9, y: cy + (q.y - cy) * 0.88 })),
  );

  const eyeR = bbox([p(33), p(133), p(159), p(145), p(130)]);
  const eyeL = bbox([p(362), p(263), p(386), p(374), p(359)]);
  carve(mask, W, H, pad(eyeR, 0.16, 1.05, 0.18));
  carve(mask, W, H, pad(eyeL, 0.16, 1.05, 0.18));
  // Brows carry hard edges and hair; both look exactly like inflammation.
  carve(mask, W, H, pad(bbox([p(70), p(63), p(105), p(66), p(107), p(55), p(46)]), 0.08, 0.4, 0.15));
  carve(mask, W, H, pad(bbox([p(300), p(293), p(334), p(296), p(336), p(285), p(276)]), 0.08, 0.4, 0.15));
  carve(mask, W, H, pad(bbox([p(61), p(291), p(0), p(17), p(13), p(14)]), 0.22, 0.35, 0.35));
  // Nostrils and the crease beside the nose are the brightest false positives
  // on an otherwise clear face: red, small, and right where skin ends.
  carve(mask, W, H, pad(bbox([p(129), p(358), p(2), p(102), p(331)]), 0.18, 0.3, 0.22));

  const lum = new Float32Array(W * H);
  const red = new Float32Array(W * H);
  for (let i = 0, j = 0; i < W * H; i++, j += 4) {
    const r = data[j];
    const g = data[j + 1];
    const b = data[j + 2];
    lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    // Chroma redness: brightness-independent, so a darker room does not read as
    // a calmer face and a side light does not read as a breakout.
    red[i] = (255 * (r - (g + b) / 2)) / (r + g + b + 1);
  }

  const rFine = Math.max(2, Math.round(faceW * 0.013));
  const rCoarse = Math.max(4, Math.round(faceW * 0.045));
  const fineLum = boxBlur(lum, W, H, rFine);
  const fineRed = boxBlur(red, W, H, rFine);
  const coarseRed = boxBlur(red, W, H, rCoarse);
  const coarseLum = boxBlur(lum, W, H, rCoarse);

  // Robust noise floor: the median residual, sampled on a grid. A mean would be
  // pulled up by the very blemishes we are looking for, which is how a face full
  // of acne used to score as clear.
  const lumResid: number[] = [];
  const redResid: number[] = [];
  let lumSum = 0;
  let samples = 0;
  for (let i = 0; i < W * H; i += 2) {
    if (!mask[i]) continue;
    if (lum[i] < 20 || lum[i] > 250) continue;
    samples++;
    lumSum += lum[i];
    lumResid.push(Math.abs(lum[i] - fineLum[i]));
    redResid.push(Math.abs(red[i] - fineRed[i]));
  }
  if (samples < 300) return emptyReport();
  const pct = (values: number[], q: number) => {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
  };
  const noiseLum = Math.max(1.5, median(lumResid));
  const noiseRed = Math.max(1.1, median(redResid));
  // Thresholds follow the face's own distribution: a soft, well-lit selfie has
  // a tight redness spread, a grainy one a wide spread, and a threshold picked
  // from a fixed constant is wrong for one of them.
  const thRed = Math.min(20, Math.max(6, pct(redResid, 0.85) * 1.25));
  const thDark = Math.min(18, Math.max(5, pct(lumResid, 0.85) * 1.1));

  const soft = new Uint8Array(W * H);
  const flags = new Uint8Array(W * H);
  const strength = new Float32Array(W * H);
  let flaggedPixels = 0;
  let softPixels = 0;
  for (let i = 0; i < W * H; i++) {
    if (!mask[i]) continue;
    if (lum[i] < 20 || lum[i] > 250) continue;
    const fineRedRes = red[i] - fineRed[i];
    const coarseRedRes = red[i] - coarseRed[i];
    const fineDarkRes = fineLum[i] - lum[i];
    const coarseDarkRes = coarseLum[i] - lum[i];
    // A spot has to be redder than its surroundings, and a little darker.
    // Darkness alone turned stubble and shadows into 25 "spots" on a clear face.
    const redScore = Math.max(fineRedRes / thRed, coarseRedRes / (thRed * 1.35));
    const darkScore = Math.max(fineDarkRes / thDark, coarseDarkRes / (thDark * 1.35));
    if (redScore > 1 && darkScore > 0.45) {
      flags[i] = 1;
      strength[i] = redScore;
      flaggedPixels++;
    } else if (redScore > 0.75 && darkScore > 0.3) {
      // Sub-threshold redness still counts as inflammation. Wide, faint
      // breakouts live here, and counting them separately is what stops
      // "nothing passed the threshold" from reading as a clear face.
      soft[i] = 1;
      softPixels++;
    }
  }

  // Blob pass: only contiguous clusters count as spots, and only clusters big
  // enough to be more than a pixel of grain.
  const minBlob = Math.max(10, Math.round(faceW * faceW * 0.00026));
  const visited = new Uint8Array(W * H);
  const stack: number[] = [];
  const blobs: { x: number; y: number; area: number }[] = [];
  const regionSpots: number[] = [0, 0, 0, 0];
  const regionSkin = [0, 0, 0, 0];
  const browY = p(9).y;
  const mouthY = p(11).y;
  const midX = (p(234).x + p(454).x) / 2;
  const bandOf = (y: number, x: number) => (y < browY ? 0 : y < mouthY ? (x < midX ? 1 : 2) : 3);

  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (!mask[i]) continue;
      regionSkin[bandOf(y, x)]++;
    }
  }

  let spotPixels = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const start = y * W + x;
      if (!flags[start] || visited[start]) continue;
      let area = 0;
      let sumX = 0;
      let sumY = 0;
      let sumStrength = 0;
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
        sumStrength += strength[i];
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
      // A real spot is red *at its core*; slivers along an edge are not.
      if (area < minBlob || sumStrength / area < 1.15) continue;
      const bx = sumX / area;
      const by = sumY / area;
      blobs.push({ x: bx, y: by, area });
      spotPixels += area;
      regionSpots[bandOf(by, bx)]++;
    }
  }

  blobs.sort((a, b) => b.area - a.area);
  const spots = blobs.length;
  const skinPixels = Math.max(1, regionSkin.reduce((a, b) => a + b, 0));
  const blemishShare = spotPixels / skinPixels;
  const softShare = softPixels / samples;
  const regions: Record<SkinRegion, number> = {
    forehead: regionSpots[0],
    left: regionSpots[1],
    right: regionSpots[2],
    chin: regionSpots[3],
  };

  // Clarity blends coverage, cluster count and sub-threshold inflammation. All
  // three are continuous, so a face cannot jump from 40% to 95% because the
  // threshold happened to miss its particular kind of breakout.
  const sev = clamp01(
    blemishShare / 0.09 * 0.62 + Math.min(1, spots / 26) * 0.26 + Math.min(1, softShare / 0.22) * 0.16,
  );
  const clarity = Math.max(0.05, clamp01(1 - Math.pow(sev, 0.85)));

  const meanLum = lumSum / samples;
  // High-frequency residual is the honest texture signal: acne, scarring and
  // dry patches all raise it. The 75th percentile, not the median: most of a
  // face is smooth cheek, and the median would report that and nothing else.
  const texture = clamp01(1 - Math.max(0, pct(lumResid, 0.75) - 1.6) / 2.6);

  // ---- Under eyes -------------------------------------------------------
  // A band under each lower lid, compared against a blur wide enough to follow
  // the lighting but not wide enough to swallow the circle.
  const patchLums = (px: number, py: number, hw: number, hh: number): number[] => {
    const out: number[] = [];
    const ax = Math.max(1, Math.round(px - hw));
    const bx = Math.min(W - 2, Math.round(px + hw));
    const ay = Math.max(1, Math.round(py - hh));
    const by = Math.min(H - 2, Math.round(py + hh));
    for (let y = ay; y <= by; y++) {
      for (let x = ax; x <= bx; x++) {
        const i = y * W + x;
        if (!mask[i] || lum[i] < 18 || lum[i] > 250) continue;
        out.push(lum[i]);
      }
    }
    return out;
  };

  const underEyesMarks: UnderEyeMark[] = [];
  const darkness: number[] = [];
  for (const eye of [eyeR, eyeL]) {
    const w = Math.max(4, eye.x1 - eye.x0);
    const h = Math.max(3, eye.y1 - eye.y0);
    const ex = (eye.x0 + eye.x1) / 2;
    const y0 = eye.y1 + h * 0.55;
    const y1 = eye.y1 + h * 2.0;
    const under = patchLums(ex, (y0 + y1) / 2, w * 0.46, (y1 - y0) / 2);
    if (under.length < 60) continue;
    // Reference at the same height, on the temple and beside the nose, so a
    // light from above (or a generally shadowed face) cannot pass for circles.
    // Several candidate patches, and the brighter half of them: if one lands on
    // hair or the shadow beside the nose, taking it would invent a circle.
    const refs = [1.15, 1.45]
      .flatMap((k) => [ex - w * k, ex + w * k])
      .map((rx) => patchLums(rx, (y0 + y1) / 2, w * 0.28, (y1 - y0) / 2))
      .filter((v) => v.length > 30)
      .map((v) => median(v))
      .sort((a, b) => a - b);
    if (!refs.length) continue;
    const underLum = median(under);
    const refLum = refs[Math.floor(refs.length * 0.6)];
    if (refLum < 20) continue;
    // Median of the band, so a stray eyelash or a glint cannot set the number.
    const d = Math.max(0, (refLum - underLum) / refLum);
    darkness.push(d);
    underEyesMarks.push({
      x: ex / W,
      y: (y0 + y1) / 2 / H,
      rx: (w * 0.46) / W,
      ry: ((y1 - y0) / 2) / H,
      darkness: d,
    });
  }
  const underDark = darkness.length ? darkness.reduce((a, b) => a + b, 0) / darkness.length : 0;
  // 3.5% darker than the skin beside it is a faint circle; 20% is severe.
  const underEyes = clamp01(1 - (underDark - 0.035) / 0.2);

  const notes: string[] = [];
  if (spots >= 3) notes.push(`${spots} visible spots`);
  if (clarity < 0.45) notes.push('Active breakouts');
  if (texture < 0.5) notes.push('Textured surface');
  if (underEyes < 0.5) notes.push('Under-eye shadows');

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
    marks: {
      spots: blobs.slice(0, 40).map((b) => ({
        x: b.x / W,
        y: b.y / H,
        r: (Math.sqrt(b.area / Math.PI) / W) * 1.6,
      })),
      underEyes: underEyesMarks,
    },
  };
}
