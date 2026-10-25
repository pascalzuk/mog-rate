import { analyzeFace, type ScanResult } from '@/lib/faceMetrics';
import { detectFace, loadFaceLandmarker } from '@/lib/faceLandmarker';

// Same width the live capture uses, so a photo and a scan are measured with the
// same amount of pixel detail behind them.
export const ANALYSIS_WIDTH = 800;

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('That image could not be decoded.'));
    img.src = url;
  });
}

/**
 * Runs the same measurement as a live scan on a picked image: full-resolution
 * decode, downscale to the analysis width, landmarks, pixels, one read.
 *
 * A photo cannot be guided through the five poses, so it is measured once and
 * the result is labelled as a photo rather than a scan.
 */
export async function scanPhoto(file: File): Promise<{ result: ScanResult; photo: string }> {
  if (!file.type.startsWith('image/')) throw new Error('That file is not an image.');

  const url = URL.createObjectURL(file);
  try {
    const img = await loadImage(url);
    if (!img.naturalWidth || !img.naturalHeight) throw new Error('That image is empty.');

    // Never upscale: a 400px photo gains nothing from being blown up to 800.
    const w = Math.min(ANALYSIS_WIDTH, img.naturalWidth);
    const h = Math.max(1, Math.round((w * img.naturalHeight) / img.naturalWidth));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('This browser cannot read images.');
    ctx.drawImage(img, 0, 0, w, h);

    const landmarker = await loadFaceLandmarker();
    const landmarks = detectFace(landmarker, canvas, performance.now());
    if (!landmarks) {
      throw new Error('No face in that photo. Use one where your whole face is visible.');
    }

    const pixels = ctx.getImageData(0, 0, w, h);
    const result = analyzeFace({ landmarks, width: w, height: h, pixels }, 1);
    return { result, photo: canvas.toDataURL('image/jpeg', 0.9) };
  } finally {
    URL.revokeObjectURL(url);
  }
}
