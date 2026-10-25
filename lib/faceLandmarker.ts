import type { FaceLandmarker, NormalizedLandmark } from '@mediapipe/tasks-vision';

const WASM_DIR = '/mediapipe/wasm';
const MODEL_PATH = '/models/face_landmarker.task';

let instance: Promise<FaceLandmarker> | null = null;

/**
 * Loads MediaPipe Face Landmarker (468 mesh points + 10 iris points) once.
 * Runs locally in the browser: the photo and the model never leave the device.
 * GPU is used when available, CPU as a fallback for older phones.
 */
export function loadFaceLandmarker(): Promise<FaceLandmarker> {
  if (!instance) {
    instance = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
      const fileset = await FilesetResolver.forVisionTasks(WASM_DIR);
      const base = {
        runningMode: 'VIDEO' as const,
        numFaces: 1,
        minFaceDetectionConfidence: 0.4,
        minFacePresenceConfidence: 0.4,
        minTrackingConfidence: 0.4,
      };
      try {
        return await FaceLandmarker.createFromOptions(fileset, {
          ...base,
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
        });
      } catch {
        return await FaceLandmarker.createFromOptions(fileset, {
          ...base,
          baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
        });
      }
    })().catch((err) => {
      instance = null;
      throw err;
    });
  }
  return instance;
}

/** Runs detection on a canvas / video frame and returns the first face's landmarks. */
export function detectFace(
  landmarker: FaceLandmarker,
  source: HTMLCanvasElement | HTMLVideoElement,
  timestamp: number,
): NormalizedLandmark[] | null {
  const result = landmarker.detectForVideo(source, timestamp);
  return result.faceLandmarks?.[0] ?? null;
}
