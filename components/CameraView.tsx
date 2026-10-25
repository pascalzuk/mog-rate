'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowsClockwise, ImageSquare, X } from '@phosphor-icons/react';
import type { FaceLandmarker, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { detectFace, loadFaceLandmarker } from '@/lib/faceLandmarker';
import { analyzeFace, mergeScans } from '@/lib/faceMetrics';
import type { ScanResult } from '@/lib/faceMetrics';

type Box = { x: number; y: number; w: number; h: number };

type Props = {
  onScanned: (result: ScanResult, photo: string) => void;
  onClose: () => void;
};

const MINT = '#4cff9b';
const SAMPLES = 7;
const SAMPLE_GAP = 130;
const ANALYSIS_WIDTH = 640;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

function faceBox(landmarks: NormalizedLandmark[]): Box {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const padX = (maxX - minX) * 0.06;
  const padY = (maxY - minY) * 0.06;
  const x = Math.max(0, minX - padX);
  const y = Math.max(0, minY - padY);
  return { x, y, w: Math.min(1, maxX + padX) - x, h: Math.min(1, maxY + padY) - y };
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

function drawOverlay(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement | null,
  current: { box: Box | null; ratio: number },
  target: { box: Box | null },
  mirrored: boolean,
  scanning: boolean,
) {
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (!cw || !ch) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  const pw = Math.round(cw * dpr);
  const ph = Math.round(ch * dpr);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  const prev = current.box;
  const next = target.box;
  const eased = next
    ? prev
      ? {
          x: prev.x + (next.x - prev.x) * 0.24,
          y: prev.y + (next.y - prev.y) * 0.24,
          w: prev.w + (next.w - prev.w) * 0.24,
          h: prev.h + (next.h - prev.h) * 0.24,
        }
      : { ...next }
    : null;
  current.box = eased;

  if (!eased || !video || !video.videoWidth) {
    current.ratio = 0;
    // Idle guide: an oval where the face should go.
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.16)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([7, 9]);
    ctx.beginPath();
    ctx.ellipse(cw / 2, ch / 2, cw * 0.31, ch * 0.34, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    return;
  }

  // object-cover mapping from video space to the on-screen stage
  const scale = Math.max(cw / video.videoWidth, ch / video.videoHeight);
  const dw = video.videoWidth * scale;
  const dh = video.videoHeight * scale;
  const ox = (cw - dw) / 2;
  const oy = (ch - dh) / 2;

  let x = ox + eased.x * dw;
  const y = oy + eased.y * dh;
  const w = eased.w * dw;
  const h = eased.h * dh;
  if (mirrored) x = cw - x - w;

  const inset = 14;
  const left = Math.max(inset, x);
  const top = Math.max(inset, y);
  const bw = Math.max(56, Math.min(cw - inset, x + w) - left);
  const bh = Math.max(56, Math.min(ch - inset, y + h) - top);
  current.ratio = bh / ch;
  const corner = Math.min(bw, bh) * 0.2;
  const radius = Math.min(bw, bh) * 0.16;

  ctx.save();
  roundedRect(ctx, left, top, bw, bh, radius);
  ctx.fillStyle = scanning ? 'rgba(76, 255, 155, 0.12)' : 'rgba(76, 255, 155, 0.05)';
  ctx.fill();
  ctx.strokeStyle = scanning ? 'rgba(76, 255, 155, 0.75)' : 'rgba(76, 255, 155, 0.24)';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = MINT;
  ctx.lineWidth = scanning ? 4 : 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = 'rgba(76, 255, 155, 0.7)';
  ctx.shadowBlur = scanning ? 20 : 12;
  const corners: Array<[number, number, number, number]> = [
    [left, top, 1, 1],
    [left + bw, top, -1, 1],
    [left, top + bh, 1, -1],
    [left + bw, top + bh, -1, -1],
  ];
  for (const [cx, cy, dx, dy] of corners) {
    ctx.beginPath();
    ctx.moveTo(cx + dx * corner, cy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx, cy + dy * corner);
    ctx.stroke();
  }
  ctx.restore();

  // centre tick marks, so the box reads as a viewfinder rather than a sticker
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.22)';
  ctx.lineWidth = 1;
  const cx = left + bw / 2;
  const cy = top + bh / 2;
  ctx.beginPath();
  ctx.moveTo(cx - 7, cy);
  ctx.lineTo(cx + 7, cy);
  ctx.moveTo(cx, cy - 7);
  ctx.lineTo(cx, cy + 7);
  ctx.stroke();
  ctx.restore();
}

export default function CameraView({ onScanned, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef(0);
  const smoothRef = useRef<{ box: Box | null; ratio: number }>({ box: null, ratio: 0 });
  const targetRef = useRef<{ box: Box | null }>({ box: null });
  const distRef = useRef<'far' | 'good' | 'close' | null>(null);
  const lastSeenRef = useRef(0);
  const lastDetectRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const faceOnRef = useRef(false);
  const facingRef = useRef<'user' | 'environment'>('user');
  const scanningRef = useRef(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const hintTimer = useRef<number | null>(null);

  const [facing, setFacing] = useState<'user' | 'environment'>('user');
  const [stageAspect, setStageAspect] = useState(0.74);
  const [modelReady, setModelReady] = useState(false);
  const [faceOn, setFaceOn] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [distance, setDistance] = useState<'far' | 'good' | 'close' | null>(null);

  facingRef.current = facing;

  const flashHint = useCallback((message: string) => {
    setHint(message);
    if (hintTimer.current) window.clearTimeout(hintTimer.current);
    hintTimer.current = window.setTimeout(() => setHint(null), 3200);
  }, []);

  useEffect(() => {
    let alive = true;
    loadFaceLandmarker()
      .then((lm) => {
        if (!alive) return;
        landmarkerRef.current = lm;
        setModelReady(true);
      })
      .catch(() => {
        if (alive) setCamError('Scanner failed to load. Check your connection and retry.');
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    setCamError(null);
    setFaceOn(false);
    faceOnRef.current = false;
    targetRef.current.box = null;
    smoothRef.current.box = null;
    smoothRef.current.ratio = 0;
    distRef.current = null;
    setDistance(null);

    const open = async (constraints: MediaStreamConstraints) =>
      navigator.mediaDevices.getUserMedia(constraints);

    const start = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('unsupported');
        streamRef.current?.getTracks().forEach((t) => t.stop());

        // Ask for a tall frame. A square or 16:9 track is what made the old
        // preview feel zoomed: cropping a wide sensor into a portrait screen
        // shows only the middle quarter of the image.
        let stream: MediaStream;
        try {
          stream = await open({
            audio: false,
            video: { facingMode: facing, width: { ideal: 1080 }, height: { ideal: 1440 } },
          });
        } catch {
          stream = await open({ audio: false, video: { facingMode: facing } });
        }

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          await video.play().catch(() => undefined);
        }
      } catch (err) {
        if (cancelled) return;
        const name = (err as { name?: string })?.name;
        if (name === 'NotAllowedError') {
          setCamError('Camera access is off. Allow it in Settings, or use a photo instead.');
        } else if (name === 'NotFoundError') {
          setCamError('No camera found. Use a photo from your library instead.');
        } else {
          setCamError('Could not open the camera. Use a photo instead.');
        }
      }
    };

    void start();
    return () => {
      cancelled = true;
    };
  }, [facing]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (hintTimer.current) window.clearTimeout(hintTimer.current);
    };
  }, []);

  const onLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;
    const aspect = video.videoWidth / video.videoHeight;
    // Never show less than about half of the sensor's horizontal field.
    setStageAspect(clamp(aspect, 0.66, 0.95));
  }, []);

  // Live tracking loop: detect ~14x/sec, draw the lock box every animation frame.
  useEffect(() => {
    if (!modelReady) return;
    let stopped = false;
    const work = document.createElement('canvas');
    const ctx = work.getContext('2d');

    const frame = () => {
      if (stopped) return;
      const now = performance.now();
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      const overlay = overlayRef.current;

      if (
        video &&
        landmarker &&
        ctx &&
        video.readyState >= 2 &&
        video.videoWidth > 0 &&
        now - lastDetectRef.current > 80 &&
        video.currentTime !== lastVideoTimeRef.current
      ) {
        lastDetectRef.current = now;
        lastVideoTimeRef.current = video.currentTime;
        const w = 400;
        const h = Math.max(1, Math.round((w * video.videoHeight) / video.videoWidth));
        if (work.width !== w || work.height !== h) {
          work.width = w;
          work.height = h;
        }
        ctx.drawImage(video, 0, 0, w, h);
        try {
          const landmarks = detectFace(landmarker, work, now);
          if (landmarks) {
            targetRef.current.box = faceBox(landmarks);
            lastSeenRef.current = now;
          } else if (now - lastSeenRef.current > 700) {
            targetRef.current.box = null;
          }
        } catch {
          // dropped frame, next tick retries
        }
      }

      if (overlay) {
        drawOverlay(
          overlay,
          video,
          smoothRef.current,
          targetRef.current,
          facingRef.current === 'user',
          scanningRef.current,
        );
      }

      const locked = performance.now() - lastSeenRef.current < 520;
      if (locked !== faceOnRef.current) {
        faceOnRef.current = locked;
        setFaceOn(locked);
      }

      const ratio = smoothRef.current.ratio;
      const nextDist: 'far' | 'good' | 'close' | null = !locked
        ? null
        : ratio < 0.3
          ? 'far'
          : ratio > 0.82
            ? 'close'
            : 'good';
      if (nextDist !== distRef.current) {
        distRef.current = nextDist;
        setDistance(nextDist);
      }

      rafRef.current = requestAnimationFrame(frame);
    };

    rafRef.current = requestAnimationFrame(frame);
    return () => {
      stopped = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, [modelReady]);

  const scan = useCallback(async () => {
    if (scanningRef.current) return;
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !landmarker || video.readyState < 2 || !video.videoWidth) {
      flashHint('Camera is still warming up.');
      return;
    }
    if (!faceOnRef.current) {
      flashHint('Line your face up in the box first.');
      return;
    }

    scanningRef.current = true;
    setScanning(true);
    setProgress(0);

    const canvas = document.createElement('canvas');
    const w = ANALYSIS_WIDTH;
    const h = Math.max(1, Math.round((w * video.videoHeight) / video.videoWidth));
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    const samples: ScanResult[] = [];
    let bestShot = '';
    let bestPose = Number.POSITIVE_INFINITY;

    try {
      for (let i = 0; i < SAMPLES; i++) {
        await wait(i === 0 ? 40 : SAMPLE_GAP);
        if (!ctx || video.readyState < 2) continue;
        ctx.drawImage(video, 0, 0, w, h);
        let landmarks: NormalizedLandmark[] | null = null;
        try {
          landmarks = detectFace(landmarker, canvas, performance.now());
        } catch {
          continue;
        }
        if (!landmarks) continue;

        const pixels = ctx.getImageData(0, 0, w, h);
        let result: ScanResult;
        try {
          result = analyzeFace({ landmarks, width: w, height: h, pixels });
        } catch {
          continue;
        }
        samples.push(result);

        const poseCost = result.pose.yaw + result.pose.roll;
        if (poseCost < bestPose) {
          bestPose = poseCost;
          bestShot = canvas.toDataURL('image/jpeg', 0.86);
        }
        setProgress((i + 1) / SAMPLES);
      }
    } finally {
      scanningRef.current = false;
    }

    if (samples.length < 3) {
      setScanning(false);
      setProgress(0);
      flashHint('Lost your face mid-scan. Hold still and try again.');
      return;
    }

    if (!bestShot && ctx) bestShot = canvas.toDataURL('image/jpeg', 0.86);
    const merged = mergeScans(samples);
    onScanned(merged, bestShot);
  }, [flashHint, onScanned]);

  const onPickFile = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (!file) return;
      setScanning(true);
      setProgress(0.4);
      try {
        const landmarker = await loadFaceLandmarker();
        const url = URL.createObjectURL(file);
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const i = new Image();
          i.onload = () => resolve(i);
          i.onerror = () => reject(new Error('Could not read that photo'));
          i.src = url;
        });
        // Same working width as the live path, so a photo and a camera frame of
        // the same face produce comparable pixel statistics.
        const w = ANALYSIS_WIDTH;
        const h = Math.max(1, Math.round((w * img.naturalHeight) / img.naturalWidth));
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const c = canvas.getContext('2d', { willReadFrequently: true });
        if (!c) throw new Error('Canvas unavailable');
        c.drawImage(img, 0, 0, w, h);
        URL.revokeObjectURL(url);

        const landmarks = detectFace(landmarker, canvas, performance.now());
        if (!landmarks) {
          setScanning(false);
          setProgress(0);
          flashHint('No face found in that photo.');
          return;
        }
        const pixels = c.getImageData(0, 0, w, h);
        setProgress(0.9);
        onScanned(analyzeFace({ landmarks, width: w, height: h, pixels }), canvas.toDataURL('image/jpeg', 0.86));
      } catch {
        setScanning(false);
        setProgress(0);
        flashHint('Could not read that photo. Try another.');
      }
    },
    [flashHint, onScanned],
  );

  const status = camError
    ? 'CAMERA UNAVAILABLE'
    : !modelReady
      ? 'LOADING SCANNER'
      : scanning
        ? `MEASURING ${Math.round(progress * SAMPLES)}/${SAMPLES}`
        : faceOn
          ? 'FACE LOCKED'
          : 'LOOKING FOR FACE';

  return (
    <div className="fixed inset-0 z-40 flex flex-col bg-ink text-bone">
      <header className="safe-top relative z-20 flex items-center justify-between px-5 pb-3 pt-2">
        <button
          type="button"
          aria-label="Close camera"
          onClick={onClose}
          className="grid h-10 w-10 place-items-center rounded-full border border-white/12 bg-white/[0.04] text-white/80 transition active:scale-95"
        >
          <X size={17} weight="bold" />
        </button>

        <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5">
          <span
            className={`h-1.5 w-1.5 rounded-full ${faceOn ? 'bg-mint' : camError ? 'bg-red-400' : 'bg-white/40'}`}
            style={faceOn ? { boxShadow: '0 0 8px #4cff9b' } : undefined}
          />
          <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/60">{status}</span>
        </div>

        <div className="h-10 w-10" aria-hidden />
      </header>

      <div className="relative z-10 flex flex-1 flex-col px-4 pt-1">
        <div
          className="relative w-full shrink-0 overflow-hidden rounded-[26px] bg-black ring-1 ring-white/10"
          style={{ aspectRatio: String(stageAspect) }}
        >
          <video
            ref={videoRef}
            onLoadedMetadata={onLoadedMetadata}
            className="absolute inset-0 h-full w-full object-cover"
            style={{ transform: facing === 'user' ? 'scaleX(-1)' : undefined }}
            playsInline
            muted
            autoPlay
          />
          <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />

          <div className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-black/55 to-transparent" />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-28 bg-gradient-to-t from-black/70 to-transparent" />

          {camError ? (
            <div className="absolute inset-0 grid place-items-center px-8 text-center">
              <p className="max-w-[280px] text-[13px] leading-relaxed text-white/70">{camError}</p>
            </div>
          ) : null}

          {hint ? (
            <div className="pointer-events-none absolute inset-x-0 top-5 flex justify-center px-6">
              <div className="rounded-full bg-bone px-4 py-2 text-[12px] font-semibold text-ink shadow-xl">
                {hint}
              </div>
            </div>
          ) : null}

          {scanning ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-ink/45 backdrop-blur-[1px]">
              <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-mint">Hold still</p>
              <p className="mt-2 font-mono text-[34px] font-bold tabular-nums leading-none text-bone">
                {Math.round(progress * 100)}%
              </p>
              <p className="mt-3 max-w-[220px] text-center text-[11px] leading-relaxed text-white/50">
                Sampling {SAMPLES} frames and measuring each one
              </p>
            </div>
          ) : null}

          <div className="absolute inset-x-0 bottom-3 flex justify-center">
            <p className="rounded-full bg-black/45 px-3 py-1.5 font-mono text-[9px] uppercase tracking-[0.24em] text-white/60 backdrop-blur-sm">
              {faceOn ? 'Locked, tap to scan' : 'Fill the box with your face'}
            </p>
          </div>
        </div>

        <div className="mt-5 grid grid-cols-3 gap-3 border-t border-white/10 px-1 pt-4">
          <div>
            <p className="font-mono text-[8px] uppercase tracking-[0.18em] text-white/25">Face</p>
            <p className={`mt-1 text-[11px] font-semibold ${faceOn ? 'text-mint' : 'text-white/40'}`}>
              {faceOn ? 'In frame' : 'Searching'}
            </p>
          </div>
          <div>
            <p className="font-mono text-[8px] uppercase tracking-[0.18em] text-white/25">Distance</p>
            <p
              className={`mt-1 text-[11px] font-semibold ${
                distance === 'good' ? 'text-mint' : distance ? 'text-white/70' : 'text-white/40'
              }`}
            >
              {distance === 'good'
                ? 'Good'
                : distance === 'far'
                  ? 'Move closer'
                  : distance === 'close'
                    ? 'Step back'
                    : 'Waiting'}
            </p>
          </div>
          <div>
            <p className="font-mono text-[8px] uppercase tracking-[0.18em] text-white/25">Light</p>
            <p className="mt-1 text-[11px] font-semibold text-white/40">Face a window</p>
          </div>
        </div>
      </div>

      <div className="safe-bottom relative z-10 mt-auto px-6 pb-2 pt-6">
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={scanning}
            aria-label="Choose a photo instead"
            className="grid h-12 w-12 place-items-center rounded-full border border-white/12 text-white/60 transition active:scale-95 disabled:opacity-40"
          >
            <ImageSquare size={20} />
          </button>

          <button
            type="button"
            onClick={scan}
            disabled={scanning || !modelReady || !!camError}
            aria-label="Scan my face"
            className="relative grid h-[76px] w-[76px] place-items-center rounded-full transition active:scale-95 disabled:opacity-50"
          >
            <span
              className="absolute inset-0 rounded-full border-2 transition-colors"
              style={{ borderColor: faceOn ? MINT : 'rgba(255,255,255,0.3)' }}
            />
            <span
              className="absolute inset-[7px] rounded-full transition-colors"
              style={{ background: faceOn ? MINT : 'rgba(255,255,255,0.75)' }}
            />
            <span className="relative font-mono text-[10px] font-bold tracking-[0.12em] text-ink">
              {scanning ? '···' : 'SCAN'}
            </span>
          </button>

          <button
            type="button"
            aria-label="Switch camera"
            disabled={scanning}
            onClick={() => setFacing((f) => (f === 'user' ? 'environment' : 'user'))}
            className="grid h-12 w-12 place-items-center rounded-full border border-white/12 text-white/60 transition active:scale-95 disabled:opacity-40"
          >
            <ArrowsClockwise size={20} weight="bold" />
          </button>
        </div>
      </div>

      <input ref={fileRef} type="file" accept="image/*" onChange={onPickFile} className="hidden" />
    </div>
  );
}
