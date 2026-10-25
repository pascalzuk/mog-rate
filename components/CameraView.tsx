'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowsClockwise,
  ImageSquare,
  X,
} from '@phosphor-icons/react';
import type { FaceLandmarker, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { detectFace, loadFaceLandmarker } from '@/lib/faceLandmarker';
import { analyzeFace, estimatePose, mergeScans } from '@/lib/faceMetrics';
import type { HeadPose, ScanResult } from '@/lib/faceMetrics';

type Box = { x: number; y: number; w: number; h: number };
type View = { landmarks: NormalizedLandmark[]; width: number; height: number };

type Props = {
  onScanned: (result: ScanResult, photo: string) => void;
  onUpload: () => void;
  onClose: () => void;
};

const MINT = '#4cff9b';
// Nine frames over about 1.2s. Six finished so fast the read felt like a guess,
// and a median over nine is much harder for a blink or a smear to move.
const SAMPLES = 9;
const SAMPLE_GAP = 130;
// 800px wide is roughly 1.5x the pixels of the old 640: small papules survive
// the downsample instead of turning into a soft blur the detector cannot see.
const ANALYSIS_WIDTH = 800;
const HOLD_MS = 420;
const STEP_TIMEOUT = 9000;

const STEPS = [
  { key: 'front', title: 'Look straight', hint: 'Chin level, eyes on the lens' },
  { key: 'side1', title: 'Rotate left', hint: 'Turn until you feel the stretch' },
  { key: 'side2', title: 'Rotate right', hint: 'Sweep past centre to the other side' },
  { key: 'tilt1', title: 'Chin up', hint: 'Hold it for a beat' },
  { key: 'tilt2', title: 'Chin down', hint: 'Almost there' },
  { key: 'scan', title: 'Face the camera', hint: 'Hold still while it reads you' },
] as const;

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
  const padX = (maxX - minX) * 0.05;
  const padY = (maxY - minY) * 0.05;
  const x = Math.max(0, minX - padX);
  const y = Math.max(0, minY - padY);
  return { x, y, w: Math.min(1, maxX + padX) - x, h: Math.min(1, maxY + padY) - y };
}

function drawOverlay(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement | null,
  smooth: { box: Box | null; ratio: number },
  target: { box: Box | null },
  mirrored: boolean,
  active: boolean,
) {
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (!cw || !ch) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 3);
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  const prev = smooth.box;
  const next = target.box;
  const eased = next
    ? prev
      ? {
          x: prev.x + (next.x - prev.x) * 0.26,
          y: prev.y + (next.y - prev.y) * 0.26,
          w: prev.w + (next.w - prev.w) * 0.26,
          h: prev.h + (next.h - prev.h) * 0.26,
        }
      : { ...next }
    : null;
  smooth.box = eased;

  if (!eased || !video || !video.videoWidth) {
    smooth.ratio = 0;
    return;
  }

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

  const inset = 10;
  const left = Math.max(inset, x);
  const top = Math.max(inset, y);
  const bw = Math.max(48, Math.min(cw - inset, x + w) - left);
  const bh = Math.max(48, Math.min(ch - inset, y + h) - top);
  smooth.ratio = bh / ch;

  const radius = Math.min(bw, bh) * 0.14;
  const corner = Math.min(bw, bh) * 0.18;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(left + radius, top);
  ctx.arcTo(left + bw, top, left + bw, top + bh, radius);
  ctx.arcTo(left + bw, top + bh, left, top + bh, radius);
  ctx.arcTo(left, top + bh, left, top, radius);
  ctx.arcTo(left, top, left + bw, top, radius);
  ctx.closePath();
  ctx.strokeStyle = active ? 'rgba(76,255,155,0.55)' : 'rgba(255,255,255,0.28)';
  ctx.lineWidth = 1;
  ctx.setLineDash(active ? [] : [5, 6]);
  ctx.stroke();
  ctx.restore();

  if (active) {
    ctx.save();
    ctx.strokeStyle = MINT;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    const cs: Array<[number, number, number, number]> = [
      [left, top, 1, 1],
      [left + bw, top, -1, 1],
      [left, top + bh, 1, -1],
      [left + bw, top + bh, -1, -1],
    ];
    for (const [cx, cy, dx, dy] of cs) {
      ctx.beginPath();
      ctx.moveTo(cx + dx * corner, cy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx, cy + dy * corner);
      ctx.stroke();
    }
    ctx.restore();
  }
}

export default function CameraView({ onScanned, onUpload, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const stageHostRef = useRef<HTMLDivElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef(0);
  const smoothRef = useRef<{ box: Box | null; ratio: number }>({ box: null, ratio: 0 });
  const targetRef = useRef<{ box: Box | null }>({ box: null });
  const lastSeenRef = useRef(0);
  const lastDetectRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const lastLandmarksRef = useRef<NormalizedLandmark[] | null>(null);
  const faceOnRef = useRef(false);
  const profileRef = useRef<View | null>(null);
  const poseRef = useRef<HeadPose>({ yaw: 0, roll: 0, pitch: 0 });
  const basePoseRef = useRef<HeadPose | null>(null);
  const facingRef = useRef<'user' | 'environment'>('user');
  const aspectRef = useRef(0.75);
  const stepRef = useRef(0);
  const holdStartRef = useRef(0);
  const stepStartRef = useRef(0);
  const sideSignRef = useRef(0);
  const tiltSignRef = useRef(0);
  const viewsRef = useRef<View[]>([]);
  const scanningRef = useRef(false);
  const doneRef = useRef(false);
  const hintTimer = useRef<number | null>(null);

  const [facing, setFacing] = useState<'user' | 'environment'>('user');
  const [aspect, setAspect] = useState(0.75);
  const [stage, setStage] = useState({ w: 0, h: 0 });
  const [modelReady, setModelReady] = useState(false);
  const [faceOn, setFaceOn] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [hold, setHold] = useState(0);
  const [yawNow, setYawNow] = useState(0);
  const yawStateRef = useRef(0);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [slow, setSlow] = useState(false);
  const [warm, setWarm] = useState(true);
  const warmRef = useRef(true);

  facingRef.current = facing;
  aspectRef.current = aspect;

  const flashHint = useCallback((message: string) => {
    setHint(message);
    if (hintTimer.current) window.clearTimeout(hintTimer.current);
    hintTimer.current = window.setTimeout(() => setHint(null), 3000);
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
    targetRef.current.box = null;
    smoothRef.current.box = null;
    smoothRef.current.ratio = 0;

    const open = (constraints: MediaStreamConstraints) => navigator.mediaDevices.getUserMedia(constraints);

    const start = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('unsupported');
        streamRef.current?.getTracks().forEach((t) => t.stop());

        // Tall request on purpose: a square or 16:9 track is what makes a
        // portrait preview look zoomed, because cover crops away most of it.
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
        setCamError(
          name === 'NotAllowedError'
            ? 'Camera access is off. Allow it in Settings, or use a photo.'
            : name === 'NotFoundError'
              ? 'No camera found. Use a photo instead.'
              : 'Could not open the camera. Use a photo instead.',
        );
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

  // Size the viewfinder to whatever space is left, so nothing ever needs a scroll.
  useEffect(() => {
    const host = stageHostRef.current;
    if (!host) return;
    const resize = () => {
      const rect = host.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const ar = clamp(aspectRef.current, 0.62, 0.95);
      let w = Math.min(rect.width, rect.height * ar);
      let h = w / ar;
      if (h > rect.height) {
        h = rect.height;
        w = h * ar;
      }
      setStage({ w: Math.round(w), h: Math.round(h) });
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(host);
    return () => ro.disconnect();
  }, [aspect]);

  const onLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return;
    setAspect(clamp(video.videoWidth / video.videoHeight, 0.62, 0.95));
  }, []);

  const runScan = useCallback(
    async (views: View[]) => {
      if (scanningRef.current || doneRef.current) return;
      scanningRef.current = true;
      setScanning(true);
      setProgress(0);

      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      if (!video || !landmarker || video.readyState < 2) {
        scanningRef.current = false;
        setScanning(false);
        flashHint('Camera is still warming up.');
        return;
      }

      const canvas = document.createElement('canvas');
      const w = ANALYSIS_WIDTH;
      const h = Math.max(1, Math.round((w * video.videoHeight) / video.videoWidth));
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });

      const samples: ScanResult[] = [];
      let bestShot = '';
      let bestPose = Number.POSITIVE_INFINITY;
      let heroIndex = 0;
      const profile = profileRef.current ?? undefined;

      try {
        for (let i = 0; i < SAMPLES; i++) {
          await wait(i === 0 ? 60 : SAMPLE_GAP);
          if (!ctx || video.readyState < 2) continue;
          ctx.drawImage(video, 0, 0, w, h);
          let landmarks: NormalizedLandmark[] | null = null;
          try {
            landmarks = detectFace(landmarker, canvas, performance.now());
          } catch {
            continue;
          }
          if (!landmarks) continue;

          // Every frame gets pixels: skipping some would mix symmetry scores
          // measured with the flip test and without it, and the median would
          // then disagree with the reported flip number.
          const pixels = ctx.getImageData(0, 0, w, h);
          let result: ScanResult;
          try {
            result = analyzeFace(
              { landmarks, width: w, height: h, pixels, profile: profile ? { ...profile } : undefined },
              views.length + 1,
            );
          } catch {
            continue;
          }
          samples.push(result);

          const cost = Math.abs(result.pose.yaw) + Math.abs(result.pose.pitch);
          if (cost < bestPose) {
            bestPose = cost;
            bestShot = canvas.toDataURL('image/jpeg', 0.86);
            heroIndex = samples.length - 1;
          }
          setProgress((i + 1) / SAMPLES);
        }
      } finally {
        scanningRef.current = false;
      }

      if (samples.length < 3) {
        doneRef.current = false;
        setScanning(false);
        setProgress(0);
        stepRef.current = 5;
        setStep(5);
        holdStartRef.current = 0;
        flashHint('Lost your face. Hold still and try again.');
        return;
      }

      doneRef.current = true;
      if (!bestShot && ctx) bestShot = canvas.toDataURL('image/jpeg', 0.86);
      if ((window as unknown as { __MOG_DEBUG?: boolean }).__MOG_DEBUG) {
        (window as unknown as { __mogFrames?: unknown }).__mogFrames = samples.map((s) => ({
          flip: Math.round(s.raw.flipDifference * 1000) / 1000,
          dev: Math.round(s.raw.symmetryDev * 10000) / 10000,
          sym: Math.round((s.metrics.find((m) => m.key === 'symmetry')?.score ?? 0) * 1000) / 1000,
          clarity: Math.round(s.skin.clarity * 1000) / 1000,
        }));
      }
      const merged = mergeScans(samples, views.length + 1, heroIndex);
      onScanned(merged, bestShot);
    },
    [flashHint, onScanned],
  );

  const captureView = useCallback((): View | null => {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !landmarker || video.readyState < 2) return null;
    const w = ANALYSIS_WIDTH;
    const h = Math.max(1, Math.round((w * video.videoHeight) / video.videoWidth));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, w, h);
    const landmarks = detectFace(landmarker, canvas, performance.now());
    return landmarks ? { landmarks, width: w, height: h } : null;
  }, []);

  const advance = useCallback(
    (index: number, pose: HeadPose) => {
      const view = captureView();
      if (view) {
        if (index === 1 || index === 2) {
          viewsRef.current.push(view);
          if (!profileRef.current) profileRef.current = view;
        }
      }
      const base = basePoseRef.current ?? pose;
      if (index === 1) sideSignRef.current = Math.sign(pose.yaw - base.yaw);
      if (index === 3) tiltSignRef.current = Math.sign(pose.pitch - base.pitch);

      const next = Math.min(STEPS.length - 1, index + 1);
      stepRef.current = next;
      setStep(next);
      holdStartRef.current = 0;
      stepStartRef.current = performance.now();
      setHold(0);
      setSlow(false);
      if (next === STEPS.length - 1) flashHint('Head straight, hold still');
    },
    [captureView, flashHint],
  );

  const skipGuide = useCallback(() => {
    if (scanning || doneRef.current) return;
    stepRef.current = STEPS.length - 1;
    setStep(STEPS.length - 1);
    holdStartRef.current = 0;
    stepStartRef.current = performance.now();
    setHold(0);
    flashHint('Guided views skipped');
  }, [scanning, flashHint]);

  // Detection + guide loop
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
        now - lastDetectRef.current > 75 &&
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
            if (warmRef.current) {
              warmRef.current = false;
              setWarm(false);
            }
            lastLandmarksRef.current = landmarks;
            targetRef.current.box = faceBox(landmarks);
            lastSeenRef.current = now;
            poseRef.current = estimatePose(landmarks, video.videoWidth / video.videoHeight);
            if ((window as unknown as { __MOG_DEBUG?: boolean }).__MOG_DEBUG) {
              (window as unknown as { __mogPose?: unknown }).__mogPose = {
                yaw: Math.round(poseRef.current.yaw),
                pitch: Math.round(poseRef.current.pitch),
              };
            }

            if (!scanningRef.current && !doneRef.current) {
              const index = stepRef.current;
              const pose = poseRef.current;
              // Everything is relative to the pose the user arrived in, because
              // a landmark model always carries a few degrees of its own bias.
              if (!basePoseRef.current) basePoseRef.current = pose;
              const base = basePoseRef.current;
              const dy = pose.yaw - base.yaw;
              const rel = Math.round(dy);
              if (Math.abs(rel - yawStateRef.current) >= 3) {
                yawStateRef.current = rel;
                setYawNow(rel);
              }
              const dp = pose.pitch - base.pitch;
              const front = Math.abs(dy) < 12 && Math.abs(dp) < 12;
              const matches =
                index === 0
                  ? front
                  : index === 1
                    ? Math.abs(dy) > 15
                    : index === 2
                      ? Math.abs(dy) > 15 && Math.sign(dy) !== sideSignRef.current
                      : index === 3
                        ? Math.abs(dp) > 8
                        : index === 4
                          ? Math.abs(dp) > 8 && Math.sign(dp) !== tiltSignRef.current
                          : front;

              if (matches) {
                if (!holdStartRef.current) holdStartRef.current = now;
                const p = (now - holdStartRef.current) / HOLD_MS;
                setHold((prev) => (Math.abs(prev - p) > 0.1 || p >= 1 ? Math.min(1, p) : prev));
                if (p >= 1) {
                  if (index === 5) {
                    setHold(1);
                    void runScan([...viewsRef.current]);
                  } else {
                    advance(index, pose);
                  }
                }
              } else if (holdStartRef.current) {
                holdStartRef.current = 0;
                setHold(0);
              }

              if (now - stepStartRef.current > STEP_TIMEOUT) {
                setSlow((prev) => (prev ? prev : true));
              }
            }
          } else if (now - lastSeenRef.current > 700) {
            targetRef.current.box = null;
            holdStartRef.current = 0;
          }
        } catch {
          // dropped frame
        }
      }

      if (overlay) {
        drawOverlay(
          overlay,
          video,
          smoothRef.current,
          targetRef.current,
          facingRef.current === 'user',
          performance.now() - lastSeenRef.current < 500,
        );
      }

      const locked = performance.now() - lastSeenRef.current < 500;
      if (locked !== faceOnRef.current) {
        faceOnRef.current = locked;
        setFaceOn(locked);
      }

      rafRef.current = requestAnimationFrame(frame);
    };

    rafRef.current = requestAnimationFrame(frame);
    return () => {
      stopped = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, [modelReady, advance, runScan]);

  const current = STEPS[step];
  const ring = ((step + hold) / (STEPS.length - 1)) * 100;
  const arrows = [null, ArrowLeft, ArrowRight, ArrowUp, ArrowDown, null];
  const Arrow = arrows[step];
  const guideDone = step === STEPS.length - 1;

  return (
    // Height comes from 100dvh rather than inset-0 alone: on iOS the layout
    // viewport runs under the browser chrome, so an inset-0 fixed screen pushes
    // the shutter below the fold. 100dvh tracks the visible area.
    <div className="fixed inset-x-0 top-0 z-40 flex flex-col overflow-hidden bg-ink text-bone" style={{ height: '100dvh' }}>
      <header className="safe-top flex items-center justify-between px-4 pb-1">
        <button
          type="button"
          aria-label="Close camera"
          onClick={onClose}
          className="grid h-9 w-9 place-items-center text-white/70 active:scale-95"
        >
          <X size={17} weight="bold" />
        </button>

        <span className="font-mono text-[9px] uppercase tracking-[0.16em] text-white/40">
          {camError
            ? 'No camera'
            : warm
              ? 'Starting'
              : scanning
                ? 'Measuring'
                : guideDone
                  ? 'Final frame'
                  : `View ${Math.min(step + 1, 5)} of 5`}
        </span>

        <button
          type="button"
          onClick={skipGuide}
          disabled={scanning || guideDone || !!camError}
          className="h-9 px-1 font-mono text-[10px] uppercase tracking-[0.14em] text-white/40 active:text-white disabled:opacity-25"
        >
          Skip
        </button>
      </header>

      <div ref={stageHostRef} className="relative min-h-0 flex-1 px-3">
        <div className="absolute inset-0 flex items-center justify-center">
          <div
            className="relative overflow-hidden rounded-[22px] bg-black ring-1 ring-white/10"
            style={{ width: stage.w || undefined, height: stage.h || undefined }}
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

            {camError ? (
              <div className="absolute inset-0 grid place-items-center px-6 text-center">
                <p className="max-w-[260px] text-[13px] leading-relaxed text-white/70">{camError}</p>
              </div>
            ) : null}

            {warm && !camError ? (
              <div className="pointer-events-none absolute inset-0 grid place-items-center">
                <span className="font-mono text-[9px] uppercase tracking-[0.24em] text-white/55">
                  Starting camera
                </span>
              </div>
            ) : null}

            {hint ? (
              <div className="pointer-events-none absolute inset-x-0 top-4 flex justify-center px-5">
                <span className="rounded-full bg-bone px-3.5 py-1.5 text-[11px] font-semibold text-ink shadow-lg">
                  {hint}
                </span>
              </div>
            ) : null}

            {Arrow && !scanning && !camError ? (
              <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center">
                <span className="flex items-center gap-2 bg-black/45 px-3 py-1.5 text-white/70 backdrop-blur-sm">
                  <Arrow size={13} weight="bold" />
                  <span className="font-mono text-[8.5px] uppercase tracking-[0.16em]">
                    {step === 1 ? 'Left' : step === 2 ? 'Right' : step === 3 ? 'Up' : 'Down'}
                  </span>
                </span>
              </div>
            ) : null}

            {scanning ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-ink/55">
                <p className="font-mono text-[9px] uppercase tracking-[0.26em] text-mint">Hold still</p>
                <p className="mt-2 text-[36px] font-black leading-none tabular-nums tracking-[-0.03em]">
                  {Math.round(progress * 100)}%
                </p>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      <div className="px-5 pt-2">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h1 className="text-[23px] font-black leading-[1.04] tracking-[-0.035em]">
              {warm ? 'Warming up' : scanning ? 'Measuring' : current.title}
            </h1>
            <p className="mt-1.5 max-w-[250px] text-[12px] leading-snug text-white/45">
              {warm
                ? 'Loading the scanner and opening the camera'
                : slow && !guideDone
                  ? 'Not moving? Tap Skip to keep going.'
                  : current.hint}
            </p>
          </div>
          <div className="shrink-0 pt-0.5 text-right">
            <p className="font-mono text-[8px] uppercase tracking-[0.16em] text-white/25">Head</p>
            <p className="mt-0.5 text-[16px] font-bold leading-none tabular-nums text-white/80">
              {scanning ? '···' : `${yawNow > 0 ? '+' : ''}${yawNow}°`}
            </p>
          </div>
        </div>

        <div className="mt-3 flex gap-1">
          {[0, 1, 2, 3, 4].map((i) => {
            const fill = i < step ? 1 : i === step ? Math.min(1, hold) : 0;
            return (
              <span key={i} className="h-[2px] flex-1 overflow-hidden bg-white/12">
                <span
                  className="block h-full bg-mint transition-[width] duration-100"
                  style={{ width: `${(step > i ? 1 : fill) * 100}%` }}
                />
              </span>
            );
          })}
        </div>
      </div>

      <div className="safe-bottom flex items-center justify-between px-7 pb-1 pt-3">
        <button
          type="button"
          onClick={onUpload}
          disabled={scanning}
          aria-label="Upload a photo instead"
          className="grid h-11 w-11 place-items-center rounded-full border border-white/10 text-white/55 active:scale-95 disabled:opacity-40"
        >
          <ImageSquare size={19} />
        </button>

        <div className="relative grid h-[68px] w-[68px] place-items-center" aria-hidden>
          <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full -rotate-90">
            <circle cx="50" cy="50" r="46" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="1.5" />
            <circle
              cx="50"
              cy="50"
              r="46"
              fill="none"
              stroke={MINT}
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeDasharray={`${(ring / 100) * 289} 289`}
            />
          </svg>
          <span
            className="grid h-[52px] w-[52px] place-items-center rounded-full text-[9px] font-bold tracking-[0.1em] text-ink transition-colors"
            style={{ background: faceOn ? MINT : 'rgba(255,255,255,0.82)' }}
          >
            {warm || scanning ? '···' : faceOn ? 'HOLD' : 'FACE'}
          </span>
        </div>

        <button
          type="button"
          aria-label="Switch camera"
          disabled={scanning}
          onClick={() => setFacing((f) => (f === 'user' ? 'environment' : 'user'))}
          className="grid h-11 w-11 place-items-center rounded-full border border-white/10 text-white/55 active:scale-95 disabled:opacity-40"
        >
          <ArrowsClockwise size={19} weight="bold" />
        </button>
      </div>

    </div>
  );
}
