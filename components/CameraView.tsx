'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { FaceLandmarker, NormalizedLandmark } from '@mediapipe/tasks-vision';
import { detectFace, loadFaceLandmarker } from '@/lib/faceLandmarker';
import { analyzeFace, type ScanResult } from '@/lib/faceMetrics';

type Box = { x: number; y: number; w: number; h: number };

type Props = {
  onScanned: (result: ScanResult, photo: string) => void;
  onClose: () => void;
};

const LOCK_COLOR = '#4cff9b';
const MAX_EDGE = 1000;

/** Bounding box of the whole face mesh, normalized to the video frame, with a little breathing room. */
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
  return {
    x,
    y,
    w: Math.min(1, maxX + padX) - x,
    h: Math.min(1, maxY + padY) - y,
  };
}

function drawOverlay(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement | null,
  current: { box: Box | null },
  target: { box: Box | null },
  mirrored: boolean,
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

  // Ease the box toward the newest detection so it glides instead of snapping.
  const prev = current.box;
  const next = target.box;
  const eased = next
    ? prev
      ? {
          x: prev.x + (next.x - prev.x) * 0.22,
          y: prev.y + (next.y - prev.y) * 0.22,
          w: prev.w + (next.w - prev.w) * 0.22,
          h: prev.h + (next.h - prev.h) * 0.22,
        }
      : { ...next }
    : null;
  current.box = eased;
  if (!eased || !video || !video.videoWidth) return;

  // object-cover mapping from video space to the on-screen preview
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

  // Keep the lock box on screen even when the face fills the frame.
  const inset = 12;
  const left = Math.max(inset, x);
  const top = Math.max(inset, y);
  const bw = Math.max(48, Math.min(cw - inset, x + w) - left);
  const bh = Math.max(48, Math.min(ch - inset, y + h) - top);

  const radius = Math.min(bw, bh) * 0.18;
  const corner = Math.min(bw, bh) * 0.24;

  // soft fill + faint outline
  ctx.save();
  ctx.beginPath();
  const r = Math.min(radius, 42);
  ctx.moveTo(left + r, top);
  ctx.arcTo(left + bw, top, left + bw, top + bh, r);
  ctx.arcTo(left + bw, top + bh, left, top + bh, r);
  ctx.arcTo(left, top + bh, left, top, r);
  ctx.arcTo(left, top, left + bw, top, r);
  ctx.closePath();
  ctx.fillStyle = 'rgba(76, 255, 155, 0.06)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(76, 255, 155, 0.28)';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();

  // corner brackets
  ctx.save();
  ctx.strokeStyle = LOCK_COLOR;
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = 'rgba(76, 255, 155, 0.65)';
  ctx.shadowBlur = 14;
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
}

export default function CameraView({ onScanned, onClose }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const rafRef = useRef(0);
  const smoothRef = useRef<{ box: Box | null }>({ box: null });
  const targetRef = useRef<{ box: Box | null }>({ box: null });
  const lastSeenRef = useRef(0);
  const lastDetectRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const faceOnRef = useRef(false);
  const facingRef = useRef<'user' | 'environment'>('user');
  const fileRef = useRef<HTMLInputElement | null>(null);
  const hintTimer = useRef<number | null>(null);

  const [facing, setFacing] = useState<'user' | 'environment'>('user');
  const [modelReady, setModelReady] = useState(false);
  const [faceOn, setFaceOn] = useState(false);
  const [camError, setCamError] = useState<string | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  facingRef.current = facing;

  const flashHint = useCallback((message: string) => {
    setHint(message);
    if (hintTimer.current) window.clearTimeout(hintTimer.current);
    hintTimer.current = window.setTimeout(() => setHint(null), 3200);
  }, []);

  // Load the model while the camera warms up.
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

  // Camera stream
  useEffect(() => {
    let cancelled = false;
    setCamError(null);
    setFaceOn(false);
    faceOnRef.current = false;
    targetRef.current.box = null;
    smoothRef.current.box = null;

    const start = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('unsupported');
        streamRef.current?.getTracks().forEach((t) => t.stop());
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: facing,
            width: { ideal: 1280 },
            height: { ideal: 1280 },
          },
        });
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
          setCamError('Camera access is off. Allow it in Settings, or pick a photo below.');
        } else if (name === 'NotFoundError') {
          setCamError('No camera found. Pick a photo from your library instead.');
        } else {
          setCamError('Could not open the camera. Pick a photo from your library instead.');
        }
      }
    };

    void start();
    return () => {
      cancelled = true;
    };
  }, [facing]);

  // Always release the camera when this screen goes away.
  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (hintTimer.current) window.clearTimeout(hintTimer.current);
    };
  }, []);

  // Detect + draw loop
  useEffect(() => {
    if (!modelReady) return;
    let stopped = false;
    const work = document.createElement('canvas');
    const ctx = work.getContext('2d', { willReadFrequently: false });

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
        now - lastDetectRef.current > 70 &&
        video.currentTime !== lastVideoTimeRef.current
      ) {
        lastDetectRef.current = now;
        lastVideoTimeRef.current = video.currentTime;
        const w = 384;
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
          } else if (now - lastSeenRef.current > 600) {
            targetRef.current.box = null;
          }
        } catch {
          // A dropped frame is fine; the next tick retries.
        }
      }

      if (overlay) {
        drawOverlay(overlay, video, smoothRef.current, targetRef.current, facingRef.current === 'user');
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
  }, [modelReady]);

  const scan = useCallback(async () => {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (busy) return;
    if (!video || !landmarker || video.readyState < 2 || !video.videoWidth) {
      flashHint('Camera is still warming up.');
      return;
    }
    if (!faceOnRef.current) {
      flashHint('Center your face inside the frame.');
      return;
    }

    setBusy(true);
    try {
      const scale = Math.min(1, MAX_EDGE / Math.max(video.videoWidth, video.videoHeight));
      const w = Math.round(video.videoWidth * scale);
      const h = Math.round(video.videoHeight * scale);
      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const c = canvas.getContext('2d');
      if (!c) throw new Error('Canvas unavailable');
      c.drawImage(video, 0, 0, w, h);

      const landmarks = detectFace(landmarker, canvas, performance.now());
      if (!landmarks) {
        flashHint('Lost your face. Hold still and try again.');
        setBusy(false);
        return;
      }

      const result = analyzeFace(landmarks, w, h);
      onScanned(result, canvas.toDataURL('image/jpeg', 0.9));
    } catch {
      flashHint('Scan failed. Try again.');
      setBusy(false);
    }
  }, [busy, flashHint, onScanned]);

  const onPickFile = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = '';
      if (!file) return;
      setBusy(true);
      try {
        const landmarker = await loadFaceLandmarker();
        const url = URL.createObjectURL(file);
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
          const i = new Image();
          i.onload = () => resolve(i);
          i.onerror = () => reject(new Error('Could not read that photo'));
          i.src = url;
        });
        const scale = Math.min(1, MAX_EDGE / Math.max(img.naturalWidth, img.naturalHeight));
        const w = Math.max(1, Math.round(img.naturalWidth * scale));
        const h = Math.max(1, Math.round(img.naturalHeight * scale));
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const c = canvas.getContext('2d');
        if (!c) throw new Error('Canvas unavailable');
        c.drawImage(img, 0, 0, w, h);
        URL.revokeObjectURL(url);

        const landmarks = detectFace(landmarker, canvas, performance.now());
        if (!landmarks) {
          flashHint('No face found in that photo.');
          setBusy(false);
          return;
        }
        onScanned(analyzeFace(landmarks, w, h), canvas.toDataURL('image/jpeg', 0.9));
      } catch {
        flashHint('Could not read that photo. Try another.');
        setBusy(false);
      }
    },
    [flashHint, onScanned],
  );

  const status = camError
    ? 'Camera unavailable'
    : !modelReady
      ? 'Loading scanner'
      : faceOn
        ? 'Face locked'
        : 'Looking for your face';

  return (
    <div className="fixed inset-0 z-40 flex flex-col bg-black">
      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full object-cover"
        style={{ transform: facing === 'user' ? 'scaleX(-1)' : undefined }}
        playsInline
        muted
        autoPlay
      />
      <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full" />

      {/* readability gradients */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-black/80 to-transparent" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-64 bg-gradient-to-t from-black/90 via-black/50 to-transparent" />

      {/* top bar */}
      <div className="safe-top relative z-10 flex items-center justify-between px-4 pt-2">
        <button
          type="button"
          aria-label="Close camera"
          onClick={onClose}
          className="grid h-10 w-10 place-items-center rounded-full border border-white/15 bg-black/40 text-bone backdrop-blur"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
            <path d="M2 2l12 12M14 2L2 14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>

        <div className="flex items-center gap-2 rounded-full border border-white/10 bg-black/45 px-3 py-1.5 backdrop-blur">
          <span
            className={`h-1.5 w-1.5 rounded-full ${faceOn ? 'bg-mint' : camError ? 'bg-red-400' : 'bg-white/50'}`}
            style={faceOn ? { boxShadow: '0 0 8px #4cff9b' } : undefined}
          />
          <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-white/70">{status}</span>
        </div>

        <button
          type="button"
          aria-label="Switch camera"
          onClick={() => setFacing((f) => (f === 'user' ? 'environment' : 'user'))}
          className="grid h-10 w-10 place-items-center rounded-full border border-white/15 bg-black/40 text-bone backdrop-blur"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path
              d="M4 9a8 8 0 0113.6-3M20 15a8 8 0 01-13.6 3"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
            />
            <path d="M18 3v4h-4M6 21v-4h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>

      {/* idle guide */}
      {!faceOn && !camError ? (
        <div className="pointer-events-none absolute inset-0 z-0 grid place-items-center">
          <div className="h-[46vh] w-[70vw] max-w-[300px] rounded-[50%] border border-dashed border-white/20" />
        </div>
      ) : null}

      {/* hint toast */}
      {hint ? (
        <div className="pointer-events-none absolute inset-x-0 top-24 z-20 flex justify-center px-6">
          <div className="rounded-full bg-white px-4 py-2 text-[12px] font-semibold text-black shadow-lg">{hint}</div>
        </div>
      ) : null}

      {camError ? (
        <div className="absolute inset-x-0 top-1/2 z-20 -translate-y-1/2 px-8 text-center">
          <p className="mx-auto max-w-[280px] text-[13px] leading-relaxed text-white/70">{camError}</p>
        </div>
      ) : null}

      {/* bottom controls */}
      <div className="safe-bottom relative z-10 mt-auto px-6 pb-3">
        <div className="mb-5 flex justify-center">
          <p className="rounded-full bg-black/35 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.28em] text-white/55 backdrop-blur-sm">
            {faceOn ? 'Locked · tap to rate' : 'Line your face up in the frame'}
          </p>
        </div>

        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            aria-label="Choose a photo instead"
            className="grid h-12 w-12 place-items-center rounded-full border border-white/15 bg-black/35 text-white/70 backdrop-blur"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
              <rect x="3" y="5" width="18" height="14" rx="3" stroke="currentColor" strokeWidth="1.6" />
              <circle cx="8.5" cy="10" r="1.6" fill="currentColor" />
              <path d="M4 17l5-4.5 4 3.5 3-2.5 4 3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>

          <button
            type="button"
            onClick={scan}
            disabled={busy || !modelReady}
            aria-label="Scan my face"
            className="relative grid h-[78px] w-[78px] place-items-center rounded-full transition active:scale-95 disabled:opacity-60"
          >
            <span
              className="absolute inset-0 rounded-full border-2"
              style={{ borderColor: faceOn ? LOCK_COLOR : 'rgba(255,255,255,0.55)' }}
            />
            <span
              className="absolute inset-[6px] rounded-full"
              style={{ background: faceOn ? LOCK_COLOR : 'rgba(255,255,255,0.85)' }}
            />
            <span className="relative font-mono text-[10px] font-bold tracking-[0.1em] text-black">
              {busy ? '···' : 'SCAN'}
            </span>
          </button>

          <div className="h-12 w-12" aria-hidden />
        </div>

        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="mx-auto mt-4 block text-[11px] text-white/35 underline-offset-4 active:underline"
        >
          Use a photo instead
        </button>
      </div>

      <input ref={fileRef} type="file" accept="image/*" onChange={onPickFile} className="hidden" />
    </div>
  );
}
