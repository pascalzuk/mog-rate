'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import AnalyzingView from '@/components/AnalyzingView';
import CameraView from '@/components/CameraView';
import HomeView from '@/components/HomeView';
import ResultView from '@/components/ResultView';
import type { ScanResult } from '@/lib/faceMetrics';
import { loadFaceLandmarker } from '@/lib/faceLandmarker';

type Phase = 'home' | 'camera' | 'analyzing' | 'result';

export default function MogApp() {
  const [phase, setPhase] = useState<Phase>('home');
  const [result, setResult] = useState<ScanResult | null>(null);
  const [photo, setPhoto] = useState('');
  const timer = useRef<number | null>(null);

  useEffect(() => {
    // Start the ~3.6 MB model + WASM download while the user is still reading
    // the home screen. Waiting for the camera to open before fetching it is what
    // made the first few seconds of the camera look frozen.
    void loadFaceLandmarker().catch(() => undefined);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, []);

  const handleScanned = useCallback((next: ScanResult, shot: string) => {
    setResult(next);
    setPhoto(shot);
    setPhase('analyzing');
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      setPhase('result');
      window.scrollTo({ top: 0 });
    }, 1250);
  }, []);

  return (
    <main className="app-shell grain min-h-[100dvh]">
      {/* Solid ink behind the iOS status bar. Even with an opaque status bar a
          standalone app can still show a hairline of the page up there, and
          anything scrolling under it reads as blurred. */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-x-0 top-0 z-[60] bg-ink"
        style={{ height: 'env(safe-area-inset-top)' }}
      />

      {phase === 'home' ? <HomeView onScan={() => setPhase('camera')} /> : null}

      {phase === 'result' && result ? (
        <ResultView
          result={result}
          photo={photo}
          onAgain={() => setPhase('camera')}
          onHome={() => setPhase('home')}
        />
      ) : null}

      {phase === 'camera' ? (
        <CameraView onScanned={handleScanned} onClose={() => setPhase('home')} />
      ) : null}

      {phase === 'analyzing' ? <AnalyzingView photo={photo} /> : null}
    </main>
  );
}
