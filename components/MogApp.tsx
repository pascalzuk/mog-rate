'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import AnalyzingView from '@/components/AnalyzingView';
import CameraView from '@/components/CameraView';
import HomeView from '@/components/HomeView';
import ResultView from '@/components/ResultView';
import type { ScanResult } from '@/lib/faceMetrics';

type Phase = 'home' | 'camera' | 'analyzing' | 'result';

export default function MogApp() {
  const [phase, setPhase] = useState<Phase>('home');
  const [result, setResult] = useState<ScanResult | null>(null);
  const [photo, setPhoto] = useState('');
  const timer = useRef<number | null>(null);

  useEffect(() => {
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
