'use client';

import { useEffect, useState } from 'react';

const STEPS = ['Reading head pose', 'Mapping landmarks', 'Counting spots', 'Checking symmetry', 'Scoring'];

export default function AnalyzingView({ photo }: { photo: string }) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => setStep((s) => Math.min(STEPS.length - 1, s + 1)), 260);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="fixed inset-0 z-40 overflow-hidden bg-ink">
      {photo ? (
        <img
          src={photo}
          alt=""
          className="absolute inset-0 h-full w-full scale-110 object-cover opacity-25 blur-[3px]"
        />
      ) : null}
      <div className="absolute inset-0 bg-gradient-to-b from-ink/90 via-ink/75 to-ink" />

      <div className="relative flex h-full flex-col items-center justify-center px-8">
        <div className="relative aspect-[3/4] w-[56vw] max-w-[220px] overflow-hidden rounded-2xl border border-white/12">
          {photo ? <img src={photo} alt="" className="h-full w-full object-cover" /> : null}
          <div className="scanline absolute inset-x-0 h-16 bg-gradient-to-b from-transparent via-signal/30 to-transparent" />
          <span className="absolute left-2 top-2 h-3.5 w-3.5 border-l-2 border-t-2 border-signal" />
          <span className="absolute right-2 top-2 h-3.5 w-3.5 border-r-2 border-t-2 border-signal" />
          <span className="absolute bottom-2 left-2 h-3.5 w-3.5 border-b-2 border-l-2 border-signal" />
          <span className="absolute bottom-2 right-2 h-3.5 w-3.5 border-b-2 border-r-2 border-signal" />
        </div>

        <div className="mt-8 w-full max-w-[260px]">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-[9.5px] uppercase tracking-[0.18em] text-white/55">
              {STEPS[step]}
            </span>
            <span className="font-mono text-[9.5px] tabular-nums text-signal">{step + 1}/5</span>
          </div>
          <div className="mt-2.5 h-[2px] w-full overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-signal transition-all duration-300 ease-out"
              style={{ width: `${((step + 1) / STEPS.length) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
