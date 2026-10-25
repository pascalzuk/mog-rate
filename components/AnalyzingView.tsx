'use client';

import { useEffect, useState } from 'react';

const STEPS = ['Loading scanner', 'Mapping 478 landmarks', 'Measuring proportions', 'Scoring'];

export default function AnalyzingView({ photo }: { photo: string }) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => setStep((s) => Math.min(STEPS.length - 1, s + 1)), 400);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="fixed inset-0 z-40 overflow-hidden bg-ink">
      <img
        src={photo}
        alt=""
        className="absolute inset-0 h-full w-full scale-110 object-cover opacity-40 blur-[2px]"
      />
      <div className="absolute inset-0 bg-gradient-to-b from-ink/80 via-ink/60 to-ink" />

      <div className="relative flex h-full flex-col items-center justify-center px-8">
        <div className="relative aspect-[3/4] w-[62vw] max-w-[260px] overflow-hidden rounded-[28px] border border-white/15">
          <img src={photo} alt="" className="h-full w-full object-cover" />
          <div className="scanline absolute inset-x-0 h-24 bg-gradient-to-b from-transparent via-acid/35 to-transparent" />
          <div className="absolute inset-0 ring-1 ring-inset ring-acid/30" />
          <span className="absolute left-2 top-2 h-4 w-4 border-l-2 border-t-2 border-acid" />
          <span className="absolute right-2 top-2 h-4 w-4 border-r-2 border-t-2 border-acid" />
          <span className="absolute bottom-2 left-2 h-4 w-4 border-b-2 border-l-2 border-acid" />
          <span className="absolute bottom-2 right-2 h-4 w-4 border-b-2 border-r-2 border-acid" />
        </div>

        <div className="mt-10 w-full max-w-[280px]">
          <div className="h-px w-full overflow-hidden bg-white/10">
            <div className="h-px bg-acid transition-all duration-300 ease-out" style={{ width: `${(step + 1) * 25}%` }} />
          </div>
          <div className="mt-4 flex items-baseline justify-between">
            <span className="font-mono text-[10px] uppercase tracking-[0.24em] text-white/60">{STEPS[step]}</span>
            <span className="font-mono text-[10px] tabular-nums text-acid">{step + 1}/4</span>
          </div>
        </div>
      </div>
    </div>
  );
}
