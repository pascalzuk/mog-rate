'use client';

import { useEffect, useState } from 'react';

const STEPS = [
  'Correcting head angle',
  'Mapping 478 landmarks',
  'Measuring 9 categories',
  'Reading skin pixels',
  'Settling score',
];

export default function AnalyzingView({ photo }: { photo: string }) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const id = window.setInterval(() => setStep((s) => Math.min(STEPS.length - 1, s + 1)), 280);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="fixed inset-0 z-40 overflow-hidden bg-ink">
      {photo ? (
        <img
          src={photo}
          alt=""
          className="absolute inset-0 h-full w-full scale-110 object-cover opacity-30 blur-[3px]"
        />
      ) : null}
      <div className="absolute inset-0 bg-gradient-to-b from-ink/85 via-ink/70 to-ink" />

      <div className="relative flex h-full flex-col items-center justify-center px-8">
        <div className="relative aspect-[3/4] w-[60vw] max-w-[250px] overflow-hidden rounded-3xl border border-white/12">
          {photo ? <img src={photo} alt="" className="h-full w-full object-cover" /> : null}
          <div className="scanline absolute inset-x-0 h-20 bg-gradient-to-b from-transparent via-acid/30 to-transparent" />
          <span className="absolute left-2.5 top-2.5 h-4 w-4 border-l-2 border-t-2 border-acid" />
          <span className="absolute right-2.5 top-2.5 h-4 w-4 border-r-2 border-t-2 border-acid" />
          <span className="absolute bottom-2.5 left-2.5 h-4 w-4 border-b-2 border-l-2 border-acid" />
          <span className="absolute bottom-2.5 right-2.5 h-4 w-4 border-b-2 border-r-2 border-acid" />
        </div>

        <div className="mt-9 w-full max-w-[280px]">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-white/60">
              {STEPS[step]}
            </span>
            <span className="font-mono text-[10px] tabular-nums text-acid">{step + 1}/5</span>
          </div>
          <div className="mt-3 h-[2px] w-full overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full rounded-full bg-acid transition-all duration-300 ease-out"
              style={{ width: `${((step + 1) / STEPS.length) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
