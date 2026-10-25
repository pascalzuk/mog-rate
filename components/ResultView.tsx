'use client';

import { useMemo, useState } from 'react';
import type { ScanResult } from '@/lib/faceMetrics';

type Props = {
  result: ScanResult;
  photo: string;
  onAgain: () => void;
  onHome: () => void;
};

export default function ResultView({ result, photo, onAgain, onHome }: Props) {
  const [shared, setShared] = useState(false);
  const pct = Math.max(0, Math.min(100, (result.score / 10) * 100));

  const shareText = useMemo(
    () => `I scored ${result.score.toFixed(1)}/10 on Mog Rate — ${result.tier}. ${result.verdict}`,
    [result],
  );

  const share = async () => {
    try {
      if (navigator.share) {
        await navigator.share({ title: 'Mog Rate', text: shareText });
        return;
      }
      await navigator.clipboard.writeText(shareText);
      setShared(true);
      window.setTimeout(() => setShared(false), 2000);
    } catch {
      // user dismissed the sheet — nothing to do
    }
  };

  return (
    <div className="relative flex min-h-[100dvh] flex-col overflow-y-auto">
      <header className="safe-top flex items-center justify-between px-5 pt-3">
        <div className="flex items-baseline gap-2">
          <span className="text-[13px] font-black tracking-[-0.02em]">MOG</span>
          <span className="font-mono text-[9px] tracking-[0.32em] text-white/35">RATE</span>
        </div>
        <button
          type="button"
          onClick={onHome}
          aria-label="Back to home"
          className="grid h-9 w-9 place-items-center rounded-full border border-white/12 text-white/60 active:bg-white/5"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden>
            <path d="M2 2l12 12M14 2L2 14" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </header>

      <div className="flex flex-1 flex-col px-6 pb-2 pt-4">
        <div className="rise flex items-center gap-3">
          <div className="relative h-[52px] w-[52px] shrink-0">
            <div
              className="absolute inset-0 rounded-full"
              style={{
                background: `conic-gradient(#d8ff4d ${pct}%, rgba(255,255,255,0.09) 0)`,
              }}
            />
            <div className="absolute inset-[2px] overflow-hidden rounded-full bg-ink">
              <img src={photo} alt="Your scan" className="h-full w-full object-cover" />
            </div>
          </div>
          <div>
            <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-white/35">Scan complete</p>
            <p className="mt-1 text-[12px] text-white/60">
              {result.pose.ok ? 'Frontal analysis · 478 points' : 'Angle detected · straighten up for a better read'}
            </p>
          </div>
        </div>

        <div className="rise rise-1 mt-7 flex items-end gap-1">
          <span className="text-[clamp(84px,26vw,124px)] font-black leading-[0.78] tracking-[-0.06em] tabular-nums">
            {result.score.toFixed(1)}
          </span>
          <span className="mb-2 font-mono text-[13px] tracking-[0.1em] text-white/30">/10</span>
        </div>

        <div className="rise rise-1 mt-4 flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-acid px-3 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-black">
            {result.tier}
          </span>
          {!result.pose.ok ? (
            <span className="rounded-full border border-white/12 px-3 py-1 font-mono text-[9px] uppercase tracking-[0.16em] text-white/45">
              Angle off
            </span>
          ) : null}
        </div>

        <p className="rise rise-2 mt-5 font-serif text-[24px] leading-[1.15] tracking-[-0.015em] text-bone">
          {result.tierNote}
        </p>
        <p className="rise rise-2 mt-2 max-w-[300px] text-[13px] leading-relaxed text-white/45">{result.verdict}</p>

        <div className="rise rise-3 mt-8">
          <div className="flex items-center justify-between">
            <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-white/30">Breakdown</p>
            <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/20">per metric</p>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-x-5 gap-y-4">
            {result.metrics.map((metric) => (
              <div key={metric.key}>
                <div className="flex items-baseline justify-between">
                  <span className="text-[10px] uppercase tracking-[0.1em] text-white/45">{metric.label}</span>
                  <span className="text-[11px] font-semibold tabular-nums text-white/80">{metric.rating.toFixed(1)}</span>
                </div>
                <div className="mt-1.5 h-[3px] w-full overflow-hidden rounded-full bg-white/8">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.max(2, (metric.rating / 10) * 100)}%`,
                      background: metric.rating >= 7.5 ? '#d8ff4d' : metric.rating >= 6 ? 'rgba(216,255,77,0.55)' : 'rgba(255,255,255,0.28)',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <p className="rise rise-3 mt-8 font-mono text-[9px] leading-relaxed tracking-[0.14em] text-white/20">
          MEASURED LOCALLY · YOUR PHOTO NEVER LEAVES THIS DEVICE
        </p>
      </div>

      <div className="safe-bottom sticky bottom-0 z-10 mt-6 bg-gradient-to-t from-ink via-ink to-transparent px-6 pb-3 pt-6">
        <button
          type="button"
          onClick={onAgain}
          className="h-[54px] w-full rounded-full bg-acid text-[12px] font-black uppercase tracking-[0.16em] text-black transition active:scale-[0.985]"
        >
          Run it again
        </button>
        <button
          type="button"
          onClick={share}
          className="mt-2 h-[50px] w-full rounded-full border border-white/12 text-[11px] font-semibold uppercase tracking-[0.16em] text-white/70 transition active:bg-white/5"
        >
          {shared ? 'Copied' : 'Share result'}
        </button>
      </div>
    </div>
  );
}
