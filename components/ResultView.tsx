'use client';

import { useMemo, useState } from 'react';
import { ArrowClockwise, ShareNetwork, X } from '@phosphor-icons/react';
import type { ScanResult } from '@/lib/faceMetrics';

type Props = {
  result: ScanResult;
  photo: string;
  onAgain: () => void;
  onHome: () => void;
};

const SEVERITY = ['', 'minor', 'notable', 'major'];

export default function ResultView({ result, photo, onAgain, onHome }: Props) {
  const [shared, setShared] = useState(false);
  const pct = Math.max(2, Math.min(100, (result.score / 10) * 100));
  const faceFlags = result.flags.filter((f) => f.kind === 'face');
  const captureFlags = result.flags.filter((f) => f.kind === 'capture');

  const shareText = useMemo(
    () => `I scored ${result.score.toFixed(1)}/10 on Mog Rate. ${result.tier}. ${result.verdict}`,
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
      // dismissed the share sheet
    }
  };

  return (
    <div
      className="relative flex min-h-[100dvh] flex-col overflow-y-auto"
      data-score={result.score}
      data-skin-clarity={result.skin.clarity.toFixed(3)}
      data-skin-blemish={result.skin.blemishShare.toFixed(3)}
      data-confidence={result.confidence}
      data-frames={result.frames}
      data-tier={result.tier}
    >
      <header className="safe-top flex items-center justify-between px-5 pt-3">
        <div className="flex items-baseline gap-2">
          <span className="text-[16px] font-black tracking-[-0.04em]">MOG</span>
          <span className="font-mono text-[9px] tracking-[0.34em] text-white/30">RATE</span>
        </div>
        <button
          type="button"
          onClick={onHome}
          aria-label="Back to home"
          className="grid h-9 w-9 place-items-center rounded-full border border-white/12 text-white/60 transition active:bg-white/5"
        >
          <X size={15} weight="bold" />
        </button>
      </header>

      <div className="flex flex-1 flex-col px-5 pb-2 pt-5">
        <div className="rise flex items-center gap-3">
          <div className="relative h-[58px] w-[58px] shrink-0">
            <div
              className="absolute inset-0 rounded-2xl"
              style={{ background: `conic-gradient(#d8ff4d ${pct}%, rgba(255,255,255,0.08) 0)` }}
            />
            <div className="absolute inset-[2px] overflow-hidden rounded-[13px] bg-ink">
              {photo ? <img src={photo} alt="Your scan" className="h-full w-full object-cover" /> : null}
            </div>
          </div>
          <div className="min-w-0">
            <p className="font-mono text-[9px] uppercase tracking-[0.26em] text-white/30">Scan complete</p>
            <p className="mt-1 text-[12px] leading-snug text-white/55">
              {result.frames > 1 ? `${result.frames} frames averaged` : 'Single frame read'}
              {result.skin.measured ? ', skin sampled from pixels' : ', landmarks only'}
            </p>
            <p className="mt-0.5 font-mono text-[9px] uppercase tracking-[0.14em] text-white/30">
              Confidence {Math.round(result.confidence * 100)}%
            </p>
          </div>
        </div>

        <div className="rise rise-1 mt-7 flex items-end gap-1.5">
          <span className="text-[clamp(80px,25vw,120px)] font-black leading-[0.78] tracking-[-0.06em] tabular-nums">
            {result.score.toFixed(1)}
          </span>
          <span className="mb-2 font-mono text-[13px] tracking-[0.08em] text-white/25">/10</span>
        </div>

        <div className="rise rise-1 mt-4 flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-acid px-3 py-1 text-[10px] font-black uppercase tracking-[0.1em] text-ink">
            {result.tier}
          </span>
          {!result.pose.ok ? (
            <span className="rounded-full border border-white/12 px-3 py-1 font-mono text-[9px] uppercase tracking-[0.14em] text-white/45">
              Angle corrected
            </span>
          ) : null}
          {result.skin.measured ? (
            <span className="rounded-full border border-white/12 px-3 py-1 font-mono text-[9px] uppercase tracking-[0.14em] text-white/45">
              Skin read
            </span>
          ) : null}
        </div>

        <p className="rise rise-2 mt-5 text-[23px] font-black leading-[1.12] tracking-[-0.035em] text-bone">
          {result.tierNote}
        </p>
        <p className="rise rise-2 mt-2 max-w-[320px] text-[13px] leading-relaxed text-white/45">
          {result.verdict}
        </p>

        <section className="rise rise-3 mt-8">
          <div className="flex items-baseline justify-between">
            <h2 className="text-[13px] font-bold tracking-[-0.01em]">Deductions</h2>
            <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-white/25">
              {faceFlags.length ? `${faceFlags.length} found` : 'none'}
            </span>
          </div>

          {faceFlags.length ? (
            <ul className="mt-3 space-y-2.5">
              {faceFlags.slice(0, 5).map((flag) => (
                <li key={flag.key} className="border-l border-white/12 pl-3">
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="text-[12.5px] font-semibold text-bone">{flag.label}</span>
                    <span className="font-mono text-[8px] uppercase tracking-[0.16em] text-white/30">
                      {SEVERITY[flag.severity]}
                    </span>
                  </div>
                  <p className="mt-0.5 text-[11.5px] leading-relaxed text-white/40">{flag.detail}</p>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-[12.5px] leading-relaxed text-white/50">
              No deductions in any measured category. That is rare.
            </p>
          )}

          {captureFlags.length ? (
            <p className="mt-3 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-[11px] leading-relaxed text-white/40">
              {captureFlags.map((f) => f.detail).join(' ')}
            </p>
          ) : null}
        </section>

        {result.skin.measured ? (
          <section className="mt-7 grid grid-cols-3 gap-3 border-y border-white/10 py-4">
            {[
              ['Clear skin', Math.round(result.skin.clarity * 100)],
              ['Texture', Math.round(result.skin.texture * 100)],
              ['Under-eyes', Math.round(result.skin.underEyes * 100)],
            ].map(([label, value]) => (
              <div key={String(label)}>
                <p className="text-[19px] font-bold leading-none tabular-nums text-bone">{value}%</p>
                <p className="mt-1.5 font-mono text-[8px] uppercase tracking-[0.14em] text-white/30">{label}</p>
              </div>
            ))}
          </section>
        ) : null}

        <section className="mt-7">
          <div className="flex items-baseline justify-between">
            <h2 className="text-[13px] font-bold tracking-[-0.01em]">Breakdown</h2>
            <span className="font-mono text-[9px] uppercase tracking-[0.18em] text-white/25">out of 10</span>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-x-5 gap-y-3.5">
            {result.metrics.map((metric) => (
              <div key={metric.key}>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate text-[10.5px] uppercase tracking-[0.06em] text-white/45">
                    {metric.label}
                  </span>
                  <span className="text-[11.5px] font-semibold tabular-nums text-white/85">
                    {metric.rating.toFixed(1)}
                  </span>
                </div>
                <div className="mt-1.5 h-[3px] w-full overflow-hidden rounded-full bg-white/[0.07]">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.max(3, (metric.rating / 10) * 100)}%`,
                      background:
                        metric.rating >= 7
                          ? '#d8ff4d'
                          : metric.rating >= 5
                            ? 'rgba(216,255,77,0.55)'
                            : 'rgba(255,255,255,0.3)',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>

        <p className="mt-7 font-mono text-[8px] leading-relaxed tracking-[0.14em] text-white/20">
          MEASURED LOCALLY · PHOTO NEVER LEAVES THIS DEVICE · GEOMETRY PLUS SKIN PIXEL HEURISTIC
        </p>
      </div>

      <div className="safe-bottom sticky bottom-0 z-10 mt-5 bg-gradient-to-t from-ink via-ink to-transparent px-5 pb-3 pt-6">
        <button
          type="button"
          onClick={onAgain}
          className="flex h-[56px] w-full items-center justify-center gap-2 rounded-full bg-acid text-[12px] font-black uppercase tracking-[0.14em] text-ink transition active:scale-[0.985]"
        >
          <ArrowClockwise size={16} weight="bold" />
          Scan again
        </button>
        <button
          type="button"
          onClick={share}
          className="mt-2 flex h-[50px] w-full items-center justify-center gap-2 rounded-full border border-white/12 text-[11px] font-semibold uppercase tracking-[0.14em] text-white/70 transition active:bg-white/5"
        >
          <ShareNetwork size={15} weight="bold" />
          {shared ? 'Copied to clipboard' : 'Share result'}
        </button>
      </div>
    </div>
  );
}
