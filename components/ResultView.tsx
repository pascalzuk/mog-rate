'use client';

import { useMemo, useState } from 'react';
import { ArrowClockwise, ArrowsLeftRight, ShareNetwork, X } from '@phosphor-icons/react';
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
  // Default to the mirrored view: it is what the user saw in the viewfinder.
  const [mirror, setMirror] = useState(true);
  const pct = Math.max(3, Math.min(100, (result.score / 10) * 100));
  const faceFlags = result.flags.filter((f) => f.kind === 'face');
  const captureFlags = result.flags.filter((f) => f.kind === 'capture');
  const symmetry = result.metrics.find((m) => m.key === 'symmetry');
  const skin = result.skin;

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
      // dismissed
    }
  };

  return (
    <div
      className="relative flex min-h-[100dvh] flex-col bg-ink"
      data-score={result.score}
      data-skin-clarity={skin.clarity.toFixed(3)}
      data-texture={skin.texture.toFixed(3)}
      data-share={skin.blemishShare.toFixed(4)}
      data-spots={skin.spots}
      data-symmetry={symmetry ? symmetry.rating : ''}
      data-flip={result.raw.flipDifference.toFixed(3)}
      data-deviation={result.raw.symmetryDev.toFixed(4)}
      data-fullness={result.raw.fullness.toFixed(3)}
      data-confidence={result.confidence}
      data-frames={result.frames}
      data-views={result.views}
      data-tier={result.tier}
    >
      <header className="safe-top flex items-center justify-between px-5 pb-1">
        <span className="font-mono text-[8.5px] uppercase tracking-[0.16em] text-white/25">
          {result.views === 1 ? 'Photo' : `${result.views} views`} · {result.frames} frames
        </span>
        <button
          type="button"
          onClick={onHome}
          aria-label="Back to home"
          className="grid h-9 w-9 place-items-center text-white/45 active:text-white"
        >
          <X size={15} weight="bold" />
        </button>
      </header>

      <div className="flex-1 px-5 pt-2">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-end gap-1.5">
              <span className="rise text-[clamp(76px,24vw,104px)] font-black leading-[0.76] tracking-[-0.065em] tabular-nums">
                {result.score.toFixed(1)}
              </span>
              <span className="mb-2 font-mono text-[11px] tracking-[0.04em] text-white/25">/10</span>
            </div>
            <p className="mt-2 text-[11px] font-black uppercase tracking-[0.1em] text-signal">
              {result.tier}
            </p>
          </div>

          <button
            type="button"
            onClick={() => setMirror((m) => !m)}
            aria-label="Flip your photo"
            className="relative h-[66px] w-[66px] shrink-0 overflow-hidden rounded-[14px] bg-panel active:scale-[0.97]"
          >
            {photo ? (
              <img
                src={photo}
                alt="Your scan"
                className="h-full w-full object-cover"
                style={{ transform: mirror ? 'scaleX(-1)' : undefined }}
              />
            ) : null}
            <span
              className="absolute inset-x-0 bottom-0 h-[4px] bg-signal"
              style={{ width: `${pct}%` }}
            />
            <span className="absolute right-1 top-1 grid h-5 w-5 place-items-center rounded-full bg-ink/70 text-white/70">
              <ArrowsLeftRight size={11} weight="bold" />
            </span>
          </button>
        </div>

        <p className="rise rise-1 mt-4 text-[18px] font-black leading-[1.15] tracking-[-0.025em]">
          {result.tierNote}
        </p>
        <p className="rise rise-1 mt-1.5 text-[12.5px] leading-relaxed text-white/45">{result.verdict}</p>

        <section className="rise rise-2 mt-5 border-t border-white/10 pt-3">
          <div className="flex items-baseline justify-between">
            <h2 className="text-[12.5px] font-bold tracking-[-0.01em]">What cost you</h2>
            <span className="font-mono text-[8.5px] uppercase tracking-[0.14em] text-white/25">
              {faceFlags.length ? `${faceFlags.length} found` : 'nothing'}
            </span>
          </div>

          {faceFlags.length ? (
            <ul className="mt-1">
              {faceFlags.slice(0, 4).map((flag) => (
                <li key={flag.key} className="flex items-baseline gap-3 border-b border-white/[0.06] py-2">
                  <span className="w-[86px] shrink-0 text-[11.5px] font-semibold leading-tight">
                    {flag.label}
                  </span>
                  <span className="flex-1 text-[10.5px] leading-snug text-white/40">{flag.detail}</span>
                  <span className="shrink-0 font-mono text-[7.5px] uppercase tracking-[0.1em] text-white/25">
                    {SEVERITY[flag.severity]}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-2 text-[12px] text-white/45">No deductions in any measured category.</p>
          )}

          {captureFlags.length ? (
            <p className="mt-2 text-[10px] leading-relaxed text-white/30">
              {captureFlags.map((f) => f.detail).join(' ')}
            </p>
          ) : null}
        </section>

        {skin.measured ? (
          <section className="mt-4">
            <div className="grid grid-cols-4 border-t border-white/10 pt-2.5">
              {[
                ['Spots', String(skin.spots)],
                ['Skin', `${Math.round(skin.clarity * 100)}%`],
                ['Texture', `${Math.round(skin.texture * 100)}%`],
                ['Under-eye', `${Math.round(skin.underEyes * 100)}%`],
              ].map(([label, value]) => (
                <div key={label}>
                  <p className="text-[15px] font-bold leading-none tabular-nums">{value}</p>
                  <p className="mt-1.5 font-mono text-[7px] uppercase tracking-[0.09em] text-white/25">
                    {label}
                  </p>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[10px] leading-snug text-white/25">
              Spots counted from skin pixels, not guessed from landmarks.
            </p>
          </section>
        ) : null}

        <section className="mt-4">
          <div className="flex items-baseline justify-between border-t border-white/10 pt-3">
            <h2 className="text-[12.5px] font-bold tracking-[-0.01em]">Breakdown</h2>
            <span className="font-mono text-[8.5px] uppercase tracking-[0.14em] text-white/25">
              out of 10
            </span>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-x-5 gap-y-2.5">
            {result.metrics.map((metric) => (
              <div key={metric.key}>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate text-[10px] uppercase tracking-[0.04em] text-white/40">
                    {metric.label}
                  </span>
                  <span className="text-[11px] font-semibold tabular-nums text-white/85">
                    {metric.rating.toFixed(1)}
                  </span>
                </div>
                <div className="mt-1 h-[2px] w-full bg-white/[0.07]">
                  <div
                    className="h-full"
                    style={{
                      width: `${Math.max(2, (metric.rating / 10) * 100)}%`,
                      background:
                        metric.rating >= 7
                          ? '#ff5a2b'
                          : metric.rating >= 5
                            ? 'rgba(255,90,43,0.45)'
                            : 'rgba(255,255,255,0.25)',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </section>

        <p className="mt-5 pb-2 font-mono text-[7.5px] leading-relaxed tracking-[0.1em] text-white/20">
          CONFIDENCE {Math.round(result.confidence * 100)}% · MEASURED ON DEVICE · PHOTO NEVER LEAVES THE PHONE
        </p>
      </div>

      <div className="safe-bottom sticky bottom-0 z-10 mt-2 border-t border-white/10 bg-ink px-5 pt-3">
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onAgain}
            className="flex h-[50px] flex-1 items-center justify-center gap-2 rounded-full bg-signal text-[11px] font-black uppercase tracking-[0.12em] text-ink transition active:scale-[0.99]"
          >
            <ArrowClockwise size={15} weight="bold" />
            Scan again
          </button>
          <button
            type="button"
            onClick={share}
            aria-label="Share result"
            className="grid h-[50px] w-[50px] place-items-center rounded-full border border-white/12 text-white/65 transition active:bg-white/5"
          >
            <ShareNetwork size={16} weight="bold" />
          </button>
        </div>
        {shared ? (
          <p className="mt-1.5 text-center font-mono text-[8px] uppercase tracking-[0.14em] text-mint">
            Copied
          </p>
        ) : null}
      </div>
    </div>
  );
}
