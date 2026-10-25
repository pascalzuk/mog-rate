'use client';

import { ArrowRight } from '@phosphor-icons/react';
import InstallPrompt from '@/components/InstallPrompt';

const SIGNALS = [
  'Symmetry',
  'Skin and spots',
  'Jawline',
  'Water retention',
  'Facial thirds',
  'Midface',
  'Canthal tilt',
  'Eye area',
  'Face width',
  'Lips',
];

export default function HomeView({ onScan }: { onScan: () => void }) {
  return (
    <div className="flex min-h-[100dvh] flex-col bg-ink px-5">
      <header className="safe-top flex items-baseline justify-between pb-2">
        <div className="flex items-baseline gap-[3px]">
          <span className="text-[16px] font-black tracking-[-0.06em]">MOG</span>
          <span className="text-[16px] font-light tracking-[0.04em] text-white/30">RATE</span>
        </div>
        <span className="font-mono text-[8.5px] uppercase tracking-[0.16em] text-white/25">
          On device
        </span>
      </header>

      <section className="flex flex-1 flex-col justify-center">
        <p className="font-mono text-[9px] uppercase tracking-[0.2em] text-white/30">
          Six views, one honest number
        </p>

        <h1 className="mt-3 text-[clamp(44px,15.5vw,64px)] font-black leading-[0.84] tracking-[-0.055em]">
          How hard
          <br />
          do you <span className="text-signal">mog</span>
          <span className="text-signal">?</span>
        </h1>

        <p className="mt-4 max-w-[260px] text-[13px] leading-relaxed text-white/45">
          Turn your head through the guide, then face the camera. Ratios, skin, symmetry and
          fullness measured on the phone.
        </p>

        <button
          type="button"
          onClick={onScan}
          className="mt-6 flex h-[54px] w-full items-center justify-between rounded-full bg-signal pl-6 pr-2 text-ink transition active:scale-[0.99]"
        >
          <span className="text-[13px] font-black uppercase tracking-[0.1em]">Start scan</span>
          <span className="grid h-[42px] w-[42px] place-items-center rounded-full bg-ink text-signal">
            <ArrowRight size={17} weight="bold" />
          </span>
        </button>
      </section>

      <section className="pb-2">
        <div className="grid grid-cols-2 gap-x-4 border-t border-white/10 pt-3">
          {SIGNALS.map((signal) => (
            <p key={signal} className="py-[3px] text-[11px] leading-tight text-white/40">
              {signal}
            </p>
          ))}
        </div>
      </section>

      <div className="pb-3 pt-3">
        <InstallPrompt />
        <p className="mt-3 font-mono text-[7.5px] leading-relaxed tracking-[0.11em] text-white/20">
          RATIOS AND PIXEL HEURISTICS. NOT A DIAGNOSIS, NOT A RATING OF YOUR WORTH.
        </p>
      </div>
    </div>
  );
}
