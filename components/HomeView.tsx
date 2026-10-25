'use client';

import { Camera } from '@phosphor-icons/react';
import InstallPrompt from '@/components/InstallPrompt';

const MEASURED = [
  'Symmetry',
  'Jawline',
  'Skin clarity',
  'Proportions',
  'Midface',
  'Canthal tilt',
  'Eye area',
  'Face width',
  'Lips',
];

export default function HomeView({ onScan }: { onScan: () => void }) {
  return (
    <div className="flex min-h-[100dvh] flex-col px-5">
      <header className="safe-top flex items-center justify-between pt-3">
        <div className="flex items-baseline gap-2">
          <span className="text-[16px] font-black tracking-[-0.04em]">MOG</span>
          <span className="font-mono text-[9px] tracking-[0.34em] text-white/30">RATE</span>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-white/10 px-2.5 py-1">
          <span className="h-1 w-1 rounded-full bg-mint" />
          <span className="font-mono text-[8px] uppercase tracking-[0.2em] text-white/45">On device</span>
        </div>
      </header>

      <section className="flex flex-1 flex-col justify-center py-10">
        <p className="font-mono text-[9px] uppercase tracking-[0.32em] text-white/28">Live face scan</p>
        <h1 className="mt-5 text-[clamp(46px,14.5vw,66px)] font-black leading-[0.86] tracking-[-0.055em]">
          How hard
          <br />
          do you <span className="text-acid">mog?</span>
        </h1>
        <p className="mt-5 max-w-[290px] text-[13.5px] leading-relaxed text-white/45">
          Nine measurements from a live camera scan, including real skin clarity. Scored on your phone, never
          uploaded.
        </p>

        <button
          type="button"
          onClick={onScan}
          className="mt-9 flex h-[58px] w-full items-center justify-center gap-2.5 rounded-full bg-acid text-[13px] font-black uppercase tracking-[0.14em] text-ink transition active:scale-[0.985]"
        >
          <Camera size={19} weight="fill" />
          Start scan
        </button>
      </section>

      <div className="space-y-4 pb-4">
        <InstallPrompt />

        <div className="rounded-2xl border border-white/10 p-4">
          <p className="font-mono text-[8px] uppercase tracking-[0.26em] text-white/25">What gets measured</p>
          <div className="mt-3 grid grid-cols-3 gap-x-3 gap-y-2.5">
            {MEASURED.map((item) => (
              <span key={item} className="text-[10.5px] leading-tight text-white/55">
                {item}
              </span>
            ))}
          </div>
        </div>

        <p className="pb-2 font-mono text-[8px] leading-relaxed tracking-[0.16em] text-white/20">
          FIXED GEOMETRY RUBRIC, NOT A MEDICAL OR SCIENTIFIC ASSESSMENT
        </p>
      </div>
    </div>
  );
}
