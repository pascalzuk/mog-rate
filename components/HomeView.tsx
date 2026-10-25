'use client';

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full border border-white/10 px-2.5 py-1 font-mono text-[8px] uppercase tracking-[0.18em] text-white/40">
      {children}
    </span>
  );
}

export default function HomeView({ onScan }: { onScan: () => void }) {
  return (
    <div className="flex min-h-[100dvh] flex-col">
      <header className="safe-top flex items-center justify-between px-5 pt-3">
        <div className="flex items-baseline gap-2">
          <span className="text-[15px] font-black tracking-[-0.03em]">MOG</span>
          <span className="font-mono text-[9px] tracking-[0.32em] text-white/35">RATE</span>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-white/10 px-2.5 py-1">
          <span className="live-dot h-1 w-1 rounded-full bg-mint" />
          <span className="font-mono text-[8px] uppercase tracking-[0.2em] text-white/45">On-device</span>
        </div>
      </header>

      <div className="flex flex-1 flex-col px-6">
        <div className="rise mt-9">
          <p className="font-mono text-[9px] uppercase tracking-[0.34em] text-white/30">Self scan · No uploads</p>
          <h1 className="mt-4 text-[clamp(42px,13.5vw,62px)] font-black leading-[0.88] tracking-[-0.055em]">
            How hard
            <br />
            do you <span className="font-serif font-normal italic tracking-[-0.02em] text-acid">mog?</span>
          </h1>
          <p className="mt-4 max-w-[272px] text-[13px] leading-relaxed text-white/45">
            One photo. 478 facial landmarks measured right here on your phone. A rating out of ten before you head out.
          </p>
        </div>

        <div className="my-auto flex flex-col items-center py-8">
          <button type="button" onClick={onScan} aria-label="Start face scan" className="scan-cta">
            <span className="scan-orbit" aria-hidden />
            <span className="scan-glow" aria-hidden />
            <span className="scan-core">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden>
                <path
                  d="M4 8.5A2.5 2.5 0 016.5 6h1.2l1-2h6.6l1 2h1.2A2.5 2.5 0 0120 8.5v8A2.5 2.5 0 0117.5 19h-11A2.5 2.5 0 014 16.5v-8z"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinejoin="round"
                />
                <circle cx="12" cy="12.5" r="3.4" stroke="currentColor" strokeWidth="1.8" />
              </svg>
              <span className="mt-1 font-mono text-[9px] font-bold uppercase tracking-[0.18em]">Start scan</span>
            </span>
          </button>
          <p className="mt-6 font-mono text-[9px] uppercase tracking-[0.26em] text-white/25">Tap to open camera</p>
        </div>

        <div className="pb-2">
          <div className="flex flex-wrap gap-1.5">
            <Chip>478 landmarks</Chip>
            <Chip>On-device</Chip>
            <Chip>0 uploads</Chip>
          </div>
          <p className="mt-4 font-mono text-[8px] leading-relaxed tracking-[0.16em] text-white/20">
            GOOD LIGHT · FRONT CAMERA · NEUTRAL FACE
          </p>
        </div>
      </div>
    </div>
  );
}
