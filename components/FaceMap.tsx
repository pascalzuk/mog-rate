'use client';

import { useState } from 'react';
import { ArrowsLeftRight, MagnifyingGlassPlus, X } from '@phosphor-icons/react';
import type { FaceMap } from '@/lib/faceMetrics';

const SIGNAL = '#ff5a2b';
const AMBER = '#ffb020';
const VIOLET = '#a694ff';

function Marks({ map, detail }: { map: FaceMap; detail: boolean }) {
  const worst = [...map.pairs].sort((a, b) => b.off - a.off).slice(0, 2);
  const worstSet = new Set(worst);
  const underDark =
    map.underEyes.length > 0
      ? map.underEyes.reduce((s, u) => s + u.darkness, 0) / map.underEyes.length
      : 0;

  return (
    <g>
      {/* The face's own midline: glabella to chin. Features that sit off it are
          what the eye reads as uneven. */}
      <polyline
        points={map.midline.map((p) => `${p.x * map.width},${p.y * map.height}`).join(' ')}
        fill="none"
        stroke="rgba(255,255,255,0.45)"
        strokeWidth={detail ? 1.4 : 1}
        strokeDasharray={detail ? '5 5' : '3 4'}
        vectorEffect="non-scaling-stroke"
      />

      {map.pairs.map((pair, i) => {
        const hot = worstSet.has(pair);
        const ax = pair.a.x * map.width;
        const ay = pair.a.y * map.height;
        const bx = pair.b.x * map.width;
        const by = pair.b.y * map.height;
        return (
          <g key={i}>
            <line
              x1={ax}
              y1={ay}
              x2={bx}
              y2={by}
              stroke={hot ? SIGNAL : 'rgba(255,255,255,0.22)'}
              strokeWidth={hot ? 1.6 : 1}
              strokeDasharray={hot ? undefined : '3 4'}
              vectorEffect="non-scaling-stroke"
            />
            <circle cx={ax} cy={ay} r={detail ? 2.6 : 1.8} fill={hot ? SIGNAL : 'rgba(255,255,255,0.6)'} />
            <circle cx={bx} cy={by} r={detail ? 2.6 : 1.8} fill={hot ? SIGNAL : 'rgba(255,255,255,0.6)'} />
          </g>
        );
      })}

      {map.underEyes.map((u, i) => (
        <ellipse
          key={`u${i}`}
          cx={u.x * map.width}
          cy={u.y * map.height}
          rx={u.rx * map.width}
          ry={u.ry * map.height}
          fill={u.darkness > 0.07 ? 'rgba(166,148,255,0.16)' : 'none'}
          stroke={u.darkness > 0.05 ? VIOLET : 'rgba(166,148,255,0.45)'}
          strokeWidth={detail ? 1.6 : 1.1}
          strokeDasharray={detail ? undefined : '4 3'}
          vectorEffect="non-scaling-stroke"
        />
      ))}

      {map.spots.map((s, i) => (
        <circle
          key={`s${i}`}
          cx={s.x * map.width}
          cy={s.y * map.height}
          r={Math.max(detail ? 5 : 3, s.r * map.width)}
          fill="rgba(255,176,32,0.14)"
          stroke={AMBER}
          strokeWidth={detail ? 1.5 : 1}
          vectorEffect="non-scaling-stroke"
        />
      ))}

      {detail ? (
        <g>
          {map.underEyes.map((u, i) => (
            <text
              key={`ut${i}`}
              x={u.x * map.width}
              y={u.y * map.height + u.ry * map.height + 14}
              textAnchor="middle"
              fill={VIOLET}
              fontSize={Math.max(12, map.width * 0.026)}
              fontFamily="ui-monospace, monospace"
            >
              {Math.round(u.darkness * 100)}% darker
            </text>
          ))}
        </g>
      ) : null}
      {underDark > 0 && !detail ? null : null}
    </g>
  );
}

export function FacePhoto({
  map,
  photo,
  mirror,
  detail,
  className,
}: {
  map: FaceMap;
  photo: string;
  mirror: boolean;
  detail: boolean;
  className?: string;
}) {
  return (
    <div
      className={`relative overflow-hidden bg-black ${className ?? ''}`}
      style={{ aspectRatio: `${map.width} / ${map.height}` }}
    >
      <div className="absolute inset-0" style={{ transform: mirror ? 'scaleX(-1)' : undefined }}>
        {/* contain + meet letterbox identically, so the marks stay on the face
            even when the box is a different shape than the photo. */}
        <img src={photo} alt="Your face with scan marks" className="h-full w-full object-contain" />
        <svg
          viewBox={`0 0 ${map.width} ${map.height}`}
          preserveAspectRatio="xMidYMid meet"
          className="absolute inset-0 h-full w-full"
        >
          <Marks map={map} detail={detail} />
        </svg>
      </div>
    </div>
  );
}

export function FaceMapCard({
  map,
  photo,
  onOpen,
}: {
  map: FaceMap;
  photo: string;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label="Open the face map"
      className="flex w-full items-center gap-3 border-t border-white/10 pt-3 text-left active:opacity-80"
    >
      <FacePhoto map={map} photo={photo} mirror={false} detail={false} className="w-[72px] shrink-0 rounded-[10px]" />
      <div className="min-w-0 flex-1">
        <p className="text-[12.5px] font-bold tracking-[-0.01em]">Face map</p>
        <p className="mt-0.5 text-[10.5px] leading-snug text-white/40">
          {map.spots.length ? `${map.spots.length} spots` : 'No spots'} ·{' '}
          {map.underEyes.length
            ? `${Math.round(
                (map.underEyes.reduce((s, u) => s + u.darkness, 0) / map.underEyes.length) * 100,
              )}% under-eye shadow`
            : 'under-eyes clear'}{' '}
          · asymmetry marked
        </p>
      </div>
      <MagnifyingGlassPlus size={16} className="shrink-0 text-white/40" />
    </button>
  );
}

export function FaceMapSheet({
  map,
  photo,
  onClose,
}: {
  map: FaceMap;
  photo: string;
  onClose: () => void;
}) {
  const [mirror, setMirror] = useState(false);
  const spots = map.spots.length;
  const under =
    map.underEyes.length > 0
      ? map.underEyes.reduce((s, u) => s + u.darkness, 0) / map.underEyes.length
      : 0;
  const worst = map.pairs.reduce((m, p) => Math.max(m, p.off), 0);

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-ink text-bone" style={{ height: '100dvh' }}>
      <header className="safe-top flex items-center justify-between px-5 pb-2 pt-1">
        <div>
          <p className="text-[13px] font-bold">Face map</p>
          <p className="font-mono text-[8px] uppercase tracking-[0.14em] text-white/30">
            measured on this frame
          </p>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={() => setMirror((m) => !m)}
            aria-label="Flip the photo"
            className="grid h-9 w-9 place-items-center text-white/55 active:text-white"
          >
            <ArrowsLeftRight size={16} weight="bold" />
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close the face map"
            className="grid h-9 w-9 place-items-center text-white/70 active:text-white"
          >
            <X size={16} weight="bold" />
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 items-center justify-center px-4">
        <FacePhoto map={map} photo={photo} mirror={mirror} detail className="w-full" />
      </div>

      <div className="safe-bottom px-5 pt-3">
        <div className="flex gap-4 border-t border-white/10 pt-3">
          <div className="flex-1">
            <p className="font-mono text-[8px] uppercase tracking-[0.12em] text-white/30">Asymmetry</p>
            <p className="mt-1 text-[13px] font-bold tabular-nums">
              {worst < 0.02 ? 'even' : `${(worst * 100).toFixed(1)}% off axis`}
            </p>
          </div>
          <div className="flex-1">
            <p className="font-mono text-[8px] uppercase tracking-[0.12em] text-white/30">Spots</p>
            <p className="mt-1 text-[13px] font-bold tabular-nums text-[#ffb020]">{spots}</p>
          </div>
          <div className="flex-1">
            <p className="font-mono text-[8px] uppercase tracking-[0.12em] text-white/30">Under-eye</p>
            <p className="mt-1 text-[13px] font-bold tabular-nums text-[#a694ff]">
              {under > 0.005 ? `${Math.round(under * 100)}% darker` : 'clear'}
            </p>
          </div>
        </div>
        <p className="mt-2 pb-1 text-[10px] leading-snug text-white/30">
          Red lines join features that do not sit evenly across the midline. Orange rings are counted
          spots; violet ellipses are the bands read for dark circles. Values come from this frame, so
          they can differ slightly from the median of the scan.
        </p>
      </div>
    </div>
  );
}
