'use client';

import { useCallback, useEffect, useState } from 'react';
import {
  CheckCircle,
  Compass,
  DeviceMobile,
  DotsThreeVertical,
  Plus,
  ShareNetwork,
  X,
} from '@phosphor-icons/react';

type InstallEvent = Event & {
  prompt: () => Promise<void>;
  userChoice?: Promise<{ outcome: 'accepted' | 'dismissed' }>;
};

const DISMISS_KEY = 'mog:install-dismissed';

function detectIos() {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  const iphone = /iphone|ipad|ipod/i.test(ua);
  const ipadOs = navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1;
  return iphone || ipadOs;
}

function detectStandalone() {
  if (typeof window === 'undefined') return false;
  const standalone = window.matchMedia('(display-mode: standalone)').matches;
  const iosStandalone = (navigator as Navigator & { standalone?: boolean }).standalone === true;
  return standalone || iosStandalone;
}

function Step({ icon, title, body }: { icon: React.ReactNode; title: string; body: React.ReactNode }) {
  return (
    <li className="flex gap-3">
      <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full border border-white/10 text-signal">
        {icon}
      </span>
      <div>
        <p className="text-[12.5px] font-semibold">{title}</p>
        <p className="mt-0.5 text-[11.5px] leading-relaxed text-white/40">{body}</p>
      </div>
    </li>
  );
}

export default function InstallPrompt() {
  const [ready, setReady] = useState(false);
  const [standalone, setStandalone] = useState(false);
  const [ios, setIos] = useState(false);
  const [dismissed, setDismissed] = useState(true);
  const [open, setOpen] = useState(false);
  const [deferred, setDeferred] = useState<InstallEvent | null>(null);

  useEffect(() => {
    setStandalone(detectStandalone());
    setIos(detectIos());
    try {
      setDismissed(window.localStorage.getItem(DISMISS_KEY) === '1');
    } catch {
      setDismissed(false);
    }
    setReady(true);

    const mq = window.matchMedia('(display-mode: standalone)');
    const onChange = () => setStandalone(detectStandalone());
    mq.addEventListener('change', onChange);
    window.addEventListener('appinstalled', onChange);

    const onPrompt = (event: Event) => {
      event.preventDefault();
      setDeferred(event as InstallEvent);
    };
    window.addEventListener('beforeinstallprompt', onPrompt);

    return () => {
      mq.removeEventListener('change', onChange);
      window.removeEventListener('appinstalled', onChange);
      window.removeEventListener('beforeinstallprompt', onPrompt);
    };
  }, []);

  const close = useCallback(() => setOpen(false), []);

  const dismiss = useCallback(() => {
    setDismissed(true);
    setOpen(false);
    try {
      window.localStorage.setItem(DISMISS_KEY, '1');
    } catch {
      // private mode
    }
  }, []);

  const install = useCallback(async () => {
    if (deferred) {
      await deferred.prompt();
      await deferred.userChoice?.catch(() => undefined);
      setDeferred(null);
      return;
    }
    setOpen(true);
  }, [deferred]);

  if (!ready || standalone || dismissed) return null;

  return (
    <>
      <div className="flex items-center gap-3 rounded-2xl border border-white/10 px-3.5 py-3">
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-signal/10 text-signal">
          <DeviceMobile size={16} weight="fill" />
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-[12px] font-semibold">Add to home screen</p>
          <p className="mt-0.5 text-[10.5px] text-white/40">Full screen, no browser bar</p>
        </div>
        <button
          type="button"
          onClick={install}
          className="h-8 shrink-0 rounded-full bg-bone px-3 text-[9.5px] font-bold uppercase tracking-[0.08em] text-ink active:scale-95"
        >
          {deferred ? 'Install' : ios ? 'How' : 'Install'}
        </button>
        <button type="button" onClick={dismiss} aria-label="Dismiss" className="shrink-0 text-white/25">
          <X size={13} weight="bold" />
        </button>
      </div>

      {open ? (
        <div
          className="fixed inset-0 z-50 flex items-end"
          role="dialog"
          aria-modal="true"
          aria-label="Add to home screen"
        >
          <button type="button" aria-label="Close" onClick={close} className="absolute inset-0 bg-ink/85" />
          <div className="safe-bottom relative w-full rounded-t-[26px] border-t border-white/10 bg-panel px-5 pb-3 pt-3">
            <div className="mx-auto mb-4 h-1 w-10 rounded-full bg-white/15" />
            <div className="flex items-start justify-between gap-4">
              <h2 className="text-[20px] font-black leading-tight tracking-[-0.03em]">
                Add Mog to your home screen
              </h2>
              <button
                type="button"
                onClick={close}
                aria-label="Close"
                className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-white/10 text-white/55"
              >
                <X size={14} weight="bold" />
              </button>
            </div>

            <ol className="mt-4 space-y-3.5">
              {ios ? (
                <>
                  <Step
                    icon={<Compass size={14} weight="bold" />}
                    title="Open this page in Safari"
                    body="Chrome and Instagram browsers on iPhone cannot install it."
                  />
                  <Step
                    icon={<ShareNetwork size={14} weight="bold" />}
                    title="Tap the Share button"
                    body="The square with the arrow pointing up, bottom toolbar."
                  />
                  <Step
                    icon={<Plus size={14} weight="bold" />}
                    title="Choose Add to Home Screen"
                    body="Scroll the sheet until you see it, then tap Add."
                  />
                  <Step
                    icon={<CheckCircle size={14} weight="bold" />}
                    title="Launch it from the new icon"
                    body="Full screen, no browser chrome."
                  />
                </>
              ) : (
                <>
                  <Step
                    icon={<Compass size={14} weight="bold" />}
                    title="Open this page in Chrome"
                    body="Desktop or Android Chrome can install it from the menu."
                  />
                  <Step
                    icon={<DotsThreeVertical size={14} weight="bold" />}
                    title="Open the browser menu"
                    body="The three dots in the top right corner."
                  />
                  <Step
                    icon={<Plus size={14} weight="bold" />}
                    title="Tap Install app"
                    body="It lands on your home screen with its own icon."
                  />
                </>
              )}
            </ol>

            <button
              type="button"
              onClick={close}
              className="mt-5 h-11 w-full rounded-full bg-bone text-[11px] font-bold uppercase tracking-[0.12em] text-ink active:scale-[0.99]"
            >
              Got it
            </button>
          </div>
        </div>
      ) : null}
    </>
  );
}
