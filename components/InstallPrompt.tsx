'use client';

import { useCallback, useEffect, useState } from 'react';
import { CheckCircle, Compass, DeviceMobile, DotsThreeVertical, Plus, ShareNetwork, X } from '@phosphor-icons/react';

type InstallEvent = Event & {
  prompt: () => Promise<void>;
  userChoice?: Promise<{ outcome: 'accepted' | 'dismissed' }>;
};

const DISMISS_KEY = 'mog:install-dismissed';

function detectIos() {
  if (typeof navigator === 'undefined') return false;
  const ua = navigator.userAgent;
  const iphone = /iphone|ipad|ipod/i.test(ua);
  // iPadOS reports as a Mac; the touch points give it away.
  const ipadOs = navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1;
  return iphone || ipadOs;
}

function detectStandalone() {
  if (typeof window === 'undefined') return false;
  const standalone = window.matchMedia('(display-mode: standalone)').matches;
  const iosStandalone = (navigator as Navigator & { standalone?: boolean }).standalone === true;
  return standalone || iosStandalone;
}

function Step({
  icon,
  title,
  body,
}: {
  icon: React.ReactNode;
  title: string;
  body: React.ReactNode;
}) {
  return (
    <li className="flex gap-3">
      <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full border border-white/12 bg-white/[0.04] text-acid">
        {icon}
      </span>
      <div>
        <p className="text-[13px] font-semibold text-bone">{title}</p>
        <p className="mt-0.5 text-[12px] leading-relaxed text-white/45">{body}</p>
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
      // private mode, nothing to persist
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
      <div className="rounded-2xl border border-white/10 bg-white/[0.035] p-4">
        <div className="flex items-start gap-3">
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-acid/12 text-acid">
            <DeviceMobile size={18} weight="fill" />
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-[13px] font-semibold text-bone">Keep it on your home screen</p>
            <p className="mt-1 text-[11.5px] leading-relaxed text-white/45">
              Opens full screen, no browser bar, no address bar. Works like an app.
            </p>
            <div className="mt-3 flex items-center gap-2">
              <button
                type="button"
                onClick={install}
                className="h-9 rounded-full bg-acid px-4 text-[11px] font-bold uppercase tracking-[0.1em] text-ink transition active:scale-[0.97]"
              >
                {deferred ? 'Install app' : ios ? 'Show me how' : 'Add to home screen'}
              </button>
              <button
                type="button"
                onClick={dismiss}
                className="h-9 rounded-full px-3 text-[11px] font-semibold text-white/40 transition active:bg-white/5"
              >
                Not now
              </button>
            </div>
          </div>
        </div>
      </div>

      {open ? (
        <div className="fixed inset-0 z-50 flex items-end" role="dialog" aria-modal="true" aria-label="Add to home screen">
          <button
            type="button"
            aria-label="Close"
            onClick={close}
            className="absolute inset-0 bg-ink/80 backdrop-blur-sm"
          />
          <div className="safe-bottom relative w-full rounded-t-[28px] border-t border-white/10 bg-[#0e0e12] px-5 pb-4 pt-3">
            <div className="mx-auto mb-4 h-1 w-10 rounded-full bg-white/15" />
            <div className="flex items-start justify-between">
              <div>
                <p className="font-mono text-[9px] uppercase tracking-[0.28em] text-white/30">Install</p>
                <h2 className="mt-2 text-[22px] font-black leading-tight tracking-[-0.03em]">
                  Add Mog to your home screen
                </h2>
              </div>
              <button
                type="button"
                onClick={close}
                aria-label="Close"
                className="grid h-9 w-9 place-items-center rounded-full border border-white/12 text-white/60 active:bg-white/5"
              >
                <X size={15} weight="bold" />
              </button>
            </div>

            <ol className="mt-5 space-y-4">
              {ios ? (
                <>
                  <Step
                    icon={<Compass size={16} weight="bold" />}
                    title="Open this page in Safari"
                    body="Chrome and Instagram browsers on iPhone cannot install it. Copy the link into Safari."
                  />
                  <Step
                    icon={<ShareNetwork size={16} weight="bold" />}
                    title="Tap the Share button"
                    body="The square with the arrow pointing up, in the toolbar at the bottom."
                  />
                  <Step
                    icon={<Plus size={16} weight="bold" />}
                    title="Choose Add to Home Screen"
                    body="Scroll the share sheet until you see it, then tap Add."
                  />
                  <Step
                    icon={<CheckCircle size={16} weight="bold" />}
                    title="Launch it from the new icon"
                    body="It opens full screen with no browser chrome and keeps working offline."
                  />
                </>
              ) : (
                <>
                  <Step
                    icon={<Compass size={16} weight="bold" />}
                    title="Open this page on your phone"
                    body="Chrome on Android or desktop can install it in one tap from the menu."
                  />
                  <Step
                    icon={<DotsThreeVertical size={16} weight="bold" />}
                    title="Open the browser menu"
                    body="The three dots in the top right corner."
                  />
                  <Step
                    icon={<Plus size={16} weight="bold" />}
                    title="Tap Install app or Add to Home screen"
                    body="It lands on your home screen with its own icon and no address bar."
                  />
                </>
              )}
            </ol>

            <button
              type="button"
              onClick={close}
              className="mt-6 h-12 w-full rounded-full bg-bone text-[12px] font-bold uppercase tracking-[0.14em] text-ink transition active:scale-[0.99]"
            >
              Got it
            </button>
          </div>
        </div>
      ) : null}
    </>
  );
}
