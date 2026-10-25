# MOG RATE

A mobile-first face scan that rates how hard you mog. Point your camera at your face, get a rating out of 10, a tier, and a breakdown of eight facial measurements — before you head out to dinner.

Built to be added to the iOS home screen and look like a native app.

## How it works

- **`@mediapipe/tasks-vision` Face Landmarker** — a small on-device ML model that finds 478 facial landmarks (468 mesh points + iris centres) in real time.
- The camera preview draws a **green lock box** around your face as soon as it is detected.
- On scan, the photo is measured locally: symmetry, jawline, midface ratio, canthal tilt, proportions, eye area, facial width, lips.
- Landmarks are **pose-normalised** first (roll + yaw solved from landmark depth), so holding the phone at an angle doesn't tank the score.
- **No API keys, no accounts, no image uploads.** The photo never leaves the phone, which also means zero server cost and nothing to configure on Vercel.

The scoring is a fun, opinionated geometry rubric — not science, not medical, not that serious.

## Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000. Camera access needs `localhost` or HTTPS.

```bash
npm run build
npm start
```

## Deploy to Vercel

1. Push this repo to GitHub.
2. In Vercel: **Add New → Project → Import** the repo.
3. Framework preset **Next.js**, no environment variables, click **Deploy**.

That's it. Everything is client-side; the model and WASM runtime ship from `public/`.

## Add to iOS home screen

1. Open the deployed URL in **Safari** (must be Safari for the standalone mode).
2. Tap **Share → Add to Home Screen**.
3. Launch it from the home screen — it opens full-screen, no browser chrome, with the custom icon.

The manifest and `apple-mobile-web-app-*` meta tags are already set up, including safe-area insets for the notch and home indicator.

## Notes

- The model is ~3.6 MB and the WASM runtime loads once on first camera open, then comes from cache.
- If the camera is blocked, the "Use a photo instead" fallback runs the same analysis on a picture from the library.
- Face data is never stored; results live in memory for the session.

## Stack

Next.js 15 (App Router) · React 19 · Tailwind CSS 4 · MediaPipe Tasks Vision
