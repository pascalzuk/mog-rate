# MOG RATE

A mobile face scan that rates how hard you mog. Rotate through the guided views, end facing the camera, and get a rating out of 10 with an itemised breakdown of what is helping and what is costing you.

Built to be added to the iOS home screen and to look like a native app on a 375×667 screen.

## How the scan works

1. The preview runs MediaPipe Face Landmarker (468 mesh points plus iris centres) about 14 times a second and draws a green lock box around your face. A live head-angle readout tells you how far you have turned.
2. The guide asks for five poses (straight, rotate left, rotate right, chin up, chin down). Each one advances only when the pose is actually reached, with an optional Skip. The last step brings you back to the camera.
3. The final step samples **six frames** facing forward. Each frame is measured independently and the **median** of every metric is used, so a blink or one bad tracking tick cannot decide the score.
4. Landmarks are pose normalised before measuring: the eye line is levelled (roll), then the head turn (yaw) is solved from landmark depth and undone. Vertical spans use 3D distance so a tilted head does not read as a long chin.
5. Skin is measured from **pixels**, not landmarks: the face oval is sampled, eyes/brows/lips/nostrils are excluded, and the local skin tone is subtracted to find clusters that are redder or darker than their neighbours. Spots are counted as blobs, and both coverage and count move the score.
6. Everything runs on the device. No API keys, no accounts, no uploads, no server cost, nothing to configure on Vercel.

## What gets measured

| Metric | What it looks at |
| --- | --- |
| Symmetry | Mirrored landmark pairs against a fitted mirror axis (eyelids, brows and nose tip included), plus a pixel flip that mirrors the face and diffs the halves |
| Jawline | Bigonial width relative to cheekbones, plus the gonial angle |
| Skin | Spot coverage and count, surface texture, under-eye darkness |
| Water retention | Lower-face width against face length (puffiness) |
| Proportions | Lower third against middle third |
| Midface | Interpupillary distance over eye-to-lip height |
| Canthal tilt | Angle of the outer eye corner relative to the inner one |
| Eye area | Aperture and intercanthal spacing |
| Face width | Facial width-to-height |
| Lips | Upper/lower vermilion ratio and mouth-to-nose width |

The overall score is a weighted composite on a deliberately harsh curve, with hard caps: a clearly asymmetric face or visible breakouts cannot average its way to a 9. Skin carries the heaviest cap, because a breakout is the first thing anyone sees. Every deduction is listed with the number that produced it, so a low score is explainable rather than mysterious.

This is a fixed geometry and pixel heuristic. It is not a dermatology model, not medical advice, and not science. It also cannot judge lighting, expression, or how you actually look in a room.

## Install on your phone

**iOS:** open the URL in **Safari**, tap Share, then **Add to Home Screen**. It must be Safari; Chrome and in-app browsers on iOS cannot install a standalone web app.

**Android:** Chrome shows a native install prompt, or use the menu and **Install app**.

There is no link that installs a web app on iOS in one tap. Apple does not expose that. The app detects this and shows the Share sheet instructions itself, which is the closest thing possible.

```bash
npm install
npm run dev      # http://localhost:3000 (camera needs localhost or HTTPS)
npm run build
npm start
```

## Deploy to Vercel

1. Push this repo to GitHub.
2. Vercel: **Add New → Project → Import** the repository.
3. Framework preset **Next.js**, no environment variables, **Deploy**.

The model (~3.6 MB) and the MediaPipe WASM runtime are served from `public/` with immutable cache headers, so the first camera open downloads them and every open after that is instant.

## Notes and limits

- Camera framing asks for a tall stream and caps the preview crop, so the preview is not blown up the way a square or 16:9 request looks on a portrait iPhone. If the browser still hands back a wide stream, the stage stays a portrait window rather than cropping to a sliver.
- Photo fallback: if camera permission is blocked, the same analysis runs on an image from the library.
- Skin readings are less reliable in poor light, and the result says so when the capture quality is low.
- Nothing scrolls on the camera screen. The shutter sits above the safe area on every phone down to a 375×667 SE, using `100dvh` rather than `inset-0`, which iOS otherwise stretches under the browser bar.
- Nothing is stored. Results live in memory for the session.

## Stack

Next.js 15 (App Router) · React 19 · Tailwind CSS 4 · MediaPipe Tasks Vision · Phosphor Icons
