# MOG RATE

A mobile face scan that rates how hard you mog. Rotate through the guided views, end facing the camera, and get a rating out of 10 with an itemised breakdown of what is helping and what is costing you.

Built to be added to the iOS home screen and to look like a native app on a 375×667 screen.

## How the scan works

1. The preview runs MediaPipe Face Landmarker (468 mesh points plus iris centres) about 14 times a second and draws a green lock box around your face. A live head-angle readout tells you how far you have turned.
2. The guide asks for five poses (straight, rotate left, rotate right, chin up, chin down). Each one advances only when the pose is actually reached, with an optional Skip. The last step brings you back to the camera.
3. The final step samples **nine frames** facing forward, analysed at 800 px wide. Each frame is measured independently and the **median** of every metric is used, so a blink or one bad tracking tick cannot decide the score.
4. Landmarks are pose normalised before measuring: the eye line is levelled (roll), then the head turn (yaw) is solved from landmark depth and undone. Vertical spans use 3D distance so a tilted head does not read as a long chin.
5. **Symmetry is a real flip.** The face is warped into a canonical frame (roll, scale and head turn removed), its two halves are compared as edge maps with the mirror axis fitted from the data, and the difference is reported. A face mirrored onto itself measures 0.18; a normal face sits near 0.45; a clearly uneven one is 0.7 and up. Landmark geometry only gets 40% of the weight, because the landmark model has its own symmetry prior and barely notices a narrower cheek.
6. **Skin is measured from pixels**, in chroma rather than brightness: `(r − (g+b)/2) / (r+g+b)`, so a darker room does not read as a calmer face. Two scales catch both papules and broad inflammation, flagged pixels are grouped into blobs, the threshold adapts to the image's own grain (estimated with a median, because a mean gets dragged up by the very spots it should find), and coverage, blob count and sub-threshold redness all move the score.
7. **Under-eye darkness** compares the band under each lid with the skin at the same height beside it, taking the brighter reference patches so a shadowed cheek or a bit of hair cannot invent a dark circle.
8. Everything runs on the device. No API keys, no accounts, no uploads, no server cost, nothing to configure on Vercel.

The result includes a **face map**: the captured frame with the midline, the mirrored landmark pairs that sit furthest off it, every counted spot and both under-eye bands drawn over the photo. Tap it to inspect the photo at full size.

## What gets measured

| Metric | What it looks at |
| --- | --- |
| Symmetry | A flip test on edge maps with the mirror axis fitted from the data (60%), corroborated by mirrored landmark pairs including eyelids, brows and nose tip (40%) |
| Jawline | Bigonial width relative to cheekbones, plus the gonial angle |
| Skin | Chroma spot coverage and count, sub-threshold inflammation, surface texture, under-eye darkness against a matched reference |
| Face map | Midline, pair offsets, spot centres and under-eye bands drawn over the captured frame |
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
- Upload a photo: the button on the home screen (and the library button in the camera) runs the same measurement on any image you pick, at the same analysis width. It is one read rather than nine, so the confidence is lower and the header says Photo. HEIC from an iPhone library is decoded by Safari; very large images are downscaled before measuring, never upscaled.
- Skin readings are less reliable in poor light, and the result says so when the capture quality is low.
- Nothing scrolls on the camera screen. The shutter sits above the safe area on every phone down to a 375×667 SE, using `100dvh` rather than `inset-0`, which iOS otherwise stretches under the browser bar.
- Nothing is stored. Results live in memory for the session.

## Stack

Next.js 15 (App Router) · React 19 · Tailwind CSS 4 · MediaPipe Tasks Vision · Phosphor Icons
