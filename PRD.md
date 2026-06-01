# Cricket Pose AI — Product Requirements Document (v1)

## 1. Purpose

A local, free, privacy-first desktop tool that lets an amateur cricketer upload a short video of their **fast-bowling** action and receive:

1. A text report flagging biomechanical issues with coaching corrections.
2. A dashboard of directly-measured biomechanical numbers.
3. A locally-stored history of past sessions (metrics + conclusion only).

The long-term ambition is to also cover spin bowling, batting, pro side-by-side comparison, and ML-predicted ball speed / grip efficiency / foot pressure — but v1 ships the smallest useful slice and **logs raw data** so the predictive models can be trained later.

---

## 2. Target user

A self-coaching amateur pace bowler with a phone and a laptop. They film themselves in the nets / park, upload the clip, and want feedback without paying a coach or a subscription.

---

## 3. Out of scope for v1

| Feature | Why deferred |
|---|---|
| Spin bowling analyzer | Different rule set; ship pace first. |
| Batting analyzer | Different rule set; ship pace first. |
| Chucking detection (ICC 15° elbow flex) | Needs 120–240 fps slow-mo. Standard 30 fps phone video gives only ~3 frames during the delivery swing — not enough. |
| Wrist / seam / grip orientation | MediaPipe doesn't track the ball. Needs a separate object-detection model. |
| Pro side-by-side comparison | Needs a curated reference library. v2. |
| Predicted ball speed / grip efficiency / foot pressure | Need calibration data (radar-gun-labeled clips) or sensors. v1 logs the raw inputs so a regression can be trained later. |
| Cloud, accounts, subscriptions | Strictly local-only by user requirement. |
| Real-time webcam mode | v1 is upload-only. (The existing `pose_test.py` becomes an optional dev tool.) |

---

## 4. Functional requirements

### 4.1 Input

- User uploads a single video file.
- Constraints: `.mp4` or `.mov`, ≤ 10 seconds, ≤ 30 MB.
- App rejects files outside these limits with a clear message.
- App **auto-detects the delivery frame** in the clip via wrist-velocity peak (transparent heuristic; user can override by scrubbing if v1 scope allows — otherwise just expose the detected frame number).

### 4.2 Pose pipeline

- MediaPipe Pose (BlazePose) runs over every frame of the uploaded clip.
- Camera angle is auto-classified: **side-on / front-on / back-on**. Some rules only fire from compatible angles.
- The full per-frame landmark stream is held in memory for the duration of the analysis, then discarded (not persisted).

### 4.3 Rules engine — v1 checks

Each rule emits: `{ ruleId, severity, measuredValue, threshold, status, plainEnglishMessage, frameRef }`. Rules fire only if the camera angle supports them.

| # | Rule | What's measured | Compatible angles |
|---|---|---|---|
| R1 | Front-arm angle at release | Angle of front shoulder–elbow segment vs. vertical at the release frame | side-on |
| R2 | Front-knee flexion at landing | Knee joint angle (hip-knee-ankle) at front-foot contact | side-on |
| R3 | Back-foot ankle dorsiflexion at landing | Ankle-foot-index angle at back-foot contact (heel-first vs flat) | side-on |
| R4 | Knee-over-toe alignment at landing | Lateral offset of knee from foot midline at front-foot landing | front-on |
| R5 | Distance from popping crease at landing | Best-effort foot-x-position estimate; **stretch goal** — needs crease line detection (Hough/OpenCV). Falls back to "n/a" if line undetected. | front-on / side-on |

### 4.4 Metrics dashboard — v1 (directly measured only)

| Metric | Source |
|---|---|
| Front-arm angle at release (°) | R1 measurement |
| Front-knee flexion at landing & release (°, °) | R2 measurement, plus release-frame angle |
| Hip-shoulder separation at back-foot landing (°) | Derived from hip-line vs shoulder-line orientation |
| Run-up cadence (steps/sec) | Foot-landmark vertical-velocity zero-crossings |
| Detected delivery frame number + camera angle | Diagnostic |

> **No predicted ball speed / grip / foot-pressure numbers in v1.** Those land in v2 once training data accumulates.

### 4.5 Report

Generated from rule outcomes + metrics. Format:
- **Header:** detected camera angle, delivery frame, duration analyzed.
- **Per-rule findings:** ✅ OK / ⚠️ flag, with plain-English explanation and what to try.
- **Metrics table:** every dashboard metric with its numeric value.
- **Conclusion paragraph:** 2–4 sentences synthesizing the top issue + next focus.
- Downloadable as JSON (machine-readable) for v2 training data and as a printable HTML view.

### 4.6 History

- SQLite database in app data dir.
- Schema: `sessions(id, created_at, camera_angle, metrics_json, rule_findings_json, conclusion_text)`.
- **Not stored:** uploaded videos, per-frame landmarks, derived intermediate arrays.
- UI: a "History" page lists sessions newest-first with the conclusion line; clicking opens the saved metrics + findings.

---

## 5. Non-functional requirements

- **Local-only.** No outbound network calls in v1. No external APIs.
- **Privacy.** Videos are processed in a temp dir and deleted after the report renders.
- **Performance target.** A 10s, 30fps clip analyzed in ≤ 20 s on a typical laptop CPU. No GPU required.
- **Cost.** Zero — no subscriptions, no API bills.
- **OS support.** Windows-first (user's environment). macOS/Linux as a side effect of being pure Python + a Vite SPA.

---

## 6. Architecture

```
cp-ai/
├── backend/                  # FastAPI + MediaPipe + OpenCV
│   ├── app/
│   │   ├── main.py           # FastAPI entry
│   │   ├── routes/
│   │   │   ├── analyze.py    # POST /api/analyze (upload + run pipeline)
│   │   │   └── history.py    # GET /api/sessions, GET /api/sessions/{id}
│   │   ├── pipeline/
│   │   │   ├── pose.py       # MediaPipe wrapper
│   │   │   ├── delivery.py   # Auto-detect release frame
│   │   │   ├── angle_view.py # Camera-angle classifier
│   │   │   ├── rules/        # One module per rule (R1..R5)
│   │   │   └── metrics.py    # Dashboard numbers
│   │   ├── storage/
│   │   │   └── sqlite.py     # session persistence
│   │   └── report.py         # Compose findings + conclusion
│   └── requirements.txt
├── frontend/                 # React + Vite + TypeScript
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Upload.tsx
│   │   │   ├── Report.tsx
│   │   │   └── History.tsx
│   │   ├── components/
│   │   │   ├── MetricsTable.tsx
│   │   │   ├── RuleFinding.tsx
│   │   │   └── VideoConstraints.tsx
│   │   └── api/client.ts
│   └── package.json
├── pose_test.py              # Existing webcam toy — kept as a dev sanity check
├── PRD.md                    # This document
└── README.md
```

API surface (v1):
- `POST /api/analyze` — multipart upload, returns `{ sessionId, report }`.
- `GET  /api/sessions` — list rows from SQLite.
- `GET  /api/sessions/{id}` — single session detail.

Backend deletes the uploaded video immediately after the response is rendered.

---

## 7. v1 milestones

1. **M1 — Backend pipeline skeleton.** Upload endpoint, MediaPipe pose extraction over a fixed test clip, camera-angle classification, delivery-frame detection. Return raw landmark summary.
2. **M2 — Rules R1, R2, R3.** Side-on rules and their metrics. Side-on report renders end-to-end.
3. **M3 — Rules R4, R5.** Front-on rules. Crease detection as stretch.
4. **M4 — Frontend.** React+Vite app with Upload / Report / History pages.
5. **M5 — SQLite history + JSON export.** Persist sessions; download JSON.
6. **M6 — Polish.** Error messages, constraints enforcement, perf check on a real phone clip.

---

## 8. Open questions still to confirm

These I'd like to nail down right before M1 — they don't block the PRD but they'll surface as we build:

- **Crease detection (R5):** acceptable as "n/a" when not detected, or is it a must-have for v1?
- **Manual override of detected release frame:** does the user want a slider in v1, or trust the heuristic?
- **Report download:** JSON is enough for v1, or also a printable PDF?
- **Theming:** any visual direction, or pick a clean default (Tailwind + neutral palette)?

---

*Document drafted 2026-06-01 from the requirements conversation. Awaiting user confirmation before implementation starts.*
