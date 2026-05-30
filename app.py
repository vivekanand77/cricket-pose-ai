import base64
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

MAX_VIDEO_SECONDS = 10
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

LANDMARKS = {
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
    "right_hip": 24,
    "right_knee": 26,
    "right_ankle": 28,
}


@dataclass
class JointAngle:
    name: str
    value: float
    ideal_range: tuple[float, float]
    correction_image_base64: str


class AnalyzeResponse(BaseModel):
    joint_angles: list[dict[str, Any]]
    coaching_feedback: str


class BowlingAnalyzer:
    def __init__(self) -> None:
        self.pose = None
        if mp:
            self.pose = mp.solutions.pose.Pose(static_image_mode=False)

    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return None
        cosine_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine_angle)))

    def extract_duration(self, path: Path) -> float:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps <= 0:
            return 0.0
        return float(frames / fps)

    def _encode_highlight_frame(
        self,
        frame: np.ndarray,
        points: list[tuple[int, int]],
    ) -> str:
        overlay = frame.copy()
        for x, y in points:
            cv2.circle(overlay, (x, y), 8, (0, 0, 255), -1)
        ok, buffer = cv2.imencode(".jpg", overlay)
        if not ok:
            return ""
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _extract_points(self, frame: np.ndarray) -> dict[str, list[tuple[int, int]]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if not results.pose_landmarks:
            raise HTTPException(status_code=400, detail="Could not detect body pose in video")

        h, w = frame.shape[:2]
        lm = results.pose_landmarks.landmark

        def point(name: str) -> tuple[int, int]:
            idx = LANDMARKS[name]
            return int(lm[idx].x * w), int(lm[idx].y * h)

        shoulder = point("right_shoulder")
        elbow = point("right_elbow")
        wrist = point("right_wrist")
        hip = point("right_hip")
        knee = point("right_knee")
        ankle = point("right_ankle")

        return {
            "elbow": [elbow, shoulder, wrist],
            "shoulder": [shoulder, elbow, hip],
            "knee": [knee, hip, ankle],
        }

    def analyze(self, path: Path) -> list[JointAngle]:
        if self.pose is None:
            raise HTTPException(status_code=500, detail="MediaPipe is not installed")

        cap = cv2.VideoCapture(str(path))
        frame_angles: list[dict[str, Any]] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            if not results.pose_landmarks:
                continue

            h, w = frame.shape[:2]
            lm = results.pose_landmarks.landmark

            def point(name: str) -> tuple[int, int]:
                idx = LANDMARKS[name]
                return int(lm[idx].x * w), int(lm[idx].y * h)

            shoulder = np.array(point("right_shoulder"), dtype=float)
            elbow = np.array(point("right_elbow"), dtype=float)
            wrist = np.array(point("right_wrist"), dtype=float)
            hip = np.array(point("right_hip"), dtype=float)
            knee = np.array(point("right_knee"), dtype=float)
            ankle = np.array(point("right_ankle"), dtype=float)

            elbow_angle = self._calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = self._calculate_angle(elbow, shoulder, hip)
            knee_angle = self._calculate_angle(hip, knee, ankle)
            if elbow_angle is None or shoulder_angle is None or knee_angle is None:
                continue

            frame_angles.append(
                {
                    "elbow": elbow_angle,
                    "shoulder": shoulder_angle,
                    "knee": knee_angle,
                    "frame": frame,
                }
            )

        cap.release()

        if not frame_angles:
            raise HTTPException(status_code=400, detail="Could not detect body pose in video")

        avg = {
            "elbow": float(np.mean([x["elbow"] for x in frame_angles])),
            "shoulder": float(np.mean([x["shoulder"] for x in frame_angles])),
            "knee": float(np.mean([x["knee"] for x in frame_angles])),
        }

        ideals = {
            "elbow": (145.0, 175.0),
            "shoulder": (80.0, 120.0),
            "knee": (150.0, 178.0),
        }

        representative = frame_angles[len(frame_angles) // 2]
        representative_points = self._extract_points(representative["frame"])

        output = []
        for joint in ("elbow", "shoulder", "knee"):
            output.append(
                JointAngle(
                    name=joint,
                    value=round(avg[joint], 2),
                    ideal_range=ideals[joint],
                    correction_image_base64=self._encode_highlight_frame(
                        representative["frame"], representative_points[joint]
                    ),
                )
            )

        return output


def build_feedback(angles: list[JointAngle]) -> str:
    prompt = (
        "You are a cricket bowling coach. Give concise correction tips based on these angle stats: "
        + "; ".join(
            [
                f"{a.name}: {a.value} (ideal {a.ideal_range[0]}-{a.ideal_range[1]})"
                for a in angles
            ]
        )
    )

    api_key = os.getenv("GOOGLE_API_KEY")
    if genai and api_key:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if text:
            return text

    tips = []
    for angle in angles:
        lo, hi = angle.ideal_range
        if angle.value < lo:
            tips.append(
                f"Your {angle.name} angle is {angle.value}°, below the ideal {lo}-{hi}°. "
                f"Focus on extending through release to increase this angle."
            )
        elif angle.value > hi:
            tips.append(
                f"Your {angle.name} angle is {angle.value}°, above the ideal {lo}-{hi}°. "
                f"Reduce extension to around {hi}° to lower injury risk."
            )
        else:
            tips.append(f"{angle.name.capitalize()} angle looks solid; maintain this movement pattern.")
    return " ".join(tips)


app = FastAPI(title="Cricket Pose AI")
analyzer = BowlingAnalyzer()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze-bowling", response_model=AnalyzeResponse)
async def analyze_bowling(video: UploadFile = File(...)) -> AnalyzeResponse:
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    suffix = suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = Path(tmp.name)

    try:
        duration = analyzer.extract_duration(tmp_path)
        if duration <= 0:
            raise HTTPException(status_code=400, detail="Invalid or unreadable video")
        if duration > MAX_VIDEO_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Video must be {MAX_VIDEO_SECONDS} seconds or shorter",
            )

        joint_angles = analyzer.analyze(tmp_path)
        feedback = build_feedback(joint_angles)

        return AnalyzeResponse(
            joint_angles=[
                {
                    "name": x.name,
                    "value": x.value,
                    "ideal_range": x.ideal_range,
                    "correction_image_base64": x.correction_image_base64,
                }
                for x in joint_angles
            ],
            coaching_feedback=feedback,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
