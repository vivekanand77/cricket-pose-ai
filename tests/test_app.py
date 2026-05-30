import io
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app as app_module


def create_video(path: Path, duration_seconds: int, fps: int = 10) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 64))
    for _ in range(duration_seconds * fps):
        frame = np.full((64, 64, 3), 127, dtype=np.uint8)
        writer.write(frame)
    writer.release()


client = TestClient(app_module.app)


class DummyAnalyzer:
    def __init__(self, duration: float, angles=None):
        self._duration = duration
        self._angles = angles or []

    def extract_duration(self, _path):
        return self._duration

    def analyze(self, _path):
        return self._angles


class DummyAngle:
    def __init__(self, name, value, ideal_range, correction_image_base64):
        self.name = name
        self.value = value
        self.ideal_range = ideal_range
        self.correction_image_base64 = correction_image_base64


def test_rejects_video_longer_than_10_seconds(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "analyzer", DummyAnalyzer(duration=11.0))
    monkeypatch.setattr(app_module, "build_feedback", lambda _angles: "feedback")

    video_path = tmp_path / "long.mp4"
    create_video(video_path, duration_seconds=1)

    with video_path.open("rb") as f:
        response = client.post(
            "/analyze-bowling",
            files={"video": ("long.mp4", f, "video/mp4")},
        )

    assert response.status_code == 400
    assert "10 seconds or shorter" in response.json()["detail"]


def test_rejects_unsupported_video_extension(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "analyzer", DummyAnalyzer(duration=8.0))
    monkeypatch.setattr(app_module, "build_feedback", lambda _angles: "feedback")

    fake_path = tmp_path / "bad.exe"
    fake_path.write_bytes(b"not-a-video")

    with fake_path.open("rb") as f:
        response = client.post(
            "/analyze-bowling",
            files={"video": ("bad.exe", f, "application/octet-stream")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported video format"


def test_returns_angle_analysis_and_feedback(monkeypatch, tmp_path):
    angles = [
        DummyAngle("elbow", 160.2, (145.0, 175.0), "img-a"),
        DummyAngle("shoulder", 100.0, (80.0, 120.0), "img-b"),
        DummyAngle("knee", 165.0, (150.0, 178.0), "img-c"),
    ]
    monkeypatch.setattr(app_module, "analyzer", DummyAnalyzer(duration=8.0, angles=angles))
    monkeypatch.setattr(app_module, "build_feedback", lambda _angles: "Keep chest upright.")

    video_path = tmp_path / "ok.mp4"
    create_video(video_path, duration_seconds=1)

    with video_path.open("rb") as f:
        response = client.post(
            "/analyze-bowling",
            files={"video": ("ok.mp4", f, "video/mp4")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["coaching_feedback"] == "Keep chest upright."
    assert len(body["joint_angles"]) == 3
    assert body["joint_angles"][0]["name"] == "elbow"
    assert body["joint_angles"][0]["correction_image_base64"] == "img-a"
