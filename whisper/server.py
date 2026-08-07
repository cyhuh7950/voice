"""
whisper STT 서버 — faster-whisper(CTranslate2) 백엔드.

이 파일이 하는 일은 모델 로딩과 transcribe() 구현뿐이다.
HTTP 규격은 _common/voiceapi.py 가 담당한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 컨테이너에서는 voiceapi.py 가 /app 에 복사되고, 호스트 venv 로 직접 띄울 때는
# ../_common 에서 가져온다. 두 경우 모두 같은 파일이 동작하도록 경로를 넣어준다.
_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import numpy as np  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

from voiceapi import STT_SAMPLE_RATE, EngineSpec, create_app, run  # noqa: E402

MODEL = os.getenv("WHISPER_MODEL", "base")
DEVICE = os.getenv("DEVICE", "cpu")
# int8 은 aarch64 CPU 에서 메모리와 속도 모두 유리하다. 정확도를 올리려면 int8_float32.
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.getenv("CPU_THREADS", "0"))  # 0 이면 CTranslate2 가 알아서
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "1"))      # 1 = greedy, 지연을 크게 줄인다
VAD_FILTER = os.getenv("VAD_FILTER", "1") not in ("0", "false", "no")
DOWNLOAD_ROOT = os.getenv("DOWNLOAD_ROOT") or None

_model: WhisperModel | None = None


def load() -> None:
    global _model
    _model = WhisperModel(
        MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
        download_root=DOWNLOAD_ROOT,
    )


def transcribe(
    audio: np.ndarray,
    *,
    language: str | None,
    prompt: str | None,
    temperature: float,
    task: str,
) -> dict:
    assert _model is not None
    segments, info = _model.transcribe(
        audio,
        language=language,
        task=task,                      # transcribe | translate(→영어)
        beam_size=BEAM_SIZE,
        initial_prompt=prompt,
        temperature=temperature,
        vad_filter=VAD_FILTER,
        condition_on_previous_text=False,
    )

    texts: list[str] = []
    segs: list[dict] = []
    for s in segments:                  # 제너레이터이므로 여기서 실제 추론이 돈다
        texts.append(s.text)
        segs.append(
            {
                "id": s.id,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
            }
        )

    return {
        "text": "".join(texts),
        "language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "segments": segs,
    }


spec = EngineSpec(
    name="whisper",
    kind="stt",
    model=MODEL,
    port=int(os.getenv("PORT", "8101")),
    languages=[],  # 99개 언어 자동 감지
    sample_rate=STT_SAMPLE_RATE,
    extra={
        "backend": "faster-whisper (CTranslate2)",
        "supports_translate": True,  # /v1/audio/translations 로 영어 번역 가능
        "compute_type": COMPUTE_TYPE,
        "beam_size": BEAM_SIZE,
        "vad_filter": VAD_FILTER,
        "license": "MIT (모델: OpenAI Whisper, MIT)",
    },
)

app = create_app(spec, loader=load, transcribe=transcribe)

if __name__ == "__main__":
    run(app, spec)
