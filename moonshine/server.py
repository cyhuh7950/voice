"""
moonshine STT 서버 — moonshine-voice(내장 ONNX) 백엔드.

whisper 와 달리 모델이 언어별로 따로 있다. 그래서 컨테이너 하나가 한 언어를 담당하고,
쓰려는 언어는 engine.env 의 MOONSHINE_LANGUAGE 로 정한다.
지연이 매우 낮은 대신(측정 RTF 약 0.05~0.18) 한국어는 tiny 모델만 있어 정확도가 낮다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import moonshine_voice as mv  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from moonshine_voice import Transcriber  # noqa: E402

from voiceapi import STT_SAMPLE_RATE, EngineSpec, create_app, run  # noqa: E402

LANGUAGE = os.getenv("MOONSHINE_LANGUAGE", "ko").lower()
# 빈 값이면 해당 언어의 기본 아키텍처. tiny / base / tiny_streaming / ... 로 강제할 수 있다.
# 영어 기본값은 medium_streaming(245M) 이라 무겁다. 가볍게 쓰려면 tiny 로 고정할 것.
ARCH = os.getenv("MOONSHINE_ARCH", "").strip()

_tr: Transcriber | None = None


def load() -> None:
    global _tr
    if LANGUAGE not in mv.supported_languages():
        raise RuntimeError(
            f"moonshine does not support language: {LANGUAGE} "
            f"(supported: {', '.join(mv.supported_languages())})"
        )
    arch = mv.string_to_model_arch(ARCH) if ARCH else None
    # 캐시에 없으면 이 호출이 다운로드까지 처리하고 모델 경로를 돌려준다.
    path, resolved = mv.get_model_for_language(LANGUAGE, arch)
    _tr = Transcriber(model_path=str(path), model_arch=resolved)

    spec.model = f"{LANGUAGE}/{resolved.name.lower()}"
    spec.extra["model_path"] = str(path)


def transcribe(
    audio: np.ndarray,
    *,
    language: str | None,
    prompt: str | None,
    temperature: float,
    task: str,
) -> dict:
    assert _tr is not None
    if language and language.lower() != LANGUAGE:
        raise HTTPException(
            400,
            f"This container is dedicated to '{LANGUAGE}'. To use '{language}', "
            f"change MOONSHINE_LANGUAGE in engine.env and restart, "
            f"or run a separate container for that language",
        )

    # moonshine 은 파이썬 float 리스트를 받는다 (numpy 배열이 아님).
    t = _tr.transcribe_without_streaming(audio.astype(np.float32).tolist(), STT_SAMPLE_RATE)

    segs: list[dict] = []
    parts: list[str] = []
    for ln in t.lines:
        txt = (ln.text or "").strip()
        if not txt:
            continue
        parts.append(txt)
        segs.append(
            {
                "start": round(float(ln.start_time), 3),
                "end": round(float(ln.start_time) + float(ln.duration), 3),
                "text": txt,
            }
        )

    return {"text": " ".join(parts), "language": LANGUAGE, "segments": segs}


spec = EngineSpec(
    name="moonshine",
    kind="stt",
    model=f"{LANGUAGE}/pending",
    port=int(os.getenv("PORT", "8102")),
    languages=[LANGUAGE],
    sample_rate=STT_SAMPLE_RATE,
    extra={
        "backend": "moonshine-voice (ONNX)",
        "supports_translate": False,  # 음성→영어 번역 기능 없음
        "available_languages": list(mv.supported_languages()),
        "note": "One container per language. For multiple languages, run one container per language",
        "license": "Code MIT / model Moonshine Community License (non-commercial) — check separately for commercial use",
    },
)

app = create_app(spec, loader=load, transcribe=transcribe)

if __name__ == "__main__":
    run(app, spec)
