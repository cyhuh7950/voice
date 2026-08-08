"""
supertonic TTS 서버 — Supertone supertonic(ONNX) 백엔드.

31개 언어를 한 모델로 처리하고 프리셋 보이스가 10종(M1~M5, F1~F5)이다.
이 서버(ARM 4코어)에서 한국어 기준 RTF 약 1.9 로 측정됨.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import numpy as np  # noqa: E402
import supertonic as st  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from supertonic import TTS  # noqa: E402

from voiceapi import EngineSpec, create_app, run  # noqa: E402

MODEL = os.getenv("SUPERTONIC_MODEL", st.DEFAULT_MODEL)
MODEL_DIR = os.getenv("SUPERTONIC_MODEL_DIR", "/models")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "F1")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ko")
# 5(빠름) ~ 12(품질). 기본 8.
TOTAL_STEPS = int(os.getenv("TOTAL_STEPS", "8"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "44100"))
_intra = int(os.getenv("INTRA_OP_THREADS", "0"))
_inter = int(os.getenv("INTER_OP_THREADS", "0"))

_tts: TTS | None = None
_styles: dict[str, object] = {}


def _voice_names() -> list[str]:
    """모델 폴더의 voice_styles/*.json 이 곧 사용 가능한 보이스 목록."""
    root = Path(MODEL_DIR)
    names = {p.stem for p in root.rglob("voice_styles/*.json")}
    return sorted(names)


def load() -> None:
    global _tts
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    _tts = TTS(
        model=MODEL,
        model_dir=MODEL_DIR,
        auto_download=True,
        intra_op_num_threads=_intra or None,
        inter_op_num_threads=_inter or None,
    )
    # 보이스 스타일은 재사용되므로 기동 시 한 번만 읽어 캐시한다.
    for name in _voice_names():
        try:
            _styles[name] = _tts.get_voice_style(name)
        except Exception:  # 손상된 스타일 파일은 건너뛴다
            continue
    if not _styles:
        raise RuntimeError(f"No voice styles found: {MODEL_DIR}")
    spec.extra["voices"] = sorted(_styles)


def voices() -> list[dict]:
    out = []
    for name in sorted(_styles):
        out.append(
            {
                "id": name,
                "name": name,
                # M1~M5 는 남성, F1~F5 는 여성 프리셋
                "gender": {"M": "male", "F": "female"}.get(name[:1].upper(), "unknown"),
                "language": "multilingual",
            }
        )
    return out


def synthesize(
    text: str,
    *,
    voice: str | None,
    language: str | None,
    speed: float,
) -> tuple[np.ndarray, int]:
    assert _tts is not None
    name = voice or DEFAULT_VOICE
    if name not in _styles:
        raise HTTPException(400, f"Unknown voice: {name} (available: {', '.join(sorted(_styles))})")

    lang = (language or DEFAULT_LANGUAGE).lower()
    if lang not in st.SUPPORTED_LANGUAGES and lang != st.UNKNOWN_LANGUAGE:
        raise HTTPException(
            400,
            f"Unsupported language: {lang} "
            f"(supported: {', '.join(st.SUPPORTED_LANGUAGES)}, auto-detect is '{st.UNKNOWN_LANGUAGE}')",
        )

    wav, _dur = _tts.synthesize(
        text=text,
        voice_style=_styles[name],
        lang=lang,
        total_steps=TOTAL_STEPS,
        speed=speed,
    )
    return np.asarray(wav, dtype=np.float32).reshape(-1), SAMPLE_RATE


spec = EngineSpec(
    name="supertonic",
    kind="tts",
    model=MODEL,
    port=int(os.getenv("PORT", "8201")),
    languages=list(st.SUPPORTED_LANGUAGES),
    sample_rate=SAMPLE_RATE,
    extra={
        "backend": "supertonic (ONNX Runtime)",
        "default_voice": DEFAULT_VOICE,
        "default_language": DEFAULT_LANGUAGE,
        "total_steps": TOTAL_STEPS,
        "license": "Supertone supertonic — check upstream repo for distribution terms",
    },
)

app = create_app(spec, loader=load, synthesize=synthesize, voices=voices)

if __name__ == "__main__":
    run(app, spec)
