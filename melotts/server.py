"""
melotts TTS 서버 — MeloTTS 백엔드.

moonshine 처럼 언어별로 체크포인트가 나뉘어 있어 컨테이너 하나가 한 언어를 담당한다.
언어는 engine.env 의 MELO_LANGUAGE 로 지정한다 (EN / ES / FR / ZH / JP / KR).

성능 주의: 이 서버(ARM 4코어, CPU)에서 한국어 기준 RTF 약 6 으로 측정됐다.
즉 3초 문장 합성에 20초 가까이 걸린다. 실시간 대화용으로는 supertonic 쪽이 맞고,
melotts 는 품질 비교나 배치 합성용으로 두는 것을 권한다.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import numpy as np  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from voiceapi import EngineSpec, create_app, run  # noqa: E402

# MeloTTS 는 오래된 API 를 써서 경고를 많이 뿜는다. 로그를 읽을 수 있게 눌러둔다.
warnings.filterwarnings("ignore")

LANGUAGE = os.getenv("MELO_LANGUAGE", "KR").upper()
DEVICE = os.getenv("DEVICE", "cpu")
SDP_RATIO = float(os.getenv("SDP_RATIO", "0.2"))
NOISE_SCALE = float(os.getenv("NOISE_SCALE", "0.6"))
NOISE_SCALE_W = float(os.getenv("NOISE_SCALE_W", "0.8"))

SUPPORTED = ["EN", "ES", "FR", "ZH", "JP", "KR"]

_model = None
_speakers: dict[str, int] = {}
_sample_rate = 44100


def load() -> None:
    global _model, _speakers, _sample_rate
    if LANGUAGE not in SUPPORTED:
        raise RuntimeError(f"MeloTTS 가 지원하지 않는 언어: {LANGUAGE} (가능: {', '.join(SUPPORTED)})")

    from melo.api import TTS  # import 자체가 무거워서 로딩 시점까지 미룬다

    _model = TTS(language=LANGUAGE, device=DEVICE)
    _speakers = {str(k): int(v) for k, v in _model.hps.data.spk2id.items()}
    _sample_rate = int(_model.hps.data.sampling_rate)

    spec.sample_rate = _sample_rate
    spec.extra["speakers"] = sorted(_speakers)


def voices() -> list[dict]:
    return [
        {"id": name, "name": name, "language": LANGUAGE, "gender": "unknown"}
        for name in sorted(_speakers)
    ]


def synthesize(
    text: str,
    *,
    voice: str | None,
    language: str | None,
    speed: float,
) -> tuple[np.ndarray, int]:
    assert _model is not None
    if language and language.upper() != LANGUAGE:
        raise HTTPException(
            400,
            f"이 컨테이너는 '{LANGUAGE}' 전용입니다. '{language}' 를 쓰려면 "
            f"engine.env 의 MELO_LANGUAGE 를 바꾸고 재시작하거나 언어별 컨테이너를 따로 띄우세요",
        )

    name = voice or next(iter(sorted(_speakers)))
    if name not in _speakers:
        raise HTTPException(400, f"없는 화자: {name} (가능: {', '.join(sorted(_speakers))})")

    # output_path=None 이면 파일을 쓰지 않고 오디오 배열을 그대로 돌려준다.
    audio = _model.tts_to_file(
        text,
        _speakers[name],
        output_path=None,
        sdp_ratio=SDP_RATIO,
        noise_scale=NOISE_SCALE,
        noise_scale_w=NOISE_SCALE_W,
        speed=speed,
        quiet=True,
    )
    return np.asarray(audio, dtype=np.float32).reshape(-1), _sample_rate


spec = EngineSpec(
    name="melotts",
    kind="tts",
    model=f"MeloTTS-{LANGUAGE}",
    port=int(os.getenv("PORT", "8202")),
    languages=[LANGUAGE],
    sample_rate=_sample_rate,
    extra={
        "backend": "MeloTTS (PyTorch CPU)",
        "available_languages": SUPPORTED,
        "note": "컨테이너 1개 = 언어 1개. ARM CPU 에서 RTF 약 6 이므로 실시간 용도로는 느리다",
        "license": "MIT (MeloTTS)",
    },
)

app = create_app(spec, loader=load, synthesize=synthesize, voices=voices)

if __name__ == "__main__":
    run(app, spec)
