"""
__ENGINE__ (__KIND__) 서버 — 새 엔진 추가용 템플릿.

구현할 것은 아래 두세 개뿐이다. HTTP 규격(엔드포인트/오디오 변환/인증/health)은
_common/voiceapi.py 가 처리하므로 손대지 않는다.

  load()        모델을 메모리에 올린다. 기동 시 한 번 호출된다.
  transcribe()  kind=stt 인 경우. 16kHz mono float32 numpy → {"text": ...}
  synthesize()  kind=tts 인 경우. 텍스트 → (mono float32 numpy, sample_rate)
  voices()      kind=tts 인 경우 선택. 보이스 목록.
  embed()       kind=speaker 인 경우. 16kHz mono float32 numpy → 임베딩 배열

종류를 새로 만들려면 voiceapi.py 에 라우트 설치 함수를 쓰고 register_kind() 로 등록한다.
이 엔진에만 필요한 라우트는 create_app(routes=...) 로 여기서 직접 붙일 수 있다.

TODO 를 지우면서 채우면 된다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import numpy as np  # noqa: E402

from voiceapi import STT_SAMPLE_RATE, EngineSpec, create_app, run  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "TODO-모델명")

_model = None


def load() -> None:
    global _model
    # TODO: 모델 로딩. 예)
    #   from some_engine import Model
    #   _model = Model(MODEL, device="cpu")
    raise NotImplementedError("load() 를 구현하세요")


# ---- STT 엔진이면 이걸 구현 -------------------------------------------------


def transcribe(
    audio: np.ndarray,
    *,
    language: str | None,
    prompt: str | None,
    temperature: float,
    task: str,
) -> dict:
    """audio 는 16kHz mono float32 (-1.0~1.0)."""
    # TODO: 추론. segments 는 없으면 빈 리스트여도 된다.
    raise NotImplementedError("transcribe() 를 구현하세요")
    # return {"text": "...", "language": language, "segments": []}


# ---- TTS 엔진이면 이걸 구현 -------------------------------------------------


def synthesize(
    text: str,
    *,
    voice: str | None,
    language: str | None,
    speed: float,
) -> tuple[np.ndarray, int]:
    """반환은 (mono float32 numpy, sample_rate)."""
    # TODO: 합성
    raise NotImplementedError("synthesize() 를 구현하세요")
    # return wav, 24000


def voices() -> list[dict]:
    return [{"id": "default", "name": "default", "language": "-", "gender": "unknown"}]


# ---- 화자(speaker) 엔진이면 이걸 구현 ---------------------------------------


def embed(audio: np.ndarray) -> np.ndarray:
    """audio 는 16kHz mono float32 (-1.0~1.0). 반환은 고정 차원 임베딩 1차원 배열."""
    # TODO: 임베딩 추출. 내적이 코사인 유사도가 되도록 L2 정규화해서 돌려주는 편이 좋다.
    raise NotImplementedError("embed() 를 구현하세요")


spec = EngineSpec(
    name="__ENGINE__",
    kind="__KIND__",
    model=MODEL,
    port=int(os.getenv("PORT", "__PORT__")),
    languages=[],           # 예: ["ko", "en"] / 빈 리스트면 "auto"
    sample_rate=STT_SAMPLE_RATE,  # TTS 면 출력 샘플레이트로 바꾼다
    extra={
        "backend": "TODO",
        "supports_translate": False,  # STT 에서 음성→영어 번역을 지원하면 True
        "license": "TODO",
    },
)

# 자기 종류에 해당하는 것만 넘긴다. 쓰지 않는 인자와 함수는 지울 것.
#   stt → transcribe= / tts → synthesize=, voices= / speaker → embed=
app = create_app(
    spec,
    loader=load,
    transcribe=transcribe if spec.kind == "stt" else None,
    synthesize=synthesize if spec.kind == "tts" else None,
    voices=voices if spec.kind == "tts" else None,
    embed=embed if spec.kind == "speaker" else None,
)

if __name__ == "__main__":
    run(app, spec)
