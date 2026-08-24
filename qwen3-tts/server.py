"""
qwen3-tts (tts) 서버 — Qwen3-TTS-12Hz-0.6B, GPU(AMD ROCm/HIP).

CustomVoice 모델은 프리셋 음색 9종을 제공한다 (한국어는 Sohee).
Base 모델로 바꾸면 참조 음성으로 zero-shot 복제를 하는데, 이 엔진은 프리셋 경로만 쓴다
(복제가 필요하면 cosyvoice(8203) 를 쓰는 편이 낫다).

HTTP 규격은 _common/voiceapi.py 가 처리한다. 여기서는 모델 로딩과 합성만 구현한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ROCm(WSL) 필수 환경변수. 이미지 ENV 에도 있지만 호스트 venv 로 직접 돌릴 때를 위해 여기서도 세운다.
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import numpy as np  # noqa: E402

from voiceapi import EngineSpec, create_app, run  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
# 모델 캐시는 HF_HOME 으로만 지정한다 (이미지 ENV 가 /models). load() 의 NOTE 참고.
DEVICE = os.getenv("DEVICE", "cuda:0")      # ROCm 에서도 torch API 는 "cuda" 다 (HIP 매핑)
DTYPE = os.getenv("COMPUTE_TYPE", "bfloat16")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "Sohee")      # 한국어 여성
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ko")

# API 는 언어를 "Korean" 같은 영문 이름으로 받는다. OpenAI 호환 규격의 ISO 코드와 이어준다.
_LANGUAGES = {
    "ko": "Korean",
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

_model = None
_speakers: list[str] = []


def load() -> None:
    global _model, _speakers
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[DTYPE]

    # NOTE cache_dir 를 넘기면 안 된다 (qwen-tts 0.1.1 버그).
    #   modeling_qwen3_tts.from_pretrained 는 speech_tokenizer 를
    #     snapshot_download(cache_dir=kwargs.get("cache_dir", cache_dir))  ← 우리가 준 경로로 받고
    #     cached_file(...,  cache_dir=kwargs.pop("cache_dir", None))       ← 기본 HF 캐시에서 찾는다
    #   cache_dir 는 명명 파라미터라 kwargs 에 없으므로 두 경로가 갈라지고,
    #   결국 speech_tokenizer/preprocessor_config.json 을 못 찾아 기동이 죽는다.
    #   넘기지 않으면 양쪽 모두 HF_HOME(/models) 기준으로 통일된다.
    _model = Qwen3TTSModel.from_pretrained(
        MODEL,
        device_map=DEVICE,
        dtype=dtype,
        # flash_attention_2 는 쓰지 않는다. gfx1101(RDNA3) 용 flash-attn 휠이 없다.
        attn_implementation="sdpa",
    )
    try:
        _speakers = list(_model.get_supported_speakers())
    except Exception:
        _speakers = [DEFAULT_VOICE]


def synthesize(
    text: str,
    *,
    voice: str | None,
    language: str | None,
    speed: float,
) -> tuple[np.ndarray, int]:
    speaker = voice or DEFAULT_VOICE
    lang_code = language or DEFAULT_LANGUAGE
    lang = _LANGUAGES.get(lang_code, lang_code)     # 영문 이름을 그대로 줘도 통과시킨다

    wavs, sr = _model.generate_custom_voice(text=text, language=lang, speaker=speaker)

    # 반환 형태가 배치([1, N])일 수도, 리스트일 수도 있어 1차원으로 정리한다.
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    return wav, int(sr)


def voices() -> list[dict]:
    return [
        {"id": s, "name": s, "language": "-", "gender": "unknown"}
        for s in (_speakers or [DEFAULT_VOICE])
    ]


spec = EngineSpec(
    name="qwen3-tts",
    kind="tts",
    model=MODEL,
    port=int(os.getenv("PORT", "8205")),
    languages=list(_LANGUAGES),
    sample_rate=24000,      # 실제 값은 generate 가 돌려주는 sr 을 그대로 쓴다
    extra={
        "backend": "qwen-tts (Qwen3TTSModel, torch/ROCm)",
        "voice_mode": "프리셋 음색 (CustomVoice). 한국어 기본값 Sohee",
        "speed_supported": False,       # 이 모델은 속도 조절 파라미터가 없다
        "supports_translate": False,
        "license": "업스트림 저장소 확인 필요",
    },
)

app = create_app(spec, loader=load, synthesize=synthesize, voices=voices)

if __name__ == "__main__":
    run(app, spec)
