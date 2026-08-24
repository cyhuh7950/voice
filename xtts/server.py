"""
xtts (tts) 서버 — XTTS-v2, GPU(AMD ROCm/HIP).

원본 coqui-ai/TTS 는 유지보수가 끊겼다. import 경로가 같은 유지보수 포크(coqui-tts,
idiap/coqui-ai-TTS)를 쓴다. 17개 언어 + 음성 복제를 지원한다.

라이선스 주의: 코드는 MPL 2.0 이지만 **가중치는 CPML(비상업)** 이고, Coqui Inc. 가 폐업해
상업 라이선스를 살 방법이 없다. 내부 개발용으로만 쓸 것.

HTTP 규격은 _common/voiceapi.py 가 처리한다. 여기서는 모델 로딩과 합성만 구현한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ROCm(WSL) 필수 환경변수. 이미지 ENV 에도 있지만 호스트 venv 로 직접 돌릴 때를 위해 여기서도 세운다.
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")
# XTTS-v2 가중치는 CPML 이라 라이브러리가 대화형으로 동의를 묻는다.
# 컨테이너에는 tty 가 없어 그대로 두면 기동이 멈춘다.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import numpy as np  # noqa: E402

from voiceapi import EngineSpec, create_app, run  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
DEVICE = os.getenv("DEVICE", "cuda")        # ROCm 에서도 torch API 는 "cuda" 다 (HIP 매핑)
VOICES_DIR = _here / "voices"
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ko")

# XTTS-v2 가 지원하는 17개 언어.
LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
             "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]

_tts = None
_builtin_speakers: list[str] = []
_sample_rate = 24000


def _patch_load_wav() -> None:
    """torchaudio.load 를 soundfile 기반으로 갈아끼운다.

    coqui-tts 는 오디오를 TTS/tts/models/xtts.py 의 load_audio() 로 읽고, 그게
    torchaudio.load 를 부른다. torchaudio 2.11 부터 이 함수는 torchcodec 에 위임되는데,
    PyPI 의 torchcodec 프리빌트 라이브러리는 CUDA 빌드라 ROCm 환경에서 로드되지 않는다:
        OSError: libnvrtc.so.13: cannot open shared object file
        OSError: Could not load this library: .../libtorchcodec_core4.so
    torchcodec 자체는 설치해둬야 한다 — TTS/__init__.py 가 is_torchcodec_available() 로
    존재 여부를 검사하고 없으면 import 단계에서 바로 죽는다. import 는 되고 .so 로딩만
    실패하는 상태라, 여기서 실제 디코딩 경로만 우회하면 된다.
    """
    import soundfile as sf
    import torch
    import torchaudio

    if getattr(torchaudio.load, "_patched_by_voiceapi", False):
        return

    def load(path, *args, **kwargs):
        # 경로뿐 아니라 파일 객체(BytesIO)로 넘어오는 경우도 받는다.
        src = path if hasattr(path, "read") else str(path)
        speech, sample_rate = sf.read(src, dtype="float32", always_2d=True)
        return torch.from_numpy(speech.T), sample_rate      # (N, C) -> (C, N)

    load._patched_by_voiceapi = True
    torchaudio.load = load


def _patch_resample() -> None:
    """torchaudio.functional.resample 을 GPU 대신 CPU 에서 돌린다.

    XTTS 는 참조 음성으로 화자 임베딩을 만들 때(get_speaker_embedding) 이미 GPU 에 올라간
    텐서를 리샘플한다. sinc 리샘플 커널은 커널 폭이 수백~수천인 conv1d 라 MIOpen 이 처리하지
    못하고 다음으로 죽는다:
        RuntimeError: miopenStatusUnknownError
          torchaudio/functional/functional.py _apply_sinc_resample_kernel -> conv1d

    MIOpen 전반의 문제는 아니다 (CosyVoice3 는 같은 GPU 에서 컨볼루션이 잘 돈다).
    이 연산 형태만 못 견디는 것이라 리샘플만 CPU 로 내린다. 참조 음성은 수 초짜리라
    CPU 리샘플 비용은 무시할 수준이다.
    """
    import torchaudio

    orig = torchaudio.functional.resample
    if getattr(orig, "_patched_by_voiceapi", False):
        return

    def resample(waveform, orig_freq, new_freq, *args, **kwargs):
        if getattr(waveform, "is_cuda", False):
            out = orig(waveform.detach().cpu(), orig_freq, new_freq, *args, **kwargs)
            return out.to(waveform.device)
        return orig(waveform, orig_freq, new_freq, *args, **kwargs)

    resample._patched_by_voiceapi = True
    torchaudio.functional.resample = resample


def load() -> None:
    global _tts, _builtin_speakers, _sample_rate
    from TTS.api import TTS

    _patch_load_wav()
    _patch_resample()
    _tts = TTS(MODEL).to(DEVICE)
    _builtin_speakers = list(getattr(_tts, "speakers", None) or [])
    sr = getattr(getattr(_tts, "synthesizer", None), "output_sample_rate", None)
    if sr:
        _sample_rate = int(sr)


def _resolve_voice(voice: str | None) -> tuple[str | None, str | None]:
    """voice -> (speaker, speaker_wav).

    voices/<id>.wav 가 있으면 그 파일로 복제하고, 없으면 모델 내장 화자 이름으로 본다.
    """
    vid = voice or DEFAULT_VOICE
    if vid:
        wav = VOICES_DIR / f"{vid}.wav"
        if wav.is_file():
            return None, str(wav)
        if vid in _builtin_speakers:
            return vid, None
        raise ValueError(
            f"알 수 없는 voice: {vid} (voices/{vid}.wav 또는 내장 화자여야 한다)"
        )
    # 기본값: 참조 음성이 있으면 그것을, 없으면 첫 내장 화자를 쓴다.
    wavs = sorted(VOICES_DIR.glob("*.wav"))
    if wavs:
        return None, str(wavs[0])
    if _builtin_speakers:
        return _builtin_speakers[0], None
    raise ValueError("사용할 보이스가 없다 (voices/*.wav 를 넣거나 DEFAULT_VOICE 를 지정할 것)")


def synthesize(
    text: str,
    *,
    voice: str | None,
    language: str | None,
    speed: float,
) -> tuple[np.ndarray, int]:
    speaker, speaker_wav = _resolve_voice(voice)
    lang = language or DEFAULT_LANGUAGE
    if lang not in LANGUAGES:
        raise ValueError(f"XTTS-v2 가 지원하지 않는 언어: {lang} (지원: {', '.join(LANGUAGES)})")

    wav = _tts.tts(
        text=text,
        language=lang,
        speaker=speaker,
        speaker_wav=speaker_wav,
        speed=speed,
    )
    return np.asarray(wav, dtype=np.float32).squeeze(), _sample_rate


def voices() -> list[dict]:
    out = [
        {"id": p.stem, "name": p.stem, "language": "-", "gender": "unknown"}
        for p in sorted(VOICES_DIR.glob("*.wav"))
    ]
    out += [
        {"id": s, "name": s, "language": "-", "gender": "unknown"}
        for s in _builtin_speakers
    ]
    return out


spec = EngineSpec(
    name="xtts",
    kind="tts",
    model="XTTS-v2",
    port=int(os.getenv("PORT", "8204")),
    languages=LANGUAGES,
    sample_rate=24000,      # 실제 값은 로딩 후 synthesizer.output_sample_rate 로 잡는다
    extra={
        "backend": "coqui-tts (idiap 포크, torch/ROCm)",
        "voice_mode": "내장 화자 + voices/<id>.wav 참조 음성 복제",
        "supports_translate": False,
        "license": "코드 MPL-2.0 / 가중치 CPML(비상업). Coqui Inc. 폐업으로 "
                   "상업 라이선스 구매 경로가 없다 — 내부 개발용으로만 쓸 것",
    },
)

app = create_app(spec, loader=load, synthesize=synthesize, voices=voices)

if __name__ == "__main__":
    run(app, spec)
