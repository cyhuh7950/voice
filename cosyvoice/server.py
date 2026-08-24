"""
cosyvoice (tts) 서버 — Fun-CosyVoice3-0.5B, GPU(AMD ROCm/HIP).

CosyVoice3는 SFT 프리셋 보이스가 없고 zero-shot 음성 복제만 지원한다.
그래서 "voice" 는 보이스 이름이 아니라 voices/<voice>.wav(+ .txt) 참조 프롬프트를 가리킨다.

HTTP 규격은 _common/voiceapi.py 가 처리한다. 여기서는 모델 로딩과 합성만 구현한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ROCm(WSL) 필수 환경변수. 이미지 ENV 에도 있지만 호스트 venv 로 직접 돌릴 때를 위해 여기서도 세운다.
#   HSA_ENABLE_DXG_DETECTION : 없으면 hsa_init 실패 (ROCm 7.13 미만)
#   ROCPROFILER_REGISTER_ENABLED : 없으면 torch.cuda.device_count() 에서 프로세스가 abort 한다
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")

# (CUDA 판에 있던 _ensure_cuda_libs_on_ld_path() 는 제거했다. onnxruntime-gpu 가 pip 의
#  nvidia-*-cu12 .so 를 못 찾는 문제를 우회하는 코드였는데, ROCm 에서는 onnxruntime 이
#  CPU 빌드라 해당 사항이 없다.)

_here = Path(__file__).resolve().parent
sys.path[:0] = [
    str(_here),
    str(_here.parent / "_common"),
    str(_here / "repo"),
    str(_here / "repo" / "third_party" / "Matcha-TTS"),
]

import numpy as np  # noqa: E402

from voiceapi import EngineSpec, create_app, run  # noqa: E402

MODEL_DIR = os.getenv("MODEL_NAME", "/models/Fun-CosyVoice3-0.5B")
VOICES_DIR = _here / "voices"
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "default")
# fp16 은 VRAM 과 속도에 유리하지만 ROCm 에서 수치가 무너지면 LLM 이 speech 토큰을
# 몇 개 못 내고 끝나버린다 (그러면 mel 이 3프레임짜리로 나와 보코더가 커널 크기에서 터진다).
FP16 = os.getenv("FP16", "1") == "1"

_cosyvoice = None
_voice_cache: dict[str, tuple[str, str]] = {}


def _patch_load_wav() -> None:
    """CosyVoice 의 load_wav 를 soundfile 기반으로 갈아끼운다.

    torchaudio 2.11 부터 레거시 오디오 백엔드가 빠지고 torchaudio.load 가 torchcodec 으로
    위임된다. CosyVoice 는 torchaudio.load(wav, backend='soundfile') 을 쓰기 때문에
    그대로 두면 참조 음성을 읽는 순간 ImportError: TorchCodec is required 로 죽는다.

    torchcodec 을 새로 들이는 대신 이 함수 하나만 대체한다. 추론 경로에서 torchaudio.load 를
    쓰는 곳은 여기뿐이고 (나머지는 transforms/kaldi 라 순수 torch 다), 동작은 원본과 같다.
    frontend 가 `from ... import load_wav` 로 이름을 이미 바인딩해 갔으므로 그쪽도 함께 바꾼다.
    """
    import soundfile as sf
    import torch
    import torchaudio
    from cosyvoice.cli import frontend as _frontend
    from cosyvoice.utils import file_utils

    def load_wav(wav, target_sr, min_sr=16000):
        speech, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
        speech = torch.from_numpy(speech.T)              # (N, C) -> (C, N)
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            assert sample_rate >= min_sr, \
                'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
            speech = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr)(speech)
        return speech

    file_utils.load_wav = load_wav
    _frontend.load_wav = load_wav


def load() -> None:
    global _cosyvoice
    from cosyvoice.cli.cosyvoice import AutoModel

    _patch_load_wav()
    _cosyvoice = AutoModel(model_dir=MODEL_DIR, fp16=FP16)


def _voice(voice_id: str | None) -> tuple[str, str]:
    """voice_id -> (prompt_wav_path, prompt_text). 캐시로 파일 IO 를 줄인다."""
    vid = voice_id or DEFAULT_VOICE
    cached = _voice_cache.get(vid)
    if cached:
        return cached
    wav_path = VOICES_DIR / f"{vid}.wav"
    txt_path = VOICES_DIR / f"{vid}.txt"
    if not wav_path.is_file() or not txt_path.is_file():
        raise ValueError(f"알 수 없는 voice: {vid} (voices/{vid}.wav + .txt 필요)")
    prompt_text = txt_path.read_text(encoding="utf-8").strip()
    result = (str(wav_path), prompt_text)
    _voice_cache[vid] = result
    return result


# CosyVoice3 는 입력 앞에 시스템 프롬프트 + <|endofprompt|> 를 요구한다.
# 없으면 LLM 이 곧바로 종료해서 mel 이 3프레임짜리로 나오고, 보코더가
# "Kernel size can't be greater than actual input size" 로 터진다
# (llm.py 의 `assert 151646 in text` 가 <|endofprompt|> 토큰을 검사한다).
_EOP = "<|endofprompt|>"
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

# 참조 음성(voices/<id>.wav)이 어떤 언어인지. 요청 언어가 이것과 다르면 cross-lingual 로 간다.
PROMPT_LANGUAGE = os.getenv("PROMPT_LANGUAGE", "ko")


def synthesize(
    text: str,
    *,
    voice: str | None,
    language: str | None,
    speed: float,
) -> tuple[np.ndarray, int]:
    prompt_wav, prompt_text = _voice(voice)

    # NOTE CosyVoice2 의 <|ko|> 같은 언어 태그는 CosyVoice3 에서 쓰지 않는다.
    # 허용 특수토큰이 <|endoftext|>/<|im_start|>/<|im_end|>/<|endofprompt|> 넷뿐이라
    # <|ko|> 를 넣으면 특수토큰이 아니라 그냥 글자로 읽혀버린다.
    # CosyVoice3 는 텍스트 자체의 문자 체계로 언어를 판단한다.
    if language and language != PROMPT_LANGUAGE:
        # 참조 음성과 다른 언어로 말하게 하는 경로. prompt_text 를 쓰지 않으므로
        # <|endofprompt|> 를 본문 앞에 붙인다.
        gen = _cosyvoice.inference_cross_lingual(
            f"{SYSTEM_PROMPT}{_EOP}{text}", prompt_wav, stream=False, speed=speed,
        )
    else:
        # 참조 음성과 같은 언어. 전사문이 있으므로 zero-shot 이 품질이 낫다.
        gen = _cosyvoice.inference_zero_shot(
            text, f"{SYSTEM_PROMPT}{_EOP}{prompt_text}", prompt_wav,
            stream=False, speed=speed,
        )
    out = next(iter(gen))["tts_speech"]  # torch.FloatTensor [1, N], -1.0~1.0
    wav = out.squeeze(0).cpu().numpy().astype(np.float32)
    return wav, _cosyvoice.sample_rate


def voices() -> list[dict]:
    result = []
    for wav_path in sorted(VOICES_DIR.glob("*.wav")):
        vid = wav_path.stem
        if (VOICES_DIR / f"{vid}.txt").is_file():
            result.append({"id": vid, "name": vid, "language": "-", "gender": "unknown"})
    return result


spec = EngineSpec(
    name="cosyvoice",
    kind="tts",
    model="Fun-CosyVoice3-0.5B",
    port=int(os.getenv("PORT", "8203")),
    languages=["zh", "en", "ja", "yue", "ko"],
    sample_rate=24000,  # AutoModel 로딩 후 cosyvoice.sample_rate 로 실제 값 확인
    extra={
        "backend": "CosyVoice3 (FunAudioLLM/CosyVoice)",
        "voice_mode": "zero-shot (3초 참조 음성 복제, voices/<id>.wav+.txt)",
        "supports_translate": False,
        "license": "Apache-2.0 (코드) — 모델 라이선스는 저장소 확인",
    },
)

app = create_app(spec, loader=load, synthesize=synthesize, voices=voices)

if __name__ == "__main__":
    run(app, spec)
