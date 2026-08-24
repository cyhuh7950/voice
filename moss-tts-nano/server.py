"""
moss-tts-nano (tts) 서버 — MOSS-TTS-Nano-100M, ONNX Runtime CPU.

이 엔진은 CPU 로 돈다. 배포되는 모델이 ONNX(MOSS-TTS-Nano-100M-ONNX +
MOSS-Audio-Tokenizer-Nano-ONNX)이고, 100M 짜리 소형 모델이라 CPU 로도 실용적이다
(x86_64 실측 RTF 0.541).

GPU 서버(WSL/ROCm)에 올려도 마찬가지로 CPU 다. ONNX Runtime 의 GPU 가속을 그 환경에서
쓸 수 없기 때문이다 — AMD 의 ROCm 용 ONNX 휠은 onnxruntime_migraphx 뿐이고, AMD 공식
문서가 "MIGraphX and mGPU configuration are not currently supported by WSL" 라고
못박고 있다. 덕분에 VRAM 을 쓰지 않아 GPU 엔진과 나란히 띄워둘 수 있다.

업스트림은 라이브러리 API 를 문서화하지 않아 저장소를 repo/ 로 통째로 받아
onnx_tts_runtime.OnnxTtsRuntime 을 직접 쓴다.

HTTP 규격은 _common/voiceapi.py 가 처리한다. 여기서는 모델 로딩과 합성만 구현한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
# onnx_tts_runtime 은 repo 패키지 안이 아니라 저장소 최상위에 있는 모듈이다.
sys.path[:0] = [str(_here), str(_here.parent / "_common"), str(_here / "repo")]

import numpy as np  # noqa: E402

from voiceapi import EngineSpec, create_app, run  # noqa: E402

# NOTE model_dir 을 넘기면 안 된다.
#   onnx_tts_runtime 은 model_dir 이 "기본 경로(<저장소>/models)일 때만" ONNX 자산을
#   자동으로 받는다(_is_default_model_dir 검사). /models 를 명시하면 자동 다운로드가 꺼져
#   FileNotFoundError: browser_onnx model assets not found 로 기동이 죽는다.
#   대신 Dockerfile 이 /app/repo/models 를 /models 볼륨으로 심볼릭 링크해두었다.
MODEL_DIR = os.getenv("MODEL_NAME", "") or None
VOICES_DIR = _here / "voices"
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "")          # 빈 값이면 첫 번째 내장 보이스
CPU_THREADS = int(os.getenv("CPU_THREADS", "4"))
MAX_NEW_FRAMES = int(os.getenv("MAX_NEW_FRAMES", "375"))
SAMPLE_MODE = os.getenv("SAMPLE_MODE", "fixed")         # greedy | fixed | full
# WeTextProcessing 기반 정규화. pynini 설치가 까다로워 기본은 끈다.
ENABLE_WETEXT = os.getenv("ENABLE_WETEXT", "0") == "1"

_runtime = None
_voices: list[dict] = []


def _patch_load_wav() -> None:
    """onnx_tts_runtime 이 참조 음성을 torchaudio.load 로 읽는다.

    torchaudio 2.11 부터 레거시 백엔드가 빠지고 torchcodec 으로 위임되기 때문에 그대로 두면
    음성 복제 경로에서 ImportError: TorchCodec is required 로 죽는다. soundfile 로 대체한다.
    (내장 보이스만 쓰면 이 경로를 안 타지만, prompt_audio_path 를 주는 순간 걸린다)
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


def load() -> None:
    global _runtime, _voices
    from onnx_tts_runtime import OnnxTtsRuntime

    _patch_load_wav()
    _runtime = OnnxTtsRuntime(
        model_dir=MODEL_DIR,
        thread_count=CPU_THREADS,
        max_new_frames=MAX_NEW_FRAMES,
        sample_mode=SAMPLE_MODE,
        execution_provider="cpu",
    )
    try:
        _voices = list(_runtime.list_builtin_voices())
    except Exception:
        _voices = []


def synthesize(
    text: str,
    *,
    voice: str | None,
    language: str | None,
    speed: float,
) -> tuple[np.ndarray, int]:
    # voices/<id>.wav 가 있으면 참조 음성 복제 경로를 쓴다. 업스트림이 미는 주 사용 방식이고,
    # 내장 프리셋에 한국어 음색이 없어서 한국어는 이쪽이 훨씬 낫다.
    vid = voice or DEFAULT_VOICE
    prompt_wav = None
    if vid:
        cand = VOICES_DIR / f"{vid}.wav"
        if cand.is_file():
            prompt_wav = str(cand)
            vid = None                      # prompt_audio_path 가 voice 를 덮어쓴다

    result = _runtime.synthesize(
        text=text,
        voice=vid or None,
        prompt_audio_path=prompt_wav,
        sample_mode=SAMPLE_MODE,
        max_new_frames=MAX_NEW_FRAMES,
        enable_wetext=ENABLE_WETEXT,
    )
    wav = np.asarray(result["waveform"], dtype=np.float32).squeeze()
    return wav, int(result["sample_rate"])


def voices() -> list[dict]:
    # voices/<id>.wav 참조 음성을 먼저, 그다음 모델 내장 프리셋.
    out = [
        {"id": p.stem, "name": f"{p.stem} (참조 음성 복제)", "language": "-", "gender": "unknown"}
        for p in sorted(VOICES_DIR.glob("*.wav"))
    ]
    for v in _voices:
        vid = v.get("voice") if isinstance(v, dict) else str(v)
        out.append({
            "id": vid,
            "name": (v.get("name") if isinstance(v, dict) else vid) or vid,
            "language": (v.get("language") if isinstance(v, dict) else "-") or "-",
            "gender": (v.get("gender") if isinstance(v, dict) else "unknown") or "unknown",
        })
    return out


spec = EngineSpec(
    name="moss-tts-nano",
    kind="tts",
    model="MOSS-TTS-Nano-100M-ONNX",
    port=int(os.getenv("PORT", "8207")),
    languages=["ko", "en", "zh", "ja", "de", "es", "fr", "it", "ru", "pt"],
    sample_rate=48000,      # 실제 값은 synthesize 가 돌려주는 sample_rate 를 그대로 쓴다
    extra={
        "backend": "onnx_tts_runtime.OnnxTtsRuntime (ONNX Runtime, CPU)",
        "device": "cpu",
        "device_note": "ONNX Runtime GPU 는 WSL/ROCm 에서 쓸 수 없다 (MIGraphX 미지원)",
        "voice_mode": "내장 프리셋 + 참조 음성 복제",
        "speed_supported": False,
        "supports_translate": False,
        "license": "업스트림 저장소 확인 필요",
    },
)

app = create_app(spec, loader=load, synthesize=synthesize, voices=voices)

if __name__ == "__main__":
    run(app, spec)
