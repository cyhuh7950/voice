"""
fish-speech (tts) 서버 — Fish Speech S2 (v2.0.0-beta), GPU(AMD ROCm/HIP).

라이선스 주의: **FISH AUDIO RESEARCH LICENSE** — 연구·비상업 무료.
상업 이용은 별도 유료 라이선스가 필요하다 (XTTS-v2 와 달리 구매 경로는 존재한다).

업스트림은 라이브러리 API 를 문서화하지 않고 자체 HTTP 서버(tools/api_server.py)만 제공한다.
그 서버가 쓰는 ModelManager 를 그대로 가져다 쓰고, HTTP 계층은 우리 규격(voiceapi.py)으로 덮는다.

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
# 업스트림 코드가 `from tools.server...` 처럼 저장소 루트 기준으로 import 한다.
sys.path[:0] = [str(_here), str(_here.parent / "_common"), str(_here / "repo")]

import numpy as np  # noqa: E402

from voiceapi import EngineSpec, create_app, run  # noqa: E402

MODEL_DIR = os.getenv("MODEL_NAME", "/models/s2-pro")
DECODER_CONFIG = os.getenv("DECODER_CONFIG", "modded_dac_vq")
DEVICE = os.getenv("DEVICE", "cuda")        # ROCm 에서도 torch API 는 "cuda" 다 (HIP 매핑)
HALF = os.getenv("HALF", "0") == "1"        # 켜면 fp16, 끄면 bf16
COMPILE = os.getenv("COMPILE", "0") == "1"  # torch.compile — ROCm 에서는 기본 끔
VOICES_DIR = _here / "voices"
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "default")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))

# KV 캐시 길이 상한. 업스트림은 config.max_seq_len(32768)을 그대로 캐시 크기로 쓰는데,
# 이 모델(36층 × 8헤드 × 128차원)에서는 그것만으로 약 4.8GB 다.
#   36 * 8 * 128 * 2(K,V) * 2바이트 * 32768 ≈ 4.8GB
# 3B 가중치(bf16 약 6GB)와 합치면 16GB VRAM 을 넘겨 기동이 OOM 으로 죽는다 (실제로 겪었다).
# 4096 으로 낮추면 약 604MB 가 된다. 참조 음성 토큰 + MAX_NEW_TOKENS 를 합쳐도 한참 밑이다.
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "4096"))

_engine = None
_sample_rate = 44100


def _patch_kv_cache_len() -> None:
    """setup_caches 가 잡는 KV 캐시 길이를 MAX_SEQ_LEN 으로 제한한다.

    업스트림에는 이걸 줄일 인자가 없다 (CLI 에도, ModelManager 에도 없고
    setup_caches(max_seq_len=model.config.max_seq_len) 로 하드코딩돼 있다).
    config.max_seq_len 도 같이 낮춰야 한다 — generate() 가 그 값으로 생성 상한을
    계산하기 때문에, 캐시만 줄이면 캐시 범위를 넘겨 인덱싱하게 된다.
    """
    from fish_speech.models.text2semantic import llama as _llama

    def wrap(orig):
        def setup_caches(self, max_batch_size, max_seq_len, *args, **kwargs):
            capped = min(int(max_seq_len), MAX_SEQ_LEN)
            cfg = getattr(self, "config", None)
            if cfg is not None and getattr(cfg, "max_seq_len", 0) > capped:
                cfg.max_seq_len = capped
            return orig(self, max_batch_size, capped, *args, **kwargs)

        setup_caches._patched_by_voiceapi = True
        return setup_caches

    for cls in (_llama.BaseTransformer, _llama.DualARTransformer):
        orig = cls.__dict__.get("setup_caches")
        if orig is None or getattr(orig, "_patched_by_voiceapi", False):
            continue
        cls.setup_caches = wrap(orig)


def _patch_torchaudio_compat() -> None:
    """torchaudio 2.11 에서 사라진 API 를 메우고, 오디오 로딩을 soundfile 로 돌린다.

    fish_speech/inference_engine/reference_loader.py 가 두 가지를 쓴다.
      34행  torchaudio.list_audio_backends()   → 2.11 에서 제거됨 (AttributeError)
      117행 torchaudio.load(..., backend=...)  → 2.11 은 torchcodec 에 위임하는데
            PyPI 의 torchcodec 은 CUDA 빌드라 ROCm 에서 libnvrtc.so.13 을 못 찾는다

    cosyvoice·moss-tts-nano·xtts 에 넣은 것과 같은 우회다. torchaudio 2.11 이 오디오
    입출력을 통째로 torchcodec 으로 넘긴 것이 원인이라 오디오를 읽는 엔진마다 걸린다.
    """
    import soundfile as sf
    import torch
    import torchaudio

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    if getattr(torchaudio.load, "_patched_by_voiceapi", False):
        return

    def load_audio(path, *args, **kwargs):
        kwargs.pop("backend", None)          # 우리 구현에는 백엔드 개념이 없다
        # fish-speech 는 참조 음성을 경로가 아니라 BytesIO 로 넘긴다.
        # str() 로 감싸면 그 repr 문자열을 파일 경로로 열려다 LibsndfileError 가 난다.
        src = path if hasattr(path, "read") else str(path)
        speech, sample_rate = sf.read(src, dtype="float32", always_2d=True)
        return torch.from_numpy(speech.T), sample_rate      # (N, C) -> (C, N)

    load_audio._patched_by_voiceapi = True
    torchaudio.load = load_audio


def load() -> None:
    global _engine, _sample_rate
    from tools.server.model_manager import ModelManager

    _patch_torchaudio_compat()
    _patch_kv_cache_len()

    mm = ModelManager(
        mode="tts",
        device=DEVICE,
        half=HALF,
        compile=COMPILE,
        llama_checkpoint_path=MODEL_DIR,
        decoder_checkpoint_path=str(Path(MODEL_DIR) / "codec.pth"),
        decoder_config_name=DECODER_CONFIG,
    )
    _engine = mm.tts_inference_engine

    decoder = mm.decoder_model
    sr = getattr(getattr(decoder, "spec_transform", None), "sample_rate", None)
    _sample_rate = int(sr or getattr(decoder, "sample_rate", _sample_rate))


def _reference(voice: str | None):
    """voices/<id>.wav (+ .txt 전사문) 을 ServeReferenceAudio 로 만든다."""
    from fish_speech.utils.schema import ServeReferenceAudio

    vid = voice or DEFAULT_VOICE
    if not vid:
        return []
    wav = VOICES_DIR / f"{vid}.wav"
    txt = VOICES_DIR / f"{vid}.txt"
    if not wav.is_file():
        raise ValueError(f"알 수 없는 voice: {vid} (voices/{vid}.wav 필요)")
    return [ServeReferenceAudio(
        audio=wav.read_bytes(),
        text=txt.read_text(encoding="utf-8").strip() if txt.is_file() else "",
    )]


def synthesize(
    text: str,
    *,
    voice: str | None,
    language: str | None,
    speed: float,
) -> tuple[np.ndarray, int]:
    from fish_speech.utils.schema import ServeTTSRequest

    req = ServeTTSRequest(
        text=text,
        references=_reference(voice),
        max_new_tokens=MAX_NEW_TOKENS,
        format="wav",
        streaming=False,
        use_memory_cache="on",      # 같은 참조 음성을 반복 요청할 때 재인코딩을 건너뛴다
    )

    chunks: list[np.ndarray] = []
    sample_rate = _sample_rate
    for result in _engine.inference(req):
        if result.code == "error":
            raise RuntimeError(str(result.error))
        if result.code in ("segment", "final") and result.audio is not None:
            sr, audio = result.audio
            sample_rate = int(sr)
            arr = np.asarray(audio, dtype=np.float32).squeeze()
            if arr.size:
                chunks.append(arr)

    if not chunks:
        raise RuntimeError("합성 결과가 비었다")
    # final 이 전체를 담아 오기도 하고 segment 가 조각으로 오기도 한다.
    wav = chunks[-1] if len(chunks) == 1 else np.concatenate(chunks)
    return wav, sample_rate


def voices() -> list[dict]:
    return [
        {"id": p.stem, "name": p.stem, "language": "-", "gender": "unknown"}
        for p in sorted(VOICES_DIR.glob("*.wav"))
    ]


spec = EngineSpec(
    name="fish-speech",
    kind="tts",
    model="Fish Speech S2 (s2-pro)",
    port=int(os.getenv("PORT", "8206")),
    languages=[],       # 다국어. 텍스트 자체로 판단한다
    sample_rate=44100,  # 실제 값은 로딩 후 디코더에서 잡는다
    extra={
        "backend": "fish_speech TTSInferenceEngine (torch/ROCm)",
        "voice_mode": "참조 음성 복제 (voices/<id>.wav + .txt)",
        "supports_translate": False,
        "license": "FISH AUDIO RESEARCH LICENSE — 연구·비상업 무료, "
                   "상업 이용은 별도 유료 라이선스 필요",
    },
)

app = create_app(spec, loader=load, synthesize=synthesize, voices=voices)

if __name__ == "__main__":
    run(app, spec)
