"""
zipformer (stt) 서버 — 한국어 Zipformer transducer, sherpa-onnx.

이 엔진은 CPU 로 돈다. Zipformer 는 원래 모바일/온디바이스용 모델이라 CPU 로도
실시간을 크게 앞선다 (x86_64 실측 RTF 0.010, 실시간의 100배).

GPU 서버(WSL/ROCm)에 올려도 마찬가지로 CPU 다. ONNX Runtime 의 GPU 가속을 그 환경에서
쓸 수 없기 때문이다 — AMD 가 내놓는 ROCm 용 ONNX 휠은 onnxruntime_migraphx 뿐인데
AMD 공식 문서가 "MIGraphX and mGPU configuration are not currently supported by WSL"
라고 못박고 있다. 덕분에 VRAM 을 쓰지 않아 GPU 엔진과 나란히 띄워둘 수 있다.

HTTP 규격은 _common/voiceapi.py 가 처리한다. 여기서는 모델 로딩과 추론만 구현한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import numpy as np  # noqa: E402

from voiceapi import STT_SAMPLE_RATE, EngineSpec, create_app, run  # noqa: E402

REPO = os.getenv("MODEL_NAME", "k2-fsa/sherpa-onnx-zipformer-korean-2024-06-24")
MODEL_CACHE = os.getenv("MODEL_CACHE", "/models")
NUM_THREADS = int(os.getenv("NUM_THREADS", "4"))
DECODING_METHOD = os.getenv("DECODING_METHOD", "greedy_search")
# int8 양자화본이 CPU 에서 훨씬 빠르다. 정확도를 우선하려면 0 으로 둔다.
INT8 = os.getenv("INT8", "1") == "1"

_recognizer = None


def load() -> None:
    global _recognizer
    import sherpa_onnx
    from huggingface_hub import hf_hub_download

    q = ".int8" if INT8 else ""
    files = {
        "encoder": f"encoder-epoch-99-avg-1{q}.onnx",
        # decoder 는 embedding + conv 라 양자화 이득이 없고, 원본 배포도 fp32 를 권한다.
        "decoder": "decoder-epoch-99-avg-1.onnx",
        "joiner": f"joiner-epoch-99-avg-1{q}.onnx",
        "tokens": "tokens.txt",
    }
    paths = {
        k: hf_hub_download(repo_id=REPO, filename=v, cache_dir=MODEL_CACHE)
        for k, v in files.items()
    }

    _recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=paths["encoder"],
        decoder=paths["decoder"],
        joiner=paths["joiner"],
        tokens=paths["tokens"],
        num_threads=NUM_THREADS,
        sample_rate=STT_SAMPLE_RATE,
        feature_dim=80,
        decoding_method=DECODING_METHOD,
    )


def transcribe(
    audio: np.ndarray,
    *,
    language: str | None,
    prompt: str | None,
    temperature: float,
    task: str,
) -> dict:
    """audio 는 16kHz mono float32 (-1.0~1.0)."""
    if task == "translate":
        raise ValueError("zipformer 는 번역을 지원하지 않는다 (whisper 8103 을 쓸 것)")

    stream = _recognizer.create_stream()
    stream.accept_waveform(STT_SAMPLE_RATE, audio)
    _recognizer.decode_stream(stream)
    result = stream.result

    tokens = getattr(result, "tokens", None) or []
    stamps = getattr(result, "timestamps", None) or []

    # result.text 는 띄어쓰기를 잃어버린다.
    #   tokens: [' 안', '녕', '하', '세요', ' ', '컨', ..., ' 학생', ' 테', '스트', ...]
    #   result.text: '안녕하세요컨테이너음성학생테스트입니다.'
    # 이 모델의 BPE 는 SentencePiece 라 tokens.txt 에 단어경계 표식(▁)이 2352개 들어 있고,
    # sherpa-onnx 가 그걸 토큰 앞의 공백으로 바꿔서 넘겨준다. 즉 띄어쓰기 정보는 이미
    # 예측돼 있고 text 로 합치는 과정에서만 사라진다. 그래서 토큰을 직접 이어 붙인다.
    # (한국어 띄어쓰기 교정기를 따로 붙이거나 재학습할 필요가 없다)
    text = ("".join(tokens).strip() if tokens else result.text.strip())

    segments: list[dict] = []
    # 토큰별 타임스탬프가 있으면 세그먼트로 내보낸다 (없는 모델도 있다).
    if tokens and stamps and len(tokens) == len(stamps):
        for i, (tok, ts) in enumerate(zip(tokens, stamps)):
            segments.append({
                "id": i,
                "start": round(float(ts), 3),
                "end": round(float(stamps[i + 1]), 3) if i + 1 < len(stamps) else None,
                "text": tok,
            })

    return {"text": text, "language": "ko", "segments": segments}


spec = EngineSpec(
    name="zipformer",
    kind="stt",
    model=REPO,
    port=int(os.getenv("PORT", "8105")),
    languages=["ko"],
    sample_rate=STT_SAMPLE_RATE,
    extra={
        "backend": "sherpa-onnx OfflineRecognizer (transducer)",
        "device": "cpu",
        "device_note": "ONNX Runtime GPU 는 WSL/ROCm 에서 쓸 수 없다 (MIGraphX 미지원). "
                       "Zipformer 는 온디바이스용이라 CPU 로도 실시간을 크게 앞선다",
        "quantization": "int8" if INT8 else "fp32",
        "num_threads": NUM_THREADS,
        "supports_translate": False,
        "license": "Apache-2.0 (k2-fsa/sherpa-onnx)",
    },
)

app = create_app(spec, loader=load, transcribe=transcribe)

if __name__ == "__main__":
    run(app, spec)
