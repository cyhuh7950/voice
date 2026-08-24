"""
whisper (stt) 서버 — Whisper large-v3, GPU(AMD ROCm/HIP).

백엔드가 CUDA 판과 다르다. faster-whisper(CTranslate2)는 ROCm 백엔드가 없어서
AMD GPU 에서 GPU 가속을 쓸 수 없다. 그래서 transformers 의
WhisperForConditionalGeneration + torch(ROCm) 조합으로 간다.
모델도 CTranslate2 변환본이 아니라 원본 safetensors(openai/whisper-large-v3)를 쓴다.

HTTP 규격은 _common/voiceapi.py 가 처리한다. 여기서는 모델 로딩과 추론만 구현한다.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# ROCm(WSL) 필수 환경변수. 이미지 ENV 에도 있지만 호스트 venv 로 직접 돌릴 때를 위해 여기서도 세운다.
#   HSA_ENABLE_DXG_DETECTION : 없으면 hsa_init 실패 (ROCm 7.13 미만)
#   ROCPROFILER_REGISTER_ENABLED : 없으면 torch.cuda.device_count() 에서 프로세스가 abort 한다
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import numpy as np  # noqa: E402

from voiceapi import STT_SAMPLE_RATE, EngineSpec, create_app, run  # noqa: E402

MODEL = os.getenv("MODEL_NAME", "openai/whisper-large-v3")
# 컨테이너는 /models 볼륨. 호스트 venv 로 직접 검증할 때는 MODEL_CACHE 로 덮어쓴다.
MODEL_CACHE = os.getenv("MODEL_CACHE", "/models")
DEVICE = os.getenv("DEVICE", "cuda")          # ROCm 에서도 torch API 는 "cuda" 다 (HIP 매핑)
DTYPE = os.getenv("COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))
# 30초를 넘는 오디오는 청크로 잘라 처리한다. 0 이면 순차(long-form) 디코딩.
CHUNK_LENGTH_S = float(os.getenv("CHUNK_LENGTH_S", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

_model = None
_processor = None
_torch = None


def load() -> None:
    global _model, _processor, _torch
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    _torch = torch
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[DTYPE]

    _processor = AutoProcessor.from_pretrained(MODEL, cache_dir=MODEL_CACHE)
    _model = WhisperForConditionalGeneration.from_pretrained(
        MODEL,
        cache_dir=MODEL_CACHE,
        dtype=dtype,
        low_cpu_mem_usage=True,
        # SDPA 는 ROCm 에서도 동작한다. flash_attention_2 는 gfx1101 휠이 없다.
        attn_implementation="sdpa",
    ).to(DEVICE)
    _model.eval()


_LANG_TOKEN = re.compile(r"<\|([a-z]{2,3})\|>")


def _detect_language(features) -> str | None:
    """오디오 앞 30초로 언어를 판정한다.

    transformers 는 감지된 언어를 응답으로 돌려주지 않는다. generate() 의 출력 시퀀스에서
    읽으려 해도 안 되는데, generate 가 프롬프트 토큰(<|startoftranscript|><|ko|>...)을
    잘라내고 새로 만든 토큰만 반환하기 때문이다. 그래서 detect_language() 를 따로 부른다.

    반환값은 1차원 텐서다 (tensor([50264]) 형태). 2차원으로 인덱싱하면 IndexError 가 난다."""
    ids = _model.detect_language(features)
    token = _processor.tokenizer.convert_ids_to_tokens(int(ids.flatten()[0]))
    m = _LANG_TOKEN.fullmatch(token)     # "<|ko|>" -> "ko"
    return m.group(1) if m else None


def transcribe(
    audio: np.ndarray,
    *,
    language: str | None,
    prompt: str | None,
    temperature: float,
    task: str,
) -> dict:
    """audio 는 16kHz mono float32 (-1.0~1.0)."""
    torch = _torch
    dtype = next(_model.parameters()).dtype

    # whisper 는 30초를 한 창(3000 mel frame)으로 본다. 30초 이하면 3000 으로 패딩해야 하고
    # (안 하면 "expects the mel input features to be of length 3000" 로 터진다),
    # 30초를 넘으면 자르지 말고 그대로 넣어 순차 long-form 디코딩에 태운다.
    long_form = len(audio) > 30 * STT_SAMPLE_RATE

    if long_form:
        inputs = _processor(
            audio,
            sampling_rate=STT_SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
    else:
        inputs = _processor(audio, sampling_rate=STT_SAMPLE_RATE, return_tensors="pt")

    features = inputs.input_features.to(DEVICE, dtype=dtype)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)

    # language 를 주면 그대로 쓰고, 없으면 여기서 감지해 응답에도 실어준다.
    # 감지값을 generate 에 넘겨두면 whisper 가 같은 판정을 다시 하지 않는다.
    detected = language or _detect_language(features)

    gen_kwargs: dict = {
        "task": task,
        "num_beams": BEAM_SIZE,
        "return_timestamps": True,
    }
    # return_segments 는 long-form 디코딩에만 있는 개념이다.
    if long_form:
        gen_kwargs["return_segments"] = True
    if detected:
        gen_kwargs["language"] = detected
    if temperature and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["num_beams"] = 1     # 샘플링과 빔서치는 같이 못 쓴다
    if prompt:
        gen_kwargs["prompt_ids"] = _processor.get_prompt_ids(prompt, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        out = _model.generate(features, attention_mask=attention_mask, **gen_kwargs)

    # return_segments=True 면 {"sequences": ..., "segments": [[seg, ...]]} 로 온다.
    segments: list[dict] = []
    text_parts: list[str] = []
    raw_segments = out["segments"][0] if isinstance(out, dict) and "segments" in out else None
    sequences = out["sequences"] if isinstance(out, dict) else out

    if raw_segments:
        # long-form: 30초 창마다 세그먼트가 나온다. 시작/끝 초가 그대로 들어 있다.
        for i, seg in enumerate(raw_segments):
            seg_text = _processor.decode(seg["tokens"], skip_special_tokens=True)
            text_parts.append(seg_text)
            segments.append({
                "id": i,
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "text": seg_text,
            })
    else:
        # short-form: 타임스탬프가 토큰으로 박혀 나오므로 offsets 로 뽑아 세그먼트를 만든다.
        decoded = _processor.tokenizer.decode(
            sequences[0], skip_special_tokens=True, output_offsets=True,
        )
        text_parts.append(decoded["text"])
        for i, off in enumerate(decoded.get("offsets", [])):
            start, end = off["timestamp"]
            segments.append({
                "id": i,
                "start": round(float(start), 3),
                "end": round(float(end), 3) if end is not None else None,
                "text": off["text"],
            })

    return {
        "text": "".join(text_parts).strip(),
        "language": detected or "unknown",
        "segments": segments,
    }


spec = EngineSpec(
    name="whisper",
    kind="stt",
    model=MODEL,
    port=int(os.getenv("PORT", "8103")),
    languages=[],  # whisper 는 99개 언어 자동 감지
    sample_rate=STT_SAMPLE_RATE,
    extra={
        "backend": "transformers WhisperForConditionalGeneration (torch/ROCm)",
        "backend_note": "faster-whisper(CTranslate2)는 ROCm 백엔드가 없어 사용하지 않는다",
        "device": DEVICE,
        "compute_type": DTYPE,
        "supports_translate": True,
        "license": "MIT (코드/모델 모두)",
    },
)

app = create_app(spec, loader=load, transcribe=transcribe)

if __name__ == "__main__":
    run(app, spec)
