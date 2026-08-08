"""
speaker 화자 임베딩 서버 — WeSpeaker ECAPA-TDNN(ONNX) 백엔드.

음성을 받아 화자 벡터(기본 192차원, L2 정규화)를 돌려준다. 두 벡터의 내적이 곧
코사인 유사도이므로, 호출하는 쪽은 등록된 화자와의 유사도로 "누구인지" 판단하거나
임계값 미만이면 "등록되지 않은 목소리"로 거부하면 된다.

특징 추출은 kaldi-native-fbank(80-bin fbank + CMN)로 하고 추론은 onnxruntime 이다.
torch 를 쓰지 않는다 — 같은 결과를 내지만 2GB 를 끌고 온다.

주의(실측으로 확인된 부분):
  voiceapi.decode_audio() 는 PCM 을 -1.0~1.0 으로 정규화해 돌려주는데,
  이 fbank 설정은 int16 스케일(±32768)의 파형을 전제로 검증됐다.
  그래서 특징 추출 직전에 FEATURE_SCALE 을 곱해 스케일을 되돌린다.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path[:0] = [str(_here), str(_here.parent / "_common")]

import kaldi_native_fbank as knf  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from voiceapi import EngineSpec, create_app, run  # noqa: E402

log = logging.getLogger("speaker")

# ---- 설정 (전부 engine.env 에서 온다) ---------------------------------------

MODEL = os.getenv("SPEAKER_MODEL_NAME", "wespeaker-ecapa-tdnn512-LM")
MODEL_DIR = Path(os.getenv("SPEAKER_MODEL_DIR", "/models"))
MODEL_FILE = os.getenv("SPEAKER_MODEL_FILE", "voxceleb_ECAPA512_LM.onnx")
# 받을 곳 후보. 콤마로 여러 개를 두고 앞에서부터 시도한다 (HF 가 죽으면 미러로 넘어간다).
MODEL_URLS = [u.strip() for u in os.getenv("SPEAKER_MODEL_URLS", "").split(",") if u.strip()]
# 받다 만 파일이나 HTML 오류 페이지를 모델로 착각하지 않도록 하한을 둔다.
MODEL_MIN_BYTES = int(os.getenv("SPEAKER_MODEL_MIN_BYTES", "1000000"))
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "5"))
DOWNLOAD_RETRY_WAIT = float(os.getenv("DOWNLOAD_RETRY_WAIT", "3"))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "120"))

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

# fbank 파라미터. 모델 학습 때와 같아야 하며, 하나라도 어긋나면 임베딩이 달라진다.
FBANK_NUM_BINS = int(os.getenv("FBANK_NUM_BINS", "80"))
FBANK_FRAME_LENGTH_MS = float(os.getenv("FBANK_FRAME_LENGTH_MS", "25.0"))
FBANK_FRAME_SHIFT_MS = float(os.getenv("FBANK_FRAME_SHIFT_MS", "10.0"))
FBANK_WINDOW = os.getenv("FBANK_WINDOW", "hamming")
FBANK_DITHER = float(os.getenv("FBANK_DITHER", "0.0"))
FBANK_ENERGY_FLOOR = float(os.getenv("FBANK_ENERGY_FLOOR", "0.0"))
FBANK_SNIP_EDGES = os.getenv("FBANK_SNIP_EDGES", "1") not in ("0", "false", "no")

# -1.0~1.0 파형을 int16 스케일로 되돌리는 계수 (위 docstring 참고)
FEATURE_SCALE = float(os.getenv("FEATURE_SCALE", "32768.0"))
# 채널/마이크 차이를 지우는 평균 차감(cepstral mean normalization)
APPLY_CMN = os.getenv("APPLY_CMN", "1") not in ("0", "false", "no")
# 내적만으로 코사인 유사도가 되도록 정규화해서 내보낸다
L2_NORMALIZE = os.getenv("L2_NORMALIZE", "1") not in ("0", "false", "no")

# 같은 화자로 볼 코사인 유사도 하한. /info 로 알리고 /v1/speaker/compare 기본값이 된다.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

_intra = int(os.getenv("INTRA_OP_THREADS", "0"))
_inter = int(os.getenv("INTER_OP_THREADS", "0"))

_sess: ort.InferenceSession | None = None
_input_name = ""
_output_name = ""


# ---- 모델 준비 ---------------------------------------------------------------


def _fetch(url: str, how: str, tmp: Path) -> None:
    """url 을 tmp 로 받는다. urllib 과 curl 두 경로를 둔다 (한쪽만 막히는 경우가 있다)."""
    import urllib.request

    tmp.unlink(missing_ok=True)
    if how == "urllib":
        req = urllib.request.Request(url, headers={"User-Agent": "voice-speaker/1.0"})
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r, tmp.open("wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    else:
        subprocess.run(
            ["curl", "-fsSL", "--http1.1", "--retry", "2", "-o", str(tmp), url],
            check=True,
            timeout=DOWNLOAD_TIMEOUT * 2,
        )
    size = tmp.stat().st_size
    if size < MODEL_MIN_BYTES:
        raise RuntimeError(f"받은 파일이 너무 작습니다({size} bytes). 오류 응답으로 보입니다")


def _download(dest: Path) -> None:
    """모델을 받아 dest 에 둔다.

    HuggingFace CDN 이 간헐적으로 500 을 내거나 전송을 끊는다. 그래서
    URL 후보 × (urllib, curl) 을 재시도까지 돌린다. 그래도 안 되면
    받아둔 파일을 SPEAKER_MODEL_DIR 에 직접 넣는 경로가 남아 있다.
    """
    if not MODEL_URLS:
        raise RuntimeError(
            f"모델 파일이 없고 SPEAKER_MODEL_URLS 도 비어 있습니다: {dest}\n"
            f"engine.env 에 URL 을 넣거나 {dest} 에 직접 파일을 두세요"
        )
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = ""
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        for url in MODEL_URLS:
            for how in ("urllib", "curl"):
                try:
                    _fetch(url, how, tmp)
                    size = tmp.stat().st_size
                    tmp.replace(dest)
                    log.info("모델 다운로드 완료: %s (%.1f MB, %s)", dest, size / 1e6, how)
                    return
                except Exception as exc:
                    last = f"{url} [{how}] {type(exc).__name__}: {exc}"
                    log.warning("모델 다운로드 실패 (%d/%d) %s", attempt, DOWNLOAD_RETRIES, last)
        time.sleep(DOWNLOAD_RETRY_WAIT * attempt)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(
        "모델 다운로드에 모두 실패했습니다. 업스트림(HuggingFace CDN) 장애일 수 있습니다.\n"
        f"  받을 곳: {', '.join(MODEL_URLS)}\n"
        f"  마지막 오류: {last}\n"
        f"  우회: 받아둔 {MODEL_FILE} 을 이 엔진의 models/ 폴더에 직접 넣고 재시작하세요"
    )


def _model_path() -> Path:
    """캐시에 없으면 받아온다. 다른 엔진들과 같이 /models 볼륨에 남는다."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / MODEL_FILE
    if path.exists() and path.stat().st_size >= MODEL_MIN_BYTES:
        return path
    _download(path)
    return path


def load() -> None:
    global _sess, _input_name, _output_name
    path = _model_path()
    opts = ort.SessionOptions()
    if _intra:
        opts.intra_op_num_threads = _intra
    if _inter:
        opts.inter_op_num_threads = _inter
    _sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    # 입출력 이름은 모델에서 읽는다 (소스에 박지 않는다).
    _input_name = _sess.get_inputs()[0].name
    _output_name = _sess.get_outputs()[0].name

    # 예열 겸 임베딩 차원 확인. 무음은 로그 에너지가 발산할 수 있어 아주 작은 잡음을 쓴다.
    warm = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 1e-3).astype(np.float32)
    dim = int(embed(warm).size)
    spec.extra["embedding_dim"] = dim
    spec.extra["model_file"] = str(path)
    log.info("화자 모델 로드 완료: %s (%s → %s, dim=%d)", path.name, _input_name, _output_name, dim)


# ---- 추론 --------------------------------------------------------------------


def _fbank(pcm: np.ndarray) -> np.ndarray:
    """16kHz mono 파형 → [T, num_bins] fbank. 입력은 int16 스케일."""
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.dither = FBANK_DITHER
    opts.frame_opts.window_type = FBANK_WINDOW
    opts.frame_opts.frame_length_ms = FBANK_FRAME_LENGTH_MS
    opts.frame_opts.frame_shift_ms = FBANK_FRAME_SHIFT_MS
    opts.frame_opts.snip_edges = FBANK_SNIP_EDGES
    opts.mel_opts.num_bins = FBANK_NUM_BINS
    opts.energy_floor = FBANK_ENERGY_FLOOR

    f = knf.OnlineFbank(opts)
    f.accept_waveform(SAMPLE_RATE, pcm.tolist())
    f.input_finished()
    if f.num_frames_ready <= 0:
        raise ValueError("특징 프레임이 만들어지지 않았습니다 (오디오가 너무 짧습니다)")
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)])


def embed(audio: np.ndarray) -> np.ndarray:
    """16kHz mono float32 (-1.0~1.0) → 화자 임베딩 1차원 배열."""
    assert _sess is not None
    pcm = np.asarray(audio, dtype=np.float32).reshape(-1) * FEATURE_SCALE
    feats = _fbank(pcm)
    if APPLY_CMN:
        feats = feats - feats.mean(axis=0, keepdims=True)
    out = _sess.run([_output_name], {_input_name: feats[None].astype(np.float32)})[0]
    vec = np.asarray(out, dtype=np.float32).reshape(-1)
    if L2_NORMALIZE:
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
    return vec


spec = EngineSpec(
    name=os.getenv("ENGINE_NAME", "speaker"),
    kind=os.getenv("ENGINE_KIND", "speaker"),
    model=MODEL,
    port=int(os.getenv("PORT", "8301")),
    languages=[],  # 화자 임베딩은 언어와 무관하다
    sample_rate=SAMPLE_RATE,
    extra={
        "backend": "WeSpeaker ECAPA-TDNN (ONNX Runtime)",
        "features": (
            f"kaldi fbank {FBANK_NUM_BINS}-bin / {FBANK_FRAME_LENGTH_MS:g}ms "
            f"/ {FBANK_FRAME_SHIFT_MS:g}ms hop / CMN={'on' if APPLY_CMN else 'off'}"
        ),
        "normalized": L2_NORMALIZE,
        "similarity": "cosine (정규화된 벡터의 내적)",
        # 이 서버 실측: 같은 화자 0.72~0.80, 다른 화자 0.20~0.42.
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "license": "WeSpeaker Apache-2.0 / VoxCeleb 학습 데이터 조건은 업스트림 확인 필요",
    },
)

app = create_app(spec, loader=load, embed=embed)

if __name__ == "__main__":
    run(app, spec)
