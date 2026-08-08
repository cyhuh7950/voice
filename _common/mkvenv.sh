#!/usr/bin/env bash
# 엔진 폴더에 독립 가상환경을 만들고 requirements.txt 를 설치한다.
#   사용법: _common/mkvenv.sh <engine-folder> [추가 pip 인자...]
# 컨테이너를 쓰지 않고 호스트에서 바로 돌려보거나, 의존성 충돌을 격리할 때 쓴다.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE="${1:?Specify an engine folder name (e.g. whisper)}"
shift || true
DIR="$ROOT/$ENGINE"

[[ -d "$DIR" ]] || { echo "Folder not found: $DIR" >&2; exit 1; }
[[ -f "$DIR/requirements.txt" ]] || { echo "requirements.txt not found: $DIR" >&2; exit 1; }

PY="${PYTHON:-python3}"
echo "[$ENGINE] Creating venv ($($PY -V))"
"$PY" -m venv "$DIR/.venv"

PIP="$DIR/.venv/bin/pip"
"$PIP" install --upgrade pip wheel setuptools >/dev/null
echo "[$ENGINE] Installing requirements"
"$PIP" install -r "$DIR/requirements.txt" "$@"

echo "[$ENGINE] Done — $("$DIR/.venv/bin/python" -V), $("$PIP" list --format=freeze | wc -l) packages"
