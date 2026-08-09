#!/usr/bin/env bash
#
# voicectl.sh — STT/TTS 엔진 컨테이너 제어
#
# 엔진 목록을 하드코딩하지 않는다. 하위 폴더에서 engine.env 를 찾아 자동 인식하므로
# 새 엔진을 추가하려면 폴더만 만들면 된다 (_template 참고).
#
# 설계상 지키는 것:
#   - 요청한 엔진만 뜬다. 인자 없이 start 하면 아무것도 뜨지 않는다.
#   - 엔진 간 의존성(depends_on)이 없어 하나가 죽어도 나머지는 영향받지 않는다.
#   - 포트/모델 설정의 단일 소스는 각 폴더의 engine.env 다.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 이 서버들의 공통 규약: 모든 서비스가 nginx-proxy-manager 와 같은 proxy-network 에 붙는다.
# 그래서 NPM 이 컨테이너 이름(voice-whisper 등)으로 바로 forward 할 수 있고,
# 엔진마다 VCN 보안 목록에 포트를 따로 열 필요가 없다.
NET="proxy-network"
# 엔진 종류. 종류가 늘어나면 여기에 이름만 추가한다 (포트 대역은 사람용 관례일 뿐이고,
# 종류 판단은 engine.env 의 ENGINE_KIND 로만 한다).
KINDS="stt tts speaker"
PORT_RULE="STT 81xx / TTS 82xx / speaker 83xx"

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_dim=$'\033[2m'; c_bld=$'\033[1m'; c_off=$'\033[0m'

die() { echo "${c_red}Error:${c_off} $*" >&2; exit 1; }

# ---------------------------------------------------------------- 엔진 탐색

engines() {
  local d name
  for d in "$ROOT"/*/; do
    name="$(basename "$d")"
    # _common, _template 처럼 밑줄로 시작하는 폴더는 엔진이 아니다.
    [[ $name == _* ]] && continue
    [[ -f "$d/engine.env" && -f "$d/docker-compose.yml" ]] || continue
    echo "$name"
  done
}

# engine.env 에서 KEY 값을 읽는다 (주석/공백 제거)
meta() {
  local file="$ROOT/$1/engine.env" key="$2"
  [[ -f $file ]] || return 0
  sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//p" "$file" \
    | head -1 | sed 's/[[:space:]]*$//'
}

# secrets.env 에서 값을 읽는다. 비밀값은 버전관리에서 빠져 있어 파일이 없을 수 있다.
secret() {
  local file="$ROOT/$1/secrets.env" key="$2"
  [[ -f $file ]] || return 0
  sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//p" "$file" \
    | head -1 | sed 's/[[:space:]]*$//'
}

valid_engine() {
  local e want="$1"
  for e in $(engines); do [[ $e == "$want" ]] && return 0; done
  return 1
}

require_engines() {
  [[ $# -gt 0 ]] || {
    echo "${c_ylw}Specify an engine.${c_off} By default, nothing starts unless you name it explicitly."
    echo
    echo "Available engines: $(engines | tr '\n' ' ')"
    echo "Example: $0 ${SUBCMD:-start} whisper"
    exit 1
  }
  local e
  for e in "$@"; do
    valid_engine "$e" || die "No such engine folder: $e (available: $(engines | tr '\n' ' '))"
  done
}

# docker compose 를 해당 엔진 폴더에서, engine.env 를 변수 소스로 삼아 실행
dc() {
  local e="$1"; shift
  ( cd "$ROOT/$e" && docker compose --env-file engine.env -f docker-compose.yml "$@" )
}

ensure_net() {
  # proxy-network 는 이 스크립트가 만든 것이 아니라 서버 전체가 공유하는 기존 네트워크다.
  # 없다고 함부로 만들지 않고, 확인만 하고 안내한다.
  docker network inspect "$NET" >/dev/null 2>&1 || die "Shared network '$NET' not found.
  This network is part of the server's shared infrastructure, including nginx-proxy-manager.
  If this really is a fresh server, create it first: docker network create $NET"
}

container_of() { echo "voice-$1"; }

is_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$(container_of "$1")" 2>/dev/null)" == "true" ]]
}

health_of() {
  docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}-{{end}}' \
    "$(container_of "$1")" 2>/dev/null || echo "-"
}

image_exists() {
  docker image inspect "voice-$1:latest" >/dev/null 2>&1
}

# ---------------------------------------------------------------- 명령

cmd_list() {
  printf "${c_bld}%-12s %-8s %-6s %-9s %-10s %s${c_off}\n" ENGINE KIND PORT IMAGE STATE MODEL
  local e kind port img state model key
  for e in $(engines); do
    kind="$(meta "$e" ENGINE_KIND)"
    port="$(meta "$e" PORT)"
    img=$(image_exists "$e" && echo "built" || echo "${c_dim}not built${c_off}")
    if is_running "$e"; then
      state="${c_grn}running${c_off}"
    else
      state="${c_dim}stopped${c_off}"
    fi
    # 어떤 키가 그 엔진의 "모델"인지는 엔진마다 다르다. 스크립트가 알 필요 없이
    # engine.env 의 MODEL_KEY 가 자기 키 이름을 가리킨다 (종류별 분기 없음).
    key="$(meta "$e" MODEL_KEY)"
    model=""
    [[ -n $key ]] && model="$(meta "$e" "$key")"
    printf "%-12s %-8s %-6s %-9b %-10b %s\n" "$e" "$kind" "$port" "$img" "$state" "${model:--}"
  done
  echo
  echo "${c_dim}Port convention: ${PORT_RULE}${c_off}"
}

cmd_build() {
  require_engines "$@"
  local e
  for e in "$@"; do
    echo "${c_bld}[$e] Building image${c_off}"
    dc "$e" build
  done
}

cmd_start() {
  require_engines "$@"
  ensure_net
  local e port
  for e in "$@"; do
    image_exists "$e" || { echo "${c_dim}[$e] No image found, building first${c_off}"; dc "$e" build; }
    echo "${c_bld}[$e] Starting${c_off}"
    dc "$e" up -d
    port="$(meta "$e" PORT)"
    echo "  ${c_dim}First start may take a few minutes while the model downloads.${c_off}"
    echo "  Check: curl http://localhost:${port}/health"
  done
}

cmd_stop() {
  require_engines "$@"
  local e
  for e in "$@"; do
    echo "${c_bld}[$e] Stopping${c_off}"
    dc "$e" down
  done
}

# `docker compose restart` 는 컨테이너를 그 자리에서 다시 돌릴 뿐 새 이미지를 쓰지
# 않는다. build 뒤에 restart 하면 옛 코드가 조용히 계속 돈다 — 아무 오류도 나지 않아
# 알아채기 어렵다. 여기서 재시작은 "지금 이미지로 다시 만든다"를 뜻하게 한다.
cmd_restart() {
  require_engines "$@"
  local e
  for e in "$@"; do
    echo "${c_bld}[$e] Restarting${c_off}"
    dc "$e" up -d --force-recreate
  done
}

cmd_status() {
  local targets=("$@")
  [[ ${#targets[@]} -eq 0 ]] && mapfile -t targets < <(engines)

  printf "${c_bld}%-12s %-10s %-11s %-6s %-10s %s${c_off}\n" ENGINE STATE HEALTH PORT MEMORY RESPONSE
  local e port state hz mem resp cid
  for e in "${targets[@]}"; do
    port="$(meta "$e" PORT)"
    if is_running "$e"; then
      state="${c_grn}running${c_off}"
      hz="$(health_of "$e")"
      cid="$(container_of "$e")"
      mem="$(docker stats --no-stream --format '{{.MemUsage}}' "$cid" 2>/dev/null | cut -d/ -f1 | tr -d ' ')"
      resp="$(curl -s -m 3 "http://localhost:${port}/health" 2>/dev/null \
               | sed -n 's/.*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
      [[ -z $resp ]] && resp="${c_red}no response${c_off}"
    else
      state="${c_dim}stopped${c_off}"; hz="-"; mem="-"; resp="-"
    fi
    printf "%-12s %-10b %-11s %-6s %-10s %b\n" "$e" "$state" "${hz:--}" "$port" "${mem:--}" "$resp"
  done
}

cmd_logs() {
  [[ $# -ge 1 ]] || die "Usage: $0 logs <engine> [-f]"
  local e="$1"; shift
  valid_engine "$e" || die "No such engine: $e"
  dc "$e" logs --tail 100 "$@"
}

cmd_health() {
  local targets=("$@")
  [[ ${#targets[@]} -eq 0 ]] && mapfile -t targets < <(engines)
  local e port
  for e in "${targets[@]}"; do
    port="$(meta "$e" PORT)"
    echo "${c_bld}[$e] http://localhost:${port}/health${c_off}"
    curl -s -m 5 "http://localhost:${port}/health" || echo "  ${c_red}no response${c_off}"
    echo
  done
}

# 실제 추론까지 확인한다. STT 는 TTS 로 만든 파일이나 지정한 파일을 넣어 왕복시킨다.
cmd_test() {
  [[ $# -ge 1 ]] || die "Usage: $0 test <engine> [audio file (STT/speaker) | sentence (TTS)]"
  local e="$1"; shift
  valid_engine "$e" || die "No such engine: $e"
  is_running "$e" || die "$e is not running. Start it first: $0 start $e"

  local port kind key auth=()
  port="$(meta "$e" PORT)"; kind="$(meta "$e" ENGINE_KIND)"
  key="$(secret "$e" API_KEY)"
  [[ -n $key ]] && auth=(-H "Authorization: Bearer $key")

  if [[ $kind == speaker ]]; then
    # 파일 1개면 임베딩, 2개면 두 화자 비교.
    local fa="${1:-}" fb="${2:-}"
    [[ -n $fa ]] || die "Speaker test requires an audio file: $0 test $e <file.wav> [compare-file.wav]"
    [[ -f $fa ]] || die "File not found: $fa"
    if [[ -n $fb ]]; then
      [[ -f $fb ]] || die "File not found: $fb"
      echo "${c_bld}[$e] Comparing:${c_off} $fa <-> $fb"
      curl -sS -m 300 "${auth[@]}" -F "file_a=@${fa}" -F "file_b=@${fb}" \
        "http://localhost:${port}/v1/speaker/compare" | sed 's/^/  /'
      echo
    else
      echo "${c_bld}[$e] Embedding:${c_off} $fa"
      # 임베딩 벡터 전체는 길어서 요약만 보여준다.
      curl -sS -m 300 "${auth[@]}" -F "file=@${fa}" \
        "http://localhost:${port}/v1/speaker/embed" \
        | python3 -c 'import json,sys
d = json.load(sys.stdin)
if "embedding" not in d:
    print("  " + json.dumps(d, ensure_ascii=False)); sys.exit(1)
v = d["embedding"]
print("  dim=%s / audio %ss / processing %ss" % (d["dim"], d["duration"], d["processing_s"]))
print("  first 8: " + ", ".join("%+.4f" % x for x in v[:8]))
print("  L2 norm: %.6f" % (sum(x * x for x in v) ** 0.5))'
    fi
  elif [[ $kind == tts ]]; then
    local text="${1:-Hello, this is a speech synthesis test.}"
    local out="/tmp/voicectl-${e}.wav"
    echo "${c_bld}[$e] Synthesizing:${c_off} $text"
    curl -sS -m 300 -D /tmp/voicectl-hdr "${auth[@]}" \
      -H 'Content-Type: application/json' \
      -d "$(printf '{"input":%s}' "$(printf '%s' "$text" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
      -o "$out" "http://localhost:${port}/v1/audio/speech"
    grep -i '^x-' /tmp/voicectl-hdr | sed 's/^/  /' || true
    if [[ -s $out ]]; then
      echo "  ${c_grn}Success${c_off} -> $out ($(stat -c%s "$out") bytes)"
    else
      echo "  ${c_red}Failed${c_off}"; cat "$out" 2>/dev/null; return 1
    fi
  else
    local file="${1:-}"
    [[ -n $file ]] || die "STT test requires an audio file: $0 test $e <file.wav>"
    [[ -f $file ]] || die "File not found: $file"
    echo "${c_bld}[$e] Transcribing:${c_off} $file"
    curl -sS -m 300 "${auth[@]}" -F "file=@${file}" \
      "http://localhost:${port}/v1/audio/transcriptions" | sed 's/^/  /'
    echo
  fi
}

cmd_new() {
  [[ $# -eq 3 ]] || die "Usage: $0 new <engine-name> <$(echo $KINDS | tr ' ' '|')> <port>
  Example: $0 new sherpa stt 8103
           $0 new piper  tts 8203
  Port convention: ${PORT_RULE}"
  local name="$1" kind="$2" port="$3" k found=0
  for k in $KINDS; do [[ $kind == "$k" ]] && found=1; done
  [[ $found == 1 ]] || die "Kind must be one of: $KINDS"
  [[ -d "$ROOT/$name" ]] && die "Folder already exists: $name"
  [[ -d "$ROOT/_template" ]] || die "_template folder not found"

  cp -r "$ROOT/_template" "$ROOT/$name"
  mkdir -p "$ROOT/$name/models"
  local f
  for f in engine.env Dockerfile docker-compose.yml server.py requirements.txt; do
    [[ -f "$ROOT/$name/$f" ]] || continue
    sed -i "s/__ENGINE__/$name/g; s/__KIND__/$kind/g; s/__PORT__/$port/g" "$ROOT/$name/$f"
  done
  local impl
  case "$kind" in
    stt)     impl="transcribe" ;;
    tts)     impl="synthesize" ;;
    speaker) impl="embed" ;;
    *)       impl="kind-specific function" ;;
  esac
  echo "${c_grn}$name created${c_off} ($kind, port $port)"
  echo "Fill these in, in order:"
  echo "  1) $name/requirements.txt — Python dependencies"
  echo "  2) $name/server.py — implement load() and ${impl}()"
  echo "  3) ./_common/mkvenv.sh $name   <- verify on the host first (recommended)"
  echo "  4) $0 start $name"
}

usage() {
  cat <<EOF
${c_bld}voicectl.sh${c_off} — STT/TTS engine container control

  ${c_bld}$0 list${c_off}                       list engines and status
  ${c_bld}$0 start${c_off}   <engine...>         start only the named engines (auto-builds if missing)
  ${c_bld}$0 stop${c_off}    <engine...>         stop only the named engines
  ${c_bld}$0 restart${c_off} <engine...>         restart
  ${c_bld}$0 build${c_off}   <engine...>         build images only
  ${c_bld}$0 status${c_off}  [engine...]         status/health/memory (all engines if omitted)
  ${c_bld}$0 health${c_off}  [engine...]         print raw /health response
  ${c_bld}$0 logs${c_off}    <engine> [-f]       logs
  ${c_bld}$0 test${c_off}    <engine> [input...] round-trip inference test
                              ${c_dim}(speaker engines: 1 file = embed, 2 files = compare)${c_off}
  ${c_bld}$0 new${c_off}     <name> <$(echo $KINDS | tr ' ' '|')> <port>   create a new engine folder from _template

Available engines: $(engines | tr '\n' ' ')
Engine kinds: $KINDS   ${c_dim}(port convention: ${PORT_RULE})${c_off}

${c_dim}Starting only what you need is the default — there's no command to start everything at once.${c_off}
EOF
}

SUBCMD="${1:-}"
[[ $# -gt 0 ]] && shift || true
case "$SUBCMD" in
  list|ls)   cmd_list "$@" ;;
  build)     cmd_build "$@" ;;
  start|up)  cmd_start "$@" ;;
  stop|down) cmd_stop "$@" ;;
  restart)   cmd_restart "$@" ;;
  status|ps) cmd_status "$@" ;;
  health)    cmd_health "$@" ;;
  logs)      cmd_logs "$@" ;;
  test)      cmd_test "$@" ;;
  new)       cmd_new "$@" ;;
  ""|-h|--help|help) usage ;;
  *)         die "Unknown command: $SUBCMD (help: $0 --help)" ;;
esac
