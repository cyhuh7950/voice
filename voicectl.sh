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

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_dim=$'\033[2m'; c_bld=$'\033[1m'; c_off=$'\033[0m'

die() { echo "${c_red}오류:${c_off} $*" >&2; exit 1; }

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
    echo "${c_ylw}엔진을 지정하세요.${c_off} 전체를 한꺼번에 띄우지 않는 것이 기본 동작입니다."
    echo
    echo "사용 가능한 엔진: $(engines | tr '\n' ' ')"
    echo "예) $0 ${SUBCMD:-start} whisper"
    exit 1
  }
  local e
  for e in "$@"; do
    valid_engine "$e" || die "그런 엔진 폴더가 없습니다: $e (사용 가능: $(engines | tr '\n' ' '))"
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
  docker network inspect "$NET" >/dev/null 2>&1 || die "공용 네트워크 '$NET' 가 없습니다.
  이 네트워크는 nginx-proxy-manager 를 포함한 서버 공통 인프라입니다.
  정말 없는 서버라면 먼저 만드세요:  docker network create $NET"
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
  printf "${c_bld}%-12s %-5s %-6s %-9s %-10s %s${c_off}\n" 엔진 종류 포트 이미지 상태 모델
  local e kind port img state model
  for e in $(engines); do
    kind="$(meta "$e" ENGINE_KIND)"
    port="$(meta "$e" PORT)"
    img=$(image_exists "$e" && echo "있음" || echo "${c_dim}없음${c_off}")
    if is_running "$e"; then
      state="${c_grn}실행중${c_off}"
    else
      state="${c_dim}정지${c_off}"
    fi
    case "$kind" in
      stt) model="$(meta "$e" WHISPER_MODEL)$(meta "$e" MOONSHINE_LANGUAGE)" ;;
      tts) model="$(meta "$e" SUPERTONIC_MODEL)$(meta "$e" MELO_LANGUAGE)" ;;
      *)   model="-" ;;
    esac
    printf "%-12s %-5s %-6s %-9b %-10b %s\n" "$e" "$kind" "$port" "$img" "$state" "${model:--}"
  done
  echo
  echo "${c_dim}포트 규칙: STT 81xx / TTS 82xx${c_off}"
}

cmd_build() {
  require_engines "$@"
  local e
  for e in "$@"; do
    echo "${c_bld}[$e] 이미지 빌드${c_off}"
    dc "$e" build
  done
}

cmd_start() {
  require_engines "$@"
  ensure_net
  local e port
  for e in "$@"; do
    image_exists "$e" || { echo "${c_dim}[$e] 이미지가 없어 먼저 빌드합니다${c_off}"; dc "$e" build; }
    echo "${c_bld}[$e] 기동${c_off}"
    dc "$e" up -d
    port="$(meta "$e" PORT)"
    echo "  ${c_dim}첫 기동은 모델 다운로드로 수 분 걸릴 수 있습니다.${c_off}"
    echo "  확인: curl http://localhost:${port}/health"
  done
}

cmd_stop() {
  require_engines "$@"
  local e
  for e in "$@"; do
    echo "${c_bld}[$e] 정지${c_off}"
    dc "$e" down
  done
}

cmd_restart() {
  require_engines "$@"
  local e
  for e in "$@"; do
    echo "${c_bld}[$e] 재시작${c_off}"
    dc "$e" restart
  done
}

cmd_status() {
  local targets=("$@")
  [[ ${#targets[@]} -eq 0 ]] && mapfile -t targets < <(engines)

  printf "${c_bld}%-12s %-10s %-11s %-6s %-10s %s${c_off}\n" 엔진 상태 헬스 포트 메모리 응답
  local e port state hz mem resp cid
  for e in "${targets[@]}"; do
    port="$(meta "$e" PORT)"
    if is_running "$e"; then
      state="${c_grn}실행중${c_off}"
      hz="$(health_of "$e")"
      cid="$(container_of "$e")"
      mem="$(docker stats --no-stream --format '{{.MemUsage}}' "$cid" 2>/dev/null | cut -d/ -f1 | tr -d ' ')"
      resp="$(curl -s -m 3 "http://localhost:${port}/health" 2>/dev/null \
               | sed -n 's/.*"status"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
      [[ -z $resp ]] && resp="${c_red}무응답${c_off}"
    else
      state="${c_dim}정지${c_off}"; hz="-"; mem="-"; resp="-"
    fi
    printf "%-12s %-10b %-11s %-6s %-10s %b\n" "$e" "$state" "${hz:--}" "$port" "${mem:--}" "$resp"
  done
}

cmd_logs() {
  [[ $# -ge 1 ]] || die "사용법: $0 logs <엔진> [-f]"
  local e="$1"; shift
  valid_engine "$e" || die "그런 엔진이 없습니다: $e"
  dc "$e" logs --tail 100 "$@"
}

cmd_health() {
  local targets=("$@")
  [[ ${#targets[@]} -eq 0 ]] && mapfile -t targets < <(engines)
  local e port
  for e in "${targets[@]}"; do
    port="$(meta "$e" PORT)"
    echo "${c_bld}[$e] http://localhost:${port}/health${c_off}"
    curl -s -m 5 "http://localhost:${port}/health" || echo "  ${c_red}응답 없음${c_off}"
    echo
  done
}

# 실제 추론까지 확인한다. STT 는 TTS 로 만든 파일이나 지정한 파일을 넣어 왕복시킨다.
cmd_test() {
  [[ $# -ge 1 ]] || die "사용법: $0 test <엔진> [오디오파일(STT) | 문장(TTS)]"
  local e="$1"; shift
  valid_engine "$e" || die "그런 엔진이 없습니다: $e"
  is_running "$e" || die "$e 가 실행 중이 아닙니다. 먼저 $0 start $e"

  local port kind key auth=()
  port="$(meta "$e" PORT)"; kind="$(meta "$e" ENGINE_KIND)"
  key="$(secret "$e" API_KEY)"
  [[ -n $key ]] && auth=(-H "Authorization: Bearer $key")

  if [[ $kind == tts ]]; then
    local text="${1:-안녕하세요. 음성 합성 테스트입니다.}"
    local out="/tmp/voicectl-${e}.wav"
    echo "${c_bld}[$e] 합성:${c_off} $text"
    curl -sS -m 300 -D /tmp/voicectl-hdr "${auth[@]}" \
      -H 'Content-Type: application/json' \
      -d "$(printf '{"input":%s}' "$(printf '%s' "$text" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")" \
      -o "$out" "http://localhost:${port}/v1/audio/speech"
    grep -i '^x-' /tmp/voicectl-hdr | sed 's/^/  /' || true
    if [[ -s $out ]]; then
      echo "  ${c_grn}성공${c_off} → $out ($(stat -c%s "$out") bytes)"
    else
      echo "  ${c_red}실패${c_off}"; cat "$out" 2>/dev/null; return 1
    fi
  else
    local file="${1:-}"
    [[ -n $file ]] || die "STT 테스트는 오디오 파일이 필요합니다: $0 test $e <파일.wav>"
    [[ -f $file ]] || die "파일이 없습니다: $file"
    echo "${c_bld}[$e] 인식:${c_off} $file"
    curl -sS -m 300 "${auth[@]}" -F "file=@${file}" \
      "http://localhost:${port}/v1/audio/transcriptions" | sed 's/^/  /'
    echo
  fi
}

cmd_new() {
  [[ $# -eq 3 ]] || die "사용법: $0 new <엔진명> <stt|tts> <포트>
  예) $0 new sherpa stt 8103
      $0 new piper  tts 8203
  포트 규칙: STT 81xx / TTS 82xx"
  local name="$1" kind="$2" port="$3"
  [[ $kind == stt || $kind == tts ]] || die "종류는 stt 또는 tts 여야 합니다"
  [[ -d "$ROOT/$name" ]] && die "이미 있는 폴더입니다: $name"
  [[ -d "$ROOT/_template" ]] || die "_template 폴더가 없습니다"

  cp -r "$ROOT/_template" "$ROOT/$name"
  mkdir -p "$ROOT/$name/models"
  local f
  for f in engine.env Dockerfile docker-compose.yml server.py requirements.txt; do
    [[ -f "$ROOT/$name/$f" ]] || continue
    sed -i "s/__ENGINE__/$name/g; s/__KIND__/$kind/g; s/__PORT__/$port/g" "$ROOT/$name/$f"
  done
  echo "${c_grn}$name 생성 완료${c_off} ($kind, 포트 $port)"
  echo "다음 순서로 채우세요:"
  echo "  1) $name/requirements.txt — 파이썬 의존성"
  echo "  2) $name/server.py — load() 와 $( [[ $kind == stt ]] && echo transcribe || echo synthesize )() 구현"
  echo "  3) ./_common/mkvenv.sh $name   ← 호스트에서 먼저 검증 (권장)"
  echo "  4) $0 start $name"
}

usage() {
  cat <<EOF
${c_bld}voicectl.sh${c_off} — STT/TTS 엔진 컨테이너 제어

  ${c_bld}$0 list${c_off}                       엔진 목록과 상태
  ${c_bld}$0 start${c_off}   <엔진...>           지정한 엔진만 기동 (없으면 자동 빌드)
  ${c_bld}$0 stop${c_off}    <엔진...>           지정한 엔진만 정지
  ${c_bld}$0 restart${c_off} <엔진...>           재시작
  ${c_bld}$0 build${c_off}   <엔진...>           이미지만 빌드
  ${c_bld}$0 status${c_off}  [엔진...]           상태/헬스/메모리 (생략 시 전체 조회)
  ${c_bld}$0 health${c_off}  [엔진...]           /health 원문 출력
  ${c_bld}$0 logs${c_off}    <엔진> [-f]         로그
  ${c_bld}$0 test${c_off}    <엔진> [입력]       실제 추론 왕복 테스트
  ${c_bld}$0 new${c_off}     <이름> <stt|tts> <포트>   _template 로 새 엔진 폴더 생성

사용 가능한 엔진: $(engines | tr '\n' ' ')

${c_dim}필요한 엔진만 골라 띄우는 것이 기본입니다. 전체를 한꺼번에 올리는 명령은 없습니다.${c_off}
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
  *)         die "모르는 명령: $SUBCMD (도움말: $0 --help)" ;;
esac
