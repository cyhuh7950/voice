# 벤더링 출처

이 디렉터리는 업스트림 저장소를 통째로 가져온 것이다. 원래는 별도 git 저장소(clone)
였으나, 이 저장소(voice)의 커밋 이력에 직접 편입시키기 위해 중첩된 `.git`을 지우고
평범한 파일로 바꿨다 (embedded git repository/gitlink 문제 — 서브모듈로 등록하지 않은
채 커밋하면 다른 곳에서 clone 시 이 폴더가 빈 채로 받아진다).

이후 이 폴더에 대한 수정 이력은 voice 저장소 자체의 git log 에서 확인한다.
업스트림 자체의 이후 변경사항을 받으려면 아래 정보로 다시 clone 해서 diff 할 것.

- 저장소: https://github.com/OpenMOSS/MOSS-TTS-Nano.git
- 벤더링 시점 커밋: cc7bdf19c7639c0870dab22045a33b442760f6be (main)
- 커밋 시각: 2026-07-26 20:05:56 +0800
- 벤더링 날짜: 2026-08-24
