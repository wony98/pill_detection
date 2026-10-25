# 공개용 로그 사본

원본 로그는 로컬에 보존했습니다. 아래 사본은 개인 경로만 치환했고, JSON 결과 줄은 바이트 단위로 그대로 유지했습니다. `.log`는 저장소 기본 제외 대상이므로 공개 사본 확장자는 `.txt`입니다.

치환 표식: 사용자 홈 `<HOME>`, 가상환경 `<VENV>`, 프로젝트 `<PROJECT_ROOT>`, 당시 작업 폴더 `<WORKSPACE_ROOT>`.

| 원본 이름 | 공개 사본 | 원본 SHA-256 | 공개본 SHA-256 | 치환 횟수 |
|---|---|---|---|---:|
| smoke-01.log | smoke-01.txt | `f06f87666d88afa346b7aaa7f73af096d1519dcb7a81dc2fc21ce124c604b474` | `c6021f353e69d2c72b9ea29bcd127de641e0a6889cc879dd0a169e9259e163f6` | 2 |
| full-01.log | full-01.txt | `3d5a0b4b32516b492478fd3cf2b61819564af7c2cbad8183fd6f3205ffb7e54c` | `7b5b7883a8faf11bf3081ed917d1c500b2b1a4dbc1a0d1b4647ad81907c0ef4a` | 2 |
| record-01.log | record-01.txt | `68fcf064da3021b6d4f80c65b5c6ff41727ab01454beff6667d447821b0c2537` | `3f1d7b36a95e730985501bd3bfdfe0d53186e869c607d8179c6d6e2a46763515` | 13 |
| record-02-ubuntu.log | record-02-ubuntu.txt | `a59560a2b4f573511f97b8c4defac82e841f0cdbb382f30d64d13feacc31ee8d` | `a59560a2b4f573511f97b8c4defac82e841f0cdbb382f30d64d13feacc31ee8d` | 0 |
| postprocessing-cross-environment.log | postprocessing-cross-environment.txt | `643731d808b0c9176050ecf4a04e3bd972fcb26579b117ac170bd584aafc230e` | `643731d808b0c9176050ecf4a04e3bd972fcb26579b117ac170bd584aafc230e` | 0 |
| environment-check.log | environment-check.txt | `d9fd0d47364bc44bc2b910efb70f67b21df02d2b5ddb277037105d5747f128fc` | `d9fd0d47364bc44bc2b910efb70f67b21df02d2b5ddb277037105d5747f128fc` | 0 |

규칙별 치환 횟수는 `redaction-manifest.json`에 있습니다. 숫자·평가 결과·예외 종류를 고치거나 실패를 성공으로 바꾸지 않았습니다.
원본 failure.json의 로그 참조는 당시 구조를 유지합니다. `../record-01.log`에 대응하는 공개 사본은 이 폴더의 `record-01.txt`입니다.
