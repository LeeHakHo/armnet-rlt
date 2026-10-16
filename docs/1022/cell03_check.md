# 1022 실험 (2026-10-16 ~)

이전 실험(`docs/1015`)과 분리한 새 실험 기록. 시각은 모두 UTC.

## 0. cell-03 동작 확인

**배경**
- 10-13, 10-14 cell-03 잡이 모두 시작 단계에서 실패:
  `OpenCVCameraConfig.__init__() got an unexpected keyword argument 'servo_mount'`
- 원인: `armnet-runtime` 0.3.2가 카메라 설정의 `servo_mount` 블록을 OpenCVCameraConfig 인자로 그대로 넘김.

**10-16 PyPI 릴리스**
- `armnet-runtime` 0.3.3: 카메라 설정에서 `arducam_mount`, `servo_mount`를 제거한 뒤 OpenCVCameraConfig 생성 (`context.py`). 서보 마운트 제어(`move_mount`), 안전 인터록 해제(`Cell.reset(unlock=True)`) 추가.
- `armnet-client` 0.3.5: `execute(cell="cell-03")` 인자 추가 → 지정한 셀이 빌 때까지 대기 후 그 셀에서만 실행.
- `armnet-core` 0.3.3.

**코드 변경**
- `Dockerfile.actor`, `pyproject.toml`, `uv.lock`, `scripts/setup_local.sh`: armnet-client 0.3.5, armnet-runtime 0.3.3, armnet-core 0.3.3.
- `rlt-submit-actor`, `rlt-submit-eval`: `--armnet.require-cell`에 셀 하나만 주면 `execute(cell=...)`로도 전달. 컨테이너 시작 시 셀 검사는 그대로 유지.

**확인 절차**
1. Mac(colima)에서 이미지만 빌드·푸시: `uv run python scripts/build_actor_image.py` (snoopy1은 Docker 권한 없음).
   - 컨텍스트는 `Dockerfile.actor`, `pyproject.toml`, `README.md`, `src/`만 사용 → 같은 커밋이면 ref가 같음.
   - snoopy1에서 `--dry-run`으로 예상 ref 확인: `armnet-rlt-actor:armnet-8e5aef28d2cb`
2. snoopy1에서 `--image-ref`로 cell-03에 짧은 frozen 평가 잡 제출 (`--armnet.require-cell cell-03`, 롤아웃 2~3회).
3. 확인 항목: 컨텍스트 생성 통과, 카메라 3대(top, wrist, front) 프레임, 로봇 구동, busybox 채점.

**결과**
- (대기)
