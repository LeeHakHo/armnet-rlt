# push_green_button 실험 정리 (2026-10-08 ~ 10-15)

Armnet SO-101 셀에서 busybox `push_green_button` 태스크로 돌린 정책 비교. 시각은 모두 UTC.

## 공통 평가 조건

| 항목 | 값 |
|---|---|
| 셀 | **cell-01** (busybox, SO-101 단일 팔). 아래 20 에피소드 비교는 전부 cell-01 |
| 장면 변화 | `variation=true`, `variation_seed=1042` → 레일 위치 + 라이트박스 밝기가 에피소드마다 바뀜. 모든 모델이 같은 20개 장면 |
| 에피소드 | 20개, 최대 45초. busybox가 버튼 눌림을 자동 채점 (성공 시 즉시 종료) |
| 카메라 | 3대: top, wrist, front |
| 언어 지시 | "push the green button" (π0.5, SmolVLA) |
| 학습 GPU | snoopy1 NVIDIA RTX A6000 48GB × 1 (Slurm) |
| 추론 | 셀 쪽 머신의 GPU에서 실행 (`policy_device=cuda`) |

데모 데이터 (ACT, SmolVLA 학습용): `villekuosmanen/busybox_push_green_button` — 39 에피소드, 4,836 프레임, 3 카메라.

## 결과 요약

| 모델 | 구조 | 학습 데이터 | 학습 시간 | 성공률 (20 장면) |
|---|---|---|---|---|
| π0.5 base | VLA, flow matching | busybox_multitask 데모 (사전 학습) | — (받아서 사용) | **4/20 (20%)** |
| ACT `010000` | CVAE + Transformer, ResNet18 | green button 데모 39개 | 1h43m | **5/20 (25%)** |
| ACT `030000` | 동일 | 동일 | 5h09m | **5/20 (25%)** |
| SmolVLA `005000` | VLA (SmolVLM2 500M) + flow-matching action expert | green button 데모 39개 | 2h11m | **10/20 (50%)** |
| π0.5 + RLT (streaming BC, frozen) | π0.5 고정 + RL-token actor MLP | 사람 데모 없음. 셀 롤아웃 90회 중 성공 32회만 BC | ~1h22m (롤아웃 90회 포함) | **13/20 (65%)** |
| π0.5 + RLT (TD3 온라인, frozen) | π0.5 고정 + RL-token actor MLP + critic 4개 | green button 데모 39개 + 셀 롤아웃 90회 (성공/실패 모두) | ~1h40m (데모 캐시 10분 + 롤아웃 90회, 연결 중단 약 10분 포함) | **9/20 (45%)** |

### 장면별 결과 (✓ 성공, · 실패)

| 장면 | 레일 | π0.5 base | RLT (streaming BC) | RLT (TD3) | ACT 10k | ACT 30k | SmolVLA 5k |
|---|---|---|---|---|---|---|---|
| 0 | 0.724 | ✓ | ✓ | ✓ | · | · | ✓ |
| 1 | 0.659 | · | ✓ | · | ✓ | ✓ | ✓ |
| 2 | 0.724 | ✓ | ✓ | · | · | · | ✓ |
| 3 | 0.779 | · | ✓ | ✓ | · | · | · |
| 4 | 0.703 | · | ✓ | ✓ | · | · | ✓ |
| 5 | 0.718 | · | ✓ | · | · | · | ✓ |
| 6 | 0.704 | · | · | · | · | · | ✓ |
| 7 | 0.797 | · | ✓ | ✓ | · | · | · |
| 8 | 0.768 | · | · | · | · | · | · |
| 9 | 0.649 | · | · | · | ✓ | ✓ | ✓ |
| 10 | 0.690 | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| 11 | 0.747 | · | ✓ | · | · | · | · |
| 12 | 0.531 | · | · | · | · | · | · |
| 13 | 0.819 | · | · | ✓ | · | · | · |
| 14 | 0.661 | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| 15 | 0.574 | · | · | · | · | · | · |
| 16 | 0.794 | ✓ | · | ✓ | · | · | · |
| 17 | 0.783 | · | ✓ | · | · | · | · |
| 18 | 0.682 | · | ✓ | ✓ | ✓ | ✓ | ✓ |
| 19 | 0.724 | ✓ | ✓ | · | · | · | · |
| **합계** | | **4** | **13** | **9** | **5** | **5** | **10** |

### 레일 위치별 성공

| 레일 구간 | 장면 수 | π0.5 base | RLT (streaming BC) | RLT (TD3) | ACT | SmolVLA |
|---|---|---|---|---|---|---|
| < 0.60 | 2 | 0 | 0 | 0 | 0 | 0 |
| 0.64 – 0.69 | 5 | 0 | 4 | 3 | 5 | 5 |
| 0.70 – 0.75 | 7 | 3 | 6 | 2 | 0 | 5 |
| ≥ 0.76 | 6 | 1 | 3 | 4 | 0 | 0 |

- ACT는 레일 0.65–0.69에서만 성공 → 데모 분포 근처만 외운 것으로 보임. 10k와 30k가 장면별로 완전히 같음.
- SmolVLA는 0.70–0.75까지 일반화, 0.76 이상은 실패.
- 0.76 이상에서는 RLT 두 가지만 성공 (streaming BC 3/6, TD3 4/6). TD3는 레일 0.819(장면 13)를 성공한 유일한 모델.
- TD3는 대신 0.70–0.75 구간이 2/7로 약함 (streaming BC 6/7, SmolVLA 5/7).
- 0.60 미만은 모든 모델 실패.

---

## 1. π0.5 base

**구조**
- π0.5 VLA: PaliGemma 계열 VLM + flow-matching action expert. action horizon 30, 3 카메라 입력, delta action.
- 체크포인트: Armnet 볼륨 `openpi/checkpoints/pi05_rlt_joint_succ_20261010`.

**체크포인트 계보** (HF 모델 카드 기준)
1. `pi05_base` (Physical Intelligence 파운데이션)
2. → `pravsels/pi05_busybox_multitask_minmax`: busybox_multitask 66 에피소드로 30k 스텝 파인튜닝
3. → `AutoRLBench/pi05_rlt_busybox_multitask_singlearm_minmax`: RLT stage 1 (VLA 고정, RL-token 인코더 20k 스텝)
4. 실제 사용한 `pi05_rlt_joint_succ_20261010`은 문서가 없어 어떻게 만들어졌는지 확인 안 됨.

**실행**
- RLT frozen 평가 잡(`4a397524`, 10-12 18:56–19:28, cell-01)에서 RLT와 같은 장면에 번갈아 실행 (`eval_include_base=true`, exploration 0).

**결과**: 4/20 (20%).

---

## 2. π0.5 + RLT (RL-Token, streaming BC)

**구조**
- π0.5는 고정. VLA가 만든 RL token(2048차원) + 관절 상태(6) + π0.5의 reference action chunk(30스텝)를 입력으로 받는 **actor MLP** (LayerNorm + 512×4, SiLU)가 10스텝 action chunk를 출력.
- 원래 RLT는 TD3(critic 포함)로 온라인 강화학습하지만, 이 실행은 **streaming BC**: 셀에서 성공한 롤아웃만 모아 actor를 behaviour cloning. critic은 학습하지 않음. 원래 방식(TD3)은 3절.

**어떻게 돌렸나**
- Learner: snoopy1 A6000 1장 (`rlt-local streaming-bc`), ngrok 터널로 셀의 actor와 연결.
- Actor 잡 `821c48a4`, cell-01, 10-12 09:01–10:23 (82분).
- 롤아웃 90회, `variation_seed=42` (평가 장면과 다름), exploration 1.0, 에피소드 45초.
- 라운드 스케줄 30, 30, 10, 10, 10 에피소드 → 라운드마다 300 스텝, batch 256 → 총 1,500 스텝.
- 결과 체크포인트 `step_001500` → `rlt-freeze-checkpoint`로 actor만 고정 → 평가.

**학습 중 성공률** (그 시점 정책으로 수집)

| 롤아웃 | 정책 스텝 | 성공 |
|---|---|---|
| 1–30 | 0 (초기) | 8/30 |
| 31–60 | 300 | 11/30 |
| 61–70 | 600 | 4/10 |
| 71–80 | 900 | 3/10 |
| 81–90 | 1200 | 6/10 |

최종 BC loss 0.35, 성공 transition 914개 (성공 에피소드 32개).

**평가**: 잡 `4a397524`, cell-01, seed 1042, π0.5 base와 같은 장면에서 순서를 번갈아 실행.

**결과**: **13/20 (65%)**. 같은 장면에서 base는 4/20.

**이전 시도** (참고)
- 10-08 ~ 10-10: Modal learner + cell-03(이전 세팅)으로 streaming BC를 시도.
- 10-08 잡 `935bcf5d`: 14 롤아웃 중 3 성공.
- 10-10 잡 `6961f655`: 49 롤아웃 중 4 성공, 90회를 채우지 못하고 종료.
- 그 사이 다수 잡이 셀 연결/설정 문제로 시작 단계에서 실패.

---

## 3. π0.5 + RLT (RL-Token, TD3 온라인 — 원래 방식)

**구조**
- streaming BC와 같은 actor MLP(RL token 2048 + 상태 6 + reference chunk 30스텝 → 10스텝 chunk)에 **critic 4개(ensemble)** 추가.
- π0.5 `pi05_rlt_joint_succ_20261010`은 고정 (streaming BC·π0.5 base 평가와 같은 체크포인트).
- actor/critic은 무작위 초기화에서 시작 (streaming BC 결과를 이어받지 않음). 커리큘럼 없음.
- TD3 설정: discount 0.985, UTD 5, batch 256 (온라인 50% + 데모 50%), exploration 1.0.

**학습 과정**
1. 데모 캐시: green button 데모 39개(ACT·SmolVLA와 같은 데이터)를 셀과 같은 π0.5 체크포인트로 인코딩 → 499 transition. snoopy1 A6000, 약 10분.
2. 데모 사전학습: actor+critic TD3 5,000 스텝 (1~2분, 로봇 없음).
3. 온라인 학습: cell-01 롤아웃 90회 (seed 42, 45초, 실패 장면 1회 재시도) — 성공/실패 롤아웃 모두 보상으로 학습. 최종 학습 스텝 10,636.
   - 로컬 학습 서버(`rlt-local online`, 이번에 추가) + ngrok 터널.
   - 롤아웃 12회 후 ngrok 월 대역폭 한도 초과로 연결 끊김(03:44) → 유료 전환 후 같은 학습 서버를 `--resume`(스텝 5822, transition 822개 복원)으로 이어서 나머지 78회 진행. 끊긴 동안의 롤아웃 2회는 학습에 반영되지 않아 제외.
- 잡: `1a367922`(12회, 10-15 03:33–03:45), `25a6cccf`(78회, 03:56–05:09), 모두 cell-01. `--armnet.require-cell cell-01` 사용 (다른 셀이면 로봇 구동 전에 실패).

**학습 중 성공률** (streaming BC와 같은 구간)

| 롤아웃 | TD3 | streaming BC |
|---|---|---|
| 1–30 | 8/30 | 8/30 |
| 31–60 | 15/30 | 11/30 |
| 61–90 | 5/30 | 13/30 |
| 합계 | 28/90 | 32/90 |

61–90 구간은 레일 0.64 미만 장면이 몰리고 재시도로 반복된 영향이 큼 (재개 후 롤아웃 기준 레일 0.64 미만 3/35, 0.64–0.76 18/27).

**평가**: 잡 `18d601b1`, cell-01, 10-15 05:18–05:34, seed 1042, exploration 0, base 재실행 없음. 데이터셋 `LeeHakHo/eval_rlt_push_green_button_05h19m14s_15-oct-2026`.

**결과**: **9/20 (45%)**.
- 레일 ≥0.76에서 4/6으로 모든 모델 중 최고. 장면 13(레일 0.819)은 이 모델만 성공.
- 대신 0.70–0.75에서 2/7로 streaming BC(6/7)·SmolVLA(5/7)보다 약함.
- 학습·평가 내내 hardware clamp 0회, reference 대비 deviation 평균 약 1 (평가), 버튼 오조작은 일부 있음.

---

## 4. ACT

**구조**
- LeRobot ACT: ResNet18 비전 백본 + Transformer (encoder 4층, decoder 1층, d=512) + CVAE.
- chunk 100 / 실행 100 스텝, 사전 학습 없음, 52M 파라미터.

**학습**
- LeRobot 0.5.1 `lerobot-train`, `tools/train-act.sh`.
- batch 8, lr 1e-5, 40k 스텝 예정 → 38k에서 중단.
- 10-12 20:34 시작. `010000` 22:17 (1h43m), `020000` 00:00, `030000` 01:42 (5h09m).

**평가 실행**
- `armnet-lerobot-eval` 런타임. 셀이 dict로 보내는 카메라 설정을 `OpenCVCameraConfig`로 바꾸는 패치를 넣은 이미지 `armnet-lerobot-eval:armnet-0ad12dbae464` 사용.
  - 패치 없이는 `'dict' object has no attribute 'width'`로 실패.
  - 이미지는 Mac(colima)에서 빌드. snoopy1에서는 Docker 사용 불가.
- `030000`: 잡 `ab3a18aa`, 10-13 09:03, cell-01.
- `010000`: 잡 `7f4def18`, 10-13 17:59, cell-01. SmolVLA와 학습 시간을 맞추려고 2시간 버전으로 평가.

**결과**: 010000 5/20, 030000 5/20. 장면별 결과가 동일 → 더 학습해도 개선 없음.

---

## 5. SmolVLA

**구조**
- LeRobot SmolVLA: SmolVLM2-500M VLM(비전 인코더 고정) + flow-matching action expert.
- 총 450M 파라미터 중 action expert 100M만 학습. chunk 50 / 실행 50 스텝.
- `lerobot/smolvla_base`에서 시작 (커뮤니티 SO-100/101 데이터로 사전 학습된 모델).
- 동료 제안("원래 데모로 작은 flow policy를 순수 supervised로 학습, steering head 없이")을 검증하려고 실행.

**학습**
- `tools/train-smolvla.sh` (sbatch): batch 64, lr 1e-4, 20k 스텝 예정.
- 카메라 이름 매핑: top/wrist/front → camera1/2/3.
- 첫 시도는 host RAM OOM(데이터로더가 720p 영상 3개 디코딩) → workers 8, 메모리 128G로 재실행. 피크 약 106GB.
- 10-13 09:33 시작. `005000` 11:44 (2h11m), `010000` 13:54, `015000` 16:03. 약 18k에서 중단 (loss가 0.007까지 떨어져 과적합 우려).

**평가 실행**
- ACT와 같은 이미지. `language_instruction="push the green button"`.
- `005000`: 잡 `d4a71009`, 10-13 17:30, cell-01.

**결과**: **10/20 (50%)**.
- 10-13 19:05에 같은 체크포인트가 의도치 않게 cell-01에서 한 번 더 20 에피소드 실행됨 (아래 이슈 참고). 채점 기록은 없고, 에피소드 길이로 보면 약 11/20 → 재현성은 비슷.
- `010000`, `015000`은 아직 평가 안 함.

---

## cell-03 상태

- 새 세팅(서보 카메라 마운트 등) 이후 평가 잡이 시작 단계에서 실패:
  ```
  armnet-runtime: failed to build context: OpenCVCameraConfig.__init__() got an unexpected keyword argument 'servo_mount'
  ```
- 셀이 보내는 카메라 설정의 새 항목 `servo_mount`를 공개 런타임(`armnet-runtime` 0.3.2, PyPI 최신)이 처리하지 못함. 10-13, 10-14에 세 번 재시도, 모두 같은 오류.
- 필요한 것: `servo_mount`를 지원하는 런타임 버전, 셀을 지정해 잡을 보내는 정식 방법 (JobSpec에 셀 지정 필드 없음).

## 운영 이슈 / 메모

- **취소한 잡이 실제로 실행됨 (10-13 19:05)**: cell-03 평가를 위해 cell-01에 배정되면 바로 취소하는 방식을 썼는데, 서버엔 `cancelled`로 기록됐지만 셀은 취소를 받지 못하고 20 에피소드를 무인으로 실행 → 이 방식은 사용 중단.
- **데이터셋 업로드 위치**: 평가/롤아웃 데이터셋이 `AutoRLBench` 조직이 아니라 개인 계정 `LeeHakHo/` 아래 private으로 올라가고 있음 (`hf_user` 미지정 → 토큰 소유자 이름 사용). 조직으로 올리려면 `--hf-user AutoRLBench` / `--armnet.hf_user=AutoRLBench` 지정 필요.
- **ngrok 대역폭 한도 (10-15 03:44)**: 무료 플랜 월 대역폭 초과(ERR_NGROK_725)로 학습 서버 연결이 끊김. 끊긴 뒤에도 셀 actor는 마지막 가중치로 롤아웃을 계속하므로 직접 멈춰야 함. 유료 전환 후 해결.
- **셀 지정**: RLT actor/eval은 `--armnet.require-cell cell-01`로 지정 (컨테이너 시작 시 검사, 다른 셀이면 로봇 구동 전 실패). 두 셀이 비어 있으면 cell-03에 먼저 배정되는 경우가 많아 재제출이 여러 번 필요.
- **잡 정지**: `stop_job`이 HTTP 500을 돌려줘도 실제로는 취소됨 (cell-01이 바로 `available`로 돌아온 것 확인).

## 다음 후보

- SmolVLA `010000`, `015000` 평가 (과적합 여부 확인).
- TD3 RLT 커리큘럼 켠 원래 레시피(`--variation-curriculum`)로 재실행.
- cell-03 런타임 해결 후 cell-03에서 같은 비교.
- 레일 0.60 미만 / 0.76 이상 장면을 겨냥한 데이터 수집.
