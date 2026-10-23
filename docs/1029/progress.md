# 1029 실험 진행 상황 (2026-10-16 ~ 10-22 기준)

`docs/1015`(cell-01, 20 장면 비교) 이후 cell-03으로 옮겨 진행한 내용. 시각은 모두 UTC.

## 공통 조건 (cell-03)

| 항목 | 값 |
|---|---|
| 셀 | **cell-03** (armnet-runtime 0.3.3, `--armnet.require-cell cell-03`) |
| 평가 | 10 롤아웃, exploration 0, rail 0.7, lightbox 30, 에피소드 45초, busybox 자동 채점 |
| 학습·추론 GPU | snoopy1 A6000 1장 (Slurm). 정책 서버도 같은 노드 (`--remote-policy`) |
| 데모 데이터 | `AutoRLBench/so101_busybox_push_green_button_cell03_clean250_426x240` — 250 에피소드, 62,566 프레임, 20fps |

## 결과 요약

| 모델 | 학습 방식 | 데이터 | 스텝 | 성공률 (cell-03, 10 롤아웃) |
|---|---|---|---|---|
| π0.5 + RLT streaming BC (`hyeonhoo-local-20261018-181329`) | 온라인. 성공 롤아웃만 BC | 사람 데모 없음. 롤아웃 640회 중 성공 339회 (reject 72) | 30,000 (학습기) | **6/10** |
| Multi-task DiT (`dit-clean250-20261020-1903`) | 오프라인 lerobot-train | clean250 | 30,000 | **10/10** |
| Diffusion Policy (`diffusion-clean250-20261021-0915`) | 오프라인 lerobot-train | clean250 | 30,000 | **8/10** |
| SmolVLA (`smolvla-clean250-20261021-0627`) | 오프라인 lerobot-train (`smolvla_base`에서) | clean250 | 20,000 | **4/10** |
| ACT (`act-clean250-20261021-1325`) | 오프라인 lerobot-train | clean250 | 30,000 | **0/10** |
| ACT streaming BC (`hyeonhoo-act-20261022-083946`) | 온라인. ACT 30k에서 시작, 성공 롤아웃 + clean250 50% 혼합 | 롤아웃 650회 중 성공 69회 | 30,000 (20,500에서 재개) | **0/10** (20.5k, 25k, 30k 모두) |
| π0.5 + RLT TD3 (`hyeonhoo-local-td3-20261022-173022`) | 온라인 RL | clean250 데모 캐시 + 롤아웃 | 16,450에서 중단 (오케스트레이터 500) | 미평가 |

### 체크포인트별 성공률 (오프라인 모델, `checkpoint-eval`)

| 스텝 | DiT | Diffusion | SmolVLA | ACT |
|---|---|---|---|---|
| 5,000 | 5/10 | 5/10 | 4/10 | 0/10 |
| 10,000 | 10/10 | 3/10 | 2/10 | 0/10 |
| 15,000 | 9/10 | 6/10 | 5/10 | 0/10 |
| 20,000 | 6/10 | 4/10 | 4/10 | 0/10 |
| 25,000 | 8/10 | 7/10 | — | 0/10 |
| 30,000 | **10/10** | **8/10** | — | 0/10 |

- DiT 30k는 평균 6.9초로 성공 (다른 모델 20~35초).
- ACT는 temporal ensemble 0.01(`te001`)로도 30k 0/10. 전부 45초 타임아웃.

## 1. 데이터: clean250

- 출처: streaming BC 10-18 실행의 롤아웃 640회 (`LeeHakHo/eval_rlt_push_green_button_18h14m36s_18-oct-2026`).
- 성공 에피소드 중 250개 선택 (`experiment-data/datasets/clean250/selection.json`), 390개 삭제. 10-19 편집, 10-20 426x240으로 재인코딩.
- 즉 **사람 데모 없이 이전 RLT 정책이 모은 성공 데이터**로 오프라인 baseline을 학습함.

## 2. π0.5 + RLT streaming BC (cell-03)

- 10-17 첫 실행 `hyeonhoo-local-20261017-014504`: 롤아웃 363회, 성공 146회. 30k 전에 중단.
- 10-18 `hyeonhoo-local-20261018-181329`: `--episodes-per-round 30,30,10 --steps-per-round 500 --online-steps 30000 --min-shaped-reward 1.0`. 60 라운드, 640 롤아웃, 성공 339회 (BC에 쓴 성공 에피소드 267, reject 72), 성공 transition 7,306. 최종 BC loss 0.31.
- 평가: 10-22 08:27, step_030000 frozen actor, **6/10**.

## 3. 오프라인 baseline (lerobot-train, clean250)

- 공통: batch 32, workers 8, save 5k마다. DiT는 `image_resize_shape [240,240]`. SmolVLA는 batch 64, 20k, 카메라 top/wrist/front → camera1/2/3.
- 평가는 `rlt-local checkpoint-eval`: 체크포인트마다 base 대비 달라진 텐서만 셀로 push, 10 롤아웃, `checkpoint_success.jsonl` + wandb(`armnet_snoopy`).
- 10-21 cell-eval 로그: `experiment-data/checkpoint-eval/*-cell-eval/`.

## 4. ACT streaming BC

- `rlt-local act-streaming-bc`: ACT 30k 체크포인트에서 시작, 라운드마다 500 스텝, batch 32, clean250을 배치의 50%로 섞음.
- 10-21 `234212`: 380 롤아웃, 성공 55, 7,500 스텝에서 actor idle로 종료 (셀 잡 cancelled, exit 137).
- 10-22 `083946` (queue-30000.sh 1단계): 460 롤아웃, 성공 67, 20,500 스텝. 17:29 셀 잡이 데이터셋 마무리에서 실패:
  ```
  lerobot_dataset.py:404 has_pending_frames: KeyError: 'size'
  ```
  30k 체크포인트가 없어 queue가 평가를 건너뜀. 구간별 성공: 1–50 26회, 51–100 13회, 이후 50회당 2~9회 → 학습 중 성공률이 오히려 떨어짐.
- 10-22 21:36 `tools/queue-act-sbc-30000.sh`: 20,500 평가 **0/10** → `--resume`으로 30,000까지 (190 롤아웃, 성공 2) → 25,000 **0/10**, 30,000 **0/10**. 10-23 02:01 완료.
- 학습 중 성공률 (정책 스텝 기준): 0–2.5k 39% → 2.5k–10k 12~20% → 10k–20k 2~4% → 20k–30k 0~3%. 실패는 전부 timeout, wrong button 0.
- 원인 분석 (10-23):
  - 데이터 포맷은 clean250과 일치 (fps, state/action 의미, 이미지 크기·색). loss 0.12 안정, grad norm 정상. 망각도 아님: clean250 eval l1이 0.196 → 0.167로 유지되고 온라인 데이터 l1은 0.226 → 0.097.
  - ACT의 "성공"은 평균 32초짜리(clean250은 12.5초) 머뭇거리는 궤적이고, exploration 0.1 노이즈가 버튼을 눌러준 것. 노이즈는 zero-mean이라 BC로 회귀하면 정책 자신의 "버튼 앞에서 머뭇거리기"만 남고, 느린 궤적이 배치의 절반을 차지해 45초 timeout이 늘어남. 성공이 줄면 새 데이터가 안 들어와 69개에만 계속 맞추는 악순환.
  - 같은 이유로 π0.5 streaming BC도 50%에서 평평함: 자기 성공으로 BC하면 시작 정책보다 좋아질 동력이 없음.
- 참고: ACT 30k 가중치 그대로 exploration 0.1로 돌린 첫 50 롤아웃은 26/50인데 checkpoint-eval(exploration 0)은 0/10. ACT 30k를 exploration 0.1로 재평가하면 "ACT는 안 된다"가 평가 설정 문제인지 확인 가능.

## 5. π0.5 + RLT TD3 (cell-03)

- 10-20 `hyeonhoo-local-td3-20261020-082607`: 롤아웃 90회, 성공 53회 (ngrok 재연결 1회). 중간 평가 step 6500 **8/20**. step 7000 평가는 20회 중 16회만 busybox 채점, 나머지 scored_by None → 0/20은 채점 실패로 무효.
- 10-22 17:30 `hyeonhoo-local-td3-20261022-173022` (queue-30000.sh 3단계, Slurm 68928): 데모 사전학습 5,000 + 온라인 25,000 목표. 21:30 오케스트레이터가 HTTP 500을 반환해 actor 잡이 끊김. 큐가 5회 재시도했지만 셀이 비지 않아 모두 QUEUED로 끝났고 6회 한도에서 학습기 scancel. 최종 16,450 스텝, 롤아웃 531회, 성공 72회 (30회 단위 2~7회로 평평). 체크포인트 step_016000까지, 평가 안 함.

## 6. DSRL (π0.5 노이즈 steering, 10-23 구현)

- Wagenmaker 외 2025 (arXiv 2506.15799). π0.5 가중치는 고정하고, actor가 RL 토큰 + 관절 상태 → flow의 초기 노이즈 (30×6, ±2 클립)를 내면 π0.5가 그 노이즈로 denoise한 청크를 로봇이 실행. 행동이 항상 π0.5 분포 안에 머무는 것이 RLT TD3(별도 행동 정책)와의 차이.
- 구현: `--algorithm dsrl` (learner `rlt-local online`, actor `rlt-submit-actor`). `src/armnet_rlt/dsrl.py`에 actor/critic/loss, 정책 서버에 `SetSteering` RPC와 prefix/denoise 분리 JIT. actor는 정책 서버 안에서 평가되어 청크당 왕복 1회 유지. transition의 `action`은 노이즈(180차원), `reference_action`은 그 노이즈로 만든 청크. 데모 캐시·사전학습 없음, critic은 RLT와 같은 TD3 앙상블.
- 실행: `tools/queue-dsrl.sh` (learner+정책 서버 → actor 1200 롤아웃, exploration std 1.0 → 최신 체크포인트 freeze → 10 롤아웃 평가). actor 이미지는 `dsrl.py`가 들어간 새 이미지가 필요.

## 다음 후보

- 새 actor 이미지 빌드 후 `tools/queue-dsrl.sh` 실행.
- TD3를 16k에서 재개할지 결정. 성공률이 평평하므로 exploration·커리큘럼 조정 없이는 의미 없음.
- ACT 30k를 exploration 0.1로 재평가 (위 참고).
- 모든 모델을 같은 20 장면(variation seed)으로 재평가. 현재는 rail 0.7 고정 10회라 10-15 표와 직접 비교 불가.
- `docs/1022/cell03_check.md`가 삭제된 상태(미커밋). 되살릴지 결정.
