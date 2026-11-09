# Physics-Guided GNN 입자 시뮬레이터

유체-강체 상호작용을 그래프 신경망으로 예측하는 연구 코드다. JKIIS 게재확정 논문과
ICLR 2027 투고본의 수치가 이 저장소의 스크립트에서 나왔다.

## 방법

입자 상태(위치, 속도, 유형)를 반경 그래프로 묶고, 그래프 신경망이 다음 스텝의 정규화
속도 변화량을 예측한 뒤 Forward Euler로 적분한다. 강체와 벽은 예측 대상이 아니라 매
스텝 정답 위치로 다시 놓이는 단방향 경계조건이다.

MeshGraphNet 백본에 두 가지를 더했다.

- 어텐션 게이트 alpha_ij: 엣지마다 학습되는 스칼라 가중치
- 유형 가중치 w_ij (beta=2.0): 경계 입자가 걸린 엣지의 메시지를 상수배로 키운다. 학습 파라미터가 없다.

## 결과

Water-3D, 400스텝 롤아웃, 시드 3개 평균이다. 이탈률은 도메인 밖으로 나간 입자 비율,
`|KE-1|`은 초기 운동에너지 대비 편차다.

| 구성 | 이탈률 (%) | \|KE-1\| |
|---|---|---|
| 보정 벽 + 유형 가중치 (제안) | **1.29 ± 0.32** | 11.9 |
| 같은 벽 + 백본만 | 13.89 ± 5.96 | 103 |
| GNS 거리 특징 (메타데이터 박스) | 35.81 ± 8.32 | 155 |
| 메타데이터 박스 위에 벽 | 74.04 ± 9.41 | 1551 |
| 벽 없음 | 86.96 ± 5.68 | 559 |

학습 파라미터는 제안 모델 557,453개, GNS 재구현 1,590,787개다.

모델과 학습 예산을 고정한 채 경계 입자의 배치만 바꾸면 결과가 위 표만큼 벌어진다.
메타데이터에 적힌 경계(0.125)에 벽을 두면 유체가 벽 바깥에 존재해 장기 롤아웃이
붕괴한다. 실측 용기 위치에 여유를 주고 두 겹으로 쌓은 구성만 유효하다.

자체 데이터(30 시뮬레이션, 77스텝)에서는 수조 벽 유출이 8~12%에서 0%로 떨어졌고
검증 손실은 0.05718이다.

## 구조

```
src/models/
  hybrid_gnn_v4.py     제안 모델 (557,453 파라미터, 10층)
  gns_baseline.py      GNS 재구현 (1,590,787 파라미터)
  hybrid_gnn_v6.py     WaterRamps 변형
  gnn.py, gat.py, hybrid_gnn.py, hybrid_gnn_tuned.py    v3p2 이전 계보
src/simulation/
  rigid_body.py, dynamic_rigid_body.py   강체 운동
  flip_simulator.py    격자 솔버 (논문 데이터 생성에 쓰이지 않았다)

scripts/data_generation/
  v3p2_20260506_043045_generate_sph_with_rigid_bodies.py   자체 데이터 생성기
  gns_tfrecord_reader.py                                   TFRecord 파서
  v5_convert_water3d.py, v6_20261021_145750_convert_waterramps.py
scripts/training/
  v4_20260707_102544_train_v4.py        기본 드라이버. 나머지가 이것을 import 한다
  v4p1_20260712_001555_train_wall.py    벽 입자 온라인 증강
  v4p2_20260824_train_split.py          시뮬레이션 단위 분할, 유형 홀드아웃
  v5_20260820_022909_train_water3d.py   Water-3D
  v5p1_20260903_train_gns.py            GNS 베이스라인
  v6_20261021_145750_train_waterramps.py
scripts/validation/
  v4_20260707_102544_rollout_eval.py    77스텝 롤아웃 4지표
  v4_20260707_102544_bc_compliance.py   경계조건 준수
  v5_20260820_022909_rollout_water3d.py Water-3D 롤아웃
  v5p3_20260926_cost_profile.py         계산 비용
scripts/visualization/
```

## 실행

PyTorch 2.12.1+cu126, Python 3.13, CUDA GPU 1개. PyTorch Geometric 대신 `index_add_`로
직접 구현했고, TensorFlow 없이 TFRecord를 파싱한다. 의존성은 `requirements.txt`에 있다.

```bash
# 자체 데이터
python scripts/data_generation/v3p2_20260506_043045_generate_sph_with_rigid_bodies.py
python scripts/training/v4p1_20260712_001555_train_wall.py --tag wall_hybrid
python scripts/validation/v4p1_20260712_001555_rollout_wall_eval.py

# Water-3D
python scripts/data_generation/v5_convert_water3d.py
python scripts/training/v5_20260820_022909_train_water3d.py --tag w3d_wall2
python scripts/validation/v5_20260820_022909_rollout_water3d.py
```

학습 드라이버 인자: `--tag`, `--noise-std`, `--pushforward-prob`, `--no-attention`,
`--no-type-weighting`, `--resume-from`, `--resume-log`. 절제 실험은 뒤의 두 플래그로
구성을 끈다.

평가는 네 지표(`vel_mse`, `pos_mse`, `|KE-1|`, `mom_drift`)와 경계조건 준수를 같이
본다. 속도 MSE만 보면 에너지를 소멸시키는 과감쇠 모델을 좋게 평가하게 된다. GPU
`index_add_`가 비결정적이라 롤아웃은 3회 평균으로 보고한다.

데이터셋(111 GB), 체크포인트(396 MB), 로그(42 MB)는 담지 않았다. 자체 데이터는 위
생성기로 만들고, Water-3D와 WaterRamps는 GNS 공개 저장소에서 받는다.

## 한계

- 공개 벤치마크 검증은 Water-3D 한 종류다. WaterRamps 확장은 `v6_...` 스크립트로 진행 중이다.
- GNS 베이스라인은 80k 스텝 예산으로만 학습했다. 원 논문은 20M 스텝이고 학습 곡선이 아직 수렴하지 않았다.
- 표본은 궤적 3개, 시드 3개다.
- 위치 MSE는 GNS가 더 낮다(0.19 대 0.30). 다만 그 시점 GNS 입자의 36%가 도메인을 이탈한 상태다.
- 자체 SPH 솔버는 코드 실측값(K=150, rho_0=400, 입자질량 0.10, h=0.1, 점성 0.1, 중력 -9.81, dt=0.01)을 쓴다. 표면장력 항은 구현되어 있지 않다.

## 라이선스

MIT. 저작권자는 제주대학교 Soft Artificial Life Laboratory 박승민이다.
MIT는 특허권을 주지 않는다.
