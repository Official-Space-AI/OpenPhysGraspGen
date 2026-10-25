# OpenPhysGraspGen

[English README](README.md)

OpenPhysGraspGen은 **변형체(deformable object)** 파지 데이터를 물리 라벨과 함께
생성하는 라이브러리입니다. 측정값으로 재료 모델을 보정(calibration)하고, 보정한
파라미터를 데이터 생성용 시뮬레이터로 옮긴 뒤, 파지마다 접촉력과 내부 응력을
기록합니다. 정책이 "어디를 잡는가"뿐 아니라 "얼마나 세게 쥐고 있는가"까지 학습할
수 있게 하는 것이 목적입니다.

SpaceAI PhysGraspGen 프로젝트의 공개 부분이며, 순수 Python·NumPy로 작성되어
있습니다. 센서 레이어는 [NVIDIA Newton](https://github.com/newton-physics/newton)
(VBD solver)의 접촉 버퍼를 읽습니다.

> **현재 상태: 공개 베타(`0.1.0b0`).** 마이너 버전 사이에 API가 바뀔 수 있습니다.
> 릴리스 후보(RC)에 들어갈 내용은 [로드맵](#로드맵)을 보세요.

## 구성

| 모듈 | 기능 |
|---|---|
| `constitutive` | 초탄성 구성방정식(Yeoh, Mooney–Rivlin, 3항 Ogden): 변형 에너지, Cauchy 응력, von Mises 응력 |
| `formulations`, `generation` | 측정 응답에 맞춰 재료 파라미터 보정, 보정 모델에서 생산(데이터 생성) 모델로 파라미터 전이, 파손 라벨이 붙은 학습 샘플 생성 |
| `estimation`, `confidence` | 로봇 관측에서 재료와 내부 응력을 함께 추정하고, 추정이 믿을 만하지 않으면 동작 크기를 제한하는 confidence gate |
| `sensors` | `DeformableContactSensor`, `SensorRecorder`: Newton 시뮬레이션 프레임에서 접촉점별 힘과 그리퍼·바닥 합력 라벨 생성 |
| `records` | 위 단계들이 주고받는 데이터 레코드 |

고정밀 솔버 코어, 고객 데이터, 내부 도구는 이 저장소에 포함되지 않습니다.
[공개 범위](#공개-범위)를 참고하세요.

## 설치

Python 3.10 이상이 필요합니다. 패키지 이름은 `openphysgraspgen`이고, `1.0.0rc1`부터 PyPI에 올라갑니다.

```bash
pip install openphysgraspgen
```

그 전까지는 GitHub에서 설치합니다.

```bash
pip install "git+https://github.com/Official-Space-AI/OpenPhysGraspGen.git"
```

## 빠른 시작

보정 → 전이 → 데이터 생성 → 추정 → confidence gate 전체 흐름을 작은 합성 문제로
실행합니다.

```bash
openphysgraspgen --fracture-threshold 0.25 --confidence-threshold 0.5
```

구성방정식을 직접 계산하는 예시와 Newton 접촉 라벨 기록 예시는
[영문 README](README.md#quick-start)에 있습니다.

## 로드맵

릴리스는 과제 계획을 따릅니다.

| 릴리스 | 목표 시점 | 내용 |
|---|---|---|
| 베타 `0.1.0b0` | 2026년 8월 | 구성방정식, 접촉 센서, 보정·전이 파이프라인 (2026-08-25 공개) |
| `1.0.0rc1` | 2026년 10월 | 파지 데이터 recorder, PyPI 패키지, Hugging Face 공개 데이터셋(100만 프레임)과 검사 스크립트, 데이터셋으로 하는 PG-T1 벤치마크 재현, Docker 이미지, 모든 릴리스의 CI와 SBOM |
| `1.0.0` | 2026년 11월 | 벤치마크, 튜토리얼, 라이선스 검증 |

## 공개 범위

| 구분 | 위치 | 라이선스 |
|---|---|---|
| 라이브러리 (이 저장소) | GitHub | Apache-2.0 |
| 파지 데이터셋 | Hugging Face Hub (`1.0.0rc1`부터) | CC-BY-4.0 |
| 벤치마크와 재현 스크립트 | 이 저장소 (PG-T1은 `1.0.0rc1`, 나머지는 `1.0.0`) | Apache-2.0 |
| 고정밀 솔버 코어 | 비공개 | 상용 |

## 개발 방식

PhysGraspGen의 일상 개발은 고객별 코드가 함께 있는 비공개 저장소에서 이루어집니다.
이 저장소는 그중 범용 부분을 리뷰를 거친 pull request와 태그된 릴리스로 받기 때문에,
커밋 이력이 프로젝트 전체 이력보다 짧습니다. 앞으로 공개 라이브러리 작업은 이곳에서
공개적으로 계획하고 리뷰합니다. 모든 변경은 이슈에서 시작해 리뷰를 거친 pull
request로 들어오고, 릴리스는 릴리스 노트와 SBOM을 붙여 배포합니다.

## 커뮤니티와 지원

- 질문·아이디어: [GitHub Discussions](https://github.com/Official-Space-AI/OpenPhysGraspGen/discussions)
- 버그: [GitHub Issues](https://github.com/Official-Space-AI/OpenPhysGraspGen/issues)
- 어디에 무엇을 물어야 하는지, 응답 시간: [COMMUNITY.md](COMMUNITY.md)
- 보안 취약점: [SECURITY.md](SECURITY.md) (공개 이슈로 올리지 마세요)

기여는 [CONTRIBUTING.md](CONTRIBUTING.md)부터 읽어 주세요. 참여하는 모든 사람은
[행동 강령](CODE_OF_CONDUCT.md)을 따릅니다. 역할과 결정 방식, 릴리스 승인 절차는
[GOVERNANCE.md](GOVERNANCE.md)에 있습니다. 질문과 이슈는 한국어로 써도 됩니다.

## 라이선스

코드는 [Apache License 2.0](LICENSE), 데이터셋은 CC-BY-4.0으로 공개합니다.
