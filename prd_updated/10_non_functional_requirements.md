# 10. 비기능적 요구사항 (Non-Functional Requirements)

> **문서 버전**: v1.0.0 | **최종 수정**: 2026-05-07

---

## 10.1 성능 요구사항 (Performance)

### 10.1.1 응답 속도

| 시나리오 | 환경 | 목표 응답 시간 |
|---------|------|--------------|
| 앱 초기 로딩 (모델 캐시 적용 후) | Streamlit Cloud | ≤ 3초 |
| 사용자 입력 → 결과 출력 | Streamlit Cloud | ≤ 2초 |
| 비교군 통계 조회 | 로컬 CSV 조회 | ≤ 0.5초 |
| 지역자원 조회 | 로컬 CSV 조회 | ≤ 0.5초 |

**측정 방법**:
```python
import time

# 추론 시간 측정
start = time.perf_counter()
result = predict_health_type(user_input, scaler, classifier, ...)
elapsed = (time.perf_counter() - start) * 1000
print(f"추론 시간: {elapsed:.1f}ms")
# 목표: ≤ 500ms
```

### 10.1.2 메모리 사용량

| 항목 | 기준 | 비고 |
|------|------|------|
| 모델 파일 전체 크기 | ≤ 80MB | Streamlit Cloud 512MB 제한 고려 |
| 배포 데이터 파일 | ≤ 5MB | 집계 통계 파일 합산 |
| 세션 메모리 (사용자 1인) | ≤ 10MB | 세션 상태 최소화 |

---

## 10.2 재현성 요구사항 (Reproducibility)

### 10.2.1 전체 시드 고정 정책

```python
# src/utils/seed.py
import random
import numpy as np

GLOBAL_SEED = 42  # 프로젝트 전체 공통 시드 (변경 금지)

def set_all_seeds(seed: int = GLOBAL_SEED) -> None:
    """
    모든 랜덤 시드를 고정합니다.
    분석·학습 스크립트 최초 실행 시 호출 필수.
    """
    random.seed(seed)
    np.random.seed(seed)
    # scikit-learn 모델은 각 생성자에 random_state=seed 직접 전달
```

### 10.2.2 설정 외부화 원칙

- **모든 하이퍼파라미터**는 `configs/*.yaml`에 정의 (코드 내 하드코딩 금지)
- **모든 판정 기준값**(BMI 임계값, 혈압 기준 등)은 `configs/rule_config.yaml`에 정의
- **모든 추천 문구**는 `configs/recommendation_config.yaml`에 정의

### 10.2.3 재현성 검증

```bash
# 동일 설정으로 2회 학습 실행 후 결과 비교
python scripts/train_models.py --seed 42
python scripts/train_models.py --seed 42

# 허용 오차: Accuracy 차이 < 0.001
# (scikit-learn은 random_state 고정 시 완전 재현 가능)
```

---

## 10.3 코드 품질 요구사항 (Code Quality)

### 10.3.1 스타일 기준

```
- PEP 8 준수 (최대 줄 길이: 100자)
- 타입 힌트 (Python type hints) 필수: 모든 public 함수
- Docstring: Google 스타일, 모든 public 함수/클래스
- 함수당 순환 복잡도 ≤ 10
```

### 10.3.2 테스트 커버리지

```
- 전체 코드 커버리지 목표: ≥ 70%
- 핵심 모듈별 커버리지:
  - calculator.py: ≥ 95%
  - rule_engine.py: ≥ 95%
  - predictor.py: ≥ 80%
  - imputer.py: ≥ 85%
  - validators.py: ≥ 90%
```

### 10.3.3 정적 분석

```bash
# 코드 스타일 검사
flake8 src/ --max-line-length=100

# 타입 체킹
mypy src/ --ignore-missing-imports

# 테스트 + 커버리지
pytest --cov=src --cov-report=html --cov-fail-under=70
```

---

## 10.4 의료적 안전성 요구사항 (Medical Safety)

### 10.4.1 금지 표현 자동 검사

```python
# tests/test_medical_safety.py
FORBIDDEN_EXPRESSIONS = [
    '고혈압입니다',
    '당뇨입니다',
    '치료가 필요합니다',
    '약을 드셔야',
    '반드시 좋아집니다',
    '개선됩니다',
    '정상으로 돌아옵니다',
    '위험형',
]

def test_no_forbidden_expressions_in_recommendations():
    """추천 문구에 금지 표현이 없는지 확인"""
    config = load_config('configs/recommendation_config.yaml')
    all_text = str(config)
    for expr in FORBIDDEN_EXPRESSIONS:
        assert expr not in all_text, f"금지 표현 발견: {expr}"

def test_disclaimer_always_present():
    """모든 결과 화면에 주의문구가 있는지 확인"""
    # UI 컴포넌트 렌더링 테스트
    pass
```

### 10.4.2 주의문구 표시 정책

- 모든 결과 화면 하단에 필수 고지 문구 표시
- 폰트 크기 최소 12px, 색상 회색 (너무 눈에 띄지 않게, 너무 안 보이지 않게)
- 결과 화면의 어떤 스크롤 위치에서도 접근 가능하도록 고정 또는 카드 최하단에 배치

---

## 10.5 문서화 요구사항 (Documentation)

### 10.5.1 README.md 필수 포함 내용

```markdown
1. 프로젝트 개요 (1~2 문장)
2. 서비스 화면 스크린샷 (최소 2장)
3. 시스템 요구사항
4. 환경 설정 (복붙 가능한 명령어)
5. 데이터 준비 방법
6. 모델 학습 방법
7. 앱 실행 방법
8. 배포 URL
9. 프로젝트 구조
10. 주요 분석 결과 요약
11. 서비스 한계 및 주의사항
12. 라이선스 및 출처
```

### 10.5.2 Jupyter 노트북 기준

```
- 모든 셀에 한글 설명 (마크다운)
- 출력 결과 반드시 포함 (실행 후 저장)
- 시각화 최소 3개 이상
- 결론 및 인사이트 마지막 셀에 정리
- 노트북 파일 크기 ≤ 10MB (이미지 크기 제한)
```

---

## 10.6 접근성 및 UX 요구사항

### 10.6.1 모바일 우선 설계 원칙

| 원칙 | 적용 방식 |
|------|----------|
| 1열 구성 | 좌우 스크롤 없는 세로 스크롤만 |
| 단계형 입력 | 기본정보 → 건강정보 → 생활습관 → 결과 순서 |
| 카드형 결과 | 각 정보를 독립 카드로 구분 |
| 큰 버튼 | 터치 최소 영역 44×44px 이상 |
| 사이드바 최소화 | 핵심 메뉴를 본문에 배치 |
| 표 최소화 | 요약 카드와 짧은 문장 중심 |

### 10.6.2 한국어 전용

- 모든 UI 텍스트는 한국어
- 오류 메시지도 한국어
- 숫자 표기: 한국 관행 준수 (소수점 `.`, 천 단위 `,`)

---

## 10.7 보안 요구사항 (Security)

```
- 사용자 입력값: 서버에 저장 금지 (세션 기반 일회성 계산)
- API 키: .env 파일에만 저장 (.gitignore 확인)
- .env 파일: 절대 git 커밋 금지
- Streamlit Secrets: 배포 시 환경변수 관리
- 원본 건강검진 데이터: GitHub 업로드 금지
```

---

## 10.8 환경 이식성 요구사항 (Portability)

```python
# 올바른 경로 사용 예시
from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', Path(__file__).parent.parent))
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
CONFIGS_DIR = PROJECT_ROOT / 'configs'

# ❌ 잘못된 예: 하드코딩 절대 경로
# DATA_DIR = '/Users/username/health_check/data'
```

- `requirements.txt`의 모든 패키지 버전 고정
- `conda environment.yaml` 포함
- Windows/macOS/Linux 모두 `pathlib.Path` 기반으로 동작
