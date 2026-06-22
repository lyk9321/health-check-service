# 08. 프로젝트 디렉토리 구조 (Directory Structure)

> **문서 버전**: v1.0.0 | **최종 수정**: 2026-05-07
> **원칙**: 이 구조에서 벗어나지 않음. 파일 추가 시 반드시 이 문서 업데이트

---

## 8.1 전체 디렉토리 트리

```
health_check/                             # 프로젝트 루트 (GitHub 저장소)
│
├── PRD/                                  # PRD 문서 모음
│   ├── 01_project_overview.md
│   ├── 02_problem_statement.md
│   ├── 03_goals_and_metrics.md
│   ├── 04_functional_requirements.md
│   ├── 05_tech_stack.md
│   ├── 06_data_requirements.md
│   ├── 07_model_architecture.md
│   ├── 08_directory_structure.md
│   ├── 09_module_specifications.md
│   ├── 10_non_functional_requirements.md
│   ├── 11_evaluation_metrics.md
│   ├── 12_development_roadmap.md
│   ├── 13_deliverables.md
│   └── 14_references.md
│
├── .streamlit/                           # Streamlit 설정
│   ├── config.toml                       # 테마, 서버 설정
│   └── secrets.toml                      # API 키 등 (git ignore)
│
├── configs/                              # 설정 파일 (git에 포함)
│   ├── model_config.yaml                 # 모델 하이퍼파라미터
│   ├── feature_config.yaml               # 피처 목록 및 설정
│   ├── rule_config.yaml                  # 룰 기반 판정 기준값
│   ├── recommendation_config.yaml        # 유형별 추천 문구
│   └── app_config.yaml                   # 앱 전반 설정
│
├── data/                                 # 데이터 (git ignore — 원본)
│   ├── raw/                              # 원본 데이터 (수정 금지)
│   │   ├── nhis_2024_sample.csv          # 건강검진정보 표본
│   │   └── README.md                     # 데이터 출처 및 라이선스 안내
│   │
│   ├── processed/                        # 전처리 완료 데이터
│   │   ├── train.csv                     # 학습 데이터
│   │   ├── val.csv                       # 검증 데이터
│   │   ├── test.csv                      # 테스트 데이터
│   │   └── cluster_labels.csv            # 군집화 결과 라벨
│   │
│   ├── external/                         # 외부 공공 데이터
│   │   ├── local_health_centers.csv      # 지역보건의료기관 현황
│   │   └── health_support_centers.csv    # 건강생활지원센터 현황
│   │
│   └── deploy/                           # 배포용 집계 데이터 (git에 포함)
│       ├── comparison_stats.csv          # 성별·연령대별 비교군 통계 (~10KB)
│       └── region_resources.csv          # 지역별 건강자원 집계 (~50KB)
│
├── models/                               # 학습 완료 모델 저장
│   ├── scaler.pkl                        # StandardScaler (git LFS 또는 직접 포함)
│   ├── kmeans.pkl                        # K-Means 군집화 모델
│   ├── classifier.pkl                    # Random Forest 분류 모델
│   └── cluster_mapping.json              # 군집 ID → 유형명 매핑
│
├── src/                                  # 소스 코드
│   ├── __init__.py
│   │
│   ├── data/                             # 데이터 처리 모듈
│   │   ├── __init__.py
│   │   ├── loader.py                     # 데이터 로딩
│   │   ├── preprocessor.py               # 전처리 파이프라인
│   │   ├── feature_engineer.py           # 파생변수 생성
│   │   ├── splitter.py                   # 데이터 분할
│   │   └── validator.py                  # 데이터 품질 검증
│   │
│   ├── models/                           # 모델 모듈
│   │   ├── __init__.py
│   │   ├── clustering.py                 # K-Means 군집화
│   │   ├── classifier.py                 # Random Forest 분류 모델
│   │   ├── loader.py                     # 모델 로딩 (캐시 포함)
│   │   └── predictor.py                  # 통합 추론 파이프라인
│   │
│   ├── core/                             # 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── calculator.py                 # BMI·완성도 계산
│   │   ├── rule_engine.py                # 룰 기반 건강지표 해석
│   │   ├── imputer.py                    # 결측값 처리 (사용자 입력)
│   │   ├── comparator.py                 # 비교군 비교 로직
│   │   ├── recommender.py                # 생활관리·운동 추천
│   │   └── local_resources.py            # 지역사회 자원 조회
│   │
│   ├── ui/                               # Streamlit UI 컴포넌트
│   │   ├── __init__.py
│   │   ├── intro_page.py                 # 서비스 소개 화면
│   │   ├── input_form.py                 # 입력 폼
│   │   ├── result_cards.py               # 결과 카드 컴포넌트
│   │   ├── comparison_chart.py           # 비교군 비교 차트
│   │   ├── local_resources_map.py        # 지역자원 지도
│   │   └── disclaimer.py                 # 주의문구 컴포넌트
│   │
│   └── utils/                            # 공통 유틸리티
│       ├── __init__.py
│       ├── config.py                     # config 파일 로딩
│       ├── logger.py                     # 로깅 설정 (loguru)
│       ├── seed.py                       # 전체 시드 고정
│       └── validators.py                 # 입력값 유효성 검사
│
├── app.py                                # Streamlit 앱 진입점
│
├── notebooks/                            # Jupyter 노트북
│   ├── 01_EDA.ipynb                      # 탐색적 데이터 분석
│   ├── 02_preprocessing.ipynb            # 전처리 파이프라인 구현·검증
│   ├── 03_clustering.ipynb               # 군집화 실험 (K 탐색, 해석)
│   ├── 04_classification.ipynb           # 분류 모델 학습·비교
│   └── 05_evaluation.ipynb               # 최종 평가·시각화
│
├── scripts/                              # 실행 스크립트
│   ├── verify_env.py                     # 환경 검증
│   ├── prepare_data.py                   # 데이터 전처리 일괄 실행
│   ├── train_models.py                   # 군집화 + 분류 모델 학습
│   ├── evaluate_models.py                # 전체 평가 실행
│   └── generate_deploy_data.py           # 배포용 집계 데이터 생성
│
├── tests/                                # 단위 테스트
│   ├── __init__.py
│   ├── test_calculator.py                # BMI·완성도 계산 테스트
│   ├── test_rule_engine.py               # 룰 기반 판정 테스트
│   ├── test_imputer.py                   # 결측값 처리 테스트
│   ├── test_comparator.py                # 비교군 비교 테스트
│   ├── test_predictor.py                 # 통합 추론 테스트
│   └── test_validators.py                # 입력 유효성 검사 테스트
│
├── reports/                              # 보고서 및 시각화 산출물
│   ├── figures/
│   │   ├── eda/                          # EDA 시각화
│   │   ├── clustering/                   # 군집 시각화 (PCA, 실루엣)
│   │   ├── classification/               # 분류 결과 (CM, 변수 중요도)
│   │   └── app_screenshots/              # 앱 스크린샷
│   ├── experiment_log.csv                # 실험별 성능 비교
│   ├── evaluation_report.json            # 최종 평가 결과
│   └── final_report.md                   # 최종 분석 보고서
│
├── README.md
├── requirements.txt
├── environment.yaml
├── .env.example
├── .gitignore
├── .gitattributes                        # Git LFS 설정
├── setup.py
└── pytest.ini
```

---

## 8.2 핵심 설정 파일 내용

### configs/rule_config.yaml

```yaml
# 룰 기반 판정 기준값 (출처: 대한비만학회, 대한고혈압학회, ADA)
bmi:
  underweight: 18.5
  normal_max: 22.9
  overweight_max: 24.9
  # 25.0 이상 → 비만

blood_pressure:
  normal_systolic: 120
  normal_diastolic: 80
  elevated_systolic: 129
  stage1_systolic: 139
  stage1_diastolic: 89
  # 140/90 이상 → 고혈압 범위

fasting_glucose:
  normal_max: 99
  impaired_max: 125
  # 126 이상 → 당뇨병 범위

total_cholesterol:
  optimal_max: 199
  borderline_max: 239
  # 240 이상 → 고콜레스테롤

waist_circumference:
  male_threshold: 90    # cm
  female_threshold: 85  # cm
```

### configs/recommendation_config.yaml

```yaml
# 유형별 생활관리 추천 문구
recommendations:
  basic:
    title: "현재 건강 상태를 잘 유지하고 있습니다"
    actions:
      - "정기 건강검진을 빠짐없이 받으세요 (연 1회 권장)"
      - "규칙적인 신체활동을 유지하세요 (주 150분 이상)"
      - "균형 잡힌 식단을 지속하세요"

  weight:
    title: "체중 관리에 집중이 필요합니다"
    actions:
      - "하루 30분 이상 걷기를 시작해보세요"
      - "식사량과 간식 섭취를 점검해보세요"
      - "체중 변화를 주 1회 이상 기록하세요"
      - "보건소 비만 관리 프로그램 이용을 권장합니다"

  blood_pressure:
    title: "혈압 관리에 집중이 필요합니다"
    actions:
      - "혈압을 정기적으로 재측정하세요 (주 2~3회)"
      - "나트륨 섭취를 줄이세요 (저염식)"
      - "규칙적인 유산소 운동이 혈압 관리에 도움이 될 수 있습니다"
      - "의료기관 또는 보건소 상담을 권장합니다"

  blood_sugar:
    title: "혈당 관리에 집중이 필요합니다"
    actions:
      - "공복혈당 재확인 및 의료기관 상담을 권장합니다"
      - "정제 탄수화물과 당류 섭취를 줄이세요"
      - "규칙적인 식사 시간을 유지하세요"
      - "신체활동이 혈당 조절에 도움이 될 수 있습니다"

  metabolic:
    title: "여러 건강 지표에 대한 복합 관리가 필요합니다"
    actions:
      - "보건소 또는 의료기관 상담을 적극 권장합니다"
      - "체중·혈압·혈당을 함께 관리하는 것이 중요합니다"
      - "금연·절주 프로그램 이용을 권장합니다"
      - "건강생활지원센터의 만성질환 예방 프로그램을 확인해보세요"

  lifestyle:
    title: "생활습관 개선이 가장 중요합니다"
    actions:
      - "금연을 강력히 권장합니다 (보건소 금연클리닉 무료 이용 가능)"
      - "음주량을 줄이거나 절주하세요"
      - "규칙적인 신체활동을 시작하세요"
      - "건강생활지원센터의 생활습관 개선 프로그램을 이용해보세요"
```

---

## 8.3 app.py 기본 구조

```python
# app.py — Streamlit 앱 진입점
import streamlit as st
from src.ui import intro_page, input_form, result_cards
from src.models.loader import load_models, load_comparison_stats
from src.utils.config import load_config

# 페이지 설정
st.set_page_config(
    page_title="건강위험 자가점검 서비스",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# 모델 및 데이터 캐시 로딩
scaler, classifier, cluster_mapping = load_models()
comparison_stats = load_comparison_stats()
config = load_config()

# 페이지 라우팅
page = st.session_state.get('page', 'intro')

if page == 'intro':
    intro_page.render()
elif page == 'input':
    input_form.render()
elif page == 'result':
    result_cards.render(
        user_input=st.session_state.get('user_input', {}),
        scaler=scaler,
        classifier=classifier,
        cluster_mapping=cluster_mapping,
        comparison_stats=comparison_stats,
        config=config,
    )
```

---

## 8.4 Git LFS 및 gitignore 설정

### .gitattributes

```
*.pkl filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
```

### pytest.ini

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --cov=src --cov-report=term-missing --cov-report=html
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning
```
