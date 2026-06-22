# 🏥 건강위험 자가점검 서비스

국민건강보험공단 건강검진 100만 명 표본 데이터를 분석하여,
사용자의 건강관리 유형을 분류하고 맞춤 생활관리 방향을 제안하는 Streamlit 웹앱입니다.

---

## 📋 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **서비스 형태** | Streamlit 기반 자가점검 웹앱 |
| **핵심 데이터** | 국민건강보험공단 건강검진정보 2024 (100만 건) |
| **모델** | 룰 기반 라벨링 + Random Forest 분류 |
| **추천 로직** | 룰 기반 해석 + 유형별 추천 + 보건복지부 신체활동 지침 |
| **배포 환경** | Streamlit Community Cloud (무료, 512MB 제한) |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 3.10 환경 생성 (conda 권장)
conda create -n health_check python=3.10.14 -y
conda activate health_check

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 다운로드

공공데이터포털에서 다음 3개 파일을 직접 다운로드 받아 `data/raw/`에 넣어주세요:

| 파일 | 출처 |
|------|------|
| `2024_국민건강보험공단_건강검진정보.CSV` | https://data.go.kr |
| `전국_지역보건의료기관_현황_20221231.csv` | https://data.go.kr |
| `전국건강생활지원센터표준데이터.csv` | https://data.go.kr |

> 인코딩은 모두 **CP949 (EUC-KR)** 입니다.

### 3. 모델 학습

```bash
# 전처리 → 룰 기반 라벨링 → 분류 모델 학습 → 저장 (약 5~10분 소요)
python -m src.modeling
```

학습이 끝나면 다음 파일이 생성됩니다:
- `model/scaler.pkl`
- `model/classifier_model.pkl`
- `model/cluster_mapping.json`
- `data/processed/comparison_group_summary.csv`
- `data/processed/cluster_profile.csv`

### 4. 웹앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속하시면 됩니다.

---

## 📁 프로젝트 구조

```
health-check-service/
├── app.py                              # Streamlit 웹앱 진입점
├── requirements.txt                    # Python 의존성
├── README.md
│
├── notebooks/                          # 분석 과정 기록
│   ├── 01_data_understanding.ipynb     # 데이터 구조 확인
│   ├── 02_preprocessing_eda.ipynb      # 전처리 + EDA
│   ├── 03_labeling_modeling.ipynb      # 룰 기반 라벨링 + 모델 학습
│   └── 04_model_evaluation.ipynb       # 모델 성능 평가
│
├── src/                                # 핵심 로직
│   ├── preprocessing.py                # 전처리 함수
│   ├── feature_engineering.py          # 비교군·군집 통계 생성
│   ├── modeling.py                     # 모델 학습 파이프라인
│   ├── recommend.py                    # 룰 엔진 + 추천
│   └── predict.py                      # 사용자 입력 → 예측
│
├── model/                              # 학습된 모델
│   ├── scaler.pkl
│   ├── classifier_model.pkl
│   └── cluster_mapping.json
│
└── data/
    ├── raw/                            # 원본 CSV (git ignore)
    └── processed/                      # 배포용 집계 데이터
        ├── comparison_group_summary.csv
        └── cluster_profile.csv
```

---

## 🎯 6가지 건강관리 유형

| 유형 | 라벨 | 주요 특성 |
|------|------|---------|
| 기본관리형 | `basic` | BMI·혈압·혈당 모두 정상 범위 |
| 체중관리형 | `weight` | BMI 또는 허리둘레 관리 필요 |
| 혈압관리형 | `blood_pressure` | 수축기·이완기혈압 관리 필요 |
| 혈당관리형 | `blood_sugar` | 공복혈당 관리 필요 |
| 대사복합관리형 | `metabolic` | 여러 지표 동시 관리 필요 |
| 생활습관관리형 | `lifestyle` | 흡연·음주 관련 요인 두드러짐 |

---

## 📊 모델 성능

| 지표 | 값 | 목표 |
|------|------|------|
| Test Accuracy | ~0.99 | ≥ 0.75 |
| Macro F1 | ~0.99 | ≥ 0.65 |

> **참고**: 임상 기준 룰로 생성한 라벨을 Random Forest가 학습하는 구조이므로 정확도가 높게 나옵니다.
> 이는 의료 진단 정확도가 아니라 임상 기준 분류의 일관성을 의미합니다.

---

## ⚠️ 주의사항

- 본 서비스는 **교육용·참고용**이며 의료 진단을 대체하지 않습니다.
- 입력 수치가 높거나 이상 소견이 의심되는 경우 의료기관 상담을 권장합니다.
- 원본 건강검진 데이터는 GitHub 업로드 금지 (라이선스 + 용량).

---

## 📚 참고 자료

| 기준 | 출처 |
|------|------|
| BMI 분류 | 대한비만학회 (2022) |
| 혈압 분류 | 대한고혈압학회 (2022) |
| 공복혈당 | 대한당뇨병학회 (2023) |
| 콜레스테롤 | 한국지질·동맥경화학회 (2022) |
| 신체활동 지침 | 보건복지부·한국건강증진개발원 (2023) |

---

## 📄 라이선스

- 코드: MIT License (포트폴리오 공개용)
- 건강검진 데이터: 공공누리 제1유형 (data.go.kr)
- 신체활동 지침서: 보건복지부·한국건강증진개발원 발행 (사업-05-2023-004-15)