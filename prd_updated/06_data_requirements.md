# 06. 데이터 요구사항 (Data Requirements)

> **문서 버전**: v1.2.0 | **최종 수정**: 2026-05-07

---

## 6.1 데이터셋 목록 및 수집 계획

| ID | 데이터셋명 | 출처 | URL | 수집 방법 | 예상 용량 |
|----|-----------|------|-----|----------|----------|
| D01 | 국민건강보험공단 건강검진정보 2024 (100만 건 표본) | 공공데이터포털 | data.go.kr | **직접 CSV 다운로드** | ~50MB |
| D02 | 전국 지역보건의료기관 현황 | 공공데이터포털 | data.go.kr | CSV 직접 다운로드 | ~2MB |
| D03 | 전국 건강생활지원센터 표준데이터 | 공공데이터포털 | data.go.kr | CSV 직접 다운로드 | ~1MB |

### 6.1.1 건강검진정보 데이터 다운로드 절차

```
1. https://data.go.kr 접속 (공공데이터포털)
2. 검색창에 "국민건강보험공단_건강검진정보" 입력
3. 해당 데이터셋 클릭
4. [다운로드] 버튼 클릭 → CSV 직접 다운로드 (별도 신청 불필요)
5. 파일 저장: data/raw/2024_국민건강보험공단_건강검진정보.CSV
```

> **인코딩**: 파일 인코딩은 **CP949 (EUC-KR)** → pandas 로딩 시 `encoding='cp949'` 필수

```python
import pandas as pd
df = pd.read_csv(
    'data/raw/2024_국민건강보험공단_건강검진정보.CSV',
    encoding='cp949'
)
```

---

## 6.2 핵심 데이터: 건강검진정보 2024 변수 명세

### 6.2.1 전체 컬럼 목록 (실제 33개)

| 번호 | 원본 컬럼명 | 내부 변수명 | 타입 | 설명 | 활용 여부 |
|------|-----------|-----------|------|------|----------|
| 1 | 기준년도 | `year` | int | 검진 연도 | 확인용 |
| 2 | 가입자일련번호 | `id` | int | 익명화된 식별자 | 미사용 |
| 3 | 시도코드 | `region_code` | int | 시도 코드 | 지역 분석 |
| 4 | 성별코드 | `gender` | int | 1=남성, 2=여성 | ✅ 필수 |
| 5 | 연령대코드(5세단위) | `age_group` | int | 5=25-29세, 6=30-34세, ..., 17=85세+ | ✅ 필수 |
| 6 | 신장(5cm단위) | `height_cm` | float | cm (5cm 단위 반올림) | ✅ 필수 |
| 7 | 체중(5kg단위) | `weight_kg` | float | kg (5kg 단위 반올림) | ✅ 필수 |
| 8 | 허리둘레 | `waist_cm` | float | cm | ✅ 권장 |
| 9 | 시력(좌) | - | float | - | ❌ 미사용 |
| 10 | 시력(우) | - | float | - | ❌ 미사용 |
| 11 | 청력(좌) | - | float | - | ❌ 미사용 |
| 12 | 청력(우) | - | float | - | ❌ 미사용 |
| 13 | 수축기혈압 | `systolic_bp` | float | mmHg | ✅ 권장 |
| 14 | 이완기혈압 | `diastolic_bp` | float | mmHg | ✅ 권장 |
| 15 | 식전혈당(공복혈당) | `fasting_glucose` | float | mg/dL | ✅ 권장 |
| 16 | 총콜레스테롤 | `total_cholesterol` | float | mg/dL (약 34% 검사자만 존재) | 선택 (룰 기반 해석 — LDL 미입력 시 보조) |
| 17 | 트리글리세라이드 | `triglyceride` | float | mg/dL | 선택 (룰 기반 해석) |
| 18 | HDL콜레스테롤 | `hdl_cholesterol` | float | mg/dL | 선택 (룰 기반 해석, 성별 기준 적용) |
| 19 | LDL콜레스테롤 | `ldl_cholesterol` | float | mg/dL | 선택 (룰 기반 해석 — 최우선) |
| 20 | 혈색소 | `hemoglobin` | float | g/dL | ❌ 미사용 |
| 21 | 요단백 | - | float | - | ❌ 미사용 |
| 22 | 혈청크레아티닌 | - | float | - | ❌ 미사용 |
| 23 | 혈청지오티(AST) | - | float | - | ❌ 미사용 |
| 24 | 혈청지피티(ALT) | - | float | - | ❌ 미사용 |
| 25 | 감마지티피 | - | float | - | ❌ 미사용 |
| 26 | 흡연상태 | `smoking_status` | int | 1=비흡연, 2=과거흡연, 3=현재흡연 | ✅ 권장 |
| 27 | 음주여부 | `drinking` | int | 0=음주없음, 1=음주함 | ✅ 권장 |
| 28-33 | 구강검진 관련 | - | - | 치아우식증, 결손치, 치석 등 | ❌ 미사용 |

### 6.2.2 연령대코드 디코딩 테이블

| 연령대코드 | 실제 연령 | 연령대코드 | 실제 연령 |
|----------|---------|----------|---------|
| 5 | 25~29세 | 11 | 55~59세 |
| 6 | 30~34세 | 12 | 60~64세 |
| 7 | 35~39세 | 13 | 65~69세 |
| 8 | 40~44세 | 14 | 70~74세 |
| 9 | 45~49세 | 15 | 75~79세 |
| 10 | 50~54세 | 16 | 80~84세 |
| 17 | 85~89세 | 18 | 90세 이상 |

> **참고**: 이 데이터는 25세 이상(연령대코드 5~) 검진자를 포함. 20~24세 데이터는 없음.

### 6.2.3 주요 변수 결측률 현황 (실제 데이터 기준)

| 변수 | 결측률 | 비고 |
|------|--------|------|
| 수축기혈압, 이완기혈압 | ~0.6% | 거의 필수 검사 |
| 식전혈당(공복혈당) | ~0.6% | 거의 필수 검사 |
| 허리둘레 | ~0.05% | 거의 필수 측정 |
| 총콜레스테롤 | **~65.8%** | 선택 검사 — 약 34%만 존재 |
| HDL, LDL, 트리글리세라이드 | **~65.8%** | 선택 검사 (총콜레스테롤과 동일) |
| 흡연상태, 음주여부 | ~0.01% | 거의 완전 |

> **지질 수치 처리 방침**: 총콜레스테롤·HDL·LDL·트리글리세라이드 모두 결측률 약 66%로  
> 모델 학습 변수에서 제외. 단, **사용자가 직접 입력한 경우에는 4종 각각의 룰 기반 해석을 제공**  
> (LDL 우선, 총콜레스테롤은 나머지 3종 미입력 시 보조 참고).  
> 비교군 통계(`comparison_group_summary.csv`)에도 지질 수치는 포함하지 않음(신뢰도 문제).

---

## 6.3 파생변수 생성

| 파생변수명 | 계산식 | 설명 |
|-----------|--------|------|
| `bmi` | `weight_kg / (height_cm / 100) ** 2` | BMI |
| `bmi_category` | BMI 기준 0~3 분류 | 0=저체중, 1=정상, 2=과체중, 3=비만 |
| `age_label` | 연령대코드 → "25-29세" 등 문자열 | 화면 표시용 |
| `hypertension_flag` | 수축기≥130 또는 이완기≥80 → 1 | 혈압 관리 필요 여부 |
| `hyperglycemia_flag` | 공복혈당≥100 → 1 | 혈당 관리 필요 여부 |
| `abdominal_obesity_flag` | 남성 허리둘레≥90cm 또는 여성≥85cm → 1 | 복부비만 여부 |
| `metabolic_risk_count` | hypertension + hyperglycemia + abdominal_obesity + bmi_category≥3 합산 | 복합 위험요인 수 (0~4) |

---

## 6.4 데이터 전처리 파이프라인

### 6.4.1 결측치 처리 전략

```python
# src/preprocessing.py

MISSING_STRATEGY = {
    # 필수 변수: 결측 시 해당 행 제거
    'gender': 'drop',
    'age_group': 'drop',
    'height_cm': 'drop',
    'weight_kg': 'drop',

    # 선택 변수: 성별·연령대 그룹 중앙값으로 대체
    'systolic_bp': 'group_median',
    'diastolic_bp': 'group_median',
    'fasting_glucose': 'group_median',
    'waist_cm': 'group_median',

    # 범주형: 최빈값으로 대체
    'smoking_status': 'mode',
    'drinking': 'mode',

    # 총콜레스테롤: 모델 학습에 미포함 (결측률 ~66%)
    # → 사용자 직접 입력 시 룰 기반 해석에만 활용
}
```

### 6.4.2 이상치 처리 기준

| 변수 | 유효 범위 | 처리 방법 |
|------|----------|----------|
| `height_cm` | 100~250 | 범위 외 → 제거 |
| `weight_kg` | 20~300 | 범위 외 → 제거 |
| `bmi` | 10~60 | 범위 외 → 제거 |
| `systolic_bp` | 60~250 | 범위 외 → 제거 |
| `diastolic_bp` | 40~150 | 범위 외 → 제거 |
| `fasting_glucose` | 50~500 | 범위 외 → 제거 |

### 6.4.3 표준화

```python
# src/modeling.py

from sklearn.preprocessing import StandardScaler
import joblib

# 군집화·분류 모델 학습 변수 (총콜레스테롤 제외)
MODEL_FEATURES = [
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'fasting_glucose',
    'waist_cm',
    'smoking_status',
    'drinking',
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[MODEL_FEATURES])

# scaler 저장 (사용자 입력 표준화에 재사용)
joblib.dump(scaler, 'model/scaler.pkl')
```

---

## 6.5 데이터 분할 규칙

```python
# src/modeling.py
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# 군집 라벨 부여 후 분류 모델용 분할 (Stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_cluster,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y_cluster
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=y_temp
)
# 결과: Train 70% / Val 15% / Test 15%
```

---

## 6.6 외부 데이터 스키마

### 6.6.1 전국 지역보건의료기관 현황 (실제 컬럼)

| 컬럼명 | 내용 |
|--------|------|
| 시도 | 시도명 (예: 서울특별시) |
| 시군구 | 시군구명 |
| 기관유형 | 보건소 / 보건지소 / 보건진료소 |
| 상위 보건기관명 | 상위 보건소명 |
| 보건기관명 | 기관명 |
| 주소 | 도로명 주소 |
| 읍면동명 | 행정동 |
| 도서지역 여부 | 예/아니오 |
| 대표 전화번호 | 전화번호 |

### 6.6.2 전국 건강생활지원센터 표준데이터 (주요 컬럼)

| 컬럼명 | 내용 |
|--------|------|
| 건강생활지원센터명 | 센터명 |
| 건강생활지원센터 유형 | 기본형 등 |
| 소재지도로명주소 | 도로명 주소 |
| 위도 / 경도 | 좌표 (folium 지도 활용) |
| 운영시작시각 / 운영종료시각 | 운영 시간 |
| 휴무일정보 | 예: 토+일+공휴일 |
| 건강증진사업내용 | 예: 신체활동+만성질환관리+영양 |
| 운영기관전화번호 | 연락처 |
| 관리기관명 | 담당 보건소명 |

---

## 6.7 비교군 집계 데이터 생성

배포 환경에서는 원본 100만 건 데이터를 포함하지 않고, **사전 집계된 통계 파일**만 사용:

```python
# scripts/generate_deploy_data.py
# 성별 × 연령대 조합별 주요 지표 통계 집계

comparison_stats = df.groupby(['gender', 'age_group']).agg(
    n=('bmi', 'count'),
    bmi_mean=('bmi', 'mean'),
    bmi_std=('bmi', 'std'),
    sbp_mean=('systolic_bp', 'mean'),
    sbp_std=('systolic_bp', 'std'),
    fbg_mean=('fasting_glucose', 'mean'),
    fbg_std=('fasting_glucose', 'std'),
    waist_mean=('waist_cm', 'mean'),
    waist_std=('waist_cm', 'std'),
    smoking_rate=('smoking_status', lambda x: (x == 3).mean()),  # 현재흡연 비율
    drinking_rate=('drinking', 'mean'),
).reset_index()

# 연령대 레이블 추가
age_label_map = {
    5:'25-29세', 6:'30-34세', 7:'35-39세', 8:'40-44세',
    9:'45-49세', 10:'50-54세', 11:'55-59세', 12:'60-64세',
    13:'65-69세', 14:'70-74세', 15:'75-79세', 16:'80-84세', 17:'85세+'
}
comparison_stats['age_label'] = comparison_stats['age_group'].map(age_label_map)

# 배포용 파일 저장 (~10KB)
comparison_stats.to_csv('data/processed/comparison_group_summary.csv', index=False)
```

---

## 6.8 데이터 품질 기준

| 항목 | 기준 |
|------|------|
| 전체 행 수 | 1,000,000 건 (2024년 표본) |
| 필수 변수 완전율 | 성별·연령대·신장·체중 99.9% 이상 |
| 혈압·혈당 완전율 | ~99.4% |
| 성별·연령대별 최소 표본 | ≥ 1,000 건 |

---

## 6.9 저작권 및 라이선스

| 데이터 | 라이선스 | 출처 |
|--------|---------|------|
| 건강검진정보 2024 표본 | 공공누리 제1유형 | 공공데이터포털 (data.go.kr) |
| 전국 지역보건의료기관 현황 | 공공누리 제1유형 | 공공데이터포털 (data.go.kr) |
| 전국 건강생활지원센터 표준데이터 | 공공누리 제1유형 | 공공데이터포털 (data.go.kr) |

> **원본 데이터 GitHub 업로드 금지**: `.gitignore`에 `data/raw/` 포함 필수  
> 배포 시 `data/processed/comparison_group_summary.csv`(~10KB)와 모델 파일만 포함
