# 07. 모델 아키텍처 (Model Architecture)

> **문서 버전**: v2.0.0 | **최종 수정**: 2026-06-18

---

## 7.1 전체 모델 파이프라인

```
[건강검진 원본 데이터 (100만 건)]
          │
          ▼
[전처리] 결측치 처리 → 이상치 제거 → 파생변수 생성 → 표준화
          │
          ▼
┌─────────────────────────────────────────────┐
│        STEP 1: 룰 기반 라벨링 (오프라인)      │
│  임상 기준 우선순위 규칙 적용                 │
│  → 각 검진자에게 건강관리 유형 라벨 부여       │
│  우선순위: 대사복합 → 혈압 → 혈당             │
│           → 체중 → 생활습관 → 기본           │
└──────────────────┬──────────────────────────┘
                   │ 라벨 생성 (임상 기준 정답)
                   ▼
┌─────────────────────────────────────────────┐
│        STEP 2: 분류 모델 학습 (오프라인)      │
│  Random Forest 학습                         │
│  입력: 표준화된 건강수치 (7개 변수)           │
│  출력: 건강관리 유형 6종 중 하나              │
└──────────────────┬──────────────────────────┘
                   │ model/scaler.pkl
                   │ model/classifier_model.pkl 저장
                   ▼
┌─────────────────────────────────────────────┐
│        STEP 3: 서비스 추론 (온라인)           │
│  사용자 입력 → 표준화 → 유형 예측             │
│  + 룰 기반 지표 해석 병행                    │
│  + data/processed/comparison_group_summary.csv│
│    에서 비교군 통계 조회                      │
│  → 통합 결과 출력                           │
└─────────────────────────────────────────────┘
```

---

## 7.2 모델 학습 변수 (MODEL_FEATURES)

총콜레스테롤은 결측률이 약 66%로 높아 모델 학습 변수에서 제외.

```python
# 룰 기반 라벨링·분류 모델 공통 입력 변수 (7개)
MODEL_FEATURES = [
    'bmi',              # 파생변수: weight / (height/100)^2
    'systolic_bp',      # 원본: 수축기혈압
    'diastolic_bp',     # 원본: 이완기혈압
    'fasting_glucose',  # 원본: 식전혈당(공복혈당)
    'waist_cm',         # 원본: 허리둘레
    'smoking_status',   # 원본: 흡연상태 (1=비흡연, 2=과거흡연, 3=현재흡연)
    'drinking',         # 원본: 음주여부 (0=없음, 1=있음)
]

# 분류 시 추가 입력 (그룹 변수 — 표준화 제외)
GROUP_FEATURES = [
    'gender',           # 원본: 성별코드 (1=남성, 2=여성)
    'age_group',        # 원본: 연령대코드(5세단위) (5=25-29세 ~ 17=85세+)
]
```

---

## 7.3 룰 기반 라벨링

### 7.3.1 설계 배경

비지도학습(K-Means)을 사용하면 데이터의 자연스러운 패턴을 찾지만, 사전에 정의한 6개 임상 유형과 일치한다는 보장이 없습니다. 실제로 K-Means를 적용했을 때 다음 문제가 발생했습니다:

- 음주 변수에 군집이 과도하게 지배되어 BMI 정상인 군집이 체중관리형으로 매핑됨
- Silhouette Score 0.21로 군집 경계가 불분명함
- Confusion Matrix가 K-Means 경계 학습 결과를 반영해 의미없는 0.97의 정확도 출력

이를 해결하기 위해 **임상 기준에 기반한 룰 기반 라벨링**으로 전환했습니다.

### 7.3.2 우선순위 규칙

```python
# src/modeling.py

def assign_labels_by_rule(df: pd.DataFrame) -> list:
    """
    임상 기준 우선순위에 따라 각 행에 건강관리 유형 라벨을 부여합니다.

    우선순위 근거:
      대사복합관리형: 위험요인 복수 존재 → 가장 시급한 복합 관리 필요
      혈압관리형:    심혈관 위험도 즉각적 → 다음 우선순위
      혈당관리형:    당뇨병 전단계·범위 → 조기 개입 중요
      체중관리형:    비만·복부비만 → 상기 위험요인 전 단계
      생활습관관리형: 흡연·음주 → 간접적 위험요인
      기본관리형:    위 항목 해당 없음
    """
    labels = []
    for _, row in df.iterrows():
        if row['metabolic_risk_count'] >= 3:
            label = 'metabolic'
        elif row['systolic_bp'] >= 130 or row['diastolic_bp'] >= 80:
            label = 'blood_pressure'
        elif row['fasting_glucose'] >= 100:
            label = 'blood_sugar'
        elif row['bmi'] >= 25 or row['abdominal_obesity_flag'] == 1:
            label = 'weight'
        elif row['smoking_status'] == 3 or row['drinking'] == 1:
            label = 'lifestyle'
        else:
            label = 'basic'
        labels.append(label)
    return labels
```

### 7.3.3 유형별 라벨 분포 검증

```python
# 라벨링 후 분포 확인 (임상 일치성 검증)
label_counts = pd.Series(y_labels).value_counts()

# 유형별 평균 수치가 임상 기준 방향과 일치하는지 확인
# 예: 혈압관리형의 평균 수축기혈압이 130 이상인지
# 예: 체중관리형의 평균 BMI가 25 이상인지
profile = df.copy()
profile['health_type'] = y_labels
print(profile.groupby('health_type')[
    ['bmi', 'systolic_bp', 'fasting_glucose']
].mean().round(1))
```

---

## 7.4 분류 모델 (Random Forest)

### 7.4.1 모델 설정

```python
# src/modeling.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # 클래스 불균형 보정
    random_state=42,
    n_jobs=-1,
)

# 교차 검증 (5-fold)
cv_scores = cross_val_score(
    rf, X_train_scaled, y_train,
    cv=5, scoring='f1_macro', n_jobs=-1
)
print(f"CV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 7.4.2 하이퍼파라미터 탐색

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train_scaled, y_train)
print(f"Best params: {grid_search.best_params_}")
```

### 7.4.3 비교 모델

| 모델 | 역할 | 선택 이유 |
|------|------|----------|
| Random Forest | 메인 분류 모델 | 변수 중요도 해석 가능, 안정적 성능 |
| Logistic Regression | 비교 베이스라인 | 해석 가능성, 빠른 학습 |
| XGBoost | 성능 비교 상한선 | 앙상블 성능 확인 |

---

## 7.5 모델 저장 및 로딩

### 7.5.1 저장 (`src/modeling.py`)

```python
import joblib
import json

# model/ 디렉토리에 저장
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(rf_best, 'model/classifier_model.pkl')

# 유형 매핑 정보 저장 (라벨명 → 한글명, 색상)
cluster_mapping = {
    'basic':          {'name': '기본관리형',     'color': '#2ECC71'},
    'weight':         {'name': '체중관리형',     'color': '#F39C12'},
    'blood_pressure': {'name': '혈압관리형',     'color': '#E74C3C'},
    'blood_sugar':    {'name': '혈당관리형',     'color': '#E67E22'},
    'metabolic':      {'name': '대사복합관리형', 'color': '#C0392B'},
    'lifestyle':      {'name': '생활습관관리형', 'color': '#9B59B6'},
}
with open('model/cluster_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_mapping, f, ensure_ascii=False, indent=2)
```

### 7.5.2 로딩 (`src/predict.py`)

```python
import streamlit as st
import joblib
import json
from pathlib import Path

MODEL_DIR = Path('model')

@st.cache_resource
def load_models():
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    classifier = joblib.load(MODEL_DIR / 'classifier_model.pkl')
    with open(MODEL_DIR / 'cluster_mapping.json', encoding='utf-8') as f:
        cluster_mapping = json.load(f)
    return scaler, classifier, cluster_mapping

@st.cache_data
def load_comparison_stats():
    return pd.read_csv('data/processed/comparison_group_summary.csv')
```

---

## 7.6 결측 입력값 처리 로직 (추론 시)

```python
# src/predict.py

def prepare_user_input(
    user_input: dict,
    comparison_stats: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    사용자 입력값을 모델 입력 벡터로 변환.
    결측 항목은 해당 성별·연령대 평균으로 대체.
    """
    gender = user_input['gender']
    age_group = user_input['age_group']

    # 비교군 평균값 조회
    group = comparison_stats[
        (comparison_stats['gender'] == gender) &
        (comparison_stats['age_group'] == age_group)
    ]

    missing_fields = []
    features = {}

    field_map = {
        'bmi': None,  # 키/몸무게로 계산
        'systolic_bp': 'sbp_mean',
        'diastolic_bp': 'sbp_mean',  # 이완기는 별도 집계 필요
        'fasting_glucose': 'fbg_mean',
        'waist_cm': 'waist_mean',
        'smoking_status': None,  # 최빈값 대체
        'drinking': None,        # 최빈값 대체
    }

    for feat in MODEL_FEATURES:
        if user_input.get(feat) is not None:
            features[feat] = user_input[feat]
        else:
            col = field_map.get(feat)
            if col and col in group.columns:
                features[feat] = group[col].values[0]
            else:
                features[feat] = 0.0
            missing_fields.append(feat)

    X = np.array([features[f] for f in MODEL_FEATURES]).reshape(1, -1)
    return X, missing_fields
```

---

## 7.7 모델 파일 크기 목표

| 파일 경로 | 예상 크기 | 목표 |
|---------|----------|------|
| `model/scaler.pkl` | ~5KB | ≤ 10KB |
| `model/classifier_model.pkl` | ~50MB (100 trees) | ≤ 80MB |
| `model/cluster_mapping.json` | ~1KB | ≤ 5KB |
| `data/processed/comparison_group_summary.csv` | ~10KB | ≤ 50KB |
| `data/processed/cluster_profile.csv` | ~5KB | ≤ 20KB |
| **합계** | **~50MB** | **≤ 80MB** |

> Streamlit Cloud 무료 플랜(512MB RAM) 제한 내에 로딩되어야 함.  
> 초과 시 `n_estimators=100` 또는 `max_depth=10` 으로 축소 검토.
