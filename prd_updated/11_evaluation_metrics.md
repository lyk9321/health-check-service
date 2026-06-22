# 11. 평가 지표 (Evaluation Metrics)

> **문서 버전**: v2.0.0 | **최종 수정**: 2026-06-18
> **원칙**: 평가는 반드시 Test set만 사용. Val set 최적화에 Test set 사용 금지

---

## 11.1 라벨 분포 분석

군집화 대신 룰 기반 라벨링을 사용하므로 Silhouette Score·Elbow Method는 적용하지 않습니다.
대신 라벨링 결과의 **임상 일치성**과 **클래스 분포**를 검증합니다.

### 11.1.1 클래스 분포 확인

```python
import pandas as pd
import matplotlib.pyplot as plt

# 유형별 라벨 수 및 비율
label_counts = pd.Series(y_labels).value_counts()
label_pct = label_counts / len(y_labels) * 100

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(label_counts.index, label_counts.values,
              color=['#2ECC71','#F39C12','#E74C3C',
                     '#E67E22','#C0392B','#9B59B6'])
ax.set_ylabel('표본 수')
ax.set_title('건강관리 유형별 라벨 분포')
for bar, pct in zip(bars, label_pct.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1000,
            f'{pct:.1f}%', ha='center', fontsize=10)
plt.tight_layout()
plt.show()
```

### 11.1.2 임상 일치성 검증

라벨링 결과가 임상 기준 방향과 실제로 일치하는지 유형별 평균 수치로 확인합니다.

| 유형 | 검증 기준 | 기대 방향 |
|------|---------|---------|
| 혈압관리형 | 평균 수축기혈압 | ≥ 130mmHg |
| 혈당관리형 | 평균 공복혈당 | ≥ 100mg/dL |
| 체중관리형 | 평균 BMI | ≥ 25.0 |
| 대사복합관리형 | 평균 위험요인 개수 | ≥ 3.0 |
| 생활습관관리형 | 현재흡연율 또는 음주율 | 타 유형 대비 높음 |
| 기본관리형 | 평균 위험요인 개수 | 전체 유형 중 최저 |

---

## 11.2 분류 모델 평가 지표

### 11.2.1 Overall Accuracy

$$\text{Accuracy} = \frac{\sum_{c=0}^{K-1} TP_c}{N_{test}}$$

### 11.2.2 Macro F1-Score

$$\text{F1}_{macro} = \frac{1}{K} \sum_{c=0}^{K-1} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

> 클래스 불균형에 관계없이 모든 클래스를 동등하게 평가

### 11.2.3 인접 유형 혼동률

$$\text{Adjacent Confusion Rate} = \frac{\sum_{|i-j|=1} CM[i,j]}{\sum_{i \neq j} CM[i,j]}$$

> 인접 유형(예: 체중관리형 → 대사복합관리형) 오분류가 전체 오분류에서 차지하는 비율

---

## 11.3 평가 실행 프로토콜

### 11.3.1 분류 모델 평가

```python
# scripts/evaluate_models.py

from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
import json

# Test set 평가 (반드시 Train/Val에 사용하지 않은 데이터)
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)

TYPE_NAMES = ['basic', 'weight', 'blood_pressure',
              'blood_sugar', 'metabolic', 'lifestyle']

metrics = {
    'overall_accuracy': float(accuracy_score(y_test, y_pred)),
    'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
    'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
    'per_class_f1': f1_score(y_test, y_pred, average=None).tolist(),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    'classification_report': classification_report(
        y_test, y_pred, target_names=TYPE_NAMES
    ),
    'n_test': len(y_test),
}

# 대사복합관리형 Recall
metabolic_idx = TYPE_NAMES.index('metabolic')
cm = confusion_matrix(y_test, y_pred)
metabolic_recall = cm[metabolic_idx, metabolic_idx] / cm[metabolic_idx].sum()
metrics['metabolic_recall'] = float(metabolic_recall)

# 결과 저장
with open('reports/evaluation_report.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
```

### 11.3.2 룰 기반 판정 정확도 검증

```python
# tests/test_rule_engine.py

import pytest
from src.core.rule_engine import interpret_blood_pressure

@pytest.mark.parametrize("systolic,diastolic,expected_category", [
    (110, 70, 'normal'),
    (125, 75, 'elevated'),
    (135, 85, 'stage1'),
    (145, 95, 'stage2'),
])
def test_blood_pressure_classification(systolic, diastolic, expected_category):
    result = interpret_blood_pressure(systolic, diastolic)
    assert result['category'] == expected_category
```

---

## 11.4 실험 비교 테이블

### 11.4.1 experiment_log.csv 작성 기준

```
- 실험마다 1행 추가 (기록된 행 수정 금지)
- 최소 비교 실험 4종:
  1. baseline: 전처리 없음, 기본 하이퍼파라미터
  2. +preprocessing: 이상치 제거 + 표준화 적용
  3. +feature_eng: 파생변수 추가
  4. final: 최적 하이퍼파라미터
```

**기대 결과 테이블 예시**:

| ID | 실험명 | 전처리 | 파생변수 | val Accuracy | test Accuracy | test F1 | 대사복합 Recall |
|----|--------|--------|---------|-------------|--------------|---------|----------------|
| 01 | baseline | 기본 | 없음 | 0.68 | 0.67 | 0.61 | 0.58 |
| 02 | +preprocessing | 이상치 제거 | 없음 | 0.72 | 0.71 | 0.66 | 0.63 |
| 03 | +feature_eng | 이상치 제거 | BMI·flag | 0.76 | 0.75 | 0.71 | 0.69 |
| 04 | final | 전체 전처리 | 전체 | **0.80** | **0.78** | **0.75** | **0.73** |

---

## 11.5 시각화 필수 산출물

### 11.5.1 라벨 분포 시각화

```python
# 1. 유형별 라벨 분포 막대그래프
# 저장: reports/figures/labeling/label_distribution.png

# 2. 유형별 주요 수치 박스플롯
# 저장: reports/figures/labeling/boxplot_bmi_by_type.png
#                                boxplot_sbp_by_type.png
#                                boxplot_fbg_by_type.png

# 3. 유형별 평균 프로파일 레이더차트 (방사형 그래프)
# 저장: reports/figures/labeling/radar_profile_by_type.png
```

**박스플롯 예시 코드**:

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col, title in zip(
    axes,
    ['bmi', 'systolic_bp', 'fasting_glucose'],
    ['BMI', '수축기혈압 (mmHg)', '공복혈당 (mg/dL)']
):
    sns.boxplot(
        data=df_labeled, x='health_type', y=col,
        order=['basic', 'weight', 'blood_pressure',
               'blood_sugar', 'metabolic', 'lifestyle'],
        ax=ax
    )
    ax.set_title(f'{title} by 건강관리 유형')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('reports/figures/labeling/boxplot_by_type.png', dpi=150)
plt.show()
```

### 11.5.2 분류 모델 시각화

```python
# 1. Confusion Matrix (정규화)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        vmin=0, vmax=1, ax=ax
    )
    ax.set_xlabel('예측 (Predicted)', fontsize=12)
    ax.set_ylabel('실제 (True)', fontsize=12)
    ax.set_title('건강관리 유형 분류 Confusion Matrix', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# 저장: reports/figures/classification/confusion_matrix.png

# 2. 변수 중요도 (Feature Importance)
importances = clf.feature_importances_
# 저장: reports/figures/classification/feature_importance.png

# 3. 클래스별 F1 비교 막대그래프
# 저장: reports/figures/classification/per_class_f1.png
```

---

## 11.6 최종 평가 리포트 구조

```json
// reports/evaluation_report.json
{
  "evaluation_date": "2026-XX-XX",
  "labeling": {
    "method": "rule_based",
    "label_distribution": {
      "basic":          {"n": 380000, "pct": 38.0},
      "weight":         {"n": 290000, "pct": 29.0},
      "blood_pressure": {"n": 150000, "pct": 15.0},
      "blood_sugar":    {"n": 100000, "pct": 10.0},
      "metabolic":      {"n":  50000, "pct":  5.0},
      "lifestyle":      {"n":  30000, "pct":  3.0}
    },
    "clinical_validation": {
      "blood_pressure_sbp_mean": 138.5,
      "blood_sugar_fbg_mean":    112.3,
      "weight_bmi_mean":          27.1,
      "metabolic_risk_mean":       3.2,
      "all_types_pass": true
    }
  },
  "classifier": {
    "model": "RandomForestClassifier",
    "n_estimators": 100,
    "max_depth": 15,
    "test_set_size": 150000,
    "overall_accuracy": 0.782,
    "macro_f1": 0.751,
    "weighted_f1": 0.774,
    "metabolic_recall": 0.731,
    "per_class": {
      "basic":          {"precision": 0.85, "recall": 0.89, "f1": 0.87},
      "weight":         {"precision": 0.80, "recall": 0.78, "f1": 0.79},
      "blood_pressure": {"precision": 0.77, "recall": 0.74, "f1": 0.75},
      "blood_sugar":    {"precision": 0.74, "recall": 0.71, "f1": 0.72},
      "metabolic":      {"precision": 0.70, "recall": 0.73, "f1": 0.71},
      "lifestyle":      {"precision": 0.68, "recall": 0.65, "f1": 0.66}
    }
  },
  "kpi_status": {
    "overall_accuracy":  {"target": 0.75, "achieved": 0.782, "passed": true},
    "macro_f1":          {"target": 0.65, "achieved": 0.751, "passed": true},
    "metabolic_recall":  {"target": 0.70, "achieved": 0.731, "passed": true},
    "inference_time_ms": {"target": 500,  "achieved": 120,   "passed": true}
  }
}
```
