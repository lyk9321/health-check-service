# 03. 목표 및 성공 지표 (Goals & Success Metrics)

> **문서 버전**: v2.0.0 | **최종 수정**: 2026-06-18

---

## 3.1 프로젝트 목표 계층

```
Level 0 (포트폴리오 목표)
└── 간호 도메인 지식 + 데이터 분석 역량을 결합한 헬스케어 MVP 구현

Level 1 (기술 목표)
├── 모델: 건강관리 유형 분류 Accuracy ≥ 0.75
├── 서비스: 필수 입력값만으로 결과 출력 가능
└── 배포: Streamlit 웹앱 URL 공개 접근 가능

Level 2 (품질 목표)
├── 재현 가능한 분석 파이프라인 (seed 고정, config 외부화)
├── 의료적 리스크 관리 (진단·처방 표현 완전 배제)
└── README만으로 환경 설정 ~ 서비스 실행 재현 가능
```

---

## 3.2 정량적 성공 지표 (KPI)

### 3.2.1 라벨 분포 품질 KPI

| 지표 | 최소 기준 | 목표 | 측정 시점 |
|------|----------|------|----------|
| 유형별 최소 표본 수 | 1,000건 | **3,000건** | 라벨링 완료 후 |
| 라벨 임상 일치성 | 각 유형 기준표 통과 | **통과** | 라벨링 완료 후 |
| 소수 클래스 비율 | ≥ 0.5% | **≥ 1%** | 라벨링 완료 후 |

> **라벨 임상 일치성 판단 기준**: 룰 기반으로 부여된 라벨과 각 유형의 평균 BMI·혈압·혈당 수치가 임상 기준 방향과 일치하는지 확인 (예: 혈압관리형의 평균 수축기혈압 ≥ 130mmHg)

### 3.2.2 분류 모델 KPI

| 지표 | 수식 | 최소 기준 | 목표 | 측정 시점 |
|------|------|----------|------|----------|
| Overall Accuracy | $\frac{\sum_c TP_c}{N}$ | 0.70 | **0.78** | 모델 학습 후 |
| Macro F1-Score | $\frac{1}{K}\sum_k F1_k$ | 0.65 | **0.74** | 모델 학습 후 |
| Weighted F1-Score | 클래스 샘플 수 가중 평균 F1 | 0.70 | **0.77** | 모델 학습 후 |
| 대사복합관리형 Recall | 복합 위험 유형 재현율 | **0.70** | 0.80 | 모델 학습 후 |
| 인접 유형 혼동률 | 인접 클래스 오분류 비율 | ≤ 0.25 | ≤ 0.18 | 모델 학습 후 |

> **대사복합관리형 Recall 우선**: 복합 위험 유형 미탐(False Negative)이 단순 오분류보다 더 큰 피해를 줄 수 있음

### 3.2.3 룰 기반 판정 KPI

| 조건 | 기준 | 검증 방법 |
|------|------|----------|
| BMI 판정 정확도 | 100% (산식 기반) | 단위 테스트 통과 |
| 혈압 판정 정확도 | 100% (기준표 기반) | 단위 테스트 통과 |
| 혈당 판정 정확도 | 100% (기준표 기반) | 단위 테스트 통과 |
| 콜레스테롤 판정 정확도 | 100% (기준표 기반) | 단위 테스트 통과 |

### 3.2.4 서비스 품질 KPI

| 지표 | 기준 | 목표 |
|------|------|------|
| 필수 입력만으로 결과 출력 | 성별+연령대+키+몸무게 입력 시 결과 제공 | **달성** |
| 결측 입력 처리 | 혈압·혈당·콜레스테롤 미입력 시 제한 안내 + 기본 결과 제공 | **달성** |
| Streamlit 앱 로딩 시간 | ≤ 5초 (첫 실행 제외) | **≤ 3초** |
| 모바일 화면 대응 | 스마트폰 세로 모드에서 좌우 스크롤 없음 | **달성** |
| 주의문구 표시 | 모든 결과 화면 하단에 의료적 고지 포함 | **달성** |

---

## 3.3 측정 방법 및 프로토콜

### 3.3.1 라벨 분포 분석

```python
import pandas as pd

# 룰 기반 라벨링 결과 분포 확인
label_counts = pd.Series(y_labels).value_counts()
label_pct = label_counts / len(y_labels) * 100

print("=== 건강관리 유형 라벨 분포 ===")
for label, count in label_counts.items():
    pct = label_pct[label]
    print(f"  {label:20s}: {count:,}건 ({pct:.1f}%)")

# 유형별 평균 수치 확인 (임상 일치성 검증)
df_labeled = df.copy()
df_labeled['health_type'] = y_labels

profile = df_labeled.groupby('health_type').agg(
    n=('bmi', 'count'),
    bmi_mean=('bmi', 'mean'),
    sbp_mean=('systolic_bp', 'mean'),
    fbg_mean=('fasting_glucose', 'mean'),
).round(1)

print("\n=== 유형별 평균 수치 (임상 일치성 확인) ===")
print(profile)
```

### 3.3.2 분류 모델 성능 계산

```python
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# Test set 평가 (반드시 Train/Val에 사용하지 않은 데이터)
y_pred = clf.predict(X_test_scaled)

metrics = {
    'overall_accuracy': accuracy_score(y_test, y_pred),
    'macro_f1': f1_score(y_test, y_pred, average='macro'),
    'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
    'per_class_f1': f1_score(y_test, y_pred, average=None),
    'confusion_matrix': confusion_matrix(y_test, y_pred),
    'classification_report': classification_report(
        y_test, y_pred,
        target_names=['basic', 'weight', 'blood_pressure',
                      'blood_sugar', 'metabolic', 'lifestyle']
    )
}

# 대사복합관리형 Recall 별도 계산
metabolic_idx = TYPE_NAMES.index('metabolic')  # 라벨명으로 인덱스 조회
cm = metrics['confusion_matrix']
metabolic_recall = cm[metabolic_idx, metabolic_idx] / cm[metabolic_idx].sum()
print(f"대사복합관리형 Recall: {metabolic_recall:.4f}")
```

### 3.3.3 비교군 통계 계산

```python
# 동일 성별·연령대 비교군 통계
def get_comparison_group_stats(df, gender, age_group):
    group = df[(df['gender'] == gender) & (df['age_group'] == age_group)]
    n = len(group)

    if n < 100:
        return None, n  # 비교군 미제공

    stats = {
        'n': n,
        'bmi_mean': group['bmi'].mean(),
        'bmi_std': group['bmi'].std(),
        'sbp_mean': group['systolic_bp'].mean(),
        'fbg_mean': group['fasting_glucose'].mean(),
        'chol_mean': group['total_cholesterol'].mean(),
        'smoking_rate': (group['smoking_status'] > 0).mean(),
    }
    return stats, n
```

---

## 3.4 정성적 목표

| 목표 | 검증 방법 |
|------|-----------|
| 간호 도메인 지식을 반영한 유형명 설계 | 유형별 평균 수치 + 도메인 해석 문서화 |
| 결측 입력에도 유연하게 동작하는 서비스 구조 | 각 입력 시나리오별 결과 출력 테스트 (5가지 케이스) |
| 의료적 리스크 배제 표현 완전 준수 | 금지 표현 목록 대조 검수 (10개 항목) |
| README만으로 재현 가능한 코드 품질 | 처음 보는 사람의 재현 테스트 |
| 포트폴리오 제출 가능 수준의 문서 품질 | 분석 노트북 5종 + README + 발표자료 |

---

## 3.5 중간 체크포인트

| 일정 | 체크포인트 | 판단 기준 |
|------|-----------|----------|
| 데이터 전처리 완료 | EDA + 전처리 완료 | 결측치·이상치 처리 완료, 파생변수 생성 완료 |
| 라벨링 완료 | 건강관리 유형 확정 | 룰 기반 라벨 분포 확인, 유형별 임상 수치 일치 검증 완료 |
| 분류 모델 완료 | 모델 학습 + 평가 완료 | Overall Accuracy ≥ 0.75 |
| 서비스 구현 완료 | Streamlit 앱 동작 | 필수 입력 → 결과 출력 동작 확인 |
| 배포 완료 | 웹앱 URL 접근 가능 | Streamlit Cloud 배포 성공 |
| 포트폴리오 완료 | 전체 산출물 완성 | README·노트북·발표자료 완성 |

---

## 3.6 실패 대응 계획

| 상황 | 판단 기준 | 대응 방안 |
|------|----------|----------|
| 분류 Accuracy < 0.65 | 경고 | 룰 기준값 재검토, XGBoost 대체, 과적합 여부 확인 |
| 소수 클래스 Recall < 0.55 | 경고 | class_weight 조정, 오버샘플링(SMOTE) 검토 |
| 대사복합관리형 Recall < 0.55 | 경고 | 해당 클래스 오버샘플링, 임계값 조정 |
| Streamlit 배포 실패 | 운영 | 로컬 실행 시연으로 대체, 배포 오류 문서화 |
| 데이터 용량 512MB 초과 | 운영 | 모델 파일 + 집계 통계만 배포, 원본 데이터 제외 |
