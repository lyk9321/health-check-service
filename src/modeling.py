"""
src/modeling.py

이 파일은 데이터를 학습시켜서 "예측 모델"을 만드는 곳이에요.

두 단계로 진행됩니다:

  STEP 1) 룰 기반 라벨링
    - 임상 기준 우선순위 규칙을 적용해 각 검진자에게 건강관리 유형 라벨을 직접 부여합니다
    - 예: 위험요인 3개 이상 → 대사복합관리형, 혈압 ≥ 130 → 혈압관리형
    - 이 라벨이 Random Forest의 정답(y)이 됩니다

  STEP 2) Random Forest 분류 모델 학습
    - STEP 1에서 만든 정답을 보고 "어떤 건강 수치가 어떤 유형인지" 패턴을 배웁니다
    - 학습이 끝나면, 새 사용자가 수치를 입력했을 때 그 사람의 유형을 예측할 수 있어요

[설계 배경]
  초기에는 K-Means 비지도학습으로 군집을 먼저 만들고 Random Forest를 학습하는 방식을 시도했으나,
  다음 문제가 발견되어 룰 기반 라벨링으로 전환했습니다:
    - 음주 변수에 군집이 과도하게 지배되어 BMI 정상인 군집이 체중관리형으로 매핑됨
    - Silhouette Score 0.21로 군집 경계가 불분명함
    - Confusion Matrix가 K-Means 경계 학습 결과를 반영해 의미없는 0.97 정확도 출력
    - 사전 정의된 임상 유형(6종)이 있는 상황에서 비지도학습 사용은 설계 모순

학습이 끝나면 3개의 파일이 model/ 폴더에 저장됩니다:
  - model/scaler.pkl              → 표준화 도구
  - model/classifier_model.pkl    → 분류 모델
  - model/cluster_mapping.json    → 유형 라벨 ↔ 한글명 매핑표
"""

import json                                      # JSON 파일 입출력
from pathlib import Path                         # 경로 다루기
import numpy as np                               # 숫자 계산
import pandas as pd                              # 표 다루기
import joblib                                    # 모델을 파일로 저장/불러오기

from sklearn.preprocessing import StandardScaler         # 데이터 표준화
from sklearn.ensemble import RandomForestClassifier      # 분류 알고리즘
from sklearn.model_selection import train_test_split     # 학습/검증/테스트 데이터 나누기
from sklearn.metrics import (                            # 모델 성능 평가 도구
    accuracy_score,           # 정확도
    f1_score,                 # F1 점수
    classification_report,    # 클래스별 성능 보고서
    confusion_matrix,         # 혼동 행렬
)

# 다른 파일에서 가져옵니다
from src.preprocessing import run_full_preprocessing
from src.feature_engineering import (
    generate_comparison_group_summary,
    generate_cluster_profile,
)


# ──────────────────────────────────────────────────────────────
# 전역 설정값
# ──────────────────────────────────────────────────────────────

# 모든 랜덤 결과를 같게 만드는 시드 (재현성 보장)
RANDOM_SEED = 42

# 모델 학습에 쓸 변수 7개 (콜레스테롤은 결측률 ~66%라 제외)
MODEL_FEATURES = [
    'bmi',              # 체질량지수
    'systolic_bp',      # 수축기혈압
    'diastolic_bp',     # 이완기혈압
    'fasting_glucose',  # 공복혈당
    'waist_cm',         # 허리둘레
    'smoking_status',   # 흡연 (1=비흡연, 2=과거, 3=현재)
    'drinking',         # 음주 (0=안함, 1=함)
]

# 건강관리 유형 정의 (임상 기준 기반, 고정값)
# K-Means처럼 학습마다 달라지지 않고 항상 동일한 매핑을 사용합니다
HEALTH_TYPE_MAPPING = {
    'basic':          {'name': '기본관리형',     'color': '#2ECC71'},
    'weight':         {'name': '체중관리형',     'color': '#F39C12'},
    'blood_pressure': {'name': '혈압관리형',     'color': '#E74C3C'},
    'blood_sugar':    {'name': '혈당관리형',     'color': '#E67E22'},
    'metabolic':      {'name': '대사복합관리형', 'color': '#C0392B'},
    'lifestyle':      {'name': '생활습관관리형', 'color': '#9B59B6'},
}


# ──────────────────────────────────────────────────────────────
# 1. 데이터 표준화 (Scaling)
# ──────────────────────────────────────────────────────────────
def fit_and_scale_features(df: pd.DataFrame) -> tuple:
    """
    각 변수의 단위가 다 다르니까(혈압은 100대, BMI는 20대) 똑같은 척도로 맞춰줍니다.
    → 평균 0, 표준편차 1로 변환하는 작업이에요. (= 표준화)

    이걸 안 하면 값이 큰 변수(혈압 120 등)가 작은 변수(BMI 23 등)보다
    Random Forest의 거리 계산에서 과도하게 영향을 미칠 수 있습니다.

    Args:
        df: 전처리 완료된 DataFrame

    Returns:
        (X_scaled, scaler) 튜플
        - X_scaled: 표준화된 데이터 (numpy 배열)
        - scaler: 학습된 StandardScaler 객체 (사용자 입력 변환에도 재사용)
    """
    # MODEL_FEATURES에 있는 컬럼들만 골라냅니다
    X = df[MODEL_FEATURES].values  # .values로 numpy 배열로 변환

    # StandardScaler 객체를 만들고
    scaler = StandardScaler()
    # fit_transform으로 학습 + 변환을 한 번에 합니다
    X_scaled = scaler.fit_transform(X)

    print(f"[fit_and_scale_features] 표준화 완료: {X_scaled.shape}")
    return X_scaled, scaler


# ──────────────────────────────────────────────────────────────
# 2. 룰 기반 라벨링
# ──────────────────────────────────────────────────────────────
def assign_labels_by_rule(df: pd.DataFrame) -> list:
    """
    임상 기준 우선순위에 따라 각 행에 건강관리 유형 라벨을 부여합니다.
    np.select()로 벡터화 처리 — iterrows() 대비 약 100배 빠름.

    우선순위 규칙 (위에서부터 차례로 검사, 처음 해당되는 유형으로 결정):
      1) 대사복합관리형: 위험요인 3개 이상 → 가장 시급한 복합 관리 필요
      2) 혈압관리형:    수축기 ≥ 130 또는 이완기 ≥ 80 → 심혈관 위험 즉각 관리
      3) 혈당관리형:    공복혈당 ≥ 100 → 당뇨병 전단계 조기 개입
      4) 체중관리형:    BMI ≥ 25 또는 복부비만 → 상기 위험요인 전 단계
      5) 생활습관관리형: 현재흡연 또는 음주 → 간접적 위험요인 관리
      6) 기본관리형:    위 항목 모두 해당 없음

    Args:
        df: 전처리 완료된 DataFrame (파생변수 포함 필수)

    Returns:
        라벨 문자열 리스트 (예: ['basic', 'metabolic', 'weight', ...])
    """
    # 조건 목록 (위에서부터 순서대로 적용, 먼저 해당되는 조건이 우선)
    conditions = [
        df['metabolic_risk_count'] >= 3,    # 우선순위 1: 위험요인이 3개 이상이면 대사복합관리형
        (df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80),    # 우선순위 2: 혈압이 고혈압 전단계 이상이면 혈압관리형
        df['fasting_glucose'] >= 100,       # 우선순위 3: 공복혈당이 공복혈당 장애 범위 이상이면 혈당관리형
        (df['bmi'] >= 25) | (df['abdominal_obesity_flag'] == 1),    # 우선순위 4: 비만(BMI≥25) 또는 복부비만이면 체중관리형
        (df['smoking_status'] == 3) | (df['drinking'] == 1),        # 우선순위 5: 현재흡연 또는 음주이면 생활습관관리형
    ]

    # 각 조건에 대응하는 라벨
    choices = [
        'metabolic',
        'blood_pressure',
        'blood_sugar',
        'weight',
        'lifestyle',
    ]

    # 어느 조건에도 해당 없으면 'basic'
    labels = np.select(conditions, choices, default='basic').tolist()   # 우선순위 6: 위 항목 모두 해당 없음 → 기본관리형

    # 라벨 분포 출력 (임상 일치성 확인용)
    label_series = pd.Series(labels)
    print("\n[assign_labels_by_rule] 유형별 라벨 분포:")
    for label, count in label_series.value_counts().items():
        pct = count / len(labels) * 100
        print(f"  - {label:20s}: {count:,}건 ({pct:.1f}%)")

    return labels


# ──────────────────────────────────────────────────────────────
# 3. Random Forest 분류 모델 학습
# ──────────────────────────────────────────────────────────────
def train_classifier(
    X_scaled: np.ndarray,
    y: list,
) -> tuple:
    """
    룰 기반으로 부여된 라벨을 정답으로 삼아 분류 모델을 학습합니다.

    Args:
        X_scaled: 표준화된 입력 데이터
        y: 룰 기반 라벨 리스트 (정답, 문자열)

    Returns:
        (classifier, metrics, test_data) 튜플
    """
    # numpy 배열로 변환 (sklearn 호환)
    y = np.array(y)
    # 데이터를 학습/검증/테스트로 나눕니다 (70/15/15)
    # stratify=y → 각 클래스 비율을 유지하면서 나누기
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"\n[train_classifier] 데이터 분할 완료:")
    print(f"  - Train: {len(X_train):,}개")
    print(f"  - Val:   {len(X_val):,}개")
    print(f"  - Test:  {len(X_test):,}개")

    # Random Forest 분류기 생성 및 학습
    # ⚠️ Streamlit Cloud 무료 플랜이 512MB라서 트리 수와 깊이를 제한합니다
    # n_estimators=200 + max_depth=None으로 하면 모델이 ~500MB가 됩니다
    # 100 + max_depth=15로 줄이면 ~50MB로 충분히 작아져요 (성능은 유사)
    clf = RandomForestClassifier(
        n_estimators=100,           # 트리 100개 (200 → 100으로 절반)
        max_depth=15,               # 깊이 15 제한 (메모리 절약)
        min_samples_split=20,       # 최소 20개 있어야 분기
        min_samples_leaf=10,        # 잎 노드에 최소 10개
        class_weight='balanced',    # 클래스 불균형 자동 보정
        random_state=RANDOM_SEED,
        n_jobs=-1,                  # CPU 전체 사용
        verbose=0,
    )

    print("[train_classifier] 학습 시작... (잠시 걸려요)")
    clf.fit(X_train, y_train)
    print("[train_classifier] 학습 완료!")

    # Test set으로 성능 평가
    y_pred = clf.predict(X_test)

    # 유형명 목록 (정렬된 순서로 — confusion matrix 레이블 고정용)
    type_names = sorted(HEALTH_TYPE_MAPPING.keys())

    # 각종 지표 계산
    metrics = {
        'overall_accuracy': float(accuracy_score(y_test, y_pred)),
        'macro_f1': float(f1_score(y_test, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_test, y_pred, average='weighted')),
        'confusion_matrix': confusion_matrix(
            y_test, y_pred, labels=type_names
        ).tolist(),
        'classification_report': classification_report(
            y_test, y_pred, target_names=type_names
        ),
        'n_test': int(len(y_test)),
    }

    print(f"\n[train_classifier] Test set 평가 결과:")
    print(f"  - Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  - Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  - Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"\n{metrics['classification_report']}")

    return clf, metrics, (X_test, y_test, y_pred)


# ──────────────────────────────────────────────────────────────
# 4. 모델 저장
# ──────────────────────────────────────────────────────────────
def save_models(
    scaler,
    classifier,
    model_dir: str = 'model',
):
    """
    학습된 모델 2개 + 매핑 정보를 디스크에 저장합니다.
    Streamlit 앱이 실행될 때 이 파일들을 다시 불러와서 예측에 사용합니다.

    저장되는 파일:
      - scaler.pkl            → 표준화 도구
      - classifier_model.pkl  → Random Forest 분류 모델
      - cluster_mapping.json  → 유형 라벨 ↔ 한글명/색상 매핑표
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 모델 파일 저장
    joblib.dump(scaler, model_dir / 'scaler.pkl')
    joblib.dump(classifier, model_dir / 'classifier_model.pkl')

    # 유형 매핑 정보 저장 (키가 문자열이므로 변환 불필요)
    with open(model_dir / 'cluster_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(HEALTH_TYPE_MAPPING, f, ensure_ascii=False, indent=2)

    print(f"\n[save_models] 모델 저장 완료 ({model_dir}):")
    for file in sorted(model_dir.iterdir()):
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name}: {size_kb:.1f} KB")


# ──────────────────────────────────────────────────────────────
# 5. 전체 학습 파이프라인 (한 번에 실행)
# ──────────────────────────────────────────────────────────────
def run_full_training(
    raw_csv_path: str = 'data/raw/2024_국민건강보험공단_건강검진정보.CSV',
    processed_dir: str = 'data/processed',
    model_dir='../model',
):
    """
    전처리 → 표준화 → 룰 기반 라벨링 → 분류 학습 → 저장까지 한 번에 실행합니다.

    이 함수만 호출하면 model/ 폴더에 모든 결과가 만들어집니다.
    """
    print("=" * 60)
    print("Step 1: 전처리")
    print("=" * 60)
    df = run_full_preprocessing(raw_csv_path)

    print("\n" + "=" * 60)
    print("Step 2: 표준화")
    print("=" * 60)
    X_scaled, scaler = fit_and_scale_features(df)

    print("\n" + "=" * 60)
    print("Step 3: 룰 기반 라벨링")
    print("=" * 60)
    y_labels = assign_labels_by_rule(df)

    print("\n" + "=" * 60)
    print("Step 4: Random Forest 분류 학습")
    print("=" * 60)
    classifier, metrics, _ = train_classifier(X_scaled, y_labels)

    print("\n" + "=" * 60)
    print("Step 5: 모델 저장")
    print("=" * 60)
    save_models(scaler, classifier, model_dir=model_dir)

    print("\n" + "=" * 60)
    print("Step 6: 비교군·유형 프로파일 CSV 생성")
    print("=" * 60)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    generate_comparison_group_summary(
        df,
        output_path=str(processed_dir / 'comparison_group_summary.csv'),
    )
    generate_cluster_profile(
        df,
        y_labels,
        HEALTH_TYPE_MAPPING,
        output_path=str(processed_dir / 'cluster_profile.csv'),
    )

    print("\n" + "=" * 60)
    print("✅ 전체 학습 파이프라인 완료!")
    print("=" * 60)

    return {
        'metrics': metrics,
        'health_type_mapping': HEALTH_TYPE_MAPPING,
    }


if __name__ == '__main__':
    run_full_training()
