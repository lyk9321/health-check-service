"""
src/predict.py

사용자가 입력한 정보를 받아서 학습된 모델로 건강관리 유형을 예측하고,
룰 기반 해석·비교군 비교·추천 문구를 모두 합쳐 최종 결과를 만드는 곳이에요.

핵심 함수: predict_health_type()
  → 사용자 입력 dict 하나 넣으면, 결과 dict 하나가 나옵니다.
  → Streamlit 앱에서 이 함수만 호출하면 모든 결과가 준비됩니다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# 룰 엔진과 추천 함수 가져오기
from src.recommend import (
    run_rule_engine,
    get_recommendations,
    get_exercise_recommendation,
    compare_with_group,
)


# 모델 학습에 쓴 변수 7개 (modeling.py와 동일해야 함!)
MODEL_FEATURES = [
    'bmi',
    'systolic_bp',
    'diastolic_bp',
    'fasting_glucose',
    'waist_cm',
    'smoking_status',
    'drinking',
]


# ──────────────────────────────────────────────────────────────
# 1. BMI 계산 (사용자 키·몸무게 → BMI)
# ──────────────────────────────────────────────────────────────
def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """
    BMI를 계산합니다. (수학식: 몸무게 / 키(m)의 제곱)

    예: 키 170cm, 몸무게 70kg → 70 / 1.7^2 = 24.2
    """
    # 입력값 유효성 검사
    if not (100 <= height_cm <= 250):
        raise ValueError(f'키는 100~250cm 사이여야 합니다 (입력값: {height_cm})')
    if not (20 <= weight_kg <= 300):
        raise ValueError(f'몸무게는 20~300kg 사이여야 합니다 (입력값: {weight_kg})')

    # 키를 m로 변환 후 BMI 계산
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # 소수점 1자리로 반올림
    return round(bmi, 1)


# ──────────────────────────────────────────────────────────────
# 2. 입력 완성도 계산 (0~100%)
# ──────────────────────────────────────────────────────────────
def calculate_completeness(user_input: dict) -> int:
    """
    사용자가 얼마나 많은 항목을 입력했는지 0~100%로 계산합니다.

    가중치:
        기본정보(성별, 연령대, 키, 몸무게): 35%
        생활습관(흡연, 음주):              15%
        혈압(수축기+이완기 둘 다):          15%
        공복혈당:                          15%
        지질 수치(4종 중 1개 이상):         10%
        허리둘레:                           5%
        운동 정보:                          5%
    """
    score = 0

    # 필수 (35%)
    required = ['gender', 'age_group', 'height_cm', 'weight_kg']
    if all(user_input.get(k) is not None for k in required):
        score += 35

    # 생활습관 (15%) - 흡연·음주 둘 다 있어야 인정
    if (user_input.get('smoking_status') is not None and
            user_input.get('drinking') is not None):
        score += 15

    # 혈압 (15%) - 수축기·이완기 둘 다 있어야 인정
    if (user_input.get('systolic_bp') is not None and
            user_input.get('diastolic_bp') is not None):
        score += 15

    # 공복혈당 (15%)
    if user_input.get('fasting_glucose') is not None:
        score += 15

    # 지질 수치 (10%) - 4종 중 하나라도 있으면 인정
    lipid_keys = ['ldl_cholesterol', 'hdl_cholesterol',
                  'triglyceride', 'total_cholesterol']
    if any(user_input.get(k) is not None for k in lipid_keys):
        score += 10

    # 허리둘레 (5%)
    if user_input.get('waist_cm') is not None:
        score += 5

    # 운동 정보 (5%)
    if user_input.get('exercise') is not None:
        score += 5

    return score


# ──────────────────────────────────────────────────────────────
# 3. 사용자 입력 → 모델 입력 벡터로 변환
# ──────────────────────────────────────────────────────────────
def prepare_user_input(
    user_input: dict,
    comparison_stats: pd.DataFrame,
) -> tuple:
    """
    사용자 입력을 모델이 이해할 수 있는 7개 변수의 numpy 배열로 변환합니다.
    빈 항목(미입력)은 같은 성별·연령대 그룹의 평균값으로 채웁니다.

    Args:
        user_input: 사용자 입력 dict
        comparison_stats: 성별·연령대별 비교군 통계 DataFrame

    Returns:
        (X, missing_fields) 튜플
        - X: shape (1, 7)인 numpy 배열 (모델 입력)
        - missing_fields: 평균값으로 대체된 항목명 리스트 (사용자 안내용)
    """
    gender = user_input.get('gender')
    age_group = user_input.get('age_group')

    # 비교군에서 사용자의 (성별, 연령대)에 해당하는 1행 찾기
    group_match = comparison_stats[
        (comparison_stats['gender'] == gender) &
        (comparison_stats['age_group'] == age_group)
    ]

    # 매칭되는 그룹이 없으면 전체 평균을 fallback으로
    if len(group_match) == 0:
        # 같은 성별 평균이라도
        same_gender = comparison_stats[comparison_stats['gender'] == gender]
        if len(same_gender) > 0:
            group_match = same_gender.head(1)
        else:
            # 진짜 아무것도 없으면 전체 첫 행 (방어 코드)
            group_match = comparison_stats.head(1)

    group = group_match.iloc[0]  # 1행을 Series로 꺼내기

    # 미입력 항목을 무엇으로 대체할지 매핑
    # (모델 변수 → 비교군 통계의 컬럼명)
    fill_map = {
        'systolic_bp':     'sbp_mean',
        'diastolic_bp':    'dbp_mean',
        'fasting_glucose': 'fbg_mean',
        'waist_cm':        'waist_mean',
    }

    # 흡연·음주는 비교군 평균 비율로 대체
    # smoking_status: 그룹 흡연율(현재흡연 비율)을 보고 적절히 1~3 중 하나로
    # drinking: 그룹 음주율 (0~1) 그대로 사용 (모델은 평균값 받아도 OK)

    missing_fields = []   # 대체된 항목 목록
    feature_values = {}   # 채워진 값들

    # BMI는 키·몸무게로 계산 (필수 입력이므로 missing 없음)
    if user_input.get('bmi') is not None:
        feature_values['bmi'] = user_input['bmi']
    elif user_input.get('height_cm') and user_input.get('weight_kg'):
        feature_values['bmi'] = calculate_bmi(
            user_input['height_cm'], user_input['weight_kg']
        )
    else:
        # 정말 정보가 없을 때만 그룹 평균으로 (이 경우엔 사실 서비스 못 씀)
        feature_values['bmi'] = group['bmi_mean']
        missing_fields.append('bmi')

    # 혈압·혈당·허리둘레: 입력값 우선, 없으면 그룹 평균
    for feat, group_col in fill_map.items():
        if user_input.get(feat) is not None:
            feature_values[feat] = user_input[feat]
        else:
            feature_values[feat] = group[group_col]
            missing_fields.append(feat)

    # 흡연: 입력 있으면 그대로, 없으면 그룹의 현재흡연 비율로 1 또는 3 추정
    if user_input.get('smoking_status') is not None:
        feature_values['smoking_status'] = user_input['smoking_status']
    else:
        # 그룹의 현재흡연 비율이 0.5 넘으면 3(현재흡연), 아니면 1(비흡연)
        # 실제로는 1이 압도적이라 거의 1이 될 거예요
        smoking_rate = group.get('smoking_rate', 0)
        feature_values['smoking_status'] = 3 if smoking_rate > 0.5 else 1
        missing_fields.append('smoking_status')

    # 음주: 입력 있으면 그대로, 없으면 그룹 음주율 (소수)
    if user_input.get('drinking') is not None:
        feature_values['drinking'] = user_input['drinking']
    else:
        drinking_rate = group.get('drinking_rate', 0)
        feature_values['drinking'] = 1 if drinking_rate >= 0.5 else 0
        missing_fields.append('drinking')

    # MODEL_FEATURES 순서대로 numpy 배열 만들기 (1행 7열)
    X = np.array(
        [feature_values[f] for f in MODEL_FEATURES],
        dtype=float,
    ).reshape(1, -1)

    return X, missing_fields


# ──────────────────────────────────────────────────────────────
# 4. 모델 + 비교군 데이터 로딩 (Streamlit 캐시용)
# ──────────────────────────────────────────────────────────────
def load_models(model_dir: str = 'model'):
    """
    저장된 모델과 매핑 정보를 모두 불러옵니다.
    Streamlit에서 한 번만 로드하고 재사용하기 위해 분리한 함수입니다.

    Returns:
        (scaler, classifier, cluster_mapping) 튜플
    """
    model_dir = Path(model_dir)

    scaler = joblib.load(model_dir / 'scaler.pkl')
    classifier = joblib.load(model_dir / 'classifier_model.pkl')

    with open(model_dir / 'cluster_mapping.json', encoding='utf-8') as f:
        cluster_mapping_raw = json.load(f)

    # 키가 이제 문자열('basic', 'weight' 등)이므로 변환 불필요
    cluster_mapping = cluster_mapping_raw

    return scaler, classifier, cluster_mapping


def load_comparison_stats(
    path: str = 'data/processed/comparison_group_summary.csv',
) -> pd.DataFrame:
    """비교군 통계 CSV를 불러옵니다."""
    return pd.read_csv(path, encoding='utf-8-sig')


# ──────────────────────────────────────────────────────────────
# 5. 메인 함수: 사용자 입력 → 종합 결과
# ──────────────────────────────────────────────────────────────
def predict_health_type(
    user_input: dict,
    scaler,
    classifier,
    cluster_mapping: dict,
    comparison_stats: pd.DataFrame,
) -> dict:
    """
    사용자 입력 → 건강관리 유형 예측 + 룰 해석 + 비교군 비교 + 추천 한 번에.

    이 함수가 서비스의 메인 엔진이에요.
    Streamlit에서는 이 함수만 호출하면 결과 카드 8개에 필요한 모든 데이터가 나옵니다.

    Args:
        user_input: 사용자 입력 dict
        scaler, classifier, cluster_mapping: 학습된 모델 자료
        comparison_stats: 비교군 통계 DataFrame

    Returns:
        결과 dict (구조는 함수 끝에서 확인)
    """
    # ── (0) BMI 자동 계산 ──
    if user_input.get('bmi') is None:
        if user_input.get('height_cm') and user_input.get('weight_kg'):
            user_input['bmi'] = calculate_bmi(
                user_input['height_cm'], user_input['weight_kg']
            )

    # ── (1) 입력 완성도 ──
    completeness = calculate_completeness(user_input)

    # ── (2) 룰 기반 판정 ──
    rule_results = run_rule_engine(user_input)

    # ── (3) 모델 예측 ──
    X, missing_fields = prepare_user_input(user_input, comparison_stats)
    X_scaled = scaler.transform(X)

    # classifier.predict()는 이제 문자열 라벨 반환 ('basic', 'weight' 등)
    health_type_label = classifier.predict(X_scaled)[0]

    # 각 클래스 확률도 가져오기 (예측 신뢰도 표시용)
    proba = classifier.predict_proba(X_scaled)[0]
    pred_proba = float(proba[list(classifier.classes_).index(health_type_label)])

    # 라벨 → 유형 정보로 변환
    health_type_info = cluster_mapping[health_type_label]
    health_type = {
        'label': health_type_label,
        'name': health_type_info['name'],
        'color': health_type_info['color'],
        'probability': round(pred_proba, 3),
    }

    # ── (4) 비교군 비교 ──
    gender = user_input.get('gender')
    age_group = user_input.get('age_group')
    group_match = comparison_stats[
        (comparison_stats['gender'] == gender) &
        (comparison_stats['age_group'] == age_group)
    ]

    if len(group_match) > 0:
        group_stats = group_match.iloc[0].to_dict()
        comparison = compare_with_group(user_input, group_stats)
        group_info = {
            'gender_label': '남성' if gender == 1 else '여성',
            'age_label': group_stats.get('age_label', ''),
            'n': int(group_stats.get('n', 0)),
        }
    else:
        comparison = []
        group_info = None

    # ── (5) 추천 문구 ──
    recommendations = get_recommendations(health_type['label'], rule_results)

    # ── (6) 운동 분석 (입력됐을 때만) ──
    exercise_result = None
    if user_input.get('exercise'):
        exercise_result = get_exercise_recommendation(user_input['exercise'])

    # ── (7) 결과 합치기 ──
    return {
        'health_type': health_type,
        'bmi': user_input.get('bmi'),
        'rule_results': rule_results,
        'comparison': comparison,
        'group_info': group_info,
        'missing_fields': missing_fields,
        'completeness': completeness,
        'recommendations': recommendations,
        'exercise': exercise_result,
    }


# ──────────────────────────────────────────────────────────────
# 6. 직접 실행 시 간단 테스트
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 모델 로딩
    scaler, classifier, cluster_mapping = load_models()
    comparison_stats = load_comparison_stats()

    # 가상의 사용자 입력
    sample_input = {
        'gender': 2,           # 여성
        'age_group': 7,        # 35-39세
        'height_cm': 165,
        'weight_kg': 70,       # BMI 약 25.7 (비만)
        'systolic_bp': 135,
        'diastolic_bp': 88,
        'fasting_glucose': 110,
        'waist_cm': 85,
        'smoking_status': 1,
        'drinking': 1,
        'ldl_cholesterol': 145,
        'hdl_cholesterol': 52,
        'exercise': {
            'frequency': '주1-2회',
            'intensity': '중강도',
            'duration_min': 30,
        },
    }

    result = predict_health_type(
        sample_input, scaler, classifier, cluster_mapping, comparison_stats
    )

    print("=" * 60)
    print(f"건강관리 유형: {result['health_type']['name']} "
          f"(확률 {result['health_type']['probability']:.1%})")
    print(f"BMI: {result['bmi']}")
    print(f"입력 완성도: {result['completeness']}%")
    print(f"위험요인 개수: {result['rule_results']['metabolic_risk_count']}")
    print()
    print("[비교군 비교]")
    if result['group_info']:
        print(f"  → {result['group_info']['age_label']} {result['group_info']['gender_label']} "
              f"검진자 {result['group_info']['n']:,}명 대비")
    for c in result['comparison']:
        print(f"  • {c['message']}")
    print()
    print("[추천]")
    for r in result['recommendations']:
        print(f"  - {r}")
    print()
    if result['exercise']:
        print("[운동 분석]")
        print(f"  {result['exercise']['message']}")
        print(f"  → {result['exercise']['suggestion']}")
