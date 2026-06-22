"""
src/feature_engineering.py

배포용 집계 데이터를 만드는 곳이에요.

왜 집계 데이터가 필요할까요?
  - 원본 100만 건은 용량이 너무 커서 Streamlit 무료 배포에 못 올라갑니다.
  - 그래서 미리 "성별×연령대별 평균 건강 수치"만 뽑아서 작은 CSV(~10KB)로 만들고,
    서비스에서는 그 CSV만 읽어서 비교군 비교를 합니다.

여기서 만드는 파일 2개:
  1) data/processed/comparison_group_summary.csv
     → 성별×연령대별 BMI/혈압/혈당/허리둘레 평균과 표준편차
     → 사용자가 본인 수치를 비교할 때 사용
  2) data/processed/cluster_profile.csv
     → 군집(건강관리 유형)별 평균 수치
     → 서비스에서 "당신은 OO관리형이고, 이 유형의 평균 수치는 이렇습니다" 안내에 사용
"""

import pandas as pd  # 표 다루는 도구
import numpy as np   # 숫자 계산 도구


# ──────────────────────────────────────────────────────────────
# 1. 비교군 통계 만들기 (성별 × 연령대별 평균)
# ──────────────────────────────────────────────────────────────
def generate_comparison_group_summary(
    df: pd.DataFrame,
    output_path: str = 'data/processed/comparison_group_summary.csv',
) -> pd.DataFrame:
    """
    성별 × 연령대 조합별로 주요 건강 수치의 평균과 표준편차를 계산합니다.

    예: 30대 여성 그룹의 BMI 평균 22.8, 표준편차 3.1
        → 사용자가 30대 여성이고 BMI 25.5라면
          "비교군 평균보다 약간 높음" 같은 안내가 가능해집니다.

    Args:
        df: 전처리 완료된 DataFrame (preprocessing.py 결과)
        output_path: 저장할 CSV 경로

    Returns:
        집계 통계 DataFrame
    """
    # groupby(['gender', 'age_group'])는 (성별, 연령대) 조합으로 묶기
    # agg(...)는 그룹별로 여러 통계를 한 번에 계산
    summary = df.groupby(['gender', 'age_group']).agg(
        # 각 줄: 새 컬럼명 = (어떤 컬럼, 어떤 함수)
        n=('bmi', 'count'),                      # 그룹 사람 수 (n)
        bmi_mean=('bmi', 'mean'),                # BMI 평균
        bmi_std=('bmi', 'std'),                  # BMI 표준편차 (얼마나 흩어져 있는지)
        sbp_mean=('systolic_bp', 'mean'),        # 수축기혈압 평균
        sbp_std=('systolic_bp', 'std'),
        dbp_mean=('diastolic_bp', 'mean'),       # 이완기혈압 평균
        dbp_std=('diastolic_bp', 'std'),
        fbg_mean=('fasting_glucose', 'mean'),    # 공복혈당 평균
        fbg_std=('fasting_glucose', 'std'),
        waist_mean=('waist_cm', 'mean'),         # 허리둘레 평균
        waist_std=('waist_cm', 'std'),
        # smoking_status==3은 "현재흡연" 코드. lambda는 "이렇게 계산해줘"라는 임시 함수
        smoking_rate=('smoking_status', lambda x: (x == 3).mean()),
        drinking_rate=('drinking', 'mean'),      # 음주 비율
    ).reset_index()  # groupby 후엔 인덱스가 다단계라서 평평하게 풀어줍니다

    # 연령대 코드(5, 6, 7, ...) → "25-29세" 같은 문자열 라벨도 추가
    age_label_map = {
        5: '25-29세',  6: '30-34세',  7: '35-39세',  8: '40-44세',
        9: '45-49세', 10: '50-54세', 11: '55-59세', 12: '60-64세',
        13: '65-69세', 14: '70-74세', 15: '75-79세', 16: '80-84세', 
        17: '85-89세', 18:'90+세'
    }
    summary['age_label'] = summary['age_group'].map(age_label_map)

    # 보기 좋게 소수점 2자리로 반올림 (n과 코드 컬럼 빼고)
    numeric_cols = [c for c in summary.columns
                    if c not in ['gender', 'age_group', 'age_label', 'n']]
    summary[numeric_cols] = summary[numeric_cols].round(2)

    # CSV로 저장 (utf-8-sig는 엑셀에서 한글 안 깨지게 해주는 인코딩)
    summary.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"[generate_comparison_group_summary] 저장 완료: {output_path}")
    print(f"  → 총 {len(summary)}개 그룹 (성별 2 × 연령대 13)")

    return summary


# ──────────────────────────────────────────────────────────────
# 2. 군집(건강관리 유형) 프로파일 만들기
# ──────────────────────────────────────────────────────────────
def generate_cluster_profile(
    df: pd.DataFrame,
    labels: list,
    health_type_mapping: dict,
    output_path: str = 'data/processed/cluster_profile.csv',
) -> pd.DataFrame:
    """
    각 건강관리 유형의 특징 요약표를 만듭니다.

    예시 결과:
      유형명          | 인원수  | BMI평균 | 혈압평균 | 혈당평균 | ...
      기본관리형       | 380,000 | 21.8   | 111    | 91      | ...
      체중관리형       | 290,000 | 27.1   | 118    | 97      | ...
      대사복합관리형   |  50,000 | 29.3   | 136    | 112     | ...

    Args:
        df: 전처리된 DataFrame (룰 기반 라벨링에 사용한 것과 같은 데이터)
        labels: 룰 기반 라벨 리스트 (문자열, 예: ['basic', 'weight', ...])
        health_type_mapping: {'basic': {'name':..., 'color':...}, ...}
        output_path: 저장 경로

    Returns:
        유형별 프로파일 DataFrame
    """
    # 원본을 건드리지 않으려고 사본 만들고
    df = df.copy()
    # 라벨을 새 컬럼으로 추가 (이미 문자열 라벨)
    df['health_type'] = labels

    # 유형별로 묶어서 평균 등을 계산
    profile = df.groupby('health_type').agg(
        n=('bmi', 'count'),
        bmi_mean=('bmi', 'mean'),
        sbp_mean=('systolic_bp', 'mean'),
        dbp_mean=('diastolic_bp', 'mean'),
        fbg_mean=('fasting_glucose', 'mean'),
        waist_mean=('waist_cm', 'mean'),
        smoking_current_rate=('smoking_status', lambda x: (x == 3).mean()),
        drinking_rate=('drinking', 'mean'),
        metabolic_risk_mean=('metabolic_risk_count', 'mean'),
    ).reset_index()

    # health_type_mapping에서 한글명과 색상 추가
    profile['health_type_name'] = profile['health_type'].map(
        lambda t: health_type_mapping.get(t, {}).get('name', t)
    )
    profile['color'] = profile['health_type'].map(
        lambda t: health_type_mapping.get(t, {}).get('color', '#999999')
    )

    # 보기 좋게 소수점 정리
    numeric_cols = [c for c in profile.columns
                    if c not in ['health_type', 'health_type_name', 'color', 'n']]
    profile[numeric_cols] = profile[numeric_cols].round(2)

    # 컬럼 순서 정리
    profile = profile[[
        'health_type', 'health_type_name', 'color', 'n',
        'bmi_mean', 'sbp_mean', 'dbp_mean', 'fbg_mean', 'waist_mean',
        'smoking_current_rate', 'drinking_rate', 'metabolic_risk_mean',
    ]]

    # 저장
    profile.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"[generate_cluster_profile] 저장 완료: {output_path}")
    print(f"  → 총 {len(profile)}개 유형")

    return profile


if __name__ == '__main__':
    # 직접 실행 테스트는 modeling.py 실행 시 같이 진행됩니다
    print("이 모듈은 modeling.py에서 호출됩니다.")
