# 09. 모듈별 함수 명세 (Module Specifications)

> **문서 버전**: v1.2.0 | **최종 수정**: 2026-05-07

---

## 9.1 src/predict.py

```python
def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """
    BMI를 계산합니다.

    Args:
        height_cm: 키 (cm), 유효 범위 100~250
        weight_kg: 몸무게 (kg), 유효 범위 20~300

    Returns:
        BMI 수치 (소수점 1자리)

    Raises:
        ValueError: 입력값이 유효 범위를 벗어난 경우

    Example:
        >>> calculate_bmi(170, 70)
        24.2
    """


def classify_bmi(bmi: float) -> dict:
    """
    BMI 수치를 분류하고 출력 문구를 반환합니다.
    (기준: 대한비만학회 2022)

    Args:
        bmi: BMI 수치

    Returns:
        {
            'category': str,   # 'underweight' | 'normal' | 'overweight' | 'obese'
            'label': str,      # '저체중' | '정상' | '과체중' | '비만'
            'message': str,    # 화면 출력 참고 문구
            'flag': int,       # 0=정상이하, 1=과체중, 2=비만 (파생변수 bmi_category)
        }
    """


def calculate_completeness(user_input: dict) -> int:
    """
    사용자 입력 완성도를 0~100 정수로 계산합니다.

    Args:
        user_input: 사용자 입력값 딕셔너리

    Returns:
        완성도 (0~100 정수)

    항목별 가중치:
        기본정보 (성별, 연령대, 키, 몸무게): 35%
        생활습관 (흡연, 음주):               15%
        혈압:                               15%
        공복혈당:                            15%
        지질 수치 (LDL/HDL/TG/총콜레스테롤 중 1개 이상): 10%
        허리둘레:                             5%
        운동 정보:                            5%
    """


def prepare_user_input(
    user_input: dict,
    comparison_stats: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    사용자 입력값을 모델 입력 벡터로 변환합니다.
    결측 항목은 해당 성별·연령대 평균으로 대체합니다.

    Args:
        user_input: 사용자 입력값 딕셔너리
        comparison_stats: 성별·연령대별 비교군 통계 DataFrame

    Returns:
        Tuple[
            X: np.ndarray (1 × 7),     # 표준화 전 입력 벡터
            missing_fields: list[str], # 대체된 항목 목록 (경고 문구 생성용)
        ]

    MODEL_FEATURES 순서:
        ['bmi', 'systolic_bp', 'diastolic_bp', 'fasting_glucose',
         'waist_cm', 'smoking_status', 'drinking']
    """


def predict_health_type(
    user_input: dict,
    scaler,
    classifier,
    cluster_mapping: dict,
    comparison_stats: pd.DataFrame,
) -> dict:
    """
    사용자 입력을 받아 건강관리 유형을 예측하고 전체 결과를 반환합니다.

    Args:
        user_input: 사용자 입력값 딕셔너리
        scaler: 학습된 StandardScaler
        classifier: 학습된 Random Forest 분류기
        cluster_mapping: 군집 ID → 유형 정보 매핑 dict
        comparison_stats: 비교군 통계 DataFrame

    Returns:
        {
            'health_type': {
                'id': int,
                'label': str,       # 'basic' | 'weight' | 'blood_pressure' | ...
                'name': str,        # '기본관리형' | '체중관리형' | ...
                'color': str,       # hex 색상
                'probability': float,
            },
            'bmi': float,
            'bmi_result': dict,
            'rule_results': dict,
            'comparison': list,
            'missing_fields': list[str],
            'completeness': int,
            'recommendations': list[str],
        }
    """
```

---

## 9.2 src/recommend.py

```python
def run_rule_engine(user_input: dict) -> dict:
    """
    BMI·혈압·혈당·콜레스테롤 기준표에 따라 룰 기반 판정을 수행합니다.
    기준 출처: 대한비만학회(BMI), 대한고혈압학회(혈압), 대한당뇨병학회(혈당),
              한국지질·동맥경화학회(콜레스테롤) 2022~2023 기준

    Args:
        user_input: 사용자 입력값 딕셔너리

    Returns:
        {
            'bmi': BMI 판정 결과,
            'blood_pressure': 혈압 판정 결과 (입력 없으면 None),
            'fasting_glucose': 혈당 판정 결과 (입력 없으면 None),
            'ldl_cholesterol': LDL 판정 결과 (입력 없으면 None),
            'hdl_cholesterol': HDL 판정 결과 (입력 없으면 None),
            'triglyceride': 트리글리세라이드 판정 결과 (입력 없으면 None),
            'total_cholesterol': 총콜레스테롤 판정 결과 (LDL/HDL/TG 모두 미입력 시만 표시),
            'waist': 허리둘레 판정 결과 (입력 없으면 None),
            'metabolic_risk_count': int,
        }
    """


def interpret_blood_pressure(systolic: float, diastolic: float) -> dict:
    """
    혈압을 대한고혈압학회 2022 기준으로 해석합니다.

    기준:
        정상:      수축기 < 120 AND 이완기 < 80
        주의:      수축기 120~129 AND 이완기 < 80
        고혈압 전단계: 수축기 130~139 OR 이완기 80~89
        고혈압 범위: 수축기 ≥ 140 OR 이완기 ≥ 90

    Returns:
        {
            'category': str,  # 'normal' | 'elevated' | 'stage1' | 'stage2'
            'label': str,
            'message': str,
            'flag': int,      # 0=정상, 1=관리필요
        }
    """


def interpret_fasting_glucose(glucose: float) -> dict:
    """
    공복혈당을 대한당뇨병학회 2023 기준으로 해석합니다.

    기준:
        정상:        < 100 mg/dL
        공복혈당 장애: 100~125 mg/dL
        당뇨병 범위:  ≥ 126 mg/dL

    Returns:
        {
            'category': str,  # 'normal' | 'impaired' | 'diabetic_range'
            'label': str,
            'message': str,
            'flag': int,
        }
    """


def interpret_ldl_cholesterol(ldl: float) -> dict:
    """
    LDL 콜레스테롤을 한국지질·동맥경화학회 2022 기준으로 해석합니다.
    심혈관 위험 1차 지표로, 낮을수록 좋습니다.

    기준:
        적정:      < 100 mg/dL
        정상 상한:  100~129 mg/dL
        경계:      130~159 mg/dL
        높음:      160~189 mg/dL
        매우 높음:  ≥ 190 mg/dL

    Returns:
        {
            'category': str,  # 'optimal' | 'near_optimal' | 'borderline' | 'high' | 'very_high'
            'label': str,
            'message': str,
            'flag': int,      # 0=정상이하, 1=경계이상 (관리 필요)
        }
    """


def interpret_hdl_cholesterol(hdl: float, gender: int) -> dict:
    """
    HDL 콜레스테롤을 한국지질·동맥경화학회 2022 기준으로 해석합니다.
    심혈관 보호 인자로, 높을수록 좋습니다.
    성별에 따라 기준이 다릅니다 (여성이 더 엄격).

    기준:
        낮음:  < 40 mg/dL (남성) / < 50 mg/dL (여성)
        보통:  40~59 mg/dL (남성) / 50~59 mg/dL (여성)
        높음:  ≥ 60 mg/dL (공통, 심혈관 보호)

    Args:
        hdl: HDL 콜레스테롤 수치 (mg/dL)
        gender: 성별 코드 (1=남성, 2=여성)

    Returns:
        {
            'category': str,  # 'low' | 'normal' | 'high'
            'label': str,
            'message': str,
            'flag': int,      # 0=보통이상, 1=낮음 (관리 필요)
        }
    """


def interpret_triglyceride(tg: float) -> dict:
    """
    트리글리세라이드(중성지방)를 한국지질·동맥경화학회 2022 기준으로 해석합니다.

    기준:
        적정:      < 150 mg/dL
        경계:      150~199 mg/dL
        높음:      200~499 mg/dL
        매우 높음:  ≥ 500 mg/dL

    Returns:
        {
            'category': str,  # 'optimal' | 'borderline' | 'high' | 'very_high'
            'label': str,
            'message': str,
            'flag': int,      # 0=적정, 1=경계이상 (관리 필요)
        }
    """


def interpret_total_cholesterol(cholesterol: float) -> dict:
    """
    총콜레스테롤을 해석합니다.
    LDL·HDL·TG 중 1개 이상 입력된 경우 이 함수를 호출하지 않음.
    4종 모두 미입력 시 보조 참고 지표로만 사용.

    기준:
        적정:   < 200 mg/dL
        경계:   200~239 mg/dL
        높음:   ≥ 240 mg/dL

    Returns:
        {
            'category': str,  # 'optimal' | 'borderline' | 'high'
            'label': str,
            'message': str,
            'flag': int,
            'is_supplementary': bool,  # 항상 True (보조 참고 표시용)
        }
    """


def interpret_cholesterol_panel(user_input: dict) -> list[dict]:
    """
    사용자 입력값에서 지질 수치 4종을 확인하고 해당 항목만 해석합니다.
    LDL → HDL → TG → 총콜레스테롤 순서로 우선순위를 적용합니다.

    Args:
        user_input: 사용자 입력값 딕셔너리

    Returns:
        입력된 항목별 판정 결과 list (미입력 항목은 포함하지 않음)

    우선순위 로직:
        1. LDL, HDL, TG 입력된 항목 각각 해석
        2. 총콜레스테롤: LDL/HDL/TG 모두 미입력 시에만 보조 참고로 해석
           (LDL/HDL/TG 중 1개라도 있으면 총콜레스테롤 해석 생략)

    Example:
        # LDL만 입력한 경우 → LDL 해석만 반환
        # LDL + TG 입력한 경우 → LDL, TG 해석 반환 (HDL, 총콜레스테롤 생략)
        # 총콜레스테롤만 입력한 경우 → 총콜레스테롤 보조 참고 해석 반환
        # 4종 모두 미입력 → 빈 리스트 반환
    """


def get_recommendations(health_type_label: str, rule_results: dict) -> list[str]:
    """
    건강관리 유형과 룰 기반 판정 결과를 종합해 생활관리 추천 문구를 반환합니다.

    Args:
        health_type_label: 건강관리 유형 라벨 ('basic', 'weight', ...)
        rule_results: run_rule_engine() 반환값

    Returns:
        list of str  # 추천 문구 목록 (최대 5~7개)
    """


def get_exercise_recommendation(exercise_input: dict) -> dict:
    """
    사용자 운동 입력을 분석하고 신체활동 지침(보건복지부·한국건강증진개발원 2023)과
    비교하여 추천 문구를 생성합니다.

    보건복지부 성인(만 19~64세) 신체활동 지침 (2023 개정판):
        - 중강도 유산소 신체활동: 주 150~300분 권장
        - 고강도 유산소 신체활동: 주 75~150분 권장 (중강도 2분 = 고강도 1분)
        - 근력 운동: 주 2일 이상
        - 앉아있는 시간 최소화

    노인(만 65세 이상) 추가 지침:
        - 위 성인 기준 동일 적용
        - 평형성 운동: 주 3일 이상 (낙상 예방)

    Args:
        exercise_input: {
            'frequency': str,     # '안함' | '주1-2회' | '주3-4회' | '매일'
            'intensity': str,     # '저강도' | '중강도' | '고강도'
            'duration_min': int,  # 1회 운동 시간 (분)
        }

    Returns:
        {
            'weekly_sessions': float,         # 주간 운동 횟수 (환산값)
            'weekly_minutes': float,          # 주간 운동 시간 (분)
            'weekly_moderate_equiv': float,   # 중강도 환산 시간 (고강도×2 적용)
            'meets_aerobic_guideline': bool,  # 유산소 권장 기준 충족 여부 (중강도 150분/주)
            'meets_strength_guideline': bool, # 근력 운동 권장 기준 충족 여부
            'message': str,                   # 화면 출력 문구
            'suggestion': str,                # 개선 제안 문구
        }

    Example (중강도 주3-4회 × 40분):
        weekly_sessions=3.5, weekly_minutes=140,
        weekly_moderate_equiv=140,
        meets_aerobic_guideline=False (140 < 150),
        message="주간 신체활동은 약 140분(중강도 환산)입니다.",
        suggestion="권장 수준(주 150분)에 가까워지고 있습니다. 주당 1~2회 10분 추가를 권장합니다."

    Example (고강도 주3-4회 × 30분):
        weekly_sessions=3.5, weekly_minutes=105,
        weekly_moderate_equiv=210 (고강도 105분 × 2),
        meets_aerobic_guideline=True (210 ≥ 150),
        message="주간 신체활동은 약 210분(중강도 환산)입니다.",
        suggestion="권장 수준을 충족하고 있습니다. 현재 수준을 유지해 주세요."
    """

    # 강도별 환산 계수 (보건복지부 신체활동 지침 2023 기준)
    # 고강도 1분 = 중강도 2분
    INTENSITY_MULTIPLIER = {
        '저강도': 0.5,   # 150분 기준 환산
        '중강도': 1.0,
        '고강도': 2.0,
    }

    # 빈도별 주간 횟수 환산
    FREQUENCY_SESSIONS = {
        '안함': 0,
        '주1-2회': 1.5,
        '주3-4회': 3.5,
        '매일': 7,
    }


def compare_with_group(
    user_input: dict,
    group_stats: dict,
) -> list[dict]:
    """
    사용자 수치를 비교군 평균과 비교합니다.
    (comparison_group_summary.csv 기반)

    Args:
        user_input: 사용자 입력값
        group_stats: 비교군 통계 (행 1개)

    Returns:
        List of {
            'metric': str,          # 'bmi' | 'systolic_bp' | 'fasting_glucose' | 'waist_cm'
            'label': str,           # 'BMI' | '수축기혈압' | '공복혈당' | '허리둘레'
            'unit': str,            # '' | 'mmHg' | 'mg/dL' | 'cm'
            'user_value': float,
            'group_mean': float,
            'group_std': float,
            'diff_ratio': float,    # (사용자 - 비교군 평균) / 비교군 표준편차
            'level': str,           # 'similar' | 'higher' | 'lower'
            'message': str,
        }

    판정 기준:
        |diff_ratio| < 0.5: 비슷한 수준 (similar)
        diff_ratio ≥ 0.5:   비교군보다 높은 수준 (higher)
        diff_ratio ≤ -0.5:  비교군보다 낮은 수준 (lower)
    """
```

---

## 9.3 src/preprocessing.py

```python
def load_raw_data(filepath: str, encoding: str = 'cp949') -> pd.DataFrame:
    """
    원본 건강검진 CSV를 로딩합니다.
    (공공데이터포털에서 다운로드한 파일: 인코딩 CP949)

    Args:
        filepath: CSV 파일 경로
        encoding: 파일 인코딩 (기본값: 'cp949')

    Returns:
        원본 DataFrame (1,000,000 행 × 33 컬럼)
    """


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 학습에 사용할 컬럼만 선택하고 내부 변수명으로 변환합니다.

    원본 컬럼 → 내부 변수명 매핑:
        성별코드 → gender
        연령대코드(5세단위) → age_group
        신장(5cm단위) → height_cm
        체중(5kg단위) → weight_kg
        허리둘레 → waist_cm
        수축기혈압 → systolic_bp
        이완기혈압 → diastolic_bp
        식전혈당(공복혈당) → fasting_glucose
        흡연상태 → smoking_status
        음주여부 → drinking

    Returns:
        선택된 컬럼만 포함한 DataFrame (내부 변수명 적용)
    """


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치를 처리합니다.

    전략:
        - gender, age_group, height_cm, weight_kg: 결측 행 제거
        - systolic_bp, diastolic_bp, fasting_glucose, waist_cm:
          성별·연령대 그룹 중앙값으로 대체
        - smoking_status, drinking: 최빈값으로 대체

    Returns:
        결측치 처리 완료된 DataFrame
    """


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    이상치를 제거합니다.

    유효 범위:
        height_cm: 100~250
        weight_kg: 20~300
        systolic_bp: 60~250
        diastolic_bp: 40~150
        fasting_glucose: 50~500
    """


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    파생변수를 생성합니다.

    생성 변수:
        bmi: weight_kg / (height_cm / 100) ** 2
        bmi_category: 0=저체중, 1=정상, 2=과체중, 3=비만
        age_label: 연령대코드 → 문자열 (예: '40-44세')
        hypertension_flag: 수축기≥130 or 이완기≥80 → 1
        hyperglycemia_flag: 공복혈당≥100 → 1
        abdominal_obesity_flag: 남성 허리≥90 or 여성 허리≥85 → 1
        metabolic_risk_count: 위 flag 합산 (0~4)
    """
```

---

## 9.4 src/feature_engineering.py

```python
def generate_comparison_group_summary(
    df: pd.DataFrame,
    output_path: str = 'data/processed/comparison_group_summary.csv',
) -> pd.DataFrame:
    """
    성별 × 연령대 조합별 주요 지표 통계를 집계하고 저장합니다.
    배포 시 원본 데이터 대신 이 파일만 사용합니다.

    Args:
        df: 전처리 완료된 건강검진 DataFrame
        output_path: 저장 경로

    Returns:
        집계 통계 DataFrame

    출력 컬럼:
        gender, age_group, age_label, n,
        bmi_mean, bmi_std,
        sbp_mean, sbp_std,
        fbg_mean, fbg_std,
        waist_mean, waist_std,
        smoking_rate (현재흡연 비율: smoking_status==3),
        drinking_rate
    """


def generate_cluster_profile(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_mapping: dict,
    output_path: str = 'data/processed/cluster_profile.csv',
) -> pd.DataFrame:
    """
    군집별 평균 수치 프로파일을 생성하고 저장합니다.
    서비스에서 "건강관리 유형 설명" 표시에 활용합니다.

    Returns:
        cluster_id, health_type_name, n, bmi_mean, sbp_mean, fbg_mean,
        waist_mean, smoking_rate, drinking_rate 컬럼을 가진 DataFrame
    """
```
