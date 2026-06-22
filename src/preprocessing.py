"""
src/preprocessing.py

건강검진 원본 데이터를 모델 학습에 쓸 수 있도록 다듬어주는 함수들이 모여 있는 파일이에요.

쉽게 말하면:
  "더러운 흙 묻은 당근(원본 CSV)을 깨끗이 씻어서, 먹기 좋게 자른 당근(전처리된 데이터)으로 바꾸는 곳"
이라고 생각하면 됩니다.

순서:
  1) 원본 CSV 읽기                  → load_raw_data()
  2) 필요한 컬럼만 골라내고 이름 바꾸기 → select_model_columns()
  3) 빈 칸(결측치) 처리              → handle_missing_values()
  4) 이상한 값(이상치) 제거          → remove_outliers()
  5) 새로운 변수 만들기 (BMI 등)     → engineer_features()
"""

# 데이터 처리에 쓸 도구들을 가져옵니다 (도구 상자 열기)
import pandas as pd  # 표(엑셀) 다루는 도구
import numpy as np   # 숫자 계산 도구


# ──────────────────────────────────────────────────────────────
# 1. 원본 CSV 파일 읽기
# ──────────────────────────────────────────────────────────────
def load_raw_data(filepath: str, encoding: str = 'cp949') -> pd.DataFrame:
    """
    원본 건강검진 CSV 파일을 읽어옵니다.

    이 데이터는 공공데이터포털에서 받은 한국어 CSV라서
    인코딩이 'cp949'(한국어 윈도우 표준)로 되어있어요.
    그냥 읽으면 한글이 깨지니까 꼭 encoding='cp949'를 넣어줘야 합니다.

    Args:
        filepath: CSV 파일 위치 (예: 'data/raw/2024_국민건강보험공단_건강검진정보.CSV')
        encoding: 파일 글자 인코딩 (기본값 'cp949')

    Returns:
        원본 DataFrame (100만 행 × 33 컬럼)
    """
    # pandas의 read_csv 함수로 CSV 파일을 표(DataFrame) 형태로 읽어옵니다
    df = pd.read_csv(filepath, encoding=encoding)

    # 몇 행 × 몇 열인지 화면에 보여줍니다 (확인용)
    print(f"[load_raw_data] 원본 데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")

    # 읽은 데이터를 함수 호출한 곳으로 돌려줍니다
    return df


# ──────────────────────────────────────────────────────────────
# 2. 필요한 컬럼만 고르기 + 이름 바꾸기
# ──────────────────────────────────────────────────────────────
def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    원본 33개 컬럼 중에서 우리가 쓸 10개만 골라냅니다.
    그리고 한국어 컬럼명을 영어 변수명으로 바꿔줍니다 (코드에서 다루기 쉽게).

    예: '성별코드' → 'gender'
        '신장(5cm단위)' → 'height_cm'

    Args:
        df: load_raw_data()로 읽은 원본 DataFrame

    Returns:
        10개 컬럼만 있는 DataFrame
    """
    # 원본 컬럼명 → 우리가 쓸 영어 이름으로 바꾸는 대조표(딕셔너리)를 만듭니다
    column_map = {
        '성별코드': 'gender',                 # 1=남성, 2=여성
        '연령대코드(5세단위)': 'age_group',     # 5=25~29세, 6=30~34세, ..., 18=90세+
        '시도코드': 'region_code',            # 시도 코드 (지역 분석용)
        '신장(5cm단위)': 'height_cm',         # 키 (cm)
        '체중(5kg단위)': 'weight_kg',         # 몸무게 (kg)
        '허리둘레': 'waist_cm',               # 허리둘레 (cm)
        '수축기혈압': 'systolic_bp',          # 위쪽 혈압 (mmHg)
        '이완기혈압': 'diastolic_bp',         # 아래쪽 혈압 (mmHg)
        '식전혈당(공복혈당)': 'fasting_glucose',  # 공복혈당 (mg/dL)
        '흡연상태': 'smoking_status',         # 1=비흡연, 2=과거흡연, 3=현재흡연
        '음주여부': 'drinking',               # 0=음주없음, 1=음주함
    }

    # 위 딕셔너리의 키(원본 컬럼명) 목록만 뽑아서, 그 컬럼들만 골라냅니다
    selected_cols = list(column_map.keys())

    # df[컬럼리스트] 형태로 쓰면 그 컬럼들만 있는 새 DataFrame이 만들어집니다
    df_selected = df[selected_cols].copy()  # .copy()는 원본을 건드리지 않고 사본을 만듭니다

    # 컬럼명을 영어로 바꿉니다 (rename은 이름만 변경, 데이터는 그대로)
    df_selected = df_selected.rename(columns=column_map)

    print(f"[select_model_columns] 선택된 컬럼: {list(df_selected.columns)}")
    return df_selected


# ──────────────────────────────────────────────────────────────
# 3. 결측치(빈 칸) 처리
# ──────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    표에서 빈 칸(NaN)을 어떻게 처리할지 정합니다.

    원칙:
      - 가장 중요한 정보(성별, 나이, 키, 몸무게)가 비어있으면 그 행은 통째로 버립니다
        → 이 정보가 없으면 BMI도 못 계산하니까요.
      - 혈압·혈당·허리둘레가 비어있으면, 같은 성별·연령대 사람들의 "중앙값"으로 채웁니다
        → 비슷한 사람들의 가운데 값으로 대신 채워주는 거예요.
      - 흡연·음주 정보가 비어있으면 가장 많이 나온 값(최빈값)으로 채웁니다.

    Args:
        df: select_model_columns()로 처리된 DataFrame

    Returns:
        빈 칸이 모두 처리된 DataFrame
    """
    # 시작 전 행 개수를 기록해둡니다 (몇 행이 버려졌는지 확인용)
    initial_rows = len(df)

    # ── (1) 필수 변수가 빈 행은 통째로 버리기 ──
    required_cols = ['gender', 'age_group', 'height_cm', 'weight_kg']

    # dropna는 빈 칸이 있는 행을 버립니다 (subset에 적은 컬럼들만 검사)
    df = df.dropna(subset=required_cols).copy()

    print(f"[handle_missing_values] 필수값 결측 제거: {initial_rows:,} → {len(df):,}행")

    # ── (2) 연속형 변수: 성별·연령대 그룹별 중앙값으로 채우기 ──
    continuous_cols = ['systolic_bp', 'diastolic_bp', 'fasting_glucose', 'waist_cm']

    for col in continuous_cols:  # 컬럼 하나씩 돌면서
        # groupby로 (성별, 연령대) 조합별로 묶고, transform('median')으로 각 그룹의 중앙값 계산
        # → 각 행마다 자기가 속한 그룹의 중앙값이 들어간 같은 길이의 시리즈가 만들어집니다
        group_median = df.groupby(['gender', 'age_group'])[col].transform('median')

        # 빈 칸을 그 그룹의 중앙값으로 채웁니다
        df[col] = df[col].fillna(group_median)

    # 그래도 혹시 남은 결측치(특정 그룹 자체가 너무 작아서 중앙값이 NaN인 경우)는 전체 중앙값으로
    for col in continuous_cols:
        df[col] = df[col].fillna(df[col].median())

    # ── (3) 범주형 변수: 최빈값(가장 자주 나오는 값)으로 채우기 ──
    categorical_cols = ['smoking_status', 'drinking']

    for col in categorical_cols:
        # mode()는 가장 많이 나온 값(최빈값)을 돌려줍니다 ([0]으로 첫 번째 것을 꺼냄)
        most_common = df[col].mode()[0]

        # 빈 칸을 최빈값으로 채웁니다
        df[col] = df[col].fillna(most_common)

    # 처리 후 결측치가 다 사라졌는지 확인 (디버깅용)
    remaining_na = df.isnull().sum().sum()
    print(f"[handle_missing_values] 처리 후 남은 결측치: {remaining_na}개")

    return df


# ──────────────────────────────────────────────────────────────
# 4. 이상치(말도 안 되는 값) 제거
# ──────────────────────────────────────────────────────────────
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    상식적으로 말이 안 되는 값들을 가진 행을 버립니다.

    예: 키 50cm? → 사람이 아닐 가능성이 큼 (입력 오류)
        몸무게 500kg? → 측정 오류

    Args:
        df: handle_missing_values()로 처리된 DataFrame

    Returns:
        이상치가 제거된 DataFrame
    """
    # 시작 행 개수
    initial_rows = len(df)

    # 각 컬럼별 유효 범위 (최소값, 최대값)
    valid_ranges = {
        'height_cm': (100, 250),       # 키: 100cm ~ 250cm
        'weight_kg': (20, 300),        # 몸무게: 20kg ~ 300kg
        'systolic_bp': (60, 250),      # 수축기혈압: 60 ~ 250
        'diastolic_bp': (40, 150),     # 이완기혈압: 40 ~ 150
        'fasting_glucose': (50, 500),  # 공복혈당: 50 ~ 500
    }

    # 각 컬럼에 대해 유효 범위 안에 있는 행만 남깁니다
    for col, (low, high) in valid_ranges.items():
        # df[col].between(low, high)는 그 범위 안에 있으면 True
        # df[조건] 형태로 쓰면 True인 행만 남깁니다
        df = df[df[col].between(low, high)].copy()

    print(f"[remove_outliers] 이상치 제거: {initial_rows:,} → {len(df):,}행")

    return df


# ──────────────────────────────────────────────────────────────
# 5. 파생변수(새 변수) 만들기
# ──────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    기존 변수들을 조합해서 새로운 의미 있는 변수를 만듭니다.

    예: 키 + 몸무게 → BMI (비만도 지수)
        혈압 → 고혈압 여부 플래그

    Args:
        df: remove_outliers()로 처리된 DataFrame

    Returns:
        파생변수가 추가된 DataFrame
    """
    df = df.copy()  # 원본을 보호하기 위해 사본을 만듭니다

    # ── (1) BMI 계산: 몸무게(kg) / 키(m)의 제곱 ──
    # 키는 cm 단위니까 100으로 나눠서 m로 바꿉니다
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)

    # 소수점 1자리로 반올림합니다
    df['bmi'] = df['bmi'].round(1)

    # ── (2) BMI 분류 (0=저체중, 1=정상, 2=과체중, 3=비만) ──
    # 대한비만학회 2022 기준
    def bmi_to_category(bmi):
        if bmi < 18.5:
            return 0  # 저체중
        elif bmi < 23.0:
            return 1  # 정상
        elif bmi < 25.0:
            return 2  # 과체중
        else:
            return 3  # 비만

    # apply로 각 행의 bmi 값에 위 함수를 적용해서 새 컬럼을 만듭니다
    df['bmi_category'] = df['bmi'].apply(bmi_to_category)

    # ── (3) 연령대 라벨 (코드 → 사람이 읽을 수 있는 문자열) ──
    age_label_map = {
        5: '25-29세',  6: '30-34세',  7: '35-39세',  8: '40-44세',
        9: '45-49세', 10: '50-54세', 11: '55-59세', 12: '60-64세',
        13: '65-69세', 14: '70-74세', 15: '75-79세', 16: '80-84세', 
        17: '85-89세', 18:'90+세'
    }
    df['age_label'] = df['age_group'].map(age_label_map)

    # ── (4) 위험 신호 플래그(0/1) 4개 ──
    # 고혈압 신호: 수축기≥130 또는 이완기≥80
    df['hypertension_flag'] = ((df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80)).astype(int)

    # 고혈당 신호: 공복혈당≥100
    df['hyperglycemia_flag'] = (df['fasting_glucose'] >= 100).astype(int)

    # 복부비만: 남성 허리≥90cm, 여성 허리≥85cm (대한비만학회 기준)
    # gender==1이 남성, gender==2가 여성
    df['abdominal_obesity_flag'] = (
        ((df['gender'] == 1) & (df['waist_cm'] >= 90)) |
        ((df['gender'] == 2) & (df['waist_cm'] >= 85))
    ).astype(int)

    # 비만 플래그: BMI 분류 코드가 3(비만)인지
    df['obesity_flag'] = (df['bmi_category'] == 3).astype(int)

    # ── (5) 대사 위험요인 개수 (0~4): 위 4개 플래그를 합칩니다 ──
    df['metabolic_risk_count'] = (
        df['hypertension_flag'] +
        df['hyperglycemia_flag'] +
        df['abdominal_obesity_flag'] +
        df['obesity_flag']
    )

    print(f"[engineer_features] 파생변수 생성 완료: bmi, bmi_category, age_label, "
          f"hypertension_flag, hyperglycemia_flag, abdominal_obesity_flag, "
          f"obesity_flag, metabolic_risk_count")

    return df


# ──────────────────────────────────────────────────────────────
# 6. 전체 파이프라인 한 번에 실행 (편의 함수)
# ──────────────────────────────────────────────────────────────
def run_full_preprocessing(
    filepath: str = 'data/raw/2024_국민건강보험공단_건강검진정보.CSV',
    encoding: str = 'cp949',
) -> pd.DataFrame:
    """
    위 1~5번 단계를 한 줄로 실행할 수 있게 묶어둔 함수입니다.

    사용 예:
        from src.preprocessing import run_full_preprocessing
        df_clean = run_full_preprocessing()
    """
    df = load_raw_data(filepath, encoding)        # 1) 원본 읽기
    df = select_model_columns(df)                  # 2) 컬럼 선택
    df = handle_missing_values(df)                 # 3) 결측치 처리
    df = remove_outliers(df)                       # 4) 이상치 제거
    df = engineer_features(df)                     # 5) 파생변수 생성

    print(f"\n✅ 전처리 완료: 최종 {len(df):,}행 × {df.shape[1]}열")
    return df


# 이 파일을 직접 실행하면 (python preprocessing.py) 아래 코드가 동작합니다
if __name__ == '__main__':
    df = run_full_preprocessing()
    print(df.head())
