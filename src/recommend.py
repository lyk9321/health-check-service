"""
src/recommend.py

이 파일은 사용자 입력 수치를 의학 기준표에 따라 해석하고,
건강관리 유형별 맞춤 생활관리 추천 문구를 만드는 곳이에요.

크게 세 가지 역할:
  1) 룰 기반 해석 (interpret_*): BMI, 혈압, 혈당, 콜레스테롤 4종
     → 표준 기준표에 맞춰 "정상/주의/관리 필요" 등을 판정
  2) 추천 문구 생성 (get_recommendations): 건강관리 유형별 다음 행동 제안
  3) 운동 분석 (get_exercise_recommendation): 보건복지부 신체활동 지침 비교

기준 출처:
  - BMI: 대한비만학회 2022
  - 혈압: 대한고혈압학회 2022
  - 혈당: 대한당뇨병학회 2023
  - 콜레스테롤: 한국지질·동맥경화학회 2022
  - 운동: 보건복지부·한국건강증진개발원 2023
"""

# 다른 파일에서 BMI 분류 함수를 가져와 재사용 (중복 방지)
# 단, 순환 import를 피하기 위해 분류 로직은 여기에 직접 작성합니다


# ──────────────────────────────────────────────────────────────
# 1. BMI 해석 (대한비만학회 2022)
# ──────────────────────────────────────────────────────────────
def interpret_bmi(bmi: float) -> dict:
    """
    BMI 수치를 해석합니다.

    기준:
        < 18.5: 저체중
        18.5 ~ 22.9: 정상
        23.0 ~ 24.9: 과체중
        ≥ 25.0: 비만
    """
    # 각 범위에 따라 결과 딕셔너리를 만들어 돌려줍니다
    if bmi < 18.5:
        return {
            'category': 'underweight',
            'label': '저체중',
            'message': '체중 증가 관련 관리가 필요할 수 있습니다.',
            'flag': 0,
        }
    elif bmi < 23.0:
        return {
            'category': 'normal',
            'label': '정상',
            'message': '체중 범위는 현재 적정 수준입니다.',
            'flag': 0,
        }
    elif bmi < 25.0:
        return {
            'category': 'overweight',
            'label': '과체중',
            'message': '체중 관리를 점검해볼 필요가 있습니다.',
            'flag': 1,
        }
    else:
        return {
            'category': 'obese',
            'label': '비만',
            'message': '체중관리가 도움이 될 수 있습니다.',
            'flag': 2,
        }


# ──────────────────────────────────────────────────────────────
# 2. 혈압 해석 (대한고혈압학회 2022)
# ──────────────────────────────────────────────────────────────
def interpret_blood_pressure(systolic: float, diastolic: float) -> dict:
    """
    수축기/이완기 혈압을 해석합니다.

    기준:
        정상:        수축기 < 120 AND 이완기 < 80
        주의:        수축기 120~129 AND 이완기 < 80
        고혈압 전단계: 수축기 130~139 OR  이완기 80~89
        고혈압 범위: 수축기 ≥ 140 OR  이완기 ≥ 90
    """
    # 가장 높은 단계부터 검사 (or 조건이 있으니 위에서부터 통과시킴)
    if systolic >= 140 or diastolic >= 90:
        return {
            'category': 'stage2',
            'label': '고혈압 범위',
            'message': '혈압 재측정 및 의료기관 상담을 권장합니다.',
            'flag': 1,
        }
    elif systolic >= 130 or diastolic >= 80:
        return {
            'category': 'stage1',
            'label': '고혈압 전단계',
            'message': '혈압 재측정 및 생활습관 관리를 권장합니다.',
            'flag': 1,
        }
    elif 120 <= systolic <= 129 and diastolic < 80:
        return {
            'category': 'elevated',
            'label': '주의',
            'message': '혈압이 높아지는 경향이 있어 관리를 권장합니다.',
            'flag': 1,
        }
    else:
        return {
            'category': 'normal',
            'label': '정상',
            'message': '현재 혈압 범위는 적정 수준입니다.',
            'flag': 0,
        }


# ──────────────────────────────────────────────────────────────
# 3. 공복혈당 해석 (대한당뇨병학회 2023)
# ──────────────────────────────────────────────────────────────
def interpret_fasting_glucose(glucose: float) -> dict:
    """
    공복혈당을 해석합니다.

    기준:
        정상:        < 100 mg/dL
        공복혈당 장애: 100~125 mg/dL
        당뇨병 범위:   ≥ 126 mg/dL
    """
    if glucose < 100:
        return {
            'category': 'normal',
            'label': '정상',
            'message': '현재 공복혈당 범위는 적정 수준입니다.',
            'flag': 0,
        }
    elif glucose < 126:
        return {
            'category': 'impaired',
            'label': '공복혈당 장애',
            'message': '공복혈당 관리가 필요할 수 있습니다.',
            'flag': 1,
        }
    else:
        return {
            'category': 'diabetic_range',
            'label': '당뇨병 범위',
            'message': '가까운 의료기관에서 확인검사를 권장합니다.',
            'flag': 1,
        }


# ──────────────────────────────────────────────────────────────
# 4. LDL 콜레스테롤 해석 (한국지질·동맥경화학회 2022)
# ──────────────────────────────────────────────────────────────
def interpret_ldl_cholesterol(ldl: float) -> dict:
    """
    LDL("나쁜 콜레스테롤")을 해석합니다. 심혈관 위험 1차 지표예요.
    """
    if ldl < 100:
        return {'category': 'optimal', 'label': '적정',
                'message': 'LDL 콜레스테롤은 적정 범위에 해당합니다.', 'flag': 0}
    elif ldl < 130:
        return {'category': 'near_optimal', 'label': '정상 상한',
                'message': 'LDL 콜레스테롤은 정상 범위에 해당합니다.', 'flag': 0}
    elif ldl < 160:
        return {'category': 'borderline', 'label': '경계',
                'message': 'LDL 콜레스테롤이 경계 범위에 해당하므로 관리 점검이 도움이 될 수 있습니다.', 'flag': 1}
    elif ldl < 190:
        return {'category': 'high', 'label': '높음',
                'message': 'LDL 콜레스테롤이 높은 범위에 해당하므로 의료기관 상담을 권장합니다.', 'flag': 1}
    else:
        return {'category': 'very_high', 'label': '매우 높음',
                'message': 'LDL 콜레스테롤이 매우 높은 범위에 해당하므로 의료기관 상담을 권장합니다.', 'flag': 1}


# ──────────────────────────────────────────────────────────────
# 5. HDL 콜레스테롤 해석 (성별에 따라 기준 다름)
# ──────────────────────────────────────────────────────────────
def interpret_hdl_cholesterol(hdl: float, gender: int) -> dict:
    """
    HDL("좋은 콜레스테롤")을 해석합니다. 높을수록 좋아요.
    여성이 남성보다 기준이 더 엄격합니다 (여성은 50 미만이 낮음).

    Args:
        hdl: HDL 수치 (mg/dL)
        gender: 1=남성, 2=여성
    """
    # 성별에 따른 "낮음" 기준치 결정
    low_threshold = 40 if gender == 1 else 50

    if hdl < low_threshold:
        return {
            'category': 'low',
            'label': '낮음',
            'message': (
                'HDL 콜레스테롤이 낮은 범위에 해당합니다. '
                'HDL은 다른 콜레스테롤과 달리 높을수록 바람직한 지표이므로, '
                '규칙적인 신체활동이 수치 향상에 도움이 될 수 있습니다.'
            ),
            'flag': 1
        }
    elif hdl < 60:
        return {
            'category': 'normal',
            'label': '보통',
            'message': (
                'HDL 콜레스테롤은 보통 범위에 해당합니다. '
                'HDL은 높을수록 바람직한 지표입니다.'
            ),
            'flag': 0
        }
    else:
        return {
            'category': 'high',
            'label': '양호',          # ← '높음(보호)' → '양호'로 변경
            'message': (
                'HDL 콜레스테롤이 높은 범위에 해당합니다. '
                'HDL은 높을수록 바람직한 지표로, 현재 수치는 긍정적인 범위에 해당합니다.'
            ),
            'flag': 0
        }


# ──────────────────────────────────────────────────────────────
# 6. 트리글리세라이드(중성지방) 해석
# ──────────────────────────────────────────────────────────────
def interpret_triglyceride(tg: float) -> dict:
    """중성지방을 해석합니다."""
    if tg < 150:
        return {'category': 'optimal', 'label': '적정',
                'message': '중성지방은 적정 범위에 해당합니다.', 'flag': 0}
    elif tg < 200:
        return {'category': 'borderline', 'label': '경계',
                'message': '중성지방이 경계 범위에 해당하므로 관리 점검이 도움이 될 수 있습니다.', 'flag': 1}
    elif tg < 500:
        return {'category': 'high', 'label': '높음',
                'message': '중성지방이 높은 범위에 해당하므로 의료기관 상담을 권장합니다.', 'flag': 1}
    else:
        return {'category': 'very_high', 'label': '매우 높음',
                'message': '중성지방이 매우 높은 범위에 해당하므로 의료기관 상담을 권장합니다.', 'flag': 1}


# ──────────────────────────────────────────────────────────────
# 7. 총콜레스테롤 해석 (보조 지표 — LDL/HDL/TG 미입력 시만 사용)
# ──────────────────────────────────────────────────────────────
def interpret_total_cholesterol(cholesterol: float) -> dict:
    """
    총콜레스테롤을 해석합니다.
    LDL/HDL/TG 중 하나라도 입력됐으면 이 함수를 호출하지 않습니다.
    """
    if cholesterol < 200:
        result = {'category': 'optimal', 'label': '적정',
                  'message': '총콜레스테롤은 적정 범위에 해당합니다.', 'flag': 0}
    elif cholesterol < 240:
        result = {'category': 'borderline', 'label': '경계',
                  'message': '총콜레스테롤이 경계 범위에 해당하므로 관리 점검이 도움이 될 수 있습니다.', 'flag': 1}
    else:
        result = {'category': 'high', 'label': '높음',
                  'message': '총콜레스테롤이 높은 범위에 해당하므로 지질 검사 결과 확인을 권장합니다.', 'flag': 1}

    # 보조 참고임을 표시 (UI에서 별도 안내 가능)
    result['is_supplementary'] = True
    return result


# ──────────────────────────────────────────────────────────────
# 8. 콜레스테롤 4종 통합 해석
# ──────────────────────────────────────────────────────────────
def interpret_cholesterol_panel(user_input: dict) -> list:
    """
    사용자가 입력한 지질 수치를 모두 해석합니다.
    LDL, HDL, TG는 입력된 항목만 각각 해석하고,
    총콜레스테롤은 LDL/HDL/TG가 모두 미입력일 때만 보조로 표시합니다.

    Returns:
        [{'metric': str, 'label': str, ...}, ...] 형태의 리스트
    """
    results = []

    # LDL 입력됐으면 해석
    if user_input.get('ldl_cholesterol') is not None:
        r = interpret_ldl_cholesterol(user_input['ldl_cholesterol'])
        r['metric'] = 'ldl'
        r['metric_label'] = 'LDL 콜레스테롤'
        r['value'] = user_input['ldl_cholesterol']
        results.append(r)

    # HDL 입력됐으면 해석 (성별 정보 필요)
    if user_input.get('hdl_cholesterol') is not None:
        # gender가 없으면 남성(1)으로 가정 (안전한 기본값)
        gender = user_input.get('gender', 1)
        r = interpret_hdl_cholesterol(user_input['hdl_cholesterol'], gender)
        r['metric'] = 'hdl'
        r['metric_label'] = 'HDL 콜레스테롤'
        r['value'] = user_input['hdl_cholesterol']
        results.append(r)

    # 트리글리세라이드 입력됐으면 해석
    if user_input.get('triglyceride') is not None:
        r = interpret_triglyceride(user_input['triglyceride'])
        r['metric'] = 'triglyceride'
        r['metric_label'] = '중성지방'
        r['value'] = user_input['triglyceride']
        results.append(r)

    # 총콜레스테롤은 LDL/HDL/TG 중 하나라도 있으면 표시 안 함 (중복 방지)
    has_specific = any(
        user_input.get(k) is not None
        for k in ['ldl_cholesterol', 'hdl_cholesterol', 'triglyceride']
    )
    if not has_specific and user_input.get('total_cholesterol') is not None:
        r = interpret_total_cholesterol(user_input['total_cholesterol'])
        r['metric'] = 'total_cholesterol'
        r['metric_label'] = '총콜레스테롤'
        r['value'] = user_input['total_cholesterol']
        results.append(r)

    return results


# ──────────────────────────────────────────────────────────────
# 9. 룰 엔진 통합 실행
# ──────────────────────────────────────────────────────────────
def run_rule_engine(user_input: dict) -> dict:
    """
    사용자 입력의 모든 룰 기반 판정을 한 번에 실행합니다.

    Returns:
        {'bmi': dict, 'blood_pressure': dict 또는 None, ...}
    """
    results = {
        'bmi': None,
        'blood_pressure': None,
        'fasting_glucose': None,
        'cholesterol_panel': [],
        'metabolic_risk_count': 0,
    }

    # BMI는 항상 계산 (필수 입력)
    if user_input.get('bmi') is not None:
        results['bmi'] = interpret_bmi(user_input['bmi'])

    # 혈압 (둘 다 입력됐을 때만)
    if (user_input.get('systolic_bp') is not None and
            user_input.get('diastolic_bp') is not None):
        results['blood_pressure'] = interpret_blood_pressure(
            user_input['systolic_bp'],
            user_input['diastolic_bp'],
        )

    # 공복혈당
    if user_input.get('fasting_glucose') is not None:
        results['fasting_glucose'] = interpret_fasting_glucose(
            user_input['fasting_glucose']
        )

    # 콜레스테롤 4종
    results['cholesterol_panel'] = interpret_cholesterol_panel(user_input)

    # 위험요인 개수: flag=1인 항목들의 개수 합산
    risk_count = 0
    if results['bmi'] and results['bmi']['flag'] >= 1:
        risk_count += 1
    if results['blood_pressure'] and results['blood_pressure']['flag'] == 1:
        risk_count += 1
    if results['fasting_glucose'] and results['fasting_glucose']['flag'] == 1:
        risk_count += 1
    # 콜레스테롤은 4종 중 하나라도 위험이면 +1 (중복 방지)
    if any(c['flag'] == 1 for c in results['cholesterol_panel']):
        risk_count += 1

    results['metabolic_risk_count'] = risk_count
    return results


# ──────────────────────────────────────────────────────────────
# 10. 건강관리 유형별 추천 문구
# ──────────────────────────────────────────────────────────────

# 유형별 기본 추천 문구 (3~5개씩)
BASE_RECOMMENDATIONS = {
    'basic': [
        '현재 건강 상태를 잘 유지하고 있습니다.',
        '정기 건강검진을 빠짐없이 받으세요 (연 1회 권장).',
        '규칙적인 신체활동을 유지하세요 (주 150분 이상 권장).',
        '균형 잡힌 식단을 지속하세요.',
    ],
    'weight': [
        '체중 관리에 집중이 필요한 시점입니다.',
        '하루 30분 이상 걷기를 시작해보세요.',
        '식사량과 간식 섭취를 점검해보세요.',
        '체중 변화를 주 1회 이상 기록해보세요.',
        '보건소 비만 관리 프로그램 이용을 권장합니다.',
    ],
    'blood_pressure': [
        '혈압 관리에 집중이 필요한 시점입니다.',
        '혈압을 정기적으로 재측정하세요 (주 2~3회 권장).',
        '나트륨 섭취를 줄이세요 (저염식).',
        '규칙적인 유산소 운동이 혈압 관리에 도움이 될 수 있습니다.',
        '의료기관 또는 보건소 상담을 권장합니다.',
    ],
    'blood_sugar': [
        '혈당 관리에 집중이 필요한 시점입니다.',
        '공복혈당 재확인 및 의료기관 상담을 권장합니다.',
        '정제 탄수화물과 당류 섭취를 줄이세요.',
        '규칙적인 식사 시간을 유지하세요.',
        '신체활동이 혈당 조절에 도움이 될 수 있습니다.',
    ],
    'metabolic': [
        '여러 건강 지표에 대한 복합 관리가 필요한 시점입니다.',
        '보건소 또는 의료기관 상담을 적극 권장합니다.',
        '체중·혈압·혈당을 함께 관리하는 것이 중요합니다.',
        '금연·절주 프로그램 이용을 권장합니다.',
        '건강생활지원센터의 만성질환 예방 프로그램 이용을 권장합니다.',
    ],
    'lifestyle': [
        '생활습관 점검이 도움이 될 수 있습니다.',
        '금연 상담 또는 금연클리닉 이용을 권장합니다.',
        '음주량을 줄이거나 절주하세요.',
        '규칙적인 신체활동을 시작하세요.',
        '건강생활지원센터의 생활습관 개선 프로그램 이용을 권장합니다.',
    ],
}


def get_recommendations(health_type_label: str, rule_results: dict) -> list:
    """
    건강관리 유형 라벨과 룰 기반 결과를 받아서 맞춤 추천 문구 리스트를 반환합니다.

    Args:
        health_type_label: 'basic', 'weight', 'blood_pressure', ...
        rule_results: run_rule_engine()의 결과

    Returns:
        추천 문구 리스트 (5~7개)
    """
    # 유형별 기본 추천을 복사 (원본 보호)
    recs = BASE_RECOMMENDATIONS.get(
        health_type_label,
        BASE_RECOMMENDATIONS['basic']
    ).copy()

    # 룰 기반 결과에 따른 추가 메시지 (중복 방지하면서 덧붙이기)
    extras = []

    # BMI가 비만 단계면 체중관리 메시지 강조
    if rule_results.get('bmi') and rule_results['bmi']['flag'] == 2:
        extras.append('BMI 25 이상으로, 체중 관리를 우선 검토하시기 바랍니다.')

    # 혈압이 stage2(140/90 이상)면 더 명확한 안내
    if (rule_results.get('blood_pressure') and
            rule_results['blood_pressure']['category'] == 'stage2'):
        extras.append('혈압 수치가 높은 범위에 해당해 의료기관 재측정·상담이 필요합니다.')

    # 혈당이 당뇨 범위면 강한 안내
    if (rule_results.get('fasting_glucose') and
            rule_results['fasting_glucose']['category'] == 'diabetic_range'):
        extras.append('공복혈당이 높은 범위에 해당해 의료기관 확인검사를 권장합니다.')

    # 위험요인이 3개 이상이면 상담 권유
    if rule_results.get('metabolic_risk_count', 0) >= 3:
        extras.append('여러 건강 지표가 동시에 관리 필요 범위에 있어, '
                      '의료기관 상담을 우선 고려하시기 바랍니다.')

    # 추가 메시지를 기본 추천 앞에 합치기 (중복 제거)
    final = []
    for msg in extras + recs:
        if msg not in final:
            final.append(msg)

    return final[:7]  # 최대 7개로 제한


# ──────────────────────────────────────────────────────────────
# 11. 운동 권장량 분석 (보건복지부 신체활동 지침 2023)
# ──────────────────────────────────────────────────────────────

# 빈도(주간 횟수) → 환산 횟수
FREQUENCY_SESSIONS = {
    '안함': 0,
    '주1-2회': 1.5,
    '주2-3회': 2.5,
    '주3-4회': 3.5,
    '매일': 7,
}

# 강도 → 중강도 환산 계수 (고강도 1분 = 중강도 2분)
INTENSITY_MULTIPLIER = {
    '저강도': 0.5,
    '중강도': 1.0,
    '고강도': 2.0,
}


def get_exercise_recommendation(exercise_input: dict) -> dict:
    """
    사용자가 입력한 운동 정보를 보건복지부 지침과 비교합니다.

    지침 (성인 만 19~64세):
      - 중강도 유산소: 주 150~300분
      - 고강도 유산소: 주 75~150분 (1분 = 중강도 2분)
      - 근력 운동: 주 2일 이상
      - 앉아있는 시간 최소화

    Args:
        exercise_input: {
            'frequency': '안함'|'주1-2회'|'주2-3회'|'주3-4회'|'매일',
            'intensity': '저강도'|'중강도'|'고강도',
            'duration_min': int (1회 운동 시간 분)
        }

    Returns:
        분석 결과 dict
    """
    freq_str = exercise_input.get('frequency', '안함')
    intensity_str = exercise_input.get('intensity', '중강도')
    duration_min = exercise_input.get('duration_min', 0) or 0

    # 환산값 계산
    weekly_sessions = FREQUENCY_SESSIONS.get(freq_str, 0)
    intensity_mult = INTENSITY_MULTIPLIER.get(intensity_str, 1.0)

    # 주간 총 운동 시간 (분)
    weekly_minutes = weekly_sessions * duration_min

    # 중강도 환산 시간 = 시간 × 강도 계수 (저:0.5, 중:1.0, 고:2.0)
    weekly_moderate_equiv = weekly_minutes * intensity_mult

    # 권장량 충족 여부
    meets_aerobic = weekly_moderate_equiv >= 150  # 중강도 150분 기준
    meets_strength = False  # 근력 운동 별도 입력은 PRD에서 미포함, 기본 False

    # 출력 메시지 생성
    if weekly_sessions == 0:
        message = '현재 정기적인 운동을 하지 않는 것으로 입력되었습니다.'
        suggestion = '하루 10분이라도 규칙적인 신체활동을 시작해보세요.'
    else:
        message = (
            f'주 {weekly_sessions:.1f}회 × {duration_min}분 ({intensity_str})으로, '
            f'주간 총 {weekly_minutes:.0f}분 운동하고 계십니다. '
            f'(중강도 환산: {weekly_moderate_equiv:.0f}분)'
        )
        if meets_aerobic:
            suggestion = (
                '권장 수준(주 150분 중강도 환산)을 충족하고 있습니다. '
                '현재 수준을 유지해 주세요.'
            )
        elif weekly_moderate_equiv >= 100:
            suggestion = (
                '권장 수준에 가까워지고 있습니다. '
                '주당 1~2회 10분 추가를 권장합니다.'
            )
        else:
            suggestion = (
                '권장 수준(주 150분)에 비해 다소 부족합니다. '
                '무리하지 않는 범위에서 점진적으로 늘려보세요.'
            )

    return {
        'weekly_sessions': weekly_sessions,
        'weekly_minutes': weekly_minutes,
        'weekly_moderate_equiv': weekly_moderate_equiv,
        'meets_aerobic_guideline': meets_aerobic,
        'meets_strength_guideline': meets_strength,
        'message': message,
        'suggestion': suggestion,
    }


# ──────────────────────────────────────────────────────────────
# 12. 비교군 비교 (성별·연령대 그룹 통계 vs 사용자 수치)
# ──────────────────────────────────────────────────────────────
def compare_with_group(user_input: dict, group_stats: dict) -> list:
    """
    사용자의 BMI/혈압/혈당/허리둘레를 비교군 평균과 비교합니다.
    (지질 수치는 결측률 ~66%라 비교군 비교에서 제외)

    Args:
        user_input: 사용자 입력값
        group_stats: 비교군 통계 (DataFrame의 1행을 dict로 변환한 것)

    Returns:
        [{'metric': str, 'level': 'similar'|'higher'|'lower', ...}, ...]
    """
    # 비교 대상: (사용자 키, 비교군 평균 키, 비교군 표준편차 키, 라벨, 단위)
    metrics_config = [
        ('bmi',             'bmi_mean',   'bmi_std',   'BMI',         ''),
        ('systolic_bp',     'sbp_mean',   'sbp_std',   '수축기혈압',   'mmHg'),
        ('fasting_glucose', 'fbg_mean',   'fbg_std',   '공복혈당',     'mg/dL'),
        ('waist_cm',        'waist_mean', 'waist_std', '허리둘레',     'cm'),
    ]

    results = []
    for user_key, mean_key, std_key, label, unit in metrics_config:
        # 사용자가 해당 항목을 입력했고 비교군 통계가 있을 때만
        user_val = user_input.get(user_key)
        if user_val is None or mean_key not in group_stats:
            continue

        group_mean = group_stats[mean_key]
        group_std = group_stats[std_key]

        # 표준편차가 0이거나 NaN이면 비교 불가 → 건너뛰기
        if not group_std or group_std == 0:
            continue

        # 차이 비율 = (사용자 - 평균) / 표준편차
        # 0.5 이상이면 "비교군보다 높음", -0.5 이하면 "낮음", 그 사이면 "비슷"
        diff_ratio = (user_val - group_mean) / group_std

        if diff_ratio > 0.5:
            level = 'higher'
            level_label = '비교군보다 높은 수준'
        elif diff_ratio < -0.5:
            level = 'lower'
            level_label = '비교군보다 낮은 수준'
        else:
            level = 'similar'
            level_label = '비슷한 수준'

        results.append({
            'metric': user_key,
            'label': label,
            'unit': unit,
            'user_value': round(user_val, 1),
            'group_mean': round(group_mean, 1),
            'group_std': round(group_std, 1),
            'diff_ratio': round(diff_ratio, 2),
            'level': level,
            'message': f'{label}: 당신 {user_val:.1f}{unit} | '
                       f'비교군 평균 {group_mean:.1f}{unit} (±{group_std:.1f}) → '
                       f'{level_label}',
        })

    return results


if __name__ == '__main__':
    # 간단한 테스트
    sample = {
        'bmi': 25.5,
        'systolic_bp': 135,
        'diastolic_bp': 88,
        'fasting_glucose': 110,
        'ldl_cholesterol': 145,
        'hdl_cholesterol': 45,
        'gender': 1,
    }
    rules = run_rule_engine(sample)
    print("룰 엔진 결과:")
    for k, v in rules.items():
        print(f"  {k}: {v}")
    print("\n추천 문구:")
    for r in get_recommendations('weight', rules):
        print(f"  - {r}")
