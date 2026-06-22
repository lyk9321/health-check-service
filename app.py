"""
app.py

Streamlit 웹앱의 진입점이에요.
`streamlit run app.py` 명령으로 실행합니다.

화면 구성:
  Step 0) 서비스 소개 (시작 버튼)
  Step 1) 기본정보 (성별, 연령대, 키, 몸무게)
  Step 2) 건강검진 수치 (혈압, 혈당, 콜레스테롤, 허리둘레)
  Step 3) 생활습관 (흡연, 음주, 운동)
  Step 4) 거주지역 (지역사회 자원 안내용)
  결과)   9개 카드
"""

import streamlit as st
import pandas as pd

# 직접 만든 모듈들 가져오기
from src.predict import (
    load_models,
    load_comparison_stats,
    predict_health_type,
    calculate_bmi,
)


# ──────────────────────────────────────────────────────────────
# 페이지 기본 설정 (브라우저 탭 제목, 아이콘, 레이아웃)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='건강위험 자가점검 서비스',
    page_icon='🏥',
    layout='centered',                  # 중앙 정렬 (모바일 친화적)
    initial_sidebar_state='collapsed',  # 사이드바 접기
)


# ──────────────────────────────────────────────────────────────
# 모델·데이터 캐시 로딩 (Streamlit 첫 실행 때 한 번만)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    """모델 로드. @st.cache_resource로 한 번만 로드되고 재사용됩니다."""
    return load_models('model')


@st.cache_data      # DataFrame 같은 데이터용 캐시
def get_comparison_stats():
    """비교군 통계 로드."""
    return load_comparison_stats('data/processed/comparison_group_summary.csv')


@st.cache_data
def get_regional_data() -> tuple:
    """
    지역보건의료기관 + 건강생활지원센터 데이터를 한 번만 읽어옵니다.
    건강생활지원센터는 주소에서 시도/시군구를 파싱합니다.
    """
    try:
        health_centers = pd.read_csv(
            'data/raw/전국_지역보건의료기관_현황_20221231.csv',
            encoding='cp949',
        )
    except FileNotFoundError:
        health_centers = pd.DataFrame()

    try:
        support_centers = pd.read_csv(
            'data/raw/전국건강생활지원센터표준데이터.csv',
            encoding='cp949',
        )

        # 주소 예: "대구광역시 달서구 ..."
        address_parts = support_centers['소재지도로명주소'].fillna('').str.split()

        support_centers['시도'] = address_parts.str[0]
        support_centers['시군구'] = address_parts.str[1]

    except FileNotFoundError:
        support_centers = pd.DataFrame()

    return health_centers, support_centers


# 서비스 시작과 동시에 로드
scaler, classifier, cluster_mapping = get_models()
comparison_stats = get_comparison_stats()
health_centers_df, support_centers_df = get_regional_data()

# 시도 목록 구성
SIDO_LIST = ['선택 안 함']
if not health_centers_df.empty:
    SIDO_LIST += sorted(health_centers_df['시도'].dropna().unique().tolist())

def get_sigungu_list(sido: str) -> list:
    """
    선택한 시도에 해당하는 시군구 목록을 만듭니다.
    지역보건의료기관 데이터와 건강생활지원센터 데이터를 합쳐서 사용합니다.
    """
    sigungu_set = set()

    if sido == '선택 안 함':
        return ['선택 안 함']

    # 지역보건의료기관 데이터: 시군구 컬럼이 있음
    if not health_centers_df.empty and '시군구' in health_centers_df.columns:
        values = (
            health_centers_df.loc[health_centers_df['시도'] == sido, '시군구']
            .dropna()
            .unique()
            .tolist()
        )
        sigungu_set.update(values)

    # 건강생활지원센터 데이터: 주소에서 파싱한 시군구 사용
    if not support_centers_df.empty and '시군구' in support_centers_df.columns:
        values = (
            support_centers_df.loc[support_centers_df['시도'] == sido, '시군구']
            .dropna()
            .unique()
            .tolist()
        )
        sigungu_set.update(values)

    return ['선택 안 함'] + sorted(sigungu_set)


# ──────────────────────────────────────────────────────────────
# 세션 상태 초기화 (페이지 라우팅용)
# ──────────────────────────────────────────────────────────────
# 'page' 변수가 없으면 'intro'(소개 화면)부터 시작
if 'page' not in st.session_state:
    st.session_state.page = 'intro'

# 사용자 입력 저장용 딕셔너리도 초기화
if 'user_input' not in st.session_state:
    st.session_state.user_input = {}


# 페이지 이동 헬퍼 함수
def go_to(page_name: str):
    """페이지를 바꾸고 화면을 다시 그립니다."""
    st.session_state.page = page_name
    st.rerun()


# ──────────────────────────────────────────────────────────────
# 공통 UI: 주의문구 (모든 결과 화면 하단에 필수)
# ──────────────────────────────────────────────────────────────
DISCLAIMER = """
> ⚠️ **주의**: 본 서비스는 공공데이터 기반 건강관리 참고 정보를 제공하는 교육용 프로젝트입니다.
> 질병의 진단, 치료, 처방 또는 의학적 판단을 대체하지 않습니다.
> 입력 수치가 높거나 이상 소견이 의심되는 경우 의료기관 또는 보건소 상담을 권장합니다.
"""


# ══════════════════════════════════════════════════════════════
# 화면 1: 서비스 소개
# ══════════════════════════════════════════════════════════════
def render_intro():
    st.title('🏥 건강위험 자가점검 서비스')
    st.markdown('### 공공데이터 기반 맞춤 건강관리 안내')

    st.markdown("""
    **이 서비스는?**

    국민건강보험공단 건강검진 100만 명 표본 데이터를 분석하여,
    당신과 비슷한 사람들이 어떤 건강관리가 필요한지 알려드립니다.

    **무엇을 알 수 있나요?**

    - 📊 당신의 건강관리 유형 (6가지 중 하나)
    - 🔍 BMI·혈압·혈당·콜레스테롤 해석
    - 👥 같은 성별·연령대 검진자 그룹과의 비교
    - 🌟 맞춤 생활관리 추천
    - 🏥 거주 지역 인근 건강관리 자원 안내

    **어떤 정보가 필요한가요?**

    - 필수: 성별, 연령대, 키, 몸무게
    - 선택: 혈압, 공복혈당, 콜레스테롤, 허리둘레, 흡연·음주, 운동 정보, 거주지역

    *모르는 항목은 입력하지 않아도 됩니다.*
    """)

    st.markdown(DISCLAIMER)
    st.markdown('---')

    if st.button('🚀 자가점검 시작하기', type='primary', use_container_width=True):
        go_to('input')


# ══════════════════════════════════════════════════════════════
# 화면 2: 입력 폼
# ══════════════════════════════════════════════════════════════
def render_input():
    st.title('📝 건강정보 입력')
    st.caption('아는 항목만 입력하셔도 됩니다.')

    # ── Step 1: 기본정보 (필수) ──
    st.subheader('1단계. 기본정보 (필수)')

    col1, col2 = st.columns(2)
    with col1:
        gender_label = st.radio(
            '성별',
            options=['남성', '여성'],
            horizontal=True,
        )
    with col2:
        age_options = {
            '25-29세':  5, '30-34세':  6, '35-39세':  7, '40-44세':  8,
            '45-49세':  9, '50-54세': 10, '55-59세': 11, '60-64세': 12,
            '65-69세': 13, '70-74세': 14, '75-79세': 15, '80-84세': 16,
            '85세 이상': 17, '90세 이상': 18
        }
        age_label = st.selectbox(
            '연령대',
            options=list(age_options.keys()),
            index=2,    # 기본값: 35-39세
        )

    col3, col4 = st.columns(2)
    with col3:
        # 실수 입력: step=0.1, format="%.1f"
        height_cm = st.number_input(
            '키 (cm)',
            min_value=100.0, max_value=250.0,
            value=165.0, step=0.1, format='%.1f',
        )
    with col4:
        # 실수 입력: step=0.1, format="%.1f"
        weight_kg = st.number_input(
            '몸무게 (kg)',
            min_value=20.0, max_value=300.0,
            value=60.0, step=0.1, format='%.1f',
        )

    # 실시간 BMI 미리 보여주기
    bmi_preview = calculate_bmi(height_cm, weight_kg)
    st.info(f'📐 입력하신 정보로 계산된 BMI: **{bmi_preview}**')

    # ── Step 2: 건강검진 수치 (선택) ──
    st.subheader('2단계. 건강검진 수치 (선택)')
    st.caption('입력한 항목만 분석에 반영됩니다. 모르면 비워두세요.')

    col5, col6 = st.columns(2)
    with col5:
        sbp = st.number_input(
            '수축기혈압 (mmHg)', min_value=0, max_value=250, value=0,
            help='0이면 미입력으로 처리됩니다',
        )
    with col6:
        dbp = st.number_input(
            '이완기혈압 (mmHg)', min_value=0, max_value=150, value=0,
            help='0이면 미입력으로 처리됩니다',
        )

    col7, col8 = st.columns(2)
    with col7:
        fbg = st.number_input(
            '공복혈당 (mg/dL)', min_value=0, max_value=500, value=0,
            help='0이면 미입력으로 처리됩니다',
        )
    with col8:
        waist = st.number_input(
            '허리둘레 (cm)', min_value=0, max_value=200, value=0,
            help='0이면 미입력으로 처리됩니다',
        )

    # 콜레스테롤 패널 (4종) - 접을 수 있는 영역
    with st.expander('💡 이상지질혈증 검사 수치 (알고 있으면 입력)'):
        st.caption('LDL·HDL·중성지방·총콜레스테롤 4종 모두 선택 입력입니다.')

        col9, col10 = st.columns(2)
        with col9:
            ldl = st.number_input('LDL 콜레스테롤 (mg/dL)',
                                  min_value=0, max_value=300, value=0)
            hdl = st.number_input('HDL 콜레스테롤 (mg/dL)',
                                  min_value=0, max_value=150, value=0)
        with col10:
            tg = st.number_input('중성지방 (mg/dL)',
                                 min_value=0, max_value=2000, value=0)
            total_chol = st.number_input('총콜레스테롤 (mg/dL)',
                                         min_value=0, max_value=500, value=0)

    # ── Step 3: 생활습관 (권장) ──
    st.subheader('3단계. 생활습관 (권장)')

    col11, col12 = st.columns(2)
    with col11:
        smoking_label = st.selectbox(
            '흡연',
            options=['선택 안 함', '비흡연', '과거흡연', '현재흡연'],
        )
    with col12:
        drinking_label = st.selectbox(
            '음주',
            options=['선택 안 함', '음주 안 함', '음주 함'],
        )

    # 운동 정보 — expander 제거, 직접 표시
    st.markdown('**🏃 운동 정보** (선택)')

    col13, col14, col15 = st.columns(3)
    with col13:
        ex_freq = st.selectbox(
            '운동 빈도',
            options=['선택 안 함', '안함', '주1-2회', '주2-3회', '주3-4회', '매일'],
        )
    with col14:
        ex_intensity = st.selectbox(
            '운동 강도',
            options=['선택 안 함', '저강도', '중강도', '고강도'],
        )
    with col15:
        ex_duration = st.number_input(
            '1회 운동 시간 (분)',
            min_value=0, max_value=300, value=0,
        )

    # 강도 설명 — expander로 제공
    with st.expander('💡 운동 강도 기준이 궁금하신가요?'):
        st.markdown("""
        **저강도** — 대화하면서 노래도 부를 수 있는 수준
        - 예: 천천히 걷기, 스트레칭, 느린 요가, 집안일(설거지·청소)

        **중강도** — 대화는 할 수 있지만 노래하기는 어려운 수준
        - 예: 빠르게 걷기, 자전거 타기, 수영, 댄스, 등산(낮은 경사)
        - 숨이 약간 차고 심박수가 올라가는 느낌

        **고강도** — 몇 마디 이상 대화하기 어려운 수준
        - 예: 달리기, 등산(가파른 경사), 빠른 수영, 축구·농구 경기, HIIT
        - 숨이 많이 차고 땀이 많이 나는 느낌

        > 보건복지부·한국건강증진개발원 신체활동 지침서 개정판(2023) 기준
        """)

    # ── Step 4: 거주지역 (선택) ──
    st.subheader('4단계. 거주지역 (선택)')
    st.caption('선택하시면 인근 보건소·건강생활지원센터 정보를 안내해드립니다.')

    selected_sido = st.selectbox(
        '거주 시·도',
        options=SIDO_LIST,
    )

    sigungu_options = get_sigungu_list(selected_sido)

    selected_sigungu = st.selectbox(
        '거주 시·군·구',
        options=sigungu_options,
        disabled=(selected_sido == '선택 안 함'),
    )

    # ── 자가점검 결과 보기 버튼 ──
    st.markdown('---')
    col_back, col_submit = st.columns([1, 3])
    with col_back:
        if st.button('← 처음으로'):
            go_to('intro')
    with col_submit:
        if st.button('🔍 자가점검 결과 보기',
                     type='primary', use_container_width=True):
            # 사용자 입력을 dict로 변환 (0이면 None으로 = 미입력 처리)
            ui = {
                'gender': 1 if gender_label == '남성' else 2,
                'age_group': age_options[age_label],
                'height_cm': float(height_cm),
                'weight_kg': float(weight_kg),
                'bmi': bmi_preview,
            }
            # 0인 입력은 None으로 처리 (미입력 = 모름)
            if sbp > 0:        ui['systolic_bp']       = float(sbp)
            if dbp > 0:        ui['diastolic_bp']      = float(dbp)
            if fbg > 0:        ui['fasting_glucose']   = float(fbg)
            if waist > 0:      ui['waist_cm']          = float(waist)
            if ldl > 0:        ui['ldl_cholesterol']   = float(ldl)
            if hdl > 0:        ui['hdl_cholesterol']   = float(hdl)
            if tg > 0:         ui['triglyceride']      = float(tg)
            if total_chol > 0: ui['total_cholesterol'] = float(total_chol)

            # 흡연·음주 코드 변환
            smoking_map = {'비흡연': 1, '과거흡연': 2, '현재흡연': 3}
            if smoking_label in smoking_map:
                ui['smoking_status'] = smoking_map[smoking_label]

            drinking_map = {'음주 안 함': 0, '음주 함': 1}
            if drinking_label in drinking_map:
                ui['drinking'] = drinking_map[drinking_label]

            # 운동 정보 (3개 다 선택돼야 의미 있음)
            if (ex_freq not in ('선택 안 함', '안함')
                    and ex_intensity != '선택 안 함'
                    and ex_duration > 0):
                ui['exercise'] = {
                    'frequency': ex_freq,
                    'intensity': ex_intensity,
                    'duration_min': int(ex_duration),
                }

            ui['region_sido'] = selected_sido if selected_sido != '선택 안 함' else None
            ui['region_sigungu'] = selected_sigungu if selected_sigungu != '선택 안 함' else None

            # 세션에 저장하고 결과 페이지로
            st.session_state.user_input = ui
            go_to('result')


# ══════════════════════════════════════════════════════════════
# 지역 자원 조회 헬퍼
# ══════════════════════════════════════════════════════════════
def get_resources_for_region(sido: str, sigungu: str | None = None) -> tuple:
    """
    선택한 시도/시군구의 보건소와 건강생활지원센터를 조회합니다.
    보건소는 기관유형='보건소'만 필터링합니다.
    """
    hc_list, sc_list = [], []

    if not sido:
        return hc_list, sc_list

    # 1) 보건소 필터링
    if not health_centers_df.empty:
        filtered = health_centers_df[
            (health_centers_df['시도'] == sido) &
            (health_centers_df['기관유형'] == '보건소')
        ]

        if sigungu and '시군구' in filtered.columns:
            filtered = filtered[filtered['시군구'] == sigungu]

        for _, row in filtered.iterrows():
            hc_list.append({
                'name': row.get('보건기관명', ''),
                'address': row.get('주소', ''),
                'phone': row.get('대표 전화번호', ''),
            })

    # 2) 건강생활지원센터 필터링
    if not support_centers_df.empty and '시도' in support_centers_df.columns:
        filtered = support_centers_df[
            support_centers_df['시도'] == sido
        ]

        if sigungu and '시군구' in filtered.columns:
            filtered = filtered[filtered['시군구'] == sigungu]

        for _, row in filtered.iterrows():
            hours = f"{row.get('운영시작시각', '')}~{row.get('운영종료시각', '')}"

            sc_list.append({
                'name': row.get('건강생활지원센터명', ''),
                'address': row.get('소재지도로명주소', ''),
                'phone': row.get('운영기관전화번호', ''),
                'hours': hours,
                'closed': row.get('휴무일정보', ''),
            })

    return hc_list, sc_list


# ══════════════════════════════════════════════════════════════
# 화면 3: 결과
# ══════════════════════════════════════════════════════════════
def render_result():
    ui = st.session_state.user_input

    if not ui:
        st.warning('입력 정보가 없습니다. 처음부터 다시 시작해주세요.')
        if st.button('처음으로'):
            go_to('intro')
        return

    # 모델 예측 + 룰 해석 + 비교군 비교 + 추천 모두 한 번에
    with st.spinner('분석 중...'):
        result = predict_health_type(
            ui, scaler, classifier, cluster_mapping, comparison_stats,
        )

    st.title('📊 자가점검 결과')

    # ── 카드 1: 요약 ──
    ht = result['health_type']
    st.markdown(
        f'<div style="background-color:{ht["color"]}22; padding:20px; '
        f'border-radius:10px; border-left:5px solid {ht["color"]};">'
        f'<h3 style="color:{ht["color"]}; margin:0;">'
        f'당신의 건강관리 유형: {ht["name"]}</h3>'
        f'<p style="margin:5px 0 0 0;">예측 신뢰도: {ht["probability"]:.1%} '
        f'| 입력 완성도: {result["completeness"]}%</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('')

    st.caption(
        "건강관리 유형은 전체 수치 패턴 기반 주된 관리 방향을 나타냅니다. "
        "혈압·혈당·콜레스테롤 등 개별 지표 상세 안내는 아래 카드를 확인하세요."
    )

    # ── 카드 2: 기본 건강지표 (BMI) ──
    with st.container(border=True):
        st.markdown('### 📐 기본 건강지표')
        bmi_result = result['rule_results']['bmi']
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric('BMI', f'{result["bmi"]}', bmi_result['label'])
        with col_b:
            st.write(bmi_result['message'])

    # ── 카드 3: 혈압·혈당 ──
    rr = result['rule_results']
    if rr['blood_pressure'] or rr['fasting_glucose']:
        with st.container(border=True):
            st.markdown('### 🩺 혈압·혈당')

            if rr['blood_pressure']:
                bp = rr['blood_pressure']
                st.write(f'**혈압**: {ui.get("systolic_bp","?")}/'
                         f'{ui.get("diastolic_bp","?")} mmHg → **{bp["label"]}**')
                st.caption(bp['message'])

            if rr['fasting_glucose']:
                fg = rr['fasting_glucose']
                st.write(f'**공복혈당**: {ui.get("fasting_glucose","?")} mg/dL '
                         f'→ **{fg["label"]}**')
                st.caption(fg['message'])

    # ── 카드 4: 지질 수치 (콜레스테롤) ──
    if rr['cholesterol_panel']:
        with st.container(border=True):
            st.markdown('### 🩸 지질 수치 (콜레스테롤)')
            for c in rr['cholesterol_panel']:
                st.write(f'**{c["metric_label"]}**: {c["value"]} mg/dL → **{c["label"]}**')
                st.caption(c['message'])
                if c.get('is_supplementary'):
                    st.caption('💡 LDL·HDL·중성지방 수치를 입력하면 더 정확한 평가가 가능합니다.')

    # ── 카드 5: 주요 관리 포인트 ──
    risk_count = rr['metabolic_risk_count']
    with st.container(border=True):
        st.markdown('### 🎯 주요 관리 포인트')
        st.write(f'관리가 필요한 지표 수: **{risk_count}개**')

        if risk_count == 0:
            st.success('현재 측정된 지표는 모두 적정 범위에 있습니다.')
        elif risk_count <= 2:
            st.warning(f'{risk_count}개 항목이 관리 필요 범위에 있습니다.')
        else:
            st.error(f'{risk_count}개 항목이 동시에 관리 필요 범위에 있어 복합 관리가 권장됩니다.')

    # ── 카드 6: 맞춤 생활관리 추천 ──
    with st.container(border=True):
        st.markdown('### 🌟 맞춤 생활관리 추천')
        for rec in result['recommendations']:
            st.write(f'• {rec}')

    # ── 카드 7: 비교군 비교 ──
    if result['comparison'] and result['group_info']:
        gi = result['group_info']
        with st.container(border=True):
            st.markdown('### 👥 비교군 비교')
            st.caption(f'{gi["age_label"]} {gi["gender_label"]} 검진자 '
                       f'그룹 (n={gi["n"]:,}명) 대비')
            for c in result['comparison']:
                # level에 따라 색상 다르게
                icon = '🔺' if c['level'] == 'higher' else ('🔻' if c['level'] == 'lower' else '🟢')
                st.write(f'{icon} **{c["label"]}**: 당신 {c["user_value"]}{c["unit"]} '
                         f'| 평균 {c["group_mean"]}{c["unit"]} (±{c["group_std"]})')

    # ── 카드 8: 운동 분석 (입력했을 때만) ──
    if result['exercise']:
        ex = result['exercise']
        with st.container(border=True):
            st.markdown('### 🏃 운동 분석 (보건복지부 신체활동 지침 2023 기준)')
            st.write(ex['message'])
            if ex['meets_aerobic_guideline']:
                st.success(ex['suggestion'])
            else:
                st.info(ex['suggestion'])

    # ── 카드 9: 지역사회 건강관리 자원 ──
    region_sido = ui.get('region_sido')
    region_sigungu = ui.get('region_sigungu')
    with st.container(border=True):
        st.markdown('### 🏥 지역사회 건강관리 자원')

        if region_sido:
            hc_list, sc_list = get_resources_for_region(region_sido, region_sigungu)

            if region_sigungu:
                st.caption(f'📍 {region_sido} {region_sigungu} 기준')
            else:
                st.caption(f'📍 {region_sido} 기준')

            if hc_list:
                st.markdown('**보건소**')
                for hc in hc_list:
                    st.write(f'**{hc["name"]}**')
                    if hc['address']:
                        st.caption(f'주소: {hc["address"]}')
                    if hc['phone']:
                        st.caption(f'전화: {hc["phone"]}')
                    st.markdown('')
            else:
                st.caption('해당 지역 보건소 정보를 불러오지 못했습니다.')

            if sc_list:
                st.markdown('**건강생활지원센터**')
                for sc in sc_list:
                    st.write(f'**{sc["name"]}**')
                    if sc['address']:
                        st.caption(f'주소: {sc["address"]}')
                    if sc['phone']:
                        st.caption(f'전화: {sc["phone"]}')
                    if sc['hours'] and sc['hours'] != '~':
                        closed_str = f' | 휴무: {sc["closed"]}' if sc['closed'] else ''
                        st.caption(f'운영: {sc["hours"]}{closed_str}')
                    st.markdown('')
            else:
                st.caption('해당 지역 건강생활지원센터 정보가 없습니다.')

            st.caption('※ 지역보건의료기관 현황(2022.12.31 기준) | 건강생활지원센터 현황 기반')

        else:
            st.caption(
                '입력 화면 4단계에서 거주 시·도와 시·군·구를 선택하시면 '
                '인근 보건소·건강생활지원센터 정보를 확인할 수 있습니다.'
            )

    # ── 주의문구 (필수) ──
    st.markdown('---')
    st.markdown(DISCLAIMER)

    # ── 다시 시작 버튼 ──
    st.markdown('---')
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button('← 입력 수정', use_container_width=True):
            go_to('input')
    with col_b:
        if st.button('🔄 처음부터 다시', use_container_width=True):
            st.session_state.user_input = {}
            go_to('intro')


# ══════════════════════════════════════════════════════════════
# 라우팅: 현재 페이지에 따라 다른 함수 호출
# ══════════════════════════════════════════════════════════════
if st.session_state.page == 'intro':
    render_intro()
elif st.session_state.page == 'input':
    render_input()
elif st.session_state.page == 'result':
    render_result()
