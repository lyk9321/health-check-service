# 14. 참고 자료 및 레퍼런스 (References)

> **문서 버전**: v1.2.0 | **최종 수정**: 2026-05-07

---

## 14.1 핵심 이론 및 알고리즘

### K-Means 군집화

```
MacQueen, J. (1967).
Some methods for classification and analysis of multivariate observations.
Proceedings of the 5th Berkeley Symposium on Mathematical Statistics
and Probability, 1(14), 281-297.

[인용 이유]
- 본 프로젝트의 건강관리 유형 도출에 사용하는 핵심 군집화 알고리즘
```

### Random Forest

```
Breiman, L. (2001).
Random Forests.
Machine Learning, 45(1), 5-32.
DOI: 10.1023/A:1010933404324

[인용 이유]
- 사용자 건강관리 유형 분류의 메인 모델
- 변수 중요도(Feature Importance) 해석에 활용
```

### 건강신념모형 (Health Belief Model)

```
Rosenstock, I. M. (1974).
Historical origins of the health belief model.
Health Education Monographs, 2(4), 328-335.
DOI: 10.1177/109019817400200403

[인용 이유]
- 서비스의 이론적 배경: 지각된 민감성, 지각된 장애, 행동 계기 등을
  서비스 설계에 반영
```

### Silhouette Analysis

```
Rousseeuw, P. J. (1987).
Silhouettes: A graphical aid to the interpretation and validation
of cluster analysis.
Journal of Computational and Applied Mathematics, 20, 53-65.
DOI: 10.1016/0377-0427(87)90125-7

[인용 이유]
- 군집화 최적 K 결정에 사용하는 Silhouette Score의 원본 논문
```

---

## 14.2 건강 기준 출처

| 기준 | 출처 | URL |
|------|------|-----|
| BMI 분류 기준 | 대한비만학회 (2022) | https://www.kosso.or.kr |
| 혈압 분류 기준 | 대한고혈압학회 (2022) | https://www.koreanhypertension.org |
| 공복혈당 기준 | 대한당뇨병학회 (2023) | https://www.diabetes.or.kr |
| 총콜레스테롤 기준 | 한국지질·동맥경화학회 (2022) | https://www.lipid.or.kr |
| 허리둘레 기준 | 대한비만학회 (2022) | https://www.kosso.or.kr |
| **신체활동 권장량** | **보건복지부·한국건강증진개발원 (2023)** | **https://www.khepi.or.kr** |

---

## 14.3 신체활동 지침 (운동 추천 로직 핵심 기준)

```
보건복지부, 한국건강증진개발원 (2023).
한국인을 위한 신체활동 지침서 개정판.
관리번호: 사업-05-2023-004-15.
서울: 한국건강증진개발원.

[인용 이유]
- src/recommend.py의 get_exercise_recommendation() 함수 내
  주간 신체활동 권장 기준의 직접적인 근거
```

**서비스에 적용한 핵심 지침 내용** (성인 만 19~64세 기준):

| 운동 유형 | 권장 기준 | 추가 건강 이점 |
|---------|---------|-------------|
| 중강도 유산소 | 주 150~300분 | 300분 이상 시 추가 이점 |
| 고강도 유산소 | 주 75~150분 | 고강도 1분 = 중강도 2분 |
| 근력 운동 | 주 2일 이상 | 신체 각 부위 고루 포함 |
| 앉아있는 시간 | 가능한 최소화 | - |

**노인(만 65세 이상) 추가 기준**:

| 운동 유형 | 권장 기준 | 목적 |
|---------|---------|------|
| 평형성 운동 | 주 3일 이상 | 낙상 예방 |

**강도 판정 기준** (본 서비스 운동 입력 UI 기반):

| 강도 | 토크 테스트 기준 | RPE 기준 | 중강도 환산 계수 |
|------|--------------|---------|--------------|
| 저강도 | 대화 및 노래 가능 | 1~4 | × 0.5 |
| 중강도 | 대화 가능, 노래 불가 | 5~6 | × 1.0 |
| 고강도 | 대화 곤란 | 7~8 | × 2.0 |

---

## 14.4 공식 문서 (Official Documentation)

| 도구 | 문서 URL | 참조 섹션 |
|------|---------|----------|
| Streamlit | https://docs.streamlit.io | API Reference, Session State, Cache |
| scikit-learn | https://scikit-learn.org/1.4 | cluster, ensemble, metrics, preprocessing |
| pandas | https://pandas.pydata.org/docs/2.2 | DataFrame, GroupBy |
| plotly | https://plotly.com/python | Bar, Box, Scatter |
| folium | https://python-visualization.github.io/folium | Map, Marker |
| joblib | https://joblib.readthedocs.io | dump, load |

---

## 14.5 데이터 출처

| 데이터셋 | URL | 접근 방법 | 라이선스 |
|---------|-----|----------|---------|
| 국민건강보험공단 건강검진정보 (2024, 100만 건 표본) | https://www.data.go.kr | **직접 CSV 다운로드** (신청 불필요) | 공공누리 제1유형 |
| 전국 지역보건의료기관 현황 (2022.12.31 기준) | https://www.data.go.kr | CSV 직접 다운로드 | 공공누리 제1유형 |
| 전국 건강생활지원센터 표준데이터 | https://www.data.go.kr | CSV 직접 다운로드 | 공공누리 제1유형 |

### 건강검진정보 데이터 상세

```
[국민건강보험공단 건강검진정보 샘플 데이터 - 공공데이터포털]
- 규모: 1,000,000 건 (2024년 기준)
- 파일 형식: CSV (인코딩: CP949)
- 컬럼 수: 33개
- 포함 변수: 성별, 연령대(5세단위), 신장, 체중, 허리둘레, 수축기혈압,
            이완기혈압, 식전혈당(공복혈당), 총콜레스테롤(~34%만 존재),
            HDL/LDL/트리글리세라이드, 혈색소, 흡연상태, 음주여부, 구강검진 항목 등
- 라이선스: 공공누리 제1유형
- 활용 제한: 출처 표기 필수, 비상업적 목적
```

---

## 14.6 참고 프로젝트 및 코드

### 유사 서비스 참고

```
[국가건강정보포털]
URL: https://health.kdca.go.kr
참고 이유: 건강지표 설명 방식, 사용자 안내 표현

[국민건강보험공단 건강IN]
URL: https://hi.nhis.or.kr
참고 이유: 건강검진 결과 해석 방식, 생활관리 추천 방식
```

### GitHub 참고 저장소

```
검색 키워드:
- "streamlit health dashboard"
- "kmeans health clustering python"
- "건강검진 데이터 분석"
- "public health data visualization streamlit"
```

---

## 14.7 학습 자료

| 제목 | 형식 | URL |
|------|------|-----|
| Streamlit 공식 튜토리얼 | 공식 가이드 | https://docs.streamlit.io/get-started |
| scikit-learn Clustering 가이드 | 공식 문서 | https://scikit-learn.org/stable/modules/clustering.html |
| K-Means Elbow Method | 블로그 | https://realpython.com/k-means-clustering-python |
| Random Forest Feature Importance | scikit-learn 예제 | https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html |
| Streamlit Session State | 공식 가이드 | https://docs.streamlit.io/develop/api-reference/caching-and-state |

---

## 14.8 법적 고지 및 라이선스

```
[라이브러리 라이선스]
- scikit-learn: BSD 3-Clause License
- Streamlit: Apache License 2.0
- pandas: BSD 3-Clause License
- plotly: MIT License
- folium: MIT License

[데이터 라이선스]
- 건강검진정보 2024 표본: 공공누리 제1유형 (data.go.kr)
  → 출처 표기 필수, 변형 가능, 비상업적 이용
  → 원본 데이터 GitHub 업로드 금지

[신체활동 지침서]
- 한국인을 위한 신체활동 지침서 개정판 (2023)
  → 발행처: 보건복지부, 한국건강증진개발원
  → 교육·참고 목적으로 기준값 인용 (비상업적 용도)

[코드 라이선스]
- 본 프로젝트 코드: MIT License (포트폴리오 공개용)

[주의사항]
- 원본 건강검진 데이터는 GitHub에 업로드 금지
- 서비스 상업화 시 데이터 제공기관 라이선스 재확인 필요
- 의료 서비스로 전환 시 의료기기 소프트웨어 관련 법령 검토 필요
```
