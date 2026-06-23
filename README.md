# 건강위험 자가점검 서비스 — 헬스케어 AI

## 한 줄 요약
검진센터 임상 경험에서 출발한 프로젝트.
국민건강보험공단 **100만 건 실데이터**로 건강관리 유형 분류 + 생활관리 방향 추천.
Random Forest Macro F1: **0.9903** — Streamlit Cloud 실서비스 운영 중.

## 왜 만들었는가 — 임상 경험에서 나온 문제 정의
검진센터에서 3년간 수검자를 직접 상담한 경험에서 출발.
결과지를 받아도 "제 수치가 나쁜 건가요?"를 반복해서 묻는 수검자를 매일 만났다.
→ 수치를 스스로 해석하고, 자신의 건강관리 방향을 찾을 수 있는 AI 서비스를 직접 만들었다.

## 핵심 설계 판단 — 임상 기준 우선

### 왜 K-Means를 버렸는가
K-Means로 군집화 시 음주 여부(0/1)로 군집이 나뉘어 임상적으로 무의미한 분류 발생.
→ 임상 기준 우선순위 규칙으로 직접 라벨 생성하는 Rule Based 방식으로 전환.
아키텍처 변경 전 PRD를 먼저 수정하고 구현 진행.

### 콜레스테롤 결측률 66% — 임상 판단으로 처리
콜레스테롤 4종 결측률 66% → 단순 제외가 아닌 설계 결정.
모델 학습 변수에서 제외하고, 사용자가 직접 입력한 경우에만 임상 기준 룰 기반 해석 제공.
→ 데이터 품질과 임상 활용을 동시에 고려한 아키텍처.

### 99.0% 정확도에 대한 솔직한 해석
Rule Based 라벨링 → RF 분류 구조에서의 정확도.
모델이 규칙을 학습한 것이므로 실제 질환 예측 성능과는 다름.
외부 검증(실제 질환 발생 데이터와 비교)이 다음 과제.

## 핵심 성과
| 지표 | 결과 |
|------|------|
| Macro F1 | **0.9903** |
| 학습 데이터 | 국민건강보험공단 건강검진 100만 건 |
| 건강관리 유형 | 6개 (Rule Based 라벨링 기반) |
| 배포 | Streamlit Cloud 실서비스 운영 중 |

## 트러블슈팅

**음주 미입력 사용자 처리**
그룹 평균 음주율(float 0.78)이 모델에 입력되는 버그 →
`if drinking_rate >= 0.5 else 0` 이진 변환 처리.
결측 대체 로직을 함수 단위로 통일.

**iterrows() 성능 병목**
100만 건 처리 5분 이상 → `np.select()` 벡터화로 수 초 단축.

## 기술 스택
`Python` `Scikit-learn` `Random Forest` `Pandas` `NumPy` `Streamlit` `GitHub`

## 서비스 링크
- 실서비스: https://health-check-service.streamlit.app
- GitHub: github.com/lyk9321/health-check-service

## 서비스 스크린샷
<img width="1105" height="1233" alt="image" src="https://github.com/user-attachments/assets/0b0ab317-3187-44f4-b1a5-e4f7d343a838" />  

서비스 안내 화면  


<img width="891" height="1168" alt="image" src="https://github.com/user-attachments/assets/18a48c2e-440c-4547-8e99-68bc141245af" />  
<img width="903" height="1299" alt="image" src="https://github.com/user-attachments/assets/5efec7db-b706-4f1c-8a0e-b40ba3c8023d" />  

대상자 건강 정보 입력 화면  


<img width="877" height="1330" alt="image" src="https://github.com/user-attachments/assets/48bcc762-a75c-49fc-892a-9c50e810f600" />  
<img width="876" height="1067" alt="image" src="https://github.com/user-attachments/assets/1c37d29a-b215-4292-abdd-2411077869c1" />  
<img width="888" height="1041" alt="image" src="https://github.com/user-attachments/assets/2556f1b6-50b5-4840-bcdb-682e09ccf4c0" />  

자가 점검 결과 화면
