# 12. 개발 로드맵 (Development Roadmap)

> **문서 버전**: v1.0.0 | **최종 수정**: 2026-05-07

---

## 12.1 4단계 마일스톤 개요

| 단계 | 핵심 목표 | 완료 기준 |
|------|----------|----------|
| **Phase 1** | 데이터 수집 및 전처리 | 전처리 완료, 파생변수 생성, EDA 노트북 완성 |
| **Phase 2** | 군집화 + 분류 모델 학습 | Silhouette ≥ 0.30, Accuracy ≥ 0.75, 모델 파일 저장 |
| **Phase 3** | Streamlit 앱 구현 | 필수 입력 → 결과 출력 전체 플로우 동작 |
| **Phase 4** | 배포 + 산출물 완성 | 웹앱 URL 공개, README·노트북·발표자료 완성 |

---

## 12.2 Phase 1: 데이터 수집 및 전처리

### 주요 작업 목록

| 작업 | 완료 기준 |
|------|----------|
| 건강검진정보 데이터 다운로드 | `data/raw/nhis_2024_sample.csv` 존재 |
| 지역보건의료기관·건강생활지원센터 데이터 수집 | `data/external/` 파일 2종 수집 완료 |
| `verify_env.py` 실행 → 환경 검증 통과 | 모든 항목 ✅ |
| 데이터 품질 검증 (`validate_data.py`) | 오류 0건 또는 수정 완료 |
| `src/data/preprocessor.py` 구현 | 단위 테스트 통과 |
| 파생변수 생성 (`feature_engineer.py`) | BMI, flag 변수 6종 생성 확인 |
| `notebooks/01_EDA.ipynb` 완성 | 분포 시각화 + 결측률 + 인사이트 |
| `notebooks/02_preprocessing.ipynb` 완성 | 전처리 전/후 비교 시각화 포함 |

**Phase 1 완료 게이트**:
- [ ] `data/processed/train.csv`, `val.csv`, `test.csv` 생성 완료
- [ ] EDA 노트북 실행 결과 저장 완료
- [ ] 성별·연령대별 표본 수 1,000건 이상 확인

---

## 12.3 Phase 2: 군집화 + 분류 모델 학습

### 주요 작업 목록

| 작업 | 완료 기준 |
|------|----------|
| K-Means K 탐색 (K=3~8) | Elbow + Silhouette 그래프 생성 |
| 최적 K 결정 및 유형명 부여 | 군집 프로파일 테이블 + 유형명 6종 확정 |
| `cluster_mapping.json` 작성 | 유형명, 색상, 추천 유형 매핑 |
| `models/kmeans.pkl`, `scaler.pkl` 저장 | 로딩 테스트 통과 |
| Random Forest 학습 (baseline) | CV Macro F1 ≥ 0.60 |
| 하이퍼파라미터 탐색 (GridSearchCV) | 최적 파라미터 확정 |
| `models/classifier.pkl` 저장 | 로딩 + 추론 테스트 통과 |
| Logistic Regression 비교 모델 학습 | 성능 비교 테이블 작성 |
| `notebooks/03_clustering.ipynb` 완성 | PCA 시각화 포함 |
| `notebooks/04_classification.ipynb` 완성 | 학습 곡선 + 변수 중요도 포함 |

**실험 실행 명령어**:
```bash
python scripts/train_models.py \
  --config configs/model_config.yaml \
  --experiment-name "exp01_baseline"

# 예상 실행 시간 (100만 건, CPU):
# 전처리: ~2분
# K-Means (k=6): ~5분
# Random Forest (200 trees): ~15~30분
```

**Phase 2 완료 게이트**:
- [ ] `models/kmeans.pkl`, `scaler.pkl`, `classifier.pkl` 저장 완료
- [ ] Test set Accuracy ≥ 0.75
- [ ] `experiment_log.csv` 4종 실험 기록 완료
- [ ] `data/deploy/comparison_stats.csv` 생성 완료 (배포용)

---

## 12.4 Phase 3: Streamlit 앱 구현

### 주요 작업 목록

| 작업 | 완료 기준 |
|------|----------|
| `src/core/rule_engine.py` 구현 | 단위 테스트 전체 통과 |
| `src/core/calculator.py` 구현 | BMI·완성도 계산 단위 테스트 통과 |
| `src/core/imputer.py` 구현 | 결측값 대체 + 경고 문구 생성 확인 |
| `src/core/comparator.py` 구현 | 비교군 조회 + 수치 비교 확인 |
| `src/core/recommender.py` 구현 | 유형별 추천 문구 출력 확인 |
| `src/core/local_resources.py` 구현 | 지역별 기관 조회 확인 |
| `src/models/predictor.py` 구현 | 샘플 입력 → JSON 결과 출력 성공 |
| `src/ui/intro_page.py` 구현 | 서비스 소개 화면 렌더링 확인 |
| `src/ui/input_form.py` 구현 | 5단계 입력 폼 동작 확인 |
| `src/ui/result_cards.py` 구현 | 8개 카드 렌더링 확인 |
| `app.py` 통합 연결 | 전체 플로우 (소개→입력→결과) 동작 |
| 모바일 화면 검증 | 스마트폰 Chrome에서 좌우 스크롤 없음 |
| 주의문구 표시 검증 | 모든 결과 화면 하단에 고지문 표시 |

**앱 실행 및 확인**:
```bash
streamlit run app.py
# 브라우저 http://localhost:8501 접속
# Chrome DevTools → 모바일 뷰 전환하여 레이아웃 확인
```

**Phase 3 완료 게이트**:
- [ ] 필수 입력(성별+연령대+키+몸무게)만으로 결과 출력 성공
- [ ] 혈압·혈당 미입력 시 제한 안내 + 기본 결과 표시 확인
- [ ] 지역 선택 시 보건기관 목록 표시 확인
- [ ] 주의문구 표시 확인
- [ ] `pytest tests/` 전체 통과

---

## 12.5 Phase 4: 배포 + 산출물 완성

### 주요 작업 목록

| 작업 | 완료 기준 |
|------|----------|
| Streamlit Cloud 배포 | 공개 URL 접근 가능 |
| 배포 앱 기능 검증 | 모든 입력 시나리오 결과 확인 |
| `notebooks/05_evaluation.ipynb` 완성 | CM, 변수 중요도, 군집 시각화 포함 |
| `reports/evaluation_report.json` 완성 | 최종 KPI 달성 확인 |
| `reports/final_report.md` 작성 | 8개 섹션 완성 |
| `README.md` 완성 | 재현성 테스트 통과 수준 |
| 발표자료 (`presentation.pptx`) | 슬라이드 15장 이상 |
| 코드 클린업 | `flake8` 오류 0건 |
| GitHub 최종 push | `v1.0.0` 태그 생성 |

**Phase 4 완료 게이트**:
- [ ] Streamlit Cloud URL 공개 접근 가능
- [ ] 전체 KPI 달성 (Accuracy ≥ 0.75, Silhouette ≥ 0.30)
- [ ] 노트북 5종 실행 결과 저장 완료
- [ ] README + 발표자료 완성

---

## 12.6 리스크 관리

| 리스크 | 발생 확률 | 대응 방안 |
|--------|---------|----------|
| 건강검진 데이터 신청 지연 | 중 | 사전 신청 완료, 지연 시 다른 공공 건강데이터 대체 검토 |
| Silhouette Score < 0.25 | 중 | 변수 조합 재검토, GMM 대안 사용 |
| 분류 Accuracy < 0.65 | 중 | 군집 수 조정, XGBoost 대체, 클래스 가중치 재조정 |
| Streamlit Cloud 512MB 초과 | 중 | n_estimators=100으로 감소, max_depth=10 제한 |
| 배포 시 모델 로딩 오류 | 낮 | joblib 버전 고정, 로컬 재현 후 배포 |
| 지역자원 데이터 형식 불일치 | 낮 | 전처리 스크립트로 통일, 컬럼명 매핑 |
