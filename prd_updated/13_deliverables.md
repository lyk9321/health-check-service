# 13. 산출물 목록 (Deliverables)

> **문서 버전**: v1.0.0 | **최종 수정**: 2026-05-07

---

## 13.1 산출물 전체 목록

| ID | 산출물명 | 파일 경로 | 완료 기준 | 상태 |
|----|---------|----------|----------|------|
| D01 | GitHub 저장소 | `https://github.com/{계정}/health-self-check` | 공개 리포 접근 가능 | ☐ |
| D02 | requirements.txt | `requirements.txt` | 패키지 버전 고정 완료 | ☐ |
| D03 | environment.yaml | `environment.yaml` | conda 환경 재현 가능 | ☐ |
| D04 | 전처리 데이터 | `data/processed/` (train/val/test.csv) | 분할 완료, 파생변수 포함 | ☐ |
| D05 | 비교군 통계 | `data/deploy/comparison_stats.csv` | 성별·연령대별 집계 완료 | ☐ |
| D06 | scaler 파일 | `models/scaler.pkl` | 로딩·변환 테스트 통과 | ☐ |
| D07 | 군집화 모델 | `models/kmeans.pkl` | 군집 라벨 생성 테스트 통과 | ☐ |
| D08 | 분류 모델 | `models/classifier.pkl` | Accuracy ≥ 0.75 | ☐ |
| D09 | 군집 매핑 | `models/cluster_mapping.json` | 유형명 6종 완성 | ☐ |
| D10 | 설정 파일들 | `configs/*.yaml` | rule·recommendation 포함 5종 | ☐ |
| D11 | EDA 노트북 | `notebooks/01_EDA.ipynb` | 실행 결과 저장 완료 | ☐ |
| D12 | 전처리 노트북 | `notebooks/02_preprocessing.ipynb` | 실행 결과 저장 완료 | ☐ |
| D13 | 군집화 노트북 | `notebooks/03_clustering.ipynb` | PCA 시각화 포함, 실행 완료 | ☐ |
| D14 | 분류 노트북 | `notebooks/04_classification.ipynb` | 실행 완료, 성능 테이블 포함 | ☐ |
| D15 | 평가 노트북 | `notebooks/05_evaluation.ipynb` | CM·변수 중요도 포함, 실행 완료 | ☐ |
| D16 | 실험 로그 | `reports/experiment_log.csv` | 4종 실험 기록 | ☐ |
| D17 | 평가 결과 JSON | `reports/evaluation_report.json` | KPI 달성 여부 포함 | ☐ |
| D18 | 최종 보고서 | `reports/final_report.md` | 8개 섹션 완성 | ☐ |
| D19 | README | `README.md` | 재현 가능 수준 | ☐ |
| D20 | 발표자료 | `reports/presentation.pptx` | 슬라이드 15장 이상 | ☐ |
| D21 | Streamlit 웹앱 | Streamlit Cloud URL | 공개 접근 가능 | ☐ |

---

## 13.2 산출물별 검수 기준

### D08: 분류 모델 (`classifier.pkl`)

```
검수 기준:
✅ test Overall Accuracy ≥ 0.75
✅ test Macro F1 ≥ 0.65
✅ 대사복합관리형 Recall ≥ 0.70
✅ 아래 코드로 로드 및 추론 성공:
   import joblib, numpy as np
   clf = joblib.load('models/classifier.pkl')
   scaler = joblib.load('models/scaler.pkl')
   sample = np.array([[23.5, 128, 82, 95, 195, 78, 0, 1]])
   pred = clf.predict(scaler.transform(sample))
   print(pred)  # 예: [1]  → 체중관리형
```

### D11~D15: Jupyter 노트북 5종

```
각 노트북 검수 기준:
✅ 모든 셀 실행 완료 (오류 없음)
✅ 출력 결과 저장 상태
✅ 한글 설명 마크다운 포함
✅ 최소 시각화 3개 이상
✅ 결론 및 인사이트 마지막 셀 작성
```

**노트북별 필수 포함 내용**:

| 노트북 | 필수 포함 내용 |
|--------|-------------|
| 01_EDA | 클래스 분포 차트, 결측률 막대그래프, 주요 변수 히스토그램, 성별·연령대별 분포 |
| 02_preprocessing | 이상치 처리 전/후 박스플롯, 표준화 전/후 비교, 파생변수 분포 |
| 03_clustering | Elbow Curve, Silhouette Score by K, PCA 2D 군집 시각화, 군집별 프로파일 표 |
| 04_classification | CV 결과 (모델별 비교), 변수 중요도 막대그래프, 학습/검증 Accuracy 비교 |
| 05_evaluation | Confusion Matrix (정규화), 클래스별 F1 비교, 오분류 사례 분석, KPI 달성 현황 |

### D18: 최종 보고서 (`final_report.md`)

```
필수 포함 섹션:
1. 프로젝트 요약 (1페이지)
2. 문제 정의 및 접근 방법
3. 데이터 수집 및 전처리 결과
4. 군집화 분석 결과 (유형별 특성)
5. 분류 모델 성능 비교 (4종 실험)
6. 최종 모델 성능 평가
7. 서비스 구현 결과 (스크린샷 포함)
8. 한계점 및 향후 개선 방향
```

### D19: README.md

```
검수 기준 (재현성 테스트):
✅ 처음 보는 사람이 환경 설정 ~ 앱 실행까지 따라할 수 있어야 함
✅ 모든 명령어는 copy-paste 가능
✅ 배포 URL 포함
✅ 앱 스크린샷 최소 2장 포함
✅ 아래 시나리오 재현 가능:
   1. conda env 생성
   2. requirements.txt 설치
   3. 모델 파일 다운로드 (Google Drive 링크)
   4. streamlit run app.py 실행
   5. 브라우저에서 결과 확인
```

### D21: Streamlit 웹앱

```
검수 기준:
✅ 공개 URL에서 접근 가능
✅ 필수 입력 (성별+연령대+키+몸무게)만으로 결과 출력
✅ 혈압·혈당 미입력 시 제한 안내 + 기본 결과 표시
✅ 모바일(375px 기준) 세로 스크롤만으로 이용 가능
✅ 주의문구 모든 결과 화면에 표시
```

---

## 13.3 최종 제출 체크리스트

```bash
# 제출 전 실행 순서

# 1. 테스트 전체 통과 확인
pytest tests/ -v --cov=src

# 2. 코드 품질 검사
flake8 src/ --max-line-length=100

# 3. 노트북 실행 결과 저장 (Kernel → Restart & Run All 후 저장)

# 4. 모델 파일 Google Drive 업로드 및 README 링크 업데이트

# 5. 최종 커밋
git add -A
git commit -m "feat: v1.0.0 최종 포트폴리오 버전"
git tag v1.0.0 -m "건강위험 자가점검 서비스 v1.0.0"
git push origin main --tags

# 6. Streamlit Cloud 재배포 확인
# → main 브랜치에 push 시 자동 재배포

# 7. 제출 확인 체크
echo "✅ GitHub: https://github.com/{계정}/health-self-check"
echo "✅ 웹앱:   https://{앱명}.streamlit.app"
echo "✅ 모델:   {Google Drive 링크}"
```
