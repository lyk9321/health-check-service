# 05. 기술 스택 (Tech Stack)

> **문서 버전**: v1.2.0 | **최종 수정**: 2026-05-07
> **원칙**: 모든 라이브러리는 오픈소스, 모든 버전은 pip install 가능한 정확한 버전으로 고정

---

## 5.1 핵심 라이브러리 (정확한 버전 고정)

```
# requirements.txt (전체 내용 — 이 파일 그대로 사용)

# ─── 웹 서비스 ────────────────────────────────────────────
streamlit==1.32.2

# ─── 데이터 처리 ──────────────────────────────────────────
pandas==2.2.1
numpy==1.26.4

# ─── 머신러닝 ─────────────────────────────────────────────
scikit-learn==1.4.1.post1

# ─── 시각화 ───────────────────────────────────────────────
matplotlib==3.8.3
seaborn==0.13.2
plotly==5.20.0

# ─── 지리 데이터 ──────────────────────────────────────────
folium==0.16.0
streamlit-folium==0.18.0

# ─── 모델 저장 ────────────────────────────────────────────
joblib==1.3.2

# ─── 유틸리티 ─────────────────────────────────────────────
pyyaml==6.0.1
python-dotenv==1.0.1
tqdm==4.66.2
loguru==0.7.2

# ─── 테스트 ───────────────────────────────────────────────
pytest==8.1.1
pytest-cov==5.0.0
```

---

## 5.2 개발 환경 명세

### 5.2.1 권장 하드웨어

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| CPU | 2코어 / 2.0GHz | 4코어 / 3.0GHz 이상 |
| RAM | 8GB | 16GB |
| 저장공간 | 20GB | 50GB |
| GPU | 불필요 | 불필요 (scikit-learn CPU 연산) |

> **GPU 불필요**: scikit-learn 기반 K-Means + Random Forest는 CPU로 100만 건 데이터 처리 충분

### 5.2.2 소프트웨어 환경

```yaml
# environment.yaml
name: health_check
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10.14
  - pip=24.0
  - pip:
    - -r requirements.txt
```

### 5.2.3 개발 도구

| 도구 | 용도 |
|------|------|
| **VS Code** | 주 개발 IDE (`.ipynb` 노트북 포함) |
| VS Code 확장: Python | Python 인터프리터 연결 |
| VS Code 확장: Jupyter | `.ipynb` 노트북 실행 |
| VS Code 확장: Pylance | 타입 체킹, 자동완성 |

> **노트북 작성 환경**: 모든 `.ipynb` 파일은 **VS Code의 Jupyter 확장**으로 작성·실행.
> Google Colab은 사용하지 않음.

---

## 5.3 환경 설정 Step-by-Step

### Step 1: conda 환경 생성

```bash
conda create -n health_check python=3.10.14 -y
conda activate health_check
```

### Step 2: 패키지 설치

```bash
pip install -r requirements.txt
```

### Step 3: VS Code 설정

```bash
# VS Code에서 Python 인터프리터 선택
# Ctrl+Shift+P → "Python: Select Interpreter" → health_check 환경 선택

# VS Code에서 Jupyter 커널 선택
# 노트북 우상단 → "Select Kernel" → health_check 환경 선택
```

**VS Code 추천 설정** (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "${env:CONDA_PREFIX}/bin/python",
  "jupyter.kernels.filter": [
    {
      "path": "${env:CONDA_PREFIX}/bin/python",
      "type": "pythonEnvironment"
    }
  ],
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

### Step 4: 설치 검증

```bash
python scripts/verify_env.py
```

**`scripts/verify_env.py` 전체 내용**:

```python
#!/usr/bin/env python3
"""환경 설정 검증 스크립트"""
import sys
import importlib

REQUIRED = {
    'streamlit': '1.32.2',
    'pandas': '2.2.1',
    'numpy': '1.26.4',
    'sklearn': None,
    'matplotlib': '3.8.3',
    'plotly': '5.20.0',
    'joblib': '1.3.2',
    'yaml': None,
}

all_ok = True
for pkg, expected_ver in REQUIRED.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, '__version__', 'N/A')
        status = '✅' if (expected_ver is None or ver == expected_ver) else '⚠️'
        print(f"{status} {pkg}: {ver}")
        if expected_ver and ver != expected_ver:
            all_ok = False
    except ImportError:
        print(f"❌ {pkg}: NOT INSTALLED")
        all_ok = False

import streamlit as st
print(f"\n✅ Streamlit: {st.__version__}")
print(f"\n{'✅ 환경 설정 완료' if all_ok else '⚠️  버전 불일치 항목 확인 필요'}")
sys.exit(0 if all_ok else 1)
```

### Step 5: 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열어 필요한 값 입력
```

**.env.example 전체 내용**:

```bash
# 프로젝트 경로 (절대 경로)
PROJECT_ROOT=/path/to/health-check-service

# 재현성 시드 (변경 금지)
RANDOM_SEED=42

# 데이터 경로
RAW_DATA_PATH=data/raw/2024_국민건강보험공단_건강검진정보.CSV
PROCESSED_DATA_PATH=data/processed/

# 지역보건의료기관 데이터 경로
LOCAL_HEALTH_DATA_PATH=data/external/전국_지역보건의료기관_현황_20221231.csv
HEALTH_SUPPORT_CENTER_PATH=data/external/전국건강생활지원센터표준데이터.csv

# 개발 모드 (DEBUG=true 시 상세 로그)
DEBUG=false
```

### Step 6: Streamlit 앱 실행 확인

```bash
streamlit run app.py
# 브라우저에서 http://localhost:8501 접속 확인
```

---

## 5.4 협업 도구 설정

### 5.4.1 GitHub 브랜치 전략

```
main            ← 최종 배포 버전
  └── develop   ← 개발 통합 브랜치
        ├── feature/data-preprocessing
        ├── feature/clustering
        ├── feature/classification
        ├── feature/streamlit-ui
        └── feature/evaluation
```

### 5.4.2 커밋 메시지 컨벤션

```
<type>(<scope>): <subject>

type:
  feat     - 새 기능
  fix      - 버그 수정
  data     - 데이터 관련 작업
  exp      - 실험 실행
  refactor - 리팩토링
  docs     - 문서 수정
  test     - 테스트 추가/수정
  deploy   - 배포 관련

예시:
  feat(clustering): K-Means 군집화 및 유형명 부여 완료
  data(preprocessing): 건강검진 데이터 결측치 처리 완료
  feat(ui): 비교군 비교 카드 컴포넌트 구현
  deploy(streamlit): Streamlit Cloud 배포 완료
```

### 5.4.3 .gitignore 필수 항목

```gitignore
# 데이터 (용량 큰 파일 — 원본 데이터 GitHub 업로드 금지)
data/raw/

# 모델 파일
model/*.pkl
model/*.joblib

# 환경 설정
.env

# Python
__pycache__/
*.egg-info/
.pytest_cache/
*.pyc

# Jupyter 체크포인트
.ipynb_checkpoints/

# VS Code
.vscode/settings.json

# OS
.DS_Store
Thumbs.db
```

---

## 5.5 Streamlit 배포 설정

### 5.5.1 Streamlit Community Cloud 배포 절차

```bash
# 1. GitHub 리포지토리 공개(Public)로 설정
# 2. https://streamlit.io/cloud 접속
# 3. New app → GitHub 리포 연결
# 4. Main file path: app.py
# 5. 환경 변수는 Streamlit Secrets에 등록
# 6. Deploy 클릭
```

### 5.5.2 `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#1A56A4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F4FA"
textColor = "#212121"
font = "sans serif"

[server]
maxUploadSize = 5
enableCORS = false
headless = true

[browser]
gatherUsageStats = false
```

> **배포 주의사항**: 원본 건강검진 데이터(100만 건 CSV, ~50MB)는 배포 파일에 포함하지 않음.  
> 배포 시에는 **전처리·집계된 통계 데이터 + 학습 완료 모델 파일(.pkl)**만 포함.

---

## 5.6 외부 서비스 설정

| 서비스 | 용도 | 비용 | 비고 |
|--------|------|------|------|
| GitHub | 코드 버전 관리, 배포 연동 | 무료 | Public 리포 |
| Streamlit Community Cloud | 웹앱 배포 | 무료 | GitHub 계정 연동 |
| 공공데이터포털 (data.go.kr) | 지역보건의료기관 데이터 | 무료 | CSV 직접 다운로드 |
| 국민건강보험공단 건강검진 데이터 | 핵심 분석 데이터 | 무료 | 공공데이터포털에서 직접 다운로드 |
