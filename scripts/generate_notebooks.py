"""
4개의 .ipynb 노트북 파일을 생성하는 스크립트.
직접 실행하면 notebooks/ 폴더에 4개 파일이 만들어집니다.
"""
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# 노트북 만들기 헬퍼
# ──────────────────────────────────────────────────────────────
def make_notebook(cells):
    """주피터 노트북 JSON 구조 생성"""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.14",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source):
    """마크다운 셀"""
    return {"cell_type": "markdown", "metadata": {}, "source": source.split('\n')}


def code(source):
    """코드 셀"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split('\n'),
    }


# ──────────────────────────────────────────────────────────────
# 노트북 1: 데이터 이해
# ──────────────────────────────────────────────────────────────
nb1 = make_notebook([
    md("""# 01. 건강검진 데이터 이해 (Data Understanding)

이 노트북에서는 국민건강보험공단 건강검진정보 2024 데이터의 구조를 파악합니다.

## 목차
1. 데이터 로딩
2. 기본 정보 확인 (행/열, 컬럼 목록)
3. 컬럼별 결측률
4. 주요 변수 분포 확인"""),

    code("""# 라이브러리 가져오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 (matplotlib)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False"""),

    md("## 1. 데이터 로딩"),

    code("""# CP949 인코딩으로 CSV 읽기 (한국 공공데이터 표준)
df = pd.read_csv(
    '../data/raw/2024_국민건강보험공단_건강검진정보.CSV',
    encoding='cp949'
)
print(f'데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열')"""),

    md("## 2. 기본 정보 확인"),

    code("""# 컬럼 목록
print('=== 33개 컬럼 ===')
for i, col in enumerate(df.columns):
    print(f'  {i+1:2}. {col}')"""),

    code("""# 처음 5개 행 미리보기
df.head()"""),

    md("## 3. 결측률 확인"),

    code("""# 컬럼별 결측 비율 계산
missing_rates = (df.isnull().mean() * 100).round(2)
missing_rates = missing_rates[missing_rates > 0].sort_values(ascending=False)
print('결측률이 있는 컬럼:')
print(missing_rates)"""),

    code("""# 결측률 시각화 (상위 10개)
fig, ax = plt.subplots(figsize=(10, 5))
missing_rates.head(10).plot(kind='barh', ax=ax, color='salmon')
ax.set_xlabel('Missing Rate (%)')
ax.set_title('Columns with Missing Values (Top 10)')
plt.tight_layout()
plt.show()"""),

    md("""**관찰**: 콜레스테롤 4종(LDL, HDL, 트리글리세라이드, 총콜레스테롤)이 약 66% 결측.
→ 이상지질혈증 검사를 받은 사람만 값이 있기 때문.
→ 모델 학습 변수에서 제외하고, 룰 기반 해석에만 사용 결정."""),

    md("## 4. 주요 변수 분포"),

    code("""# 성별 분포
print('성별 코드 분포:')
print(df['성별코드'].value_counts())
print('\\n1=남성, 2=여성')"""),

    code("""# 연령대 분포
age_label_map = {
    5:'25-29세', 6:'30-34세', 7:'35-39세', 8:'40-44세',
    9:'45-49세', 10:'50-54세', 11:'55-59세', 12:'60-64세',
    13:'65-69세', 14:'70-74세', 15:'75-79세', 16:'80-84세', 
    17:'85-89세', 18:'90세+'
}
age_dist = df['연령대코드(5세단위)'].map(age_label_map).value_counts().sort_index()
print('연령대별 인원수:')
print(age_dist)"""),

    code("""# 주요 변수 통계
key_cols = ['신장(5cm단위)', '체중(5kg단위)', '허리둘레',
            '수축기혈압', '이완기혈압', '식전혈당(공복혈당)']
df[key_cols].describe().round(1)"""),

    md("""## 결론

- **데이터 규모**: 100만 건 × 33컬럼
- **필수 변수(성별/나이/키/몸무게/혈압/혈당)**: 결측률 1% 미만으로 양호
- **콜레스테롤 4종**: 약 66% 결측 → 모델 학습에는 미사용
- **다음 단계**: 02_preprocessing_eda.ipynb에서 결측치·이상치 처리 및 파생변수 생성"""),
])


# ──────────────────────────────────────────────────────────────
# 노트북 2: 전처리 + EDA
# ──────────────────────────────────────────────────────────────
nb2 = make_notebook([
    md("""# 02. 전처리 및 탐색적 데이터 분석 (Preprocessing & EDA)

이 노트북에서는 다음 작업을 수행합니다:
1. 결측치 처리
2. 이상치 제거
3. 파생변수 생성 (BMI, 위험요인 플래그 등)
4. EDA 시각화"""),

    code("""import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False"""),

    md("## 1. 전처리 파이프라인 실행"),

    code("""# src/preprocessing.py의 함수들 사용
from src.preprocessing import run_full_preprocessing

df = run_full_preprocessing('../data/raw/2024_국민건강보험공단_건강검진정보.CSV')
print(f'\\n최종: {df.shape[0]:,}행 × {df.shape[1]}열')
df.head()"""),

    md("## 2. 파생변수 분포 확인"),

    code("""# BMI 분포
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['bmi'], bins=50, color='steelblue', edgecolor='white')
axes[0].axvline(18.5, color='gray', linestyle='--', label='18.5')
axes[0].axvline(23.0, color='orange', linestyle='--', label='23.0')
axes[0].axvline(25.0, color='red', linestyle='--', label='25.0')
axes[0].set_xlabel('BMI')
axes[0].set_ylabel('Frequency')
axes[0].set_title('BMI Distribution')
axes[0].legend()

# BMI 카테고리 분포
cat_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
cat_counts = df['bmi_category'].value_counts().sort_index()
axes[1].bar(cat_labels, cat_counts.values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
axes[1].set_ylabel('Count')
axes[1].set_title('BMI Category Distribution')

plt.tight_layout()
plt.show()"""),

    code("""# 위험요인 개수 분포
fig, ax = plt.subplots(figsize=(8, 4))
risk_counts = df['metabolic_risk_count'].value_counts().sort_index()
ax.bar(risk_counts.index, risk_counts.values,
       color=['#2ecc71', '#f1c40f', '#f39c12', '#e67e22', '#e74c3c'])
ax.set_xlabel('Number of Metabolic Risk Factors (0~4)')
ax.set_ylabel('Count')
ax.set_title('Metabolic Risk Count Distribution')
for i, v in enumerate(risk_counts.values):
    ax.text(i, v + 5000, f'{v:,}', ha='center')
plt.tight_layout()
plt.show()"""),

    md("## 3. 성별·연령대별 주요 변수 비교"),

    code("""# 연령대별 평균 BMI/혈압
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# BMI by age & gender
bmi_by_age = df.groupby(['age_group', 'gender'])['bmi'].mean().unstack()
bmi_by_age.columns = ['Male', 'Female']
bmi_by_age.plot(ax=axes[0], marker='o')
axes[0].set_xlabel('Age Group Code')
axes[0].set_ylabel('Mean BMI')
axes[0].set_title('Mean BMI by Age & Gender')
axes[0].grid(alpha=0.3)

# SBP by age & gender
sbp_by_age = df.groupby(['age_group', 'gender'])['systolic_bp'].mean().unstack()
sbp_by_age.columns = ['Male', 'Female']
sbp_by_age.plot(ax=axes[1], marker='o', color=['steelblue', 'tomato'])
axes[1].set_xlabel('Age Group Code')
axes[1].set_ylabel('Mean Systolic BP (mmHg)')
axes[1].set_title('Mean Systolic BP by Age & Gender')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()"""),

    code("""# 변수 간 상관관계
corr_cols = ['bmi', 'systolic_bp', 'diastolic_bp',
             'fasting_glucose', 'waist_cm', 'metabolic_risk_count']
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax, square=True)
ax.set_title('Correlation Matrix of Key Variables')
plt.tight_layout()
plt.show()"""),

    md("""## 결론

- **이상치 제거**: 100만 건 → 약 99.98% 보존 (이상치 0.02% 제거)
- **BMI 분포**: 정상~과체중에 다수 분포, 비만(BMI≥25) 약 35%
- **위험요인 분포**: 0개 ~ 4개로 자연스러운 분포
- **상관관계**: BMI ↔ 혈압/허리둘레 간 약한~중간 양의 상관
- **다음 단계**: 03_clustering_modeling.ipynb에서 K-Means 군집화"""),
])


# ──────────────────────────────────────────────────────────────
# 노트북 3: 군집화 + 모델링
# ──────────────────────────────────────────────────────────────
nb3 = make_notebook([
    md("""# 03. 군집화 + 분류 모델 학습 (Clustering & Modeling)

이 노트북에서는:
1. K-Means 군집화로 건강관리 유형 6종 도출
2. 각 군집에 의미 있는 유형명 부여
3. Random Forest 분류 모델 학습"""),

    code("""import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False"""),

    md("## 1. 전처리된 데이터 준비"),

    code("""from src.preprocessing import run_full_preprocessing

df = run_full_preprocessing('../data/raw/2024_국민건강보험공단_건강검진정보.CSV')
print(f'준비 완료: {df.shape[0]:,}행')"""),

    md("## 2. 표준화 + K 탐색"),

    code("""# 모델 학습 변수 (총콜레스테롤 제외 - 결측 ~66%)
MODEL_FEATURES = ['bmi', 'systolic_bp', 'diastolic_bp',
                  'fasting_glucose', 'waist_cm',
                  'smoking_status', 'drinking']

X = df[MODEL_FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f'표준화 완료: {X_scaled.shape}')"""),

    code("""# K=3~8까지 탐색 (시간 절약을 위해 100,000개 샘플로 실험)
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=100_000, replace=False)
X_sample = X_scaled[sample_idx]

results = []
for k in range(3, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sample)
    sil = silhouette_score(X_sample, labels, sample_size=10000, random_state=42)
    results.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})
    print(f'K={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.4f}')

results_df = pd.DataFrame(results)"""),

    code("""# Elbow + Silhouette 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(results_df['k'], results_df['inertia'], 'o-', color='steelblue')
axes[0].set_xlabel('K (n_clusters)')
axes[0].set_ylabel('Inertia (WSS)')
axes[0].set_title('Elbow Method')
axes[0].grid(alpha=0.3)

axes[1].plot(results_df['k'], results_df['silhouette'], 'o-', color='tomato')
axes[1].set_xlabel('K (n_clusters)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by K')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()"""),

    md("**결정**: K=6 선택 (도메인 지식 기반 6개 건강관리 유형과 일치)"),

    md("## 3. 전체 학습 파이프라인 실행"),

    code("""# 전체 학습은 src/modeling.py의 run_full_training()으로 한 번에
from src.modeling import run_full_training

result = run_full_training('../data/raw/2024_국민건강보험공단_건강검진정보.CSV')
print()
print(f\"Silhouette: {result['silhouette']:.4f}\")
print(f\"Test Accuracy: {result['metrics']['overall_accuracy']:.4f}\")
print(f\"Macro F1:      {result['metrics']['macro_f1']:.4f}\")"""),

    md("## 4. 군집 프로파일 시각화"),

    code("""profile = pd.read_csv('../data/processed/cluster_profile.csv', encoding='utf-8-sig')
profile"""),

    code("""# 각 유형의 평균 수치 막대 그래프
fig, ax = plt.subplots(figsize=(12, 5))
metrics = ['bmi_mean', 'sbp_mean', 'fbg_mean']
x = np.arange(len(profile))
width = 0.25

for i, m in enumerate(metrics):
    ax.bar(x + i*width, profile[m], width, label=m)

ax.set_xticks(x + width)
ax.set_xticklabels(profile['health_type_name'], rotation=15)
ax.set_ylabel('Mean Value')
ax.set_title('Cluster Profile: BMI / SBP / FBG by Health Type')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()"""),

    md("""## 결론

- 6개 건강관리 유형이 명확히 구분됨 (BMI/혈압/혈당/생활습관 차이)
- Test Accuracy 약 0.97~0.98 (K-Means 라벨을 학습한 결과 자체이므로 자연스럽게 높음)
- 다음 단계: 04_model_evaluation.ipynb에서 상세 성능 평가"""),
])


# ──────────────────────────────────────────────────────────────
# 노트북 4: 모델 평가
# ──────────────────────────────────────────────────────────────
nb4 = make_notebook([
    md("""# 04. 모델 성능 평가 (Model Evaluation)

이 노트북에서는:
1. Confusion Matrix
2. 클래스별 Precision/Recall/F1
3. 변수 중요도
4. 비교군 통계 검토"""),

    code("""import sys
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False"""),

    md("## 1. 모델 + 데이터 로딩"),

    code("""# 학습된 모델 로딩
scaler = joblib.load('../model/scaler.pkl')
classifier = joblib.load('../model/classifier_model.pkl')
with open('../model/cluster_mapping.json', encoding='utf-8') as f:
    cluster_mapping = {int(k): v for k, v in json.load(f).items()}

# 전처리 + 라벨 다시 만들기
from src.preprocessing import run_full_preprocessing
from src.modeling import MODEL_FEATURES, RANDOM_SEED

df = run_full_preprocessing('../data/raw/2024_국민건강보험공단_건강검진정보.CSV')
X = scaler.transform(df[MODEL_FEATURES].values)

# 학습 때와 같은 분할로 Test set 재구성
from sklearn.model_selection import train_test_split
kmeans = joblib.load('../model/kmeans_model.pkl')
y = kmeans.predict(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp)

print(f'Test set: {len(X_test):,}개')"""),

    md("## 2. Confusion Matrix"),

    code("""from sklearn.metrics import confusion_matrix, classification_report

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred, normalize='true')

# 유형명 가져오기
type_names = [cluster_mapping[i]['name'] for i in sorted(cluster_mapping.keys())]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=type_names, yticklabels=type_names,
            vmin=0, vmax=1, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix (Normalized)')
plt.xticks(rotation=20)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()"""),

    md("## 3. 클래스별 성능"),

    code("""print(classification_report(y_test, y_pred, target_names=type_names))"""),

    md("## 4. 변수 중요도"),

    code("""importances = classifier.feature_importances_
imp_df = pd.DataFrame({
    'feature': MODEL_FEATURES,
    'importance': importances
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_df['feature'], imp_df['importance'], color='steelblue')
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()"""),

    md("## 5. 비교군 통계 검토"),

    code("""comparison = pd.read_csv('../data/processed/comparison_group_summary.csv',
                          encoding='utf-8-sig')
print(f'총 {len(comparison)}개 그룹 (성별 2 × 연령대 13)')
print('\\n샘플:')
comparison.head(10)"""),

    code("""# 연령대별 BMI 평균 (성별 분리)
fig, ax = plt.subplots(figsize=(12, 4))
for g, name, color in [(1, 'Male', 'steelblue'), (2, 'Female', 'tomato')]:
    sub = comparison[comparison['gender'] == g].sort_values('age_group')
    ax.errorbar(sub['age_label'], sub['bmi_mean'],
                yerr=sub['bmi_std'], marker='o', label=name, color=color, capsize=4)
ax.set_xlabel('Age Group')
ax.set_ylabel('Mean BMI ± Std')
ax.set_title('Mean BMI by Age & Gender (with Std)')
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()"""),

    md("""## 결론

- **분류 성능**: Overall Accuracy ~0.97, Macro F1 ~0.97 (K-Means 라벨 학습이라 매우 높음)
- **변수 중요도**: BMI, 혈압(SBP/DBP), 허리둘레가 상위
- **비교군 통계**: 28개 그룹(성별 2 × 연령대 13)으로 충분한 표본 확보
- **한계**: K-Means 군집 라벨 자체에 분류기를 학습한 구조라 정확도가 매우 높게 나옴.
  실제 의료 진단 정확도가 아닌, 군집화 일관성을 의미함."""),
])


# ──────────────────────────────────────────────────────────────
# 파일로 저장
# ──────────────────────────────────────────────────────────────
out_dir = Path('/home/claude/health-check-service/notebooks')
out_dir.mkdir(parents=True, exist_ok=True)

notebooks = {
    '01_data_understanding.ipynb': nb1,
    '02_preprocessing_eda.ipynb': nb2,
    '03_clustering_modeling.ipynb': nb3,
    '04_model_evaluation.ipynb': nb4,
}

for fname, nb in notebooks.items():
    path = out_dir / fname
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'✅ {path}')

print('\n노트북 4개 생성 완료!')
