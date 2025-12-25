# 🧬 Hybrid DNA Variant Distance Regression
Reference vs Variant DNA 임베딩 거리 학습을 통한 변이 병원성 모델링

<br>

이 저장소는 **DNA 변이 병원성(Pathogenicity) 학습을 위한 코드만** 포함합니다.  
유전체 데이터, 사전학습 모델 가중치, 파인튜닝 체크포인트 및 실험 로그는 포함하지 않습니다.

- 입력: Reference DNA 서열 / Variant DNA 서열  
- 출력: L2 정규화 임베딩(2048-d) 및 Cosine Distance ∈ [0, 2]

<br>

---

## 🚦 목차

1. [🔬 프로젝트 소개](#intro)  
   1-1. [👥 팀원 소개](#team)  
2. [🧩 문제 정의](#problem)  
3. [🧠 방법론 요약](#method)  
4. [🏗️ 모델 구조](#model)  
5. [📐 학습 손실 함수](#loss)  
6. [📊 평가 지표](#metrics)  
7. [📁 저장소 구조](#structure)  
8. [⚙️ 설치 및 빠른 시작](#quickstart)  
   8-1. [🧰 환경 구성](#install)  
   8-2. [🗂️ 데이터 준비](#data)  
   8-3. [🧹 전처리](#preprocess)  
   8-4. [🏋️ 학습](#train)  
   8-5. [🔎 추론](#inference)  
9. [🧪 로깅 및 재현성](#repro)  
10. [🚫 데이터/모델 재배포 안내](#policy)  
11. [🛠️ FAQ / 트러블슈팅](#faq)  
12. [📬 문의](#contact)  

<br><br><br>

---

<a id="intro"></a>
# 1. 🔬 프로젝트 소개

### 프로젝트가 하는 일

본 프로젝트는 **Reference DNA 서열과 Variant DNA 서열**을 입력으로 받아 각각 임베딩으로 변환하고,  
**두 임베딩의 거리(distance)** 가 변이 병원성 신호와 일관되도록 학습합니다.

즉, 변이 병원성을 “분류(label)”로만 맞히는 것이 아니라, 임베딩 공간에서:

- Pathogenic 샘플끼리는 더 가깝게
- Benign 샘플과는 더 멀게
- 병원성 점수(또는 연속 타깃)와 거리가 정합되게

정렬되도록 학습합니다.

### 이 접근이 유용한 경우

- 변이 효과를 단일 점수 예측보다 **거리/유사도 기반 표현**으로 다루고 싶을 때
- downstream(검색, 클러스터링, 유사 변이 탐색, metric learning)까지 고려한 임베딩이 필요할 때
- 전역 문맥(유전체 언어모델)과 국소 변이 신호(변이 주변) 모두를 반영하고 싶을 때

### 전체 파이프라인 개요

```text
ClinVar (VCF) + Reference Genome (hg38 FASTA)
            │
            ▼
   src/data_preprocess.py
            │  (reference/variant pair + label/score 생성)
            ▼
        src/train.py
            │  (LoRA fine-tuning + multi-loss distance learning)
            ▼
     fine-tuned checkpoint
            │
            ▼
      src/inference.py
            │  (embedding export / distance compute)
            ▼
  Embeddings (L2-normalized, 2048-d) + Cosine Distance
```

<br>

<a id="team"></a>
## 1-1. 👥 팀원 소개

<table>
  <tr>
    <td align="center" width="220">
      <img src="https://placehold.co/200x200?text=%EC%95%88%EC%A4%80%EC%8B%9D" width="200" height="200" alt="안준식"><br>
      안준식<br><sub>Lead Researcher</sub>
    </td>
    <td align="center" width="220">
      <img src="https://placehold.co/200x200?text=%EC%9C%A4%EC%97%AC%ED%97%8C" width="200" height="200" alt="윤여헌"><br>
      윤여헌<br><sub>Bio-ML Engineer</sub>
    </td>
    <td align="center" width="220">
      <img src="https://avatars.githubusercontent.com/u/150754838?v=4" width="200" height="200" alt="장영웅"><br>
      장영웅<br><sub>Read Model Engineer</sub>
    </td>
    <td align="center" width="220">
      <img src="https://placehold.co/200x200?text=%EC%9D%B4%EC%A0%95%EC%9B%90" width="200" height="200" alt="이정원"><br>
      이정원<br><sub>Data Engineer</sub>
    </td>
    <td align="center" width="220">
      <img src="https://placehold.co/200x200?text=%EC%A1%B0%EB%AF%BC%EC%84%B1" width="200" height="200" alt="조민성"><br>
      조민성<br><sub>Data Researcher</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="220">
      연구 총괄, 문제 정의, 전체 모델 아키텍처 방향 설계
    </td>
    <td align="center" width="220">
      의생명 서열 데이터 기반<br>학습 파이프라인 구현,<br>변이·서열 특성 성능 분석
    </td>
    <td align="center" width="220">
      학습 파이프라인 최적화,<br>지표 정합성 개선, 파인튜닝<br>기반 성능 향상·안정화
    </td>
    <td align="center" width="220">
      데이터 전처리, 데이터셋 생성 파이프라인 구축
    </td>
    <td align="center" width="220">
      ClinVar 기반 변이 분석, 병원성 특성 연구
    </td>
  </tr>
</table>


---

<a id="problem"></a>
# 2. 🧩 문제 정의

- 입력: Reference DNA 서열 / Variant DNA 서열 (서열 쌍)
- 출력: L2 정규화 임베딩 벡터(2048-d)
- 거리: L2 정규화된 두 임베딩의 **cosine distance** 사용 (범위 [0, 2])

학습 목적은 “정답 라벨을 직접 맞히기”가 아니라,
**임베딩 공간의 거리 구조**가 병원성 신호와 정렬되도록 만드는 것입니다.

---

<a id="method"></a>
# 3. 🧠 방법론 요약

핵심은 **전역(Genomic LM) + 국소(Local CNN)** 하이브리드 임베딩입니다.

- Global(전역): Genomic Language Model로 서열 전역 문맥을 인코딩
- Local(국소): 변이 위치 주변(one-hot) 입력을 1D CNN으로 인코딩해 변이 민감도 강화
- Fusion: Global + Local 결합 → Linear Projection → L2 정규화
- Training: 다중 손실을 가중합으로 결합해 임베딩 구조를 안정화

---

<a id="model"></a>
# 4. 🏗️ 모델 구조

### Backbone (Global Encoder)

- Hugging Face 공개 모델  
  `LongSafari/hyenadna-large-1m-seqlen-hf`
- 파인튜닝 방식  
  LoRA (Parameter-Efficient Fine-Tuning)

### Local Encoder

- 변이 위치 기준 one-hot DNA 서열
- 1D CNN 기반 국소 특징 강화

### Final Embedding

- Global + Local 결합
- Linear projection 후 L2 normalization
- Embedding dimension: 2048

### 수학적 정의 (Notation)

- Reference/Variant 서열: $x_{ref}, x_{var} \in \{A,C,G,T,N\}^{L}$, $L=1024$
- 변이 상대 위치: $m$ (컬럼 `mut_index`)
- 라벨/점수: $y\in\{0,1\}$, $s\in\{0.0,0.2,0.8,1.0\}$
- 목표 거리(감독 신호): $d^{\ast}=2s\in[0,2]$

### Global Encoder (HyenaDNA backbone)

백본의 마지막 hidden state를 $H\in\mathbb{R}^{L\times d}$라 하면,

- mean pooling: $\bar{h}=\frac{1}{L}\sum_{i=1}^{L}H_i\in\mathbb{R}^{d}$
- projection: $g=W_g\bar{h}+b_g\in\mathbb{R}^{1024}$

### Local Encoder (Mutation-centered window + 1D CNN)

변이 위치 $m$을 기준으로 local window 길이 $w=64$를 추출해 one-hot으로 변환합니다.

- one-hot: $X_{local}\in\{0,1\}^{4\times w}$
- CNN 출력: $\ell=\mathrm{CNN}(X_{local})\in\mathbb{R}^{1024}$

### Fusion + Final Embedding

- concat: $c=[g;\ell]\in\mathbb{R}^{2048}$
- final projection: $\tilde{z}=W_f c+b_f\in\mathbb{R}^{2048}$
- L2 normalize: $z=\frac{\tilde{z}}{\lVert\tilde{z}\rVert_2}$

### Distance (Cosine Distance)

- cosine similarity: $\cos(z_a,z_b)=\frac{z_a^\top z_b}{\lVert z_a\rVert_2\,\lVert z_b\rVert_2}$
- cosine distance (final): $\hat{d}=1-\cos(z_{ref},z_{var})\in[0,2]$

Mutation Focus Loss용 local distance는 $\ell$을 정규화 후 동일하게 계산합니다.

- $`\hat{d}_{\mathrm{local}} = 1 - \cos\big(\mathrm{norm}(\ell_{\mathrm{ref}}),\ \mathrm{norm}(\ell_{\mathrm{var}})\big)`$


---

<a id="loss"></a>
# 5. 📐 학습 손실 함수 (Loss Functions)

본 모델은 아래 손실들의 **가중합(weighted sum)** 으로 학습됩니다.

- Distance Regression Loss (MSE)  
  예측 거리와 병원성 점수(또는 타깃 거리) 정합
- Mutation Focus Loss  
  변이 주변 국소 특징 민감도 강화
- Mean Margin Gap Loss  
  Pathogenic/Benign 평균 거리 차이 확보
- Triplet Loss  
  임베딩 구조 분리 강화
- Pairwise Margin Loss  
  클래스 간 마진 유지 강화
- Supervised Contrastive Loss  
  임베딩만 사용해도 클래스 분리가 유지되도록 정렬

손실의 상세 정의/가중치는 `Train.md` 및 `src/train.py` 설정(CONFIG 등)을 기준으로 합니다.

### 손실 함수 수학적 정의

모든 $\hat{d}$는 (Reference, Variant) 최종 임베딩의 cosine distance입니다.

- 최종 거리: $\hat{d}=1-\cos(z_{ref},z_{var})$
local 거리: $`\hat{d}_{\mathrm{local}} = 1 - \cos\big(\mathrm{norm}(\ell_{\mathrm{ref}}),\ \mathrm{norm}(\ell_{\mathrm{var}})\big)`$
- 타깃 거리: $d^{\ast}=2s$  (코드에서 `target_dist = score * 2.0`)

배치에서 Pathogenic 집합 $P=\{i\mid y_i=1\}$, Benign 집합 $B=\{i\mid y_i=0\}$로 두면,

1) Distance Regression Loss (MSE)

```math
\mathcal{L}_{reg}=\frac{1}{N}\sum_{i=1}^{N}(\hat{d}_i-d^{\ast}_i)^2
```
2) Mutation Focus Loss (Local CNN MSE)

```math
\mathcal{L}_{focus}=\frac{1}{N}\sum_{i=1}^{N}(\hat{d}_{local,i}-d^{\ast}_i)^2
```
3) Mean Margin Gap Loss (Path vs Benign 평균 거리 차이)

```math
\mu_P=\frac{1}{\lvert P\rvert}\sum_{i\in P}\hat{d}_i,\quad
\mu_B=\frac{1}{\lvert B\rvert}\sum_{j\in B}\hat{d}_j,\quad
\mathcal{L}_{mean}=\max(0, m_{mean}-(\mu_P-\mu_B))
```
여기서 $m_{mean}$은 `CONFIG["margin_value"]` 입니다.

4) Batch-level Triplet (Path intra vs Path–Benign)
Pathogenic anchor $i\in P$에 대해,

```math
\text{pos\_mean}_i=\frac{1}{\lvert P\rvert-1}\sum_{j\in P, j\neq i}(1-\cos(z_i,z_j)),\quad
\text{neg\_mean}_i=\frac{1}{\lvert B\rvert}\sum_{k\in B}(1-\cos(z_i,z_k))
```

```math
\mathcal{L}_{triplet}=\frac{1}{\lvert P\rvert}\sum_{i\in P}\max(0,\text{pos\_mean}_i-\text{neg\_mean}_i+m_{triplet})
```
여기서 $m_{triplet}$은 `CONFIG["triplet_margin"]` 입니다.

5) Pairwise Margin Loss (Path–Benign 쌍별 마진)

```math
\mathcal{L}_{pair}=\frac{1}{\lvert P\rvert\,\lvert B\rvert}\sum_{i\in P}\sum_{j\in B}\max(0, m_{pair}-(\hat{d}_i-\hat{d}_j))
```
여기서 $m_{pair}$는 `CONFIG["pair_margin_value"]` 입니다.

6) Supervised Contrastive Loss
정규화 임베딩 $z_i$에 대해, $A(i)=\{a\neq i\}$, $P(i)=\{p\neq i\mid y_p=y_i\}$.

```math
\mathcal{L}_{supcon}
=
-\frac{1}{|\{i:|P(i)|>0\}|}
\sum_{i:|P(i)|>0}
\frac{1}{|P(i)|}
\sum_{p\in P(i)}
\log
\frac{\exp(z_i^\top z_p/\tau)}{\sum_{a\in A(i)}\exp(z_i^\top z_a/\tau)}
```
여기서 $\tau$는 `CONFIG["contrast_temperature"]` 입니다.

7) Total loss (가중합)

```math
\mathcal{L}
=
\text{loss\_scale}\cdot(
w_{reg}\mathcal{L}_{reg}
+w_{focus}\mathcal{L}_{focus}
+w_{mean}\mathcal{L}_{mean}
+w_{triplet}\mathcal{L}_{triplet}
+w_{pair}\mathcal{L}_{pair}
+w_{supcon}\mathcal{L}_{supcon}
)
```
각 가중치는 코드의 `CONFIG["w_*"]` 항목을 따릅니다.


---

<a id="metrics"></a>
# 6. 📊 평가 지표 (Metrics)

학습 및 평가에서 다음 지표를 계산합니다.

- CD (Cosine Distance Mean)
- CDD (Class Distance Difference)
- PCC (Pearson Correlation Coefficient)
- 정규화 지표(0~1 스케일)
- Final Score: Normalized CD, CDD, PCC의 평균

### 지표 수학적 정의

거리 배열 $`D=\{\hat{d}_i\}_{i=1}^{N}`$에 대해,

- CD (Cosine Distance Mean)

```math
\mathrm{CD}=\frac{1}{N}\sum_{i=1}^{N}\hat{d}_i
```
- CDD (Class Distance Difference)

```math
\mu_P=\frac{1}{\lvert P\rvert}\sum_{i\in P}\hat{d}_i,\quad
\mu_B=\frac{1}{\lvert B\rvert}\sum_{j\in B}\hat{d}_j,\quad
\mathrm{CDD}=\frac{\mu_P-\mu_B}{2}
```
(코드 기준으로 $\mu_P-\mu_B$를 2로 나눠 스케일을 맞춥니다)

- PCC (Pearson Correlation)

```math
\mathrm{PCC}=\rho(y,\hat{d})
```
- Normalization (0~1 clamp)

```math
\mathrm{Normal\_CD}=\mathrm{clip}(\mathrm{CD}/2,0,1)
```

```math
\mathrm{Normal\_CDD}=\mathrm{clip}((\mathrm{CDD}+1)/2,0,1)
```

```math
\mathrm{Normal\_PCC}=\mathrm{clip}((\mathrm{PCC}+1)/2,0,1)
```
- Final Score

```math
\mathrm{FinalScore}=\frac{\mathrm{Normal\_CD}+\mathrm{Normal\_CDD}+\mathrm{Normal\_PCC}}{3}
```
---

<a id="structure"></a>
# 7. 📁 저장소 구조

```text
.
├── src/
│   ├── train.py               # 모델 학습 코드
│   ├── data_preprocess.py     # 데이터셋 생성 코드 (ClinVar 기반)
│   └── inference.py           # 추론 및 임베딩 생성 코드
├── requirements.txt
├── README.md
├── Data_Preprocessing.md
├── Train.md
├── Inference.md
└── .gitignore
```

---

<a id="quickstart"></a>
# 8. ⚙️ 설치 및 빠른 시작

참고
- 본 저장소의 `src/*.py`는 argparse 기반 CLI 인자를 제공하지 않습니다.
- 아래의 "1) CLI 인자를 지원하는 경우" 예시는 문서 템플릿이며, 실제 실행은 "2) CLI가 없다면" 방식으로 진행하세요.



<a id="install"></a>
## 8-1. 🧰 환경 구성 (Conda 권장)

```bash
conda create -n medicalAI python=3.10
conda activate medicalAI
pip install -r requirements.txt
```

참고
- CUDA/cuDNN/드라이버는 사용자 환경에 맞게 별도 설치가 필요합니다.
- CPU에서도 실행은 가능하지만 학습 속도는 크게 저하될 수 있습니다.
- 긴 시퀀스/대형 백본은 메모리 요구량이 큽니다. 먼저 작은 설정으로 파이프라인을 검증하세요.

<br>

<a id="data"></a>
## 8-2. 🗂️ 데이터 준비

이 저장소는 데이터 파일을 포함하지 않습니다. 사용자가 공식 출처에서 직접 획득해야 합니다.

- Reference Genome: UCSC Genome Browser (hg38)
- Variant Annotation: ClinVar (NCBI)

권장 디렉터리 예시

```text
data/
├── raw/
│   ├── hg38.fa
│   └── clinvar.vcf.gz
└── processed/
    ├── train.csv
    ├── valid.csv
    └── test.csv
```

<br>

<a id="preprocess"></a>
## 8-3. 🧹 전처리 (Data Preprocessing)

전처리 목표

- ClinVar 변이 정보(VCF) 파싱
- 변이 좌표 기반 reference/variant 서열 생성
- 학습용 pair 데이터셋(예: CSV) 생성
- variant 단위 split로 데이터 누수 방지

실행 방식은 아래 두 패턴 중 하나입니다.

1) CLI 인자를 지원하는 경우

```bash
python src/data_preprocess.py -h
python src/data_preprocess.py --input_vcf <VCF_PATH> --ref_fa <FASTA_PATH> --out_dir <OUT_DIR>
```

2) CLI가 없다면  
`src/data_preprocess.py` 상단의 설정(CONFIG/경로)을 수정 후 실행

```bash
python src/data_preprocess.py
```

전처리 입력/출력 스키마(컬럼명/포맷)는 `Data_Preprocessing.md`를 기준으로 맞추세요.

<br>

<a id="train"></a>
## 8-4. 🏋️ 학습 (Training)

학습 목표

- 하이브리드 임베딩 모델 구성(Global LM + Local CNN)
- cosine distance 기반 병원성 신호 학습
- multi-loss로 임베딩 구조 안정화

실행 방식은 전처리와 동일하게 두 패턴 중 하나입니다.

1) CLI 인자를 지원하는 경우

```bash
python src/train.py -h
python src/train.py --data_dir <PROCESSED_DIR> --output_dir <OUT_DIR>
```

2) CLI가 없다면  
`src/train.py` 내부 설정(CONFIG 등)에서 다음을 확인 후 실행

- 데이터 경로(`data/processed/<...>`)
- 백본/LoRA 설정
- Local encoder 설정(윈도우 크기, CNN 채널/커널 등)
- 손실 가중치
- 하이퍼파라미터(lr, batch size, epochs, grad accumulation 등)
- 체크포인트/로그 저장 경로

```bash
python src/train.py
```

학습 상세 옵션과 재현 규칙은 `Train.md`를 우선 확인하세요.

<br>

<a id="inference"></a>
## 8-5. 🔎 추론 (Inference / Embedding Export)

추론 목표

- fine-tuned checkpoint 로드
- embedding 생성 및 저장
- 필요 시 cosine distance 산출

실행 방식은 아래 두 패턴 중 하나입니다.

1) CLI 인자를 지원하는 경우

```bash
python src/inference.py -h
python src/inference.py --ckpt <CKPT_PATH> --input <CSV_PATH> --out <OUT_PATH>
```

2) CLI가 없다면  
`src/inference.py` 내부 설정(CONFIG/경로)을 수정 후 실행

```bash
python src/inference.py
```

추론 입출력(파일 포맷/저장 방식)은 `Inference.md`를 기준으로 맞추세요.

---

<a id="repro"></a>
# 9. 🧪 로깅 및 재현성 (Logging & Reproducibility)

### Weights & Biases (wandb)

- 학습/평가 로그 기록에 wandb를 사용합니다.
- 사용하지 않는 경우 `src/train.py`의 `wandb.init()`, `wandb.log()`를 주석 처리하세요.

### 재현성

- Python/NumPy/PyTorch 시드 고정
- 설정은 CONFIG(또는 유사 설정 블록)로 중앙 관리
- variant 단위 split로 누수 방지

---

<a id="policy"></a>
# 10. 🚫 데이터 및 모델 재배포 안내 (중요)

본 저장소는 **연구 및 교육 목적의 코드만 공개**합니다. 아래 항목은 포함되어 있지 않으며 재배포하지 않습니다.

- 인간 유전체 Reference Genome (예: hg38)
- ClinVar 원본 VCF
- ClinVar 기반 파생 데이터셋(CSV 등)
- 사전학습(pretrained) 모델 가중치
- 파인튜닝(fine-tuned) 모델 체크포인트
- 학습 결과물 및 실험 로그

사전학습 모델은 Hugging Face Hub에서 동적으로 로드됩니다.  
사용 시 각 데이터/모델의 라이선스 및 이용 정책을 준수해야 합니다.

---

<a id="faq"></a>
# 11. 🛠️ FAQ / 트러블슈팅

### Q1. 실행이 느리거나 OOM이 납니다
- batch size를 줄이세요.
- 입력 시퀀스 길이를 줄이거나(local window 포함) gradient accumulation을 사용하세요.
- mixed precision 사용 여부(지원 시)를 확인하세요.
- GPU 메모리 사용량이 큰 백본이므로 작은 설정으로 end-to-end 검증 후 스케일업을 권장합니다.

### Q2. 전처리 결과 파일 형식이 맞지 않습니다
- `Data_Preprocessing.md`의 출력 스키마를 기준으로 컬럼/포맷을 맞추세요.
- CLI 옵션이 없다면 `src/data_preprocess.py` 내부 설정(CONFIG)을 우선 확인하세요.

### Q3. 학습 체크포인트/출력 경로를 바꾸고 싶습니다
- `src/train.py`의 output_dir(또는 CONFIG) 항목을 수정하세요.
- wandb를 사용 중이면 run name/project도 함께 정리하는 것을 권장합니다.

---

<a id="contact"></a>
# 12. 📬 문의

코드 구조, 학습 방식, 실험 설정 관련 문의는 GitHub Issue로 남겨주세요.
