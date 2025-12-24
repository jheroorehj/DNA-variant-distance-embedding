# Hybrid DNA Variant Distance Regression

이 저장소는 **DNA 변이 병원성 학습을 위한 코드만을 포함**합니다.  
유전체 데이터, 사전학습 모델 가중치, 파인튜닝된 체크포인트는 **포함되어 있지 않습니다**.

본 프로젝트는 **Reference DNA 서열과 Variant DNA 서열 간의 거리 기반 임베딩 학습**을 목표로 하며,  
전역(genomic language model) 정보와 국소(mutation-focused) 정보를 결합한 하이브리드 구조를 사용합니다.

---

## 📄 Documentation

- [Data Preprocessing](./Data_Preprocessing.md)
- [Training Pipeline](./Train.md)
- [Inference Pipeline](./Inference.md)


## 🔬 프로젝트 개요

- **문제 정의**: DNA 변이의 병원성(Pathogenicity)을 거리 기반 임베딩으로 모델링
- **입력**: Reference DNA 서열 / Variant DNA 서열
- **출력**: 정규화된 임베딩 벡터 (Cosine Distance ∈ [0, 2])
- **핵심 아이디어**:
  - 사전학습된 Genomic Language Model을 통한 전역 문맥 인코딩
  - 변이 위치 중심의 Local CNN을 통한 국소 특징 강화
  - 다양한 감독 신호(loss)를 결합한 거리 학습

---

## 🧠 모델 구조

- **Backbone 모델**  
  - Hugging Face 공개 모델  
  - `LongSafari/hyenadna-large-1m-seqlen-hf`
- **파인튜닝 방식**  
  - LoRA(Parameter-Efficient Fine-Tuning)
- **Local Encoder**  
  - 변이 위치 기준 One-hot DNA 서열에 대한 1D CNN
- **최종 임베딩**  
  - Global Embedding + Local Embedding 결합
  - Linear Projection 후 L2 정규화
  - 최종 차원: 2048

---

## 📐 학습 손실 함수 (Loss Functions)

본 모델은 다음 손실 함수들의 가중합으로 학습됩니다.

- **Distance Regression Loss (MSE)**  
  - 예측 거리와 병원성 점수 간 정합성 학습
- **Mutation Focus Loss**  
  - 변이 주변 국소 특징에 대한 민감도 강화
- **Mean Margin Gap Loss**  
  - Pathogenic / Benign 간 평균 거리 차이 확보
- **Triplet Loss**  
  - Pathogenic vs Benign 임베딩 구조 분리
- **Pairwise Margin Loss**  
  - 클래스 간 거리 차이를 강하게 유지
- **Supervised Contrastive Loss**  
  - 테스트 시 단일 임베딩만 사용하더라도 클래스 분리가 유지되도록 정렬

---

## 📊 평가 지표 (Metrics)

학습 및 평가 과정에서 다음 지표를 계산합니다.

- **CD (Cosine Distance Mean)**  
- **CDD (Class Distance Difference)**  
- **PCC (Pearson Correlation Coefficient)**  
- **정규화 지표 (0~1 스케일)**  
- **Final Score**  
  - Normalized CD, CDD, PCC의 평균

---

## 📁 저장소 구조

```text
.
├── src/
│   ├── train.py               # 모델 학습 코드
│   ├── data_preprocess.py     # 데이터셋 생성 코드 (ClinVar 기반)
│   └── inference.py           # 추론 및 임베딩 생성 코드
├── requirements.txt
├── README.md
└── .gitignore
```
## 🚫 데이터 및 모델 재배포 안내 (중요)

본 저장소는 **연구 및 교육 목적의 코드만을 공개**합니다.  
아래 항목들은 **라이선스 및 데이터 이용 정책에 따라 본 저장소에 포함되어 있지 않습니다**.

### 포함되지 않는 항목
- 인간 유전체 Reference Genome 파일 (예: hg38)
- ClinVar 원본 VCF 파일
- ClinVar 기반으로 생성된 파생 데이터셋(CSV 등)
- 사전학습(pretrained) 모델 가중치
- 파인튜닝(fine-tuned) 모델 체크포인트
- 학습 결과물 및 실험 로그

### 데이터 출처 및 이용 조건
- **Reference Genome**: UCSC Genome Browser (hg38)
- **Variant Annotation**: ClinVar (NCBI)

해당 데이터들은 공개적으로 접근 가능하나,  
**원본 또는 파생 데이터의 재배포는 각 데이터 제공 기관의 정책에 의해 제한될 수 있습니다.**  
본 저장소는 데이터 파일을 직접 제공하지 않으며,  
사용자는 공식 출처를 통해 개별적으로 데이터를 획득해야 합니다.

---

## 🧩 사전학습 모델 안내

본 프로젝트는 Hugging Face Hub를 통해 다음 사전학습 모델을 동적으로 로드합니다.

- `LongSafari/hyenadna-large-1m-seqlen-hf`

본 저장소는 **사전학습 모델 가중치를 재배포하지 않습니다.**  
모델 사용 시에는 Hugging Face에 명시된 해당 모델의 라이선스 및 이용 조건을  
반드시 준수해야 합니다.

---

## ⚙️ 환경 설정 방법

### Conda 환경 (권장)

```bash
conda create -n medicalAI python=3.10
conda activate medicalAI
pip install -r requirements.txt
```

### 참고 사항
- CUDA, cuDNN, GPU 드라이버는 사용자 시스템 환경에 맞게 별도로 설치해야 합니다.
- CPU 환경에서도 코드 실행은 가능하지만, 학습 속도는 크게 저하될 수 있습니다.

---

## 🧪 실험 로깅 (Weights & Biases)

본 프로젝트는 **Weights & Biases (wandb)** 를 활용하여 학습 및 평가 과정의 로그를 기록합니다.

- wandb 사용을 위해서는 별도의 계정 로그인이 필요합니다.
- wandb를 사용하지 않는 경우,  
  `train.py` 파일 내 `wandb.init()` 및 `wandb.log()` 관련 코드를 주석 처리하면 됩니다.

---

## ♻️ 재현성 (Reproducibility)

- Python, NumPy, PyTorch에 대해 시드를 고정하여 실험 재현성을 확보하였습니다.
- 모든 학습 관련 설정은 코드 내 `CONFIG` 딕셔너리를 통해 중앙에서 관리됩니다.
- 데이터셋 분할은 변이(variant) 그룹 단위로 수행하여 데이터 누수를 방지합니다.

---

## 📜 라이선스

본 저장소는 **코드만을 포함한 연구 및 교육 목적의 저장소**입니다.  
데이터, 사전학습 모델, 및 파생 결과물의 사용 시에는  
각각의 원본 라이선스와 이용 정책을 반드시 준수해야 합니다.

---

## 📬 문의

코드 구조, 학습 방식, 또는 실험 설정과 관련한 문의 사항은  
GitHub Issue를 통해 남겨주시기 바랍니다.
