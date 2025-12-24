# 모델 학습 (Training Pipeline)

본 문서는 본 프로젝트에서 사용된 **Hybrid DNA Variant Distance Regression 모델의 학습 파이프라인**을 설명한다.  
본 학습 코드는 **Reference–Variant DNA 서열 쌍을 입력으로 받아**,  
임베딩 공간 상의 **거리(distance)** 가 병원성(pathogenicity)을 반영하도록 학습하는 것을 목표로 한다.

---

## 1. 학습 목표

본 모델은 DNA 변이의 병원성을 **이진 분류 문제가 아닌 거리 기반 회귀 문제**로 정의한다.

- 입력: Reference DNA 서열 / Variant DNA 서열
- 출력: L2 정규화된 임베딩 벡터
- 학습 목표:
  - Pathogenic 변이는 **큰 거리**
  - Benign 변이는 **작은 거리**
  - 두 클래스 간 평균 거리 차이(CDD)를 안정적으로 확보

---

## 2. 전체 학습 구조 개요

학습 파이프라인은 다음 구성 요소로 이루어진다.

1. Genomic Language Model 기반 전역 임베딩 추출
2. 변이 위치 중심의 Local CNN을 통한 국소 특징 추출
3. Global + Local 임베딩 결합
4. 거리 기반 감독 학습 (Distance Regression)
5. 다양한 보조 손실을 통한 임베딩 구조 정렬

---

## 3. 입력 데이터 형식

학습 데이터는 전처리 단계에서 생성된 CSV 파일을 사용한다.

각 샘플은 다음 정보를 포함한다.

- `ref_seq`: Reference DNA sequence
- `var_seq`: Variant DNA sequence
- `mut_index`: 시퀀스 내 변이 상대 위치
- `label`: 병원성 이진 라벨 (0: benign, 1: pathogenic)
- `score`: 연속적 병원성 점수 (0.0 ~ 1.0)

---

## 4. Dataset 및 입력 처리

### 4.1 Global Encoder 입력
- 전체 DNA 서열을 tokenizer를 통해 토크나이즈
- 최대 길이: 1024 bp
- padding / truncation 적용

### 4.2 Local Encoder 입력
- 변이 위치(`mut_index`) 기준으로 국소 윈도우 추출
- One-hot encoding (A, C, G, T)
- Local window size: 64 bp

이를 통해 전역 문맥 정보와 변이 중심 국소 정보를 동시에 활용한다.

---

## 5. 모델 아키텍처

### 5.1 Backbone (Global Encoder)

- 사전학습 모델:
  - `LongSafari/hyenadna-large-1m-seqlen-hf`
- Hugging Face `AutoModelForCausalLM` 사용
- 마지막 hidden state 평균 풀링(mean pooling)

### 5.2 Parameter-Efficient Fine-Tuning (LoRA)

- LoRA를 적용하여 backbone을 효율적으로 미세조정
- 주요 projection 레이어에만 low-rank adapter 적용
- 전체 파라미터 수 증가 최소화

---

### 5.3 Local CNN Encoder

변이 위치 중심의 국소 특징 강화를 위해 1D CNN을 사용한다.

- 입력: One-hot encoded local DNA sequence (4 × 64)
- Conv1D + BatchNorm + GELU
- MaxPooling을 통한 차원 축소
- 출력 차원: 1024

---

### 5.4 Embedding Fusion

- Global embedding (1024)
- Local embedding (1024)

두 벡터를 concatenation 후 linear projection을 적용하여  
최종 **2048차원 임베딩**을 생성한다.

마지막으로 L2 정규화를 적용하여 cosine distance 기반 학습이 가능하도록 한다.

---

## 6. 거리 계산 방식

Reference / Variant 임베딩 간 cosine similarity를 계산하고,  
다음과 같이 거리로 변환한다.

- `distance = 1 - cosine_similarity`
- 거리 범위: **[0, 2]**

---

## 7. 손실 함수 구성

본 모델은 단일 손실이 아닌,  
**임베딩 공간 구조를 직접 설계하기 위한 다중 손실 함수의 가중합**으로 학습된다.

### 7.1 Distance Regression Loss (MSE)
- 예측 거리와 병원성 점수 기반 목표 거리 간 MSE
- 기본 감독 신호

### 7.2 Mutation Focus Loss
- Local CNN 임베딩 거리 기반 MSE
- 변이 주변 국소 특징 민감도 강화

### 7.3 Mean Margin Gap Loss
- Pathogenic / Benign 간 평균 거리 차이가
  설정한 margin 이상 유지되도록 유도

### 7.4 Triplet Loss
- Anchor/Positive: Pathogenic
- Negative: Benign
- 클래스 간 임베딩 분리 강화

### 7.5 Pairwise Margin Loss
- 모든 Path–Benign 쌍에 대해
  거리 차이가 margin 이상이 되도록 강제

### 7.6 Supervised Contrastive Loss
- 동일 클래스 임베딩을 정렬
- 테스트 시 단일 임베딩만 사용하더라도
  클래스 분리가 유지되도록 설계

---

## 8. 학습 설정

- Batch size: 128
- Optimizer: AdamW
- Learning rate: 3e-5
- Epochs: 10
- Mixed Precision (bfloat16) 사용
- Seed 고정으로 재현성 확보

---

## 9. 평가 지표

학습 및 평가 과정에서 다음 지표를 계산한다.

- CD (Cosine Distance Mean)
- CDD (Class Distance Difference)
- PCC (Pearson Correlation Coefficient)
- Normalized CD / CDD / PCC
- Final Score (세 지표 평균)

이 지표들은 **임베딩 거리 기반 병원성 분리 성능**을 종합적으로 평가한다.

---

## 10. 로깅 및 모델 저장

- Weights & Biases (wandb)를 사용하여
  step-level / epoch-level 로그 기록
- Final Score 기준으로 최고 성능 모델 저장
- 저장 파일:
  - `checkpoints/best_hybrid_model.pt`

---

## 11. 요약

본 학습 파이프라인은 다음 원칙을 중심으로 설계되었다.

- 병원성의 거리 기반 정량화
- 전역 + 국소 정보의 결합
- 임베딩 공간 구조 직접 설계
- 다중 손실 기반 안정적 분리
- 실험 재현성 및 분석 가능성 확보

> **학습은 분류 정확도를 높이는 과정이 아니라,  
> 의미 있는 임베딩 공간을 형성하는 과정이다.**
