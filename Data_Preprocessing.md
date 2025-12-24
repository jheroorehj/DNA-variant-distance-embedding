# 데이터 전처리 (Data Preprocessing)

본 문서는 본 프로젝트에서 사용된 **DNA 변이 데이터 전처리 파이프라인**을 설명한다.  
전처리는 ClinVar 변이 데이터와 hg38 reference genome을 기반으로 수행되며,  
모델 학습을 위한 **거리 기반 임베딩 학습에 적합한 데이터셋 생성**을 목표로 한다.

---

## 1. 데이터 소스

### Reference Genome
- UCSC Genome Browser
- Human Genome Assembly: **hg38**

### Variant Annotation
- ClinVar (NCBI)
- VCF 포맷 사용

> 본 저장소에는 원본 데이터 및 파생 데이터셋을 포함하지 않는다.  
> 사용자는 각 공식 출처를 통해 데이터를 개별적으로 획득해야 한다.

---

## 2. 변이 필터링

### 2.1 변이 유형 제한
- **SNP (Single Nucleotide Polymorphism)만 사용**
- 삽입/결실(Indel) 변이는 제외

필터링 조건:
- `len(ref) == 1` 이고 `len(alt) == 1`
- 염기 문자는 {A, C, G, T} 중 하나

이는 모델 입력을 단일 염기 치환 변이에 한정하여  
해석 가능한 거리 학습을 수행하기 위함이다.

---

### 2.2 ClinVar 라벨 신뢰도 필터링

ClinVar annotation의 불확실성을 고려하여  
다음 항목이 포함된 변이는 학습 데이터에서 제외하였다.

- `conflicting`
- `uncertain significance`
- `not provided`

이는 라벨 노이즈로 인한 임베딩 공간 왜곡을 방지하기 위한 조치이다.

---

## 3. 병원성 라벨 및 점수 정의

본 프로젝트는 **이진 분류 문제가 아닌 거리 기반 회귀 문제**로  
DNA 변이의 병원성을 모델링한다.

ClinVar annotation은 다음과 같이  
연속적인 병원성 점수로 변환된다.

| ClinVar Annotation | Label | Score |
|-------------------|-------|-------|
| Pathogenic | 1 | 1.0 |
| Likely Pathogenic | 1 | 0.8 |
| Benign | 0 | 0.0 |
| Likely Benign | 0 | 0.2 |

이 score는 reference sequence와 variant sequence 간  
**cosine distance 학습의 감독 신호**로 사용된다.

---

## 4. 클래스 불균형 처리

ClinVar 데이터는 benign 변이가 병원성 변이에 비해  
압도적으로 많아 심각한 클래스 불균형 문제가 존재한다.

이를 완화하기 위해 다음과 같은 확률적 다운샘플링 전략을 적용하였다.

- **Pathogenic 변이**: 100% 유지
- **Benign 변이**: 10% 확률로 유지

이는 병원성 변이에 대한 **민감도 학습을 강화**하기 위한 설계이다.

---

## 5. 서열 컨텍스트 추출

### 5.1 서열 길이
- 고정 길이: **1024 bp**

### 5.2 Mutation-aware window sampling
- 변이 위치가 항상 시퀀스에 포함되도록 윈도우를 샘플링
- 변이 주변 최소 컨텍스트 길이를 보장

이를 통해 모델은  
변이 자체와 주변 문맥 정보를 동시에 학습한다.

---

## 6. Variant Sequence 생성

각 변이에 대해 다음 두 종류의 서열을 생성한다.

- **Reference Sequence**
  - hg38 reference genome에서 추출한 원본 서열

- **Variant Sequence**
  - 변이 위치의 염기를 alt allele로 치환한 서열

변이의 상대 위치(`mut_index`)를 함께 저장하여  
Local encoder(CNN)가 변이 중심 특징을 학습할 수 있도록 한다.

---

## 7. 데이터 증강 (Augmentation)

각 변이에 대해 서로 다른 컨텍스트를 갖는  
여러 개의 시퀀스를 생성한다.

이는 위치 편향(position bias)을 완화하고  
모델의 일반화 성능을 향상시키기 위함이다.

---

## 8. 학습 / 평가 데이터 분할

- Evaluation 비율: **10%**
- **Variant group 단위 분할** 적용

동일한 `(chrom, pos, ref, alt)` 조합의 변이는  
학습(train) 또는 평가(eval) 중 **한 곳에만 포함**된다.

이를 통해 데이터 누수(leakage)를 방지한다.

---

## 9. 출력 데이터 포맷

각 샘플은 다음 필드를 포함한다.

- `group_id`: 변이 고유 식별자
- `chrom`: 염색체
- `pos`: 변이 위치
- `ref_allele`: reference allele
- `alt_allele`: alternative allele
- `ref_seq`: reference DNA sequence
- `var_seq`: variant DNA sequence
- `label`: 이진 병원성 라벨
- `score`: 연속적 병원성 점수
- `mut_index`: 시퀀스 내 변이 상대 위치
- `strand`: DNA strand (forward)

---

## 10. 요약

본 전처리 파이프라인은 다음 원칙을 중심으로 설계되었다.

- ClinVar 라벨 신뢰도 고려
- 클래스 불균형 완화
- 변이 중심 문맥 학습
- 거리 기반 임베딩 학습
- 재현성 확보 및 데이터 누수 방지

> 전처리는 정확도를 높이기 위한 과정이 아니라,  
> **임베딩 공간의 구조를 설계하는 과정이다.**
