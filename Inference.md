# 추론 파이프라인 (Inference)

본 문서는 학습이 완료된 **Hybrid DNA Variant Distance Regression 모델**을 이용하여  
새로운 DNA 서열에 대해 **임베딩을 생성하고 제출용 결과 파일을 생성하는 추론 과정**을 설명한다.

본 추론 코드는 학습 코드와 **모델 구조 및 레이어 이름을 완전히 동일하게 유지**하여  
체크포인트 로딩 및 재현성을 보장한다.

---

## 1. 추론 목적

추론 단계의 목적은 다음과 같다.

- 입력: DNA 서열 (`seq`)
- 출력: L2 정규화된 **2048차원 임베딩 벡터**
- 용도:
  - 거리 기반 병원성 평가
  - 대회 제출용 임베딩 생성
  - 후속 분석 (distance, clustering 등)

---

## 2. 입력 데이터 형식

추론 입력 데이터는 CSV 파일 형태로 제공된다.

### 입력 파일: `test.csv`

| 컬럼명 | 설명 |
|------|------|
| ID | 샘플 고유 식별자 |
| seq | DNA 서열 문자열 |

---

## 3. Dataset 구성

### 3.1 Global 입력 (Genomic LM)

- 전체 DNA 서열을 tokenizer로 토크나이즈
- 최대 길이: 1024
- padding / truncation 적용

### 3.2 Local 입력 (Mutation-agnostic)

추론 단계에서는 변이 위치 정보가 제공되지 않으므로  
다음과 같은 전략을 사용한다.

- 서열 중앙을 기준으로 local window 추출
- Local window size: 64 bp
- One-hot encoding (A, C, G, T)

이는 학습 시 사용된 Local CNN 구조를 그대로 유지하기 위함이다.

---

## 4. 모델 구조 일치성

추론 모델은 **학습 시 사용된 모델과 레이어 이름이 완전히 동일**하다.

- Backbone: `LongSafari/hyenadna-large-1m-seqlen-hf`
- LoRA 설정 동일
- `hyena_proj`, `cnn`, `final_proj` 레이어 이름 유지
- 출력 차원: 2048

이를 통해 `state_dict`를 `strict=True` 옵션으로 안전하게 로딩한다.

---

## 5. 임베딩 생성 과정

추론 과정은 다음 단계로 이루어진다.

1. Backbone을 통한 hidden state 추출
2. Mean pooling을 통한 global embedding 생성
3. Local CNN을 통한 local embedding 생성
4. Global + Local embedding 결합
5. Linear projection
6. L2 정규화 적용

최종 출력은 **cosine distance 계산에 직접 사용 가능한 임베딩**이다.

---

## 6. 배치 추론 설정

- Batch size: 128
- DataLoader 기반 배치 처리
- GPU 사용 시 CUDA 자동 활성화
- Gradient 계산 비활성화 (`torch.no_grad()`)

이를 통해 대규모 입력 데이터에 대해서도  
효율적인 추론이 가능하다.

---

## 7. 출력 파일 형식

### 출력 파일: `submission/submission.csv`

- 첫 번째 컬럼: `ID`
- 이후 컬럼: 임베딩 벡터

```text
ID, emb_0000, emb_0001, ..., emb_2047
```
- 임베딩은 float16으로 저장
- 파일 크기 및 I/O 효율 최적화

## 8. 재현성 및 안정성

본 추론 파이프라인은 학습 단계와의 **완전한 일관성**을 유지하도록 설계되었다.

- 학습 코드와 **동일한 모델 정의** 사용
- 동일한 tokenizer 및 전처리 로직 사용
- 체크포인트 경로를 **명시적으로 지정**
- `strict=True` 옵션을 사용한 `state_dict` 로딩으로  
  모델 구조 불일치로 인한 오류를 사전에 방지

이를 통해 추론 결과의 재현성과 안정성을 확보한다.

---

## 9. 요약

본 추론 파이프라인은 다음 원칙을 따른다.

- 학습 구조와의 완전한 일치
- 임베딩 기반 후속 분석에 친화적인 설계
- 대규모 데이터 처리 가능
- 제출 및 평가 환경에 적합한 출력 포맷 제공
