import pysam
import gzip
import os
import random
import csv
from tqdm import tqdm

SEQ_LEN = 1024             
MIN_CONTEXT = 100          
AUGMENT_PER_VARIANT = 2    # 증강 횟수 

KEEP_PROB_BENIGN = 0.1     # Benign 데이터는 10%만 사용
KEEP_PROB_PATHOGENIC = 1.0 # Pathogenic 데이터는 100% 사용

REF_PATH = "data/hg38.fa"       
VCF_PATH = "data/clinvar.vcf.gz" 

TRAIN_OUTPUT_PATH = "data/train_dataset_optimized.csv"
EVAL_OUTPUT_PATH = "data/eval_dataset_optimized.csv"
SPLIT_RATIO = 0.1 # Eval 비율

# 시드 고정
random.seed(42)

def get_reverse_complement(seq):
    complement = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(complement)[::-1]

def get_label_and_score(clnsig_str):
    sig = clnsig_str.lower()
    if "conflicting" in sig or "uncertain" in sig or "not provided" in sig: return None, None

    if "pathogenic" in sig:
        if "likely" in sig: return 1, 0.8
        else: return 1, 1.0
    elif "benign" in sig:
        if "likely" in sig: return 0, 0.2
        else: return 0, 0.0
    return None, None


def generate_optimized_dataset():
    if not os.path.exists(REF_PATH) or not os.path.exists(VCF_PATH):
        print("파일 경로를 확인해주세요.")
        return

    print("Loading Reference Genome...")
    try:
        fasta = pysam.FastaFile(REF_PATH)
    except Exception as e:
        print(f"Error loading FASTA: {e}")
        return
    
    valid_chroms = set(fasta.references)
    
    fieldnames = ['group_id', 'chrom', 'pos', 'ref_allele', 'alt_allele', 
                  'ref_seq', 'var_seq', 'label', 'score', 'mut_index', 'strand']
    f_train = open(TRAIN_OUTPUT_PATH, 'w', newline='', encoding='utf-8')
    f_eval = open(EVAL_OUTPUT_PATH, 'w', newline='', encoding='utf-8')
    
    writer_train = csv.DictWriter(f_train, fieldnames=fieldnames)
    writer_eval = csv.DictWriter(f_eval, fieldnames=fieldnames)
    
    writer_train.writeheader()
    writer_eval.writeheader()

    print(f"Processing VCF (Streaming Mode)...")
    print(f"Strategies: Augment=x{AUGMENT_PER_VARIANT}, Benign_Keep={KEEP_PROB_BENIGN*100}%")
    
    cnt = {'total': 0, 'saved': 0, 'pathogenic': 0, 'benign': 0}

    with gzip.open(VCF_PATH, 'rt') as f:
        for line in tqdm(f, desc="Generating"):
            if line.startswith('#'): continue
            
            parts = line.strip().split('\t')
            try:
                chrom = parts[0]
                pos = int(parts[1])
                ref = parts[3]
                alt = parts[4]
                info = parts[7]
            except: continue
            
            # 1. 필터링 (SNP만)
            if len(ref) != 1 or len(alt) != 1: continue
            if ref not in "ACGT" or alt not in "ACGT": continue

            # 2. 염색체 확인
            target_chrom = chrom
            if chrom not in valid_chroms:
                if f"chr{chrom}" in valid_chroms: target_chrom = f"chr{chrom}"
                else: continue
            
            # 3. 라벨 확인 및 다운샘플링
            clnsig = ""
            if "CLNSIG=" in info:
                for item in info.split(';'):
                    if item.startswith("CLNSIG="):
                        clnsig = item.split('=')[1]
                        break
            
            label, score = get_label_and_score(clnsig)
            if label is None: continue

            # Pathogenic이면 무조건 저장, Benign이면 확률적으로 버림
            if label == 1:
                if random.random() > KEEP_PROB_PATHOGENIC: continue
            else:
                if random.random() > KEEP_PROB_BENIGN: continue

            # 4. 데이터 생성 (Augmentation)
            zero_pos = pos - 1
            min_start = zero_pos - (SEQ_LEN - MIN_CONTEXT)
            max_start = zero_pos - MIN_CONTEXT
            if min_start < 0: min_start = 0

            # VCF vs Ref 일치 확인
            try:
                if fasta.fetch(target_chrom, zero_pos, zero_pos+1).upper() != ref: continue
            except: continue

            # Train/Eval 결정 (Variant 그룹 단위로 분리)
            # 10% 확률로 Eval 파일에, 90% 확률로 Train 파일에 기록
            target_writer = writer_eval if random.random() < SPLIT_RATIO else writer_train
            
            group_id = f"{chrom}_{pos}_{ref}_{alt}"

            for _ in range(AUGMENT_PER_VARIANT):
                try:
                    # 랜덤 윈도우
                    if max_start < min_start: rand_start = max(0, zero_pos - SEQ_LEN // 2)
                    else: rand_start = random.randint(min_start, max_start)
                    
                    rand_end = rand_start + SEQ_LEN
                    ref_seq_full = fasta.fetch(target_chrom, rand_start, rand_end).upper()
                    
                    if len(ref_seq_full) != SEQ_LEN: continue

                    relative_mut_pos = zero_pos - rand_start
                    if not (0 <= relative_mut_pos < SEQ_LEN): continue

                    # Variant Seq 생성
                    var_seq_full = list(ref_seq_full)
                    var_seq_full[relative_mut_pos] = alt
                    var_seq_full = "".join(var_seq_full)
                    
                    row = {
                        'group_id': group_id,
                        'chrom': chrom,
                        'pos': pos,
                        'ref_allele': ref,
                        'alt_allele': alt,
                        'ref_seq': ref_seq_full,
                        'var_seq': var_seq_full,
                        'label': label,
                        'score': score,
                        'mut_index': relative_mut_pos,
                        'strand': 'forward'
                    }
                    target_writer.writerow(row)
                    cnt['saved'] += 1
                    if label == 1: cnt['pathogenic'] += 1
                    else: cnt['benign'] += 1

                except: continue
                
    f_train.close()
    f_eval.close()
    fasta.close()

    print("\n" + "="*40)
    print("Optimized Dataset Generated!")
    print(f"Total Samples Saved: {cnt['saved']}")
    print(f" - Pathogenic Samples: {cnt['pathogenic']}")
    print(f" - Benign Samples: {cnt['benign']}")
    print(f"Saved to: {TRAIN_OUTPUT_PATH}")
    print("="*40)

if __name__ == "__main__":
    generate_optimized_dataset()