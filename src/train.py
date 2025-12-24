#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from scipy.stats import pearsonr
import wandb
from tqdm import tqdm


CONFIG = {
    "project_name": "MedicalAI",
    "model_name": "LongSafari/hyenadna-large-1m-seqlen-hf",

    "max_length": 1024,
    "local_window_size": 64,

    # Output embedding dim 
    "output_dim": 2048,

    # Train config
    "batch_size": 128,
    "learning_rate": 3e-5,
    "epochs": 10,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,

    # Loss 스케일 및 가중치
    "loss_scale": 1.0,
    "w_reg": 1.0,          # Distance Regression 
    "w_focus": 0.5,        # Mutation Focus Loss (local CNN)
    "w_margin_mean": 1.0,  # mean gap margin loss
    "margin_value": 0.3,   # (0~2 스케일 상에서 Path - Benign 최소 차이)

    # Contrastive 계열 
    "use_triplet": True,
    "w_triplet": 0.7,
    "triplet_margin": 0.2,

    "use_pair_margin": True,
    "w_pair_margin": 1.0,
    "pair_margin_value": 0.3,

    # supervised contrastive on emb_var 
    "use_contrastive": True,
    "w_contrastive": 0.5,
    "contrast_temperature": 0.07,

    # Local CNN 사용 여부
    "use_local_cnn": True,
}

torch.set_float32_matmul_precision("high")


# Seed 고정
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])



# Dataset
class HybridGenomeDataset(Dataset):

    def __init__(self, csv_path, tokenizer, max_length=1024, local_size=64):
        self.data = pd.read_csv(csv_path, low_memory=False)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.local_size = local_size
        self.dna_map = {"A": 0, "C": 1, "G": 2, "T": 3}

    def __len__(self):
        return len(self.data)

    def one_hot(self, seq: str):
        seq = seq.upper()
        mat = np.zeros((4, len(seq)), dtype=np.float32)
        for i, base in enumerate(seq):
            if base in self.dna_map:
                mat[self.dna_map[base], i] = 1.0
        return mat

    def process_seq(self, seq: str, mut_idx: int):
        # gLM 입력용 토크나이즈
        tokens = self.tokenizer(
            seq,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Local CNN용 윈도우 (mut_index 기준)
        half = self.local_size // 2
        start = max(0, mut_idx - half)
        end = start + self.local_size

        local = seq[start:end]
        if len(local) < self.local_size:
            local = local + "N" * (self.local_size - len(local))

        local_oh = torch.tensor(self.one_hot(local), dtype=torch.float32)  # (4, L)

        return tokens["input_ids"].squeeze(0), local_oh

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        mut_idx = int(row["mut_index"])

        ref_ids, ref_local = self.process_seq(row["ref_seq"], mut_idx)
        var_ids, var_local = self.process_seq(row["var_seq"], mut_idx)

        label = float(row["label"])       # 0 / 1 (benign / pathogenic)
        score = float(row["score"])       # 0, 0.2, 0.8, 1.0

        return {
            "ref_ids": ref_ids,
            "ref_local": ref_local,
            "var_ids": var_ids,
            "var_local": var_local,
            "label": torch.tensor(label, dtype=torch.float32),
            "score": torch.tensor(score, dtype=torch.float32),
        }


# Model
class HybridDNA(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        print(f"Loading Backbone: {base_model_name}")

        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        if hasattr(self.backbone.config, "d_model"):
            d_model = self.backbone.config.d_model
        else:
            d_model = 256

        # LoRA
        peft_config = LoraConfig(
            inference_mode=False,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["out_proj", "dense", "c_fc", "c_proj"],
        )
        self.backbone = get_peft_model(self.backbone, peft_config)

        # gLM -> 1024
        self.hyena_proj = nn.Linear(d_model, 1024)

        # Local CNN
        if CONFIG["use_local_cnn"]:
            self.cnn = nn.Sequential(
                nn.Conv1d(4, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.MaxPool1d(2),           # L=64 -> 32
                nn.Flatten(),
                nn.Linear(128 * 32, 1024),
                nn.GELU(),
            )
        else:
            self.cnn = None

        # 1024(global) + 1024(local) -> 2048
        self.final_proj = nn.Linear(2048, CONFIG["output_dim"])

    def forward(self, input_ids, local_input, return_all=False):
        # gLM backbone
        outputs = self.backbone(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]          # (B, L, d_model)
        hidden = hidden.float()

        pooled = torch.mean(hidden, dim=1)          # (B, d_model)
        global_emb = self.hyena_proj(pooled)        # (B, 1024)

        # Local CNN
        local_input = local_input.float()

        if self.cnn is not None:
            if local_input.ndim != 3:
                raise ValueError(
                    f"local_input ndim expected 3, got {local_input.ndim}"
                )

            if local_input.shape[1] != 4 and local_input.shape[2] == 4:
                local_input = local_input.permute(0, 2, 1)

            local_emb = self.cnn(local_input)       # (B, 1024)
        else:
            local_emb = global_emb                  # fallback

        concat = torch.cat([global_emb, local_emb], dim=1)  # (B, 2048)

        final_emb = self.final_proj(concat)         # (B, 2048)
        final_emb = F.normalize(final_emb, p=2, dim=1)

        if return_all:
            return final_emb, global_emb, local_emb
        else:
            return final_emb


# Metrics (CD / CDD / PCC + Normal)
def calculate_metrics(labels, dists):
    # labels: 0/1 (benign/pathogenic)
    # dists:  cosine distance (0~2)
    labels = np.array(labels).astype(float)
    dists = np.array(dists).astype(float)

    if len(labels) == 0:
        return {
            "CD_Raw": 0.0,
            "CDD_Raw": 0.0,
            "PCC_Raw": 0.0,
            "Normal_CD": 0.0,
            "Normal_CDD": 0.0,
            "Normal_PCC": 0.0,
            "Final_Score": 0.0,
            "CD_Path": 0.0,
            "CD_Benign": 0.0,
        }

    path_mask = labels == 1
    benign_mask = labels == 0

    path_dists = dists[path_mask]
    benign_dists = dists[benign_mask]

    mean_path = float(path_dists.mean()) if path_dists.size > 0 else 0.0
    mean_benign = float(benign_dists.mean()) if benign_dists.size > 0 else 0.0

    cd_raw = float(dists.mean())
    cdd_raw = float((mean_path - mean_benign) / 2.0)

    try:
        if len(np.unique(labels)) < 2:
            pcc_raw = 0.0
        else:
            pcc_raw, _ = pearsonr(labels, dists)
            if np.isnan(pcc_raw):
                pcc_raw = 0.0
    except Exception:
        pcc_raw = 0.0

    normal_cd = float(np.clip(cd_raw / 2.0, 0.0, 1.0))
    normal_cdd = float(np.clip((cdd_raw + 1.0) / 2.0, 0.0, 1.0))
    normal_pcc = float(np.clip((pcc_raw + 1.0) / 2.0, 0.0, 1.0))

    final_score = float((normal_cd + normal_cdd + normal_pcc) / 3.0)

    return {
        "CD_Raw": cd_raw,
        "CDD_Raw": cdd_raw,
        "PCC_Raw": pcc_raw,
        "Normal_CD": normal_cd,
        "Normal_CDD": normal_cdd,
        "Normal_PCC": normal_pcc,
        "Final_Score": final_score,
        "CD_Path": mean_path / 2.0,
        "CD_Benign": mean_benign / 2.0,
    }



# 추가 Loss 함수 (Triplet / Pairwise Margin)
def triplet_loss_var(emb_var, labels, margin: float):
    
    #batch-level triplet:
    #  - anchor / positive : pathogenic var
    #  - negative : benign var

    device = emb_var.device
    labels = labels.view(-1)

    path_mask = labels == 1.0
    benign_mask = labels == 0.0

    if path_mask.sum() < 1 or benign_mask.sum() < 1:
        return torch.zeros(1, device=device)

    path_emb = emb_var[path_mask]      # (P, D)
    benign_emb = emb_var[benign_mask]  # (B, D)

    # normalize 
    path_emb = F.normalize(path_emb, p=2, dim=1)
    benign_emb = F.normalize(benign_emb, p=2, dim=1)

    # pairwise distances
    # pos: path vs path (intra-path)
    # neg: path vs benign
    pos_sim = torch.matmul(path_emb, path_emb.t())   # (P,P)
    neg_sim = torch.matmul(path_emb, benign_emb.t()) # (P,B)

    # diag 제거용 mask
    P = path_emb.size(0)
    eye = torch.eye(P, device=device)
    pos_sim = pos_sim * (1 - eye) - eye * 2.0  

    pos_dist = 1 - pos_sim   # (P,P)
    neg_dist = 1 - neg_sim   # (P,B)

    # anchor별로 평균
    pos_mean = pos_dist.mean(dim=1)   # (P,)
    neg_mean = neg_dist.mean(dim=1)   # (P,)

    loss = F.relu(pos_mean - neg_mean + margin).mean()
    return loss


def pairwise_margin_loss(dist_final, labels, margin_val: float):
    device = dist_final.device
    labels = labels.view(-1)

    path_mask = labels == 1.0
    benign_mask = labels == 0.0

    if path_mask.sum() < 1 or benign_mask.sum() < 1:
        return torch.zeros(1, device=device)

    p = dist_final[path_mask]    # (P,)
    b = dist_final[benign_mask]  # (B,)

    diff = p[:, None] - b[None, :]  # (P,B) = Path - Benign
    loss = F.relu(margin_val - diff).mean()
    return loss


# Supervised Contrastive Loss 
def supervised_contrastive_loss(embeddings, labels, temperature: float):
    device = embeddings.device
    labels = labels.view(-1)

    if embeddings.size(0) <= 1:
        return torch.zeros(1, device=device)

    # label이 하나뿐이면 contrastive 할 대상 x
    if torch.unique(labels).numel() < 2:
        return torch.zeros(1, device=device)

    # 정규화 후 similarity matrix
    emb = F.normalize(embeddings, p=2, dim=1)  # (B, D)
    sim = torch.matmul(emb, emb.t()) / temperature  # (B, B)

    batch_size = emb.size(0)
    mask = torch.eye(batch_size, device=device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # 같은 label인 것만 positive mask
    labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    positive_mask = labels_matrix & (~mask)  # 자기 자신 제외한 같은 클래스

    # exp(sim) / sum(exp(sim)) 구조
    exp_sim = torch.exp(sim)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)  # (B, B)

    # anchor별 positive 개수
    positive_count = positive_mask.sum(dim=1)  # (B,)
    valid_anchor = positive_count > 0

    if not valid_anchor.any():
        return torch.zeros(1, device=device)

    # positive에 대해서만 log_prob 평균
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_count.clamp(min=1.0)

    # valid anchor만 loss 계산
    loss = -mean_log_prob_pos[valid_anchor].mean()
    return loss


# Train Loop
def main():
    wandb.init(project=CONFIG["project_name"], config=CONFIG)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = HybridDNA(CONFIG["model_name"])
    model.to(CONFIG["device"])

    # Dataset & Loader
    train_ds = HybridGenomeDataset(
        "data/train_dataset_optimized.csv",
        tokenizer,
        CONFIG["max_length"],
        CONFIG["local_window_size"],
    )
    eval_ds = HybridGenomeDataset(
        "data/eval_dataset_optimized.csv",
        tokenizer,
        CONFIG["max_length"],
        CONFIG["local_window_size"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
    )

    use_amp = CONFIG["device"].startswith("cuda") and torch.cuda.is_available()

    print("Start Training (DistReg + Focus + Margin + Triplet + PairMargin + SupContrast)")
    best_score = -999.0
    global_step = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in loop:
            ref_ids = batch["ref_ids"].to(CONFIG["device"])
            ref_local = batch["ref_local"].to(CONFIG["device"])
            var_ids = batch["var_ids"].to(CONFIG["device"])
            var_local = batch["var_local"].to(CONFIG["device"])

            labels = batch["label"].to(CONFIG["device"])   # 0/1
            scores = batch["score"].to(CONFIG["device"])   # 0,0.2,0.8,1.0

            optimizer.zero_grad()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                # Forward
                emb_ref, glob_ref, loc_ref = model(ref_ids, ref_local, return_all=True)
                emb_var, glob_var, loc_var = model(var_ids, var_local, return_all=True)

                # Final distance
                sim_final = F.cosine_similarity(emb_ref, emb_var)  # (B,)
                dist_final = 1.0 - sim_final                       # 0~2

                # Local CNN distance (Mutation Focus 용)
                loc_ref_n = F.normalize(loc_ref, p=2, dim=1)
                loc_var_n = F.normalize(loc_var, p=2, dim=1)
                sim_local = F.cosine_similarity(loc_ref_n, loc_var_n)
                dist_local = 1.0 - sim_local

                # Distance Regression Loss (기존)
                target_dist = scores * 2.0
                loss_reg = F.mse_loss(dist_final, target_dist)

                # Mutation Focus Loss
                loss_focus = F.mse_loss(dist_local, target_dist)

                # Mean Margin Gap Loss
                labels_np = labels.detach().float()
                path_mask = labels_np == 1.0
                benign_mask = labels_np == 0.0

                loss_margin_mean = torch.zeros(
                    1, device=CONFIG["device"], dtype=torch.float32
                )

                mean_path_step = 0.0
                mean_benign_step = 0.0
                gap_step = 0.0

                if path_mask.any() and benign_mask.any():
                    path_d = dist_final[path_mask]
                    benign_d = dist_final[benign_mask]
                    mean_path = path_d.mean()
                    mean_benign = benign_d.mean()
                    gap = mean_path - mean_benign  # Path - Benign

                    mean_path_step = mean_path.detach().item()
                    mean_benign_step = mean_benign.detach().item()
                    gap_step = gap.detach().item()

                    loss_margin_mean = F.relu(CONFIG["margin_value"] - gap)

                # Triplet Loss (Path vs Benign)
                loss_triplet = torch.zeros(
                    1, device=CONFIG["device"], dtype=torch.float32
                )
                if CONFIG["use_triplet"]:
                    loss_triplet = triplet_loss_var(
                        emb_var, labels, CONFIG["triplet_margin"]
                    )

                # Pairwise Margin Loss (강한 CDD용)
                loss_pair_margin = torch.zeros(
                    1, device=CONFIG["device"], dtype=torch.float32
                )
                if CONFIG["use_pair_margin"]:
                    loss_pair_margin = pairwise_margin_loss(
                        dist_final, labels, CONFIG["pair_margin_value"]
                    )

                # Supervised Contrastive Loss (single embedding 정렬)
                loss_contrastive = torch.zeros(
                    1, device=CONFIG["device"], dtype=torch.float32
                )
                if CONFIG["use_contrastive"]:
                    loss_contrastive = supervised_contrastive_loss(
                        emb_var, labels, CONFIG["contrast_temperature"]
                    )

                # Total Loss
                total_loss = (
                    CONFIG["w_reg"] * loss_reg
                    + CONFIG["w_focus"] * loss_focus
                    + CONFIG["w_margin_mean"] * loss_margin_mean
                    + CONFIG["w_triplet"] * loss_triplet
                    + CONFIG["w_pair_margin"] * loss_pair_margin
                    + CONFIG["w_contrastive"] * loss_contrastive
                ) * CONFIG["loss_scale"]

            total_loss.backward()

            # Grad norm 모니터링
            total_norm = 0.0
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()

            # Step-wise Metrics (label 기준) 
            step_metrics = calculate_metrics(
                labels.detach().cpu().numpy(),
                dist_final.detach().cpu().numpy(),
            )

            log_dict = {
                # Wandb Train Logs 
                "Train_Loss": total_loss.item(),
                "Step_CD": step_metrics["CD_Raw"],
                "Step_CDD": step_metrics["CDD_Raw"],
                "Step_PCC": step_metrics["PCC_Raw"],
                "Step_Normal_CD": step_metrics["Normal_CD"],
                "Step_Normal_CDD": step_metrics["Normal_CDD"],
                "Step_Normal_PCC": step_metrics["Normal_PCC"],
                "Step_FinalScore": step_metrics["Final_Score"],
                "Global_Step": global_step,

                # Loss 분해 
                "Train_Loss_Reg": loss_reg.item(),
                "Train_Loss_Focus": loss_focus.item(),
                "Train_Loss_MarginMean": loss_margin_mean.item(),
                "Train_Loss_Triplet": loss_triplet.item(),
                "Train_Loss_PairMargin": loss_pair_margin.item(),
                "Train_Loss_Contrastive": loss_contrastive.item(),

                # Distance 통계 (step) 
                "Step_MeanDist_Path": mean_path_step,
                "Step_MeanDist_Benign": mean_benign_step,
                "Step_Dist_Gap": gap_step,

                # Grad / LR 
                "Grad_Norm": total_norm,
                "LR": optimizer.param_groups[0]["lr"],
            }

            wandb.log(log_dict)
            global_step += 1

        # Evaluation
        model.eval()
        all_labels, all_dists = [], []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Eval {epoch+1}"):
                ref_ids = batch["ref_ids"].to(CONFIG["device"])
                ref_local = batch["ref_local"].to(CONFIG["device"])
                var_ids = batch["var_ids"].to(CONFIG["device"])
                var_local = batch["var_local"].to(CONFIG["device"])

                labels_np = batch["label"].numpy()

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    emb_ref = model(ref_ids, ref_local, return_all=False)
                    emb_var = model(var_ids, var_local, return_all=False)
                    sim = F.cosine_similarity(emb_ref, emb_var)
                    dist = 1.0 - sim

                all_labels.extend(labels_np.tolist())
                all_dists.extend(dist.cpu().numpy().tolist())

        epoch_metrics = calculate_metrics(all_labels, all_dists)
        print(f"[Epoch {epoch+1}] Metrics: {epoch_metrics}")

        # wandb Histogram용 테이블 
        data = [
            [d, int(l)]
            for d, l in zip(all_dists, all_labels)
        ]
        table = wandb.Table(data=data, columns=["distance", "label"])

        # Path / Benign 별 히스토그램용 테이블
        all_labels_np = np.array(all_labels).astype(float)
        all_dists_np = np.array(all_dists).astype(float)
        path_mask_ep = all_labels_np == 1.0
        benign_mask_ep = all_labels_np == 0.0

        path_dists_ep = all_dists_np[path_mask_ep]
        benign_dists_ep = all_dists_np[benign_mask_ep]

        path_table = None
        benign_table = None
        if path_dists_ep.size > 0:
            path_table = wandb.Table(
                data=[[float(d)] for d in path_dists_ep],
                columns=["distance"]
            )
        if benign_dists_ep.size > 0:
            benign_table = wandb.Table(
                data=[[float(d)] for d in benign_dists_ep],
                columns=["distance"]
            )

        # Epoch-level Dist Gap
        epoch_gap = epoch_metrics["CD_Path"] - epoch_metrics["CD_Benign"]

        log_epoch = {
            # Wandb Epoch Logs
            "Epoch_CD": epoch_metrics["CD_Raw"],
            "Epoch_CDD": epoch_metrics["CDD_Raw"],
            "Epoch_PCC": epoch_metrics["PCC_Raw"],
            "Epoch_CD_Path": epoch_metrics["CD_Path"],
            "Epoch_CD_Benign": epoch_metrics["CD_Benign"],
            "Epoch_Normal_CD": epoch_metrics["Normal_CD"],
            "Epoch_Normal_CDD": epoch_metrics["Normal_CDD"],
            "Epoch_Normal_PCC": epoch_metrics["Normal_PCC"],
            "Epoch_FinalScore": epoch_metrics["Final_Score"],
            "Distance_Distribution": wandb.plot.histogram(
                table, "distance", title="Pathogenic vs Benign Distance (All)"
            ),

            # 추가 Epoch Logs 
            "Epoch_Dist_Gap": epoch_gap,
        }

        # Path/Benign 별 히스토그램
        if path_table is not None:
            log_epoch["Epoch_Path_DistHist"] = wandb.plot.histogram(
                path_table, "distance", title="Pathogenic Distance Only"
            )
        if benign_table is not None:
            log_epoch["Epoch_Benign_DistHist"] = wandb.plot.histogram(
                benign_table, "distance", title="Benign Distance Only"
            )

        wandb.log(log_epoch)

        # Best 모델 저장 
        if epoch_metrics["Final_Score"] > best_score:
            best_score = epoch_metrics["Final_Score"]
            print(f"New Best Score {best_score:.4f} — Saving model...")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                model.state_dict(),
                "checkpoints/best_hybrid_model.pt",
            )

        model.train()

    wandb.finish()


if __name__ == "__main__":
    main()
