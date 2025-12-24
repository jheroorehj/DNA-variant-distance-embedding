#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

CONFIG = {
    "model_name": "LongSafari/hyenadna-large-1m-seqlen-hf",
    "ckpt_path": "checkpoints/best_hybrid_model.pt",
    "max_length": 1024,
    "local_window": 64,
    "batch_size": 128,
    "num_workers": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dim": 2048,
}

torch.set_float32_matmul_precision("high")

# Test Dataset (test.csv = ID, seq)
class TestDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=1024, local=64):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.local = local
        self.map={"A":0,"C":1,"G":2,"T":3}

    def one_hot(self,seq):
        seq=seq.upper()
        mat=np.zeros((4,len(seq)),np.float32)
        for i,b in enumerate(seq):
            if b in self.map: mat[self.map[b],i]=1.0
        return mat

    def process(self,seq):
        tok=self.tokenizer(
            seq,max_length=self.max_len,
            truncation=True,padding="max_length",
            return_tensors="pt"
        )
        c=len(seq)//2; h=self.local//2
        s=max(0,c-h); e=s+self.local
        local=seq[s:e]+"N"*(max(0,self.local-len(seq[s:e])))
        return tok["input_ids"].squeeze(0), torch.tensor(self.one_hot(local),dtype=torch.float32)

    def __getitem__(self,idx):
        seq=self.data.iloc[idx]["seq"]
        ids,local=self.process(seq)
        return {"ID":self.data.iloc[idx]["ID"],"ids":ids,"local":local}

    def __len__(self): return len(self.data)

# MODEL (Train과 layer 이름 일치)
class HybridDNA(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.backbone=AutoModelForCausalLM.from_pretrained(
            base,trust_remote_code=True,torch_dtype=torch.bfloat16
        )
        d=self.backbone.config.d_model

        peft=LoraConfig(inference_mode=False,r=32,lora_alpha=64,lora_dropout=0.1,
                        target_modules=["out_proj","dense","c_fc","c_proj"])
        self.backbone=get_peft_model(self.backbone,peft)

        self.hyena_proj=nn.Linear(d,1024)                 # train과 동일한 layer명
        self.cnn=nn.Sequential(
            nn.Conv1d(4,64,5,padding=2),nn.BatchNorm1d(64),nn.GELU(),
            nn.Conv1d(64,128,3,padding=1),nn.BatchNorm1d(128),nn.GELU(),
            nn.MaxPool1d(2),nn.Flatten(),
            nn.Linear(128*32,1024),nn.GELU(),
        )
        self.final_proj=nn.Linear(2048,CONFIG["output_dim"])  # train과 동일

    def forward(self,ids,local):
        h=self.backbone(ids,output_hidden_states=True).hidden_states[-1].float()
        g=self.hyena_proj(h.mean(1))
        if local.ndim==2: local=local.unsqueeze(0)
        if local.shape[1]!=4: local=local.permute(0,2,1)
        l=self.cnn(local.float())
        emb=torch.cat([g,l],1)
        return F.normalize(self.final_proj(emb),p=2,dim=1)

# INFERENCE 
def inference():
    print("Loading tokenizer & model ...")
    tok=AutoTokenizer.from_pretrained(CONFIG["model_name"],trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token

    model=HybridDNA(CONFIG["model_name"])
    print(f"Loading checkpoint -> {CONFIG['ckpt_path']}")
    state=torch.load(CONFIG["ckpt_path"],map_location="cpu")
    model.load_state_dict(state,strict=True)  
    model.to(CONFIG["device"]); model.eval()

    ds=TestDataset("data/test.csv",tok,CONFIG["max_length"],CONFIG["local_window"])
    loader=DataLoader(ds,batch_size=CONFIG["batch_size"],shuffle=False,
                      num_workers=CONFIG["num_workers"],pin_memory=True)

    EMB=[]; IDs=[]
    print("🚀 Running inference ...")
    with torch.no_grad():
        for b in tqdm(loader):
            ids=b["ids"].to(CONFIG["device"])
            loc=b["local"].to(CONFIG["device"])
            emb=model(ids,loc).cpu().numpy()
            EMB.append(emb); IDs+=b["ID"]

    emb=np.vstack(EMB).astype(np.float16)        

    print(f"Saving submission -> submission.csv   shape={emb.shape}")
    cols=[f"emb_{i:04d}" for i in range(CONFIG["output_dim"])]
    df=pd.DataFrame(emb,columns=cols); df.insert(0,"ID",IDs)
    os.makedirs("submission",exist_ok=True)
    df.to_csv("submission/submission.csv",index=False)

    print("Submission Ready! (float16 compressed)")
    

if __name__=="__main__":
    inference()
