#!/usr/bin/env python
# coding: utf-8
# ESM-2 LoRA fine-tuning — per residue binding site classification
# with cross-chain attention + ContactHead + contact/attention supervision
# Updated to match PT5_train_crossattn_matrix.py architecture exactly.

import os
import os.path
import re
import copy
import random
import pickle
import multiprocessing as mp
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import transformers, datasets
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import (
    EsmModel, EsmTokenizer,
    TrainingArguments, Trainer, set_seed,
)
from evaluate import load
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

if "RANK" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10022"
    os.environ["RANK"]        = "0"
    os.environ["LOCAL_RANK"]  = "0"
    os.environ["WORLD_SIZE"]  = "1"

os.chdir("/home/nurit.haspel/Bert/Fine-Tuning/")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRAConfig:
    def __init__(self, rank=16):
        self.lora_rank             = rank
        self.lora_init_scale       = 0.01
        self.lora_modules          = r".*attention\.(self|output)"
        self.lora_layers           = r"query|key|value|dense"
        self.trainable_param_names = r".*layer_norm.*|.*lora_[ab].*|.*multi_lora.*"
        self.lora_scaling_rank     = 1


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, scaling_rank, init_scale):
        super().__init__()
        self.in_features  = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank          = rank
        self.scaling_rank  = scaling_rank
        self.weight        = linear_layer.weight
        self.bias          = linear_layer.bias
        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.randn(rank, linear_layer.in_features) * init_scale)
            self.lora_b = nn.Parameter(torch.zeros(linear_layer.out_features, rank) if init_scale >= 0
                                       else torch.randn(linear_layer.out_features, rank) * init_scale)
        if self.scaling_rank:
            self.multi_lora_a = nn.Parameter(
                torch.ones(self.scaling_rank, linear_layer.in_features)
                + torch.randn(self.scaling_rank, linear_layer.in_features) * init_scale)
            self.multi_lora_b = nn.Parameter(
                torch.zeros(linear_layer.out_features, self.scaling_rank))

    def forward(self, input):
        weight = self.weight
        if self.scaling_rank:
            weight = weight * torch.matmul(self.multi_lora_b, self.multi_lora_a) / self.scaling_rank
        if self.rank > 0:
            weight = weight + torch.matmul(self.lora_b, self.lora_a) / self.rank
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scaling_rank={self.scaling_rank}")


def modify_with_lora(transformer, config):
    n_replaced = 0
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.lora_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.lora_layers, c_name):
                    assert isinstance(layer, nn.Linear), \
                        f"Expected Linear layer but got {type(layer)}"
                    setattr(module, c_name,
                            LoRALinear(layer, config.lora_rank,
                                       config.lora_scaling_rank,
                                       config.lora_init_scale))
                    n_replaced += 1
    print(f"  LoRA: replaced {n_replaced} linear layers (rank={config.lora_rank})")
    if n_replaced == 0:
        # Print first 20 module names to debug regex mismatch
        print("  [WARN] No LoRA layers replaced! Sample module names:")
        for i, (m_name, _) in enumerate(transformer.named_modules()):
            if i < 20:
                print(f"    {m_name}")
    return transformer


# ── Cross-chain attention ─────────────────────────────────────────────────────

class CrossChainAttention(nn.Module):
    """
    Bidirectional cross-attention between chain A and chain B.
    Returns enriched hidden states AND head-averaged attention weights
    for use by the ContactHead.
    """
    def __init__(self, hidden_size=1280, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_a = nn.LayerNorm(hidden_size)
        self.norm_b = nn.LayerNorm(hidden_size)
        self.drop   = nn.Dropout(dropout)

    def _attend(self, q_proj, k_proj, v_proj, out_proj, norm,
                query, key_val, residual):
        B, Nq, d = query.shape
        Nk = key_val.shape[1]
        h, hd = self.num_heads, self.head_dim
        Q = q_proj(query).view(B, Nq, h, hd).transpose(1, 2)
        K = k_proj(key_val).view(B, Nk, h, hd).transpose(1, 2)
        V = v_proj(key_val).view(B, Nk, h, hd).transpose(1, 2)
        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.drop(attn)
        attn_avg = attn.mean(dim=1)   # (B, Nq, Nk) averaged over heads
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Nq, d)
        return norm(residual + out_proj(out)), attn_avg

    def forward(self, h_a, h_b):
        h_a_prime, attn_ab = self._attend(
            self.q_a, self.k_b, self.v_b, self.out_a, self.norm_a,
            query=h_a, key_val=h_b, residual=h_a)
        h_b_prime, attn_ba = self._attend(
            self.q_b, self.k_a, self.v_a, self.out_b, self.norm_b,
            query=h_b, key_val=h_a, residual=h_b)
        return h_a_prime, h_b_prime, attn_ab, attn_ba


# ── Contact head ──────────────────────────────────────────────────────────────

class ContactHead(nn.Module):
    """
    Predicts contact probability for each (i,j) in the binding sub-matrix
    from cross-attention weights. Uses log-relative normalisation.
    Architecture: MLP(2→64→32→1), bias=-2.0 for sparse initialisation.
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.mlp[-1].bias.data.fill_(-2.0)

    def forward(self, attn_ab, attn_ba):
        attn_ba_t = attn_ba.transpose(1, 2)
        def log_relative(x):
            mean = x.mean(dim=(-2, -1), keepdim=True)
            return torch.log(x / (mean + 1e-8) + 1e-8)
        feats  = torch.stack([log_relative(attn_ab),
                               log_relative(attn_ba_t)], dim=-1)
        return self.mlp(feats).squeeze(-1)


# ── Classification model ──────────────────────────────────────────────────────

class ClassConfig:
    def __init__(self, dropout=0.2, num_labels=2, num_heads=8,
                 cross_attn_dropout=0.1, detach_contact=False,
                 use_cross_attn=True):
        self.dropout_rate       = dropout
        self.num_labels         = num_labels
        self.num_heads          = num_heads
        self.cross_attn_dropout = cross_attn_dropout
        self.detach_contact     = detach_contact
        self.use_cross_attn     = use_cross_attn


class ESM2ForTokenClassificationWithCrossAttn(nn.Module):
    """
    ESM-2 encoder + LoRA + cross-chain attention + ContactHead + classifier.

    Uses token_type_ids (0=chain A, 1=chain B) to split chains — cleaner
    than separator-token splitting since ESM-2 natively supports segments.

    Matches PT5_train_crossattn_matrix.py feature-for-feature:
      - Sub-matrix contact prediction (only predicted binding residues)
      - Log-relative attention normalisation in ContactHead
      - Contact supervision loss (BCEWithLogitsLoss + pos_weight)
      - Attention supervision loss (KL divergence)
      - bind_thresh fallback to top-20% if too few predicted binding residues
    """
    def __init__(self, model_name, class_config):
        super().__init__()
        self.num_labels     = class_config.num_labels
        self.detach_contact = class_config.detach_contact
        self.use_cross_attn = class_config.use_cross_attn

        self.esm = EsmModel.from_pretrained(model_name, ignore_mismatched_sizes=True)
        hidden_size = self.esm.config.hidden_size   # 1280 for esm2_t33_650M

        self.cross_attn   = CrossChainAttention(hidden_size, class_config.num_heads,
                                                 class_config.cross_attn_dropout)
        self.contact_head = ContactHead()
        self.dropout      = nn.Dropout(class_config.dropout_rate)
        self.classifier   = nn.Linear(hidden_size, class_config.num_labels)

    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, labels=None, **kwargs):

        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        hidden = outputs.last_hidden_state   # (B, L, d)

        contact_logits_list = []

        if self.use_cross_attn and token_type_ids is not None:
            B, L, d = hidden.shape
            enriched = []

            for i in range(B):
                mask_a = token_type_ids[i] == 0
                mask_b = token_type_ids[i] == 1

                h_a = hidden[i][mask_a].unsqueeze(0)   # (1, n, d)
                h_b = hidden[i][mask_b].unsqueeze(0)   # (1, m, d)

                if h_a.shape[1] == 0 or h_b.shape[1] == 0:
                    enriched.append(hidden[i])
                    continue

                h_a_e, h_b_e, attn_ab, attn_ba = self.cross_attn(h_a, h_b)

                # ── Sub-matrix: restrict contacts to predicted binding residues ──
                logits_so_far = self.classifier(self.dropout(hidden[i]))
                p_bind_a = torch.softmax(logits_so_far[mask_a], dim=-1)[:, 1]
                p_bind_b = torch.softmax(logits_so_far[mask_b], dim=-1)[:, 1]

                BIND_THRESH = getattr(self, 'bind_thresh', 0.3)
                mask_bind_a = (p_bind_a.detach() >= BIND_THRESH)
                mask_bind_b = (p_bind_b.detach() >= BIND_THRESH)

                # Fallback: if too few predicted binding residues, use top-20%
                if mask_bind_a.sum() < 3:
                    k = max(3, int(0.2 * len(p_bind_a)))
                    topk = p_bind_a.detach().topk(k).indices
                    mask_bind_a = torch.zeros_like(mask_bind_a)
                    mask_bind_a[topk] = True
                if mask_bind_b.sum() < 3:
                    k = max(3, int(0.2 * len(p_bind_b)))
                    topk = p_bind_b.detach().topk(k).indices
                    mask_bind_b = torch.zeros_like(mask_bind_b)
                    mask_bind_b[topk] = True

                # Slice attention to binding sub-matrix
                attn_ab_in = attn_ab[:, mask_bind_a, :][:, :, mask_bind_b]
                attn_ba_in = attn_ba[:, mask_bind_b, :][:, :, mask_bind_a]
                if self.detach_contact:
                    attn_ab_in = attn_ab_in.detach()
                    attn_ba_in = attn_ba_in.detach()

                c_logits = self.contact_head(attn_ab_in, attn_ba_in)
                contact_logits_list.append((
                    c_logits.squeeze(0),      # (n_bind, m_bind)
                    mask_bind_a,              # (n,) chain A binding mask
                    mask_bind_b,              # (m,) chain B binding mask
                    attn_ab.squeeze(0),       # (n, m) full attention for supervision
                ))

                # Scatter enriched states back
                out = hidden[i].clone()
                out[mask_a] = h_a_e.squeeze(0)
                out[mask_b] = h_b_e.squeeze(0)
                enriched.append(out)

            hidden = torch.stack(enriched, dim=0)

        logits = self.classifier(self.dropout(hidden))
        loss   = None
        if labels is not None:
            active     = attention_mask.view(-1) == 1
            act_logits = logits.view(-1, self.num_labels)
            act_labels = torch.where(active, labels.view(-1),
                                     torch.tensor(-100).type_as(labels))
            valid_logits = act_logits[act_labels != -100]
            valid_labels = act_labels[act_labels != -100].long()
            loss = CrossEntropyLoss()(valid_logits.float(), valid_labels)

        output = TokenClassifierOutput(loss=loss, logits=logits)
        if self.training:
            return output, contact_logits_list
        return output


# ── Data collator ─────────────────────────────────────────────────────────────

class FastDataCollatorForTokenClassification:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer          = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        label_list   = [f.pop("labels")          for f in features]
        contact_list = [f.pop("contacts", [])    for f in features]

        batch = self.tokenizer.pad(features, padding=True,
                                   pad_to_multiple_of=self.pad_to_multiple_of,
                                   return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [(list(lbl) + [-100] * max_len)[:max_len] for lbl in label_list]
        )
        batch["contacts"] = contact_list

        for f, lbl, con in zip(features, label_list, contact_list):
            f["labels"]   = lbl
            f["contacts"] = con

        return batch


# ── Model builder ─────────────────────────────────────────────────────────────

ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"   # 650M — best balance of quality/speed


def ESM2_classification_model(num_labels=2, detach_contact=False,
                               use_cross_attn=True, model_name=ESM2_MODEL,
                               lora_rank=16):
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    model = ESM2ForTokenClassificationWithCrossAttn(
        model_name   = model_name,
        class_config = ClassConfig(num_labels=num_labels, num_heads=8,
                                   cross_attn_dropout=0.1,
                                   detach_contact=detach_contact,
                                   use_cross_attn=use_cross_attn),
    )

    lora_cfg = LoRAConfig(rank=lora_rank)
    model    = modify_with_lora(model, lora_cfg)
    model.lora_rank = lora_cfg.lora_rank

    # Freeze ESM-2 encoder; keep LoRA + layer norms + cross_attn + contact_head + classifier
    for p in model.esm.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if re.fullmatch(lora_cfg.trainable_param_names, name):
            p.requires_grad = True
    for name, p in model.named_parameters():
        if any(k in name for k in ["lm_head", "position_ids"]):
            p.requires_grad = False
    for p in model.cross_attn.parameters():   p.requires_grad = True
    for p in model.contact_head.parameters(): p.requires_grad = True
    for p in model.classifier.parameters():   p.requires_grad = True

    trainable     = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    cross_params  = sum(np.prod(p.size()) for p in model.cross_attn.parameters())
    contact_params = sum(np.prod(p.size()) for p in model.contact_head.parameters())
    lora_params   = sum(np.prod(p.size()) for n, p in model.named_parameters()
                        if p.requires_grad and 'lora' in n.lower())
    other_params  = trainable - lora_params
    print(f"ESM2_LoRA_CrossAttn+Contact_Classifier ({model_name})")
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  LoRA params          : {lora_params:,}")
    print(f"  Other trainable      : {other_params:,}")
    print(f"  cross-attention      : {cross_params:,}")
    print(f"  contact head         : {contact_params:,}")
    print(f"  LoRA rank            : {model.lora_rank}\n")
    return model, tokenizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seeds(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s); set_seed(s)


def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, matthews_corrcoef
    preds, labels = eval_pred
    labels = labels.reshape(-1)
    preds  = np.argmax(preds, axis=2).reshape(-1)
    preds  = preds[labels != -100]
    labels = labels[labels != -100]
    return {
        "accuracy": float(np.mean(preds == labels)),
        "f1":       float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "mcc":      float(matthews_corrcoef(labels, preds)),
    }


def load_pairs_csv(filepath):
    string_cols = ['pair_id', 'pdb_id', 'chain_A', 'chain_B',
                   'seq_A', 'seq_B', 'label_A', 'label_B', 'contacts']
    df = pd.read_csv(filepath,
                     dtype={c: str for c in string_cols},
                     lineterminator='\n')
    required = ['pair_id', 'chain_A', 'chain_B', 'seq_A', 'seq_B', 'label_A', 'label_B']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    before = len(df)
    df = df.dropna(subset=['seq_A', 'seq_B', 'label_A', 'label_B'])
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing values")
    df['seq_A'] = df['seq_A'].str.replace('?', 'X', regex=False)
    df['seq_B'] = df['seq_B'].str.replace('?', 'X', regex=False)

    def parse_labels(s):
        return [int(c) if c in ('0','1') else -100 for c in str(s)]

    df['label_A'] = df['label_A'].apply(parse_labels)
    df['label_B'] = df['label_B'].apply(parse_labels)

    def parse_contacts(s):
        if not isinstance(s, str) or s.strip() == '':
            return []
        return [tuple(int(x) for x in pair.split(','))
                for pair in s.split(';') if pair.strip()]

    if 'contacts' in df.columns:
        df['contacts'] = df['contacts'].apply(parse_contacts)
    else:
        df['contacts'] = [[] for _ in range(len(df))]

    df = df.rename(columns={
        'seq_A': 'seq_a', 'seq_B': 'seq_b',
        'label_A': 'label_a', 'label_B': 'label_b',
    })
    print(f"  Loaded {len(df)} pairs from {filepath}")
    has_contacts = df['contacts'].apply(len).sum() > 0
    print(f"  Contact labels: {'present' if has_contacts else 'absent'}")
    return df


def create_dataset_pairs(tokenizer, df, max_length=1024):
    """
    Tokenize protein pairs for ESM-2 with token_type_ids.
    ESM-2 format: [CLS] chain_A [EOS] chain_B [EOS]
    token_type_ids: 0 for chain A tokens (including CLS + first EOS),
                    1 for chain B tokens (including second EOS)
    max_length=1024: half_max = (1024-3)//2 = 510 tokens per chain
    (increased from 512 to match ProstT5's effective per-chain capacity)
    """
    half_max = (max_length - 3) // 2
    cls_id   = tokenizer.cls_token_id
    sep_id   = tokenizer.eos_token_id
    RARE     = str.maketrans("OUBZ", "XXXX")

    all_input_ids      = []
    all_attention_mask = []
    all_token_type_ids = []
    all_labels         = []
    all_contacts       = []

    for _, row in df.iterrows():
        seq_a    = row['seq_a'].upper().translate(RARE)
        seq_b    = row['seq_b'].upper().translate(RARE)
        lbl_a    = list(row['label_a'])
        lbl_b    = list(row['label_b'])
        contacts = list(row.get('contacts', []))

        ids_a = tokenizer(seq_a, add_special_tokens=False)['input_ids']
        ids_b = tokenizer(seq_b, add_special_tokens=False)['input_ids']

        # Truncate independently
        trunc_a = min(len(ids_a), half_max)
        trunc_b = min(len(ids_b), half_max)
        ids_a = ids_a[:trunc_a]; lbl_a = lbl_a[:trunc_a]
        ids_b = ids_b[:trunc_b]; lbl_b = lbl_b[:trunc_b]
        contacts = [(i, j) for i, j in contacts if i < trunc_a and j < trunc_b]

        # [CLS] chain_A [EOS] chain_B [EOS]
        input_ids = [cls_id] + ids_a + [sep_id] + ids_b + [sep_id]
        labels    = [-100]   + lbl_a + [-100]   + lbl_b + [-100]
        n_a, n_b  = len(ids_a), len(ids_b)
        token_type_ids = [0] * (1 + n_a + 1) + [1] * (n_b + 1)

        all_input_ids.append(input_ids)
        all_attention_mask.append([1] * len(input_ids))
        all_token_type_ids.append(token_type_ids)
        all_labels.append(labels)
        all_contacts.append(contacts)

    return Dataset.from_dict({
        'input_ids':       all_input_ids,
        'attention_mask':  all_attention_mask,
        'token_type_ids':  all_token_type_ids,
        'labels':          all_labels,
        'contacts':        all_contacts,
    })


# ── Trainer ───────────────────────────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """
    Class-weighted cross-entropy + label smoothing + contact loss
    + attention supervision. Identical loss logic to PT5 trainer.
    Custom _save/_load_from_checkpoint to avoid HuggingFace save_pretrained.
    """
    def __init__(self, *args, class_weights=None, label_smoothing=0.05,
                 contact_lambda=1.0, pos_weight_cap=10.0,
                 attn_supervision_lambda=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights           = class_weights
        self.label_smoothing         = label_smoothing
        self.contact_lambda          = contact_lambda
        self.pos_weight_cap          = pos_weight_cap
        self.attn_supervision_lambda = attn_supervision_lambda

    def _save(self, output_dir, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)
        trainable = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        torch.save(trainable, os.path.join(output_dir, "trainable_weights.pt"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        weights_path = os.path.join(resume_from_checkpoint, "trainable_weights.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"No trainable_weights.pt in {resume_from_checkpoint}")
        if model is None:
            model = self.model
        saved = torch.load(weights_path, map_location="cpu")
        model_params = dict(model.named_parameters())
        missing, unexpected = [], []
        for name, param in saved.items():
            if name in model_params:
                model_params[name].data = param.data
            else:
                unexpected.append(name)
        for name in model_params:
            if model_params[name].requires_grad and name not in saved:
                missing.append(name)
        if unexpected: print(f"  [WARN] Unexpected keys: {unexpected}")
        if missing:    print(f"  [WARN] Missing keys: {missing}")
        print(f"  Loaded trainable weights from {weights_path}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels   = inputs.pop("labels")
        contacts = inputs.pop("contacts", None)

        result = model(**inputs)
        if isinstance(result, tuple):
            outputs, contact_logits_list = result
        else:
            outputs, contact_logits_list = result, []
        logits = outputs.logits

        weights      = self.class_weights.to(logits.device) if self.class_weights is not None else None
        active       = inputs["attention_mask"].view(-1) == 1
        act_logits   = logits.view(-1, self.model.num_labels)
        act_labels   = torch.where(active, labels.view(-1),
                                   torch.tensor(-100).type_as(labels))
        valid_logits = act_logits[act_labels != -100]
        valid_labels = act_labels[act_labels != -100].long()

        # ── Per-residue loss with label smoothing ─────────────────────────────
        num_classes = valid_logits.size(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_labels = torch.zeros_like(valid_logits)
                smooth_labels.fill_(self.label_smoothing / num_classes)
                smooth_labels.scatter_(1, valid_labels.unsqueeze(1),
                                       1.0 - self.label_smoothing + self.label_smoothing / num_classes)
            log_probs = F.log_softmax(valid_logits.float(), dim=-1)
            if weights is not None:
                loss = (-(smooth_labels * log_probs).sum(dim=-1) * weights[valid_labels]).mean()
            else:
                loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        else:
            loss = CrossEntropyLoss(weight=weights)(valid_logits.float(), valid_labels)

        # ── Pairwise contact loss ─────────────────────────────────────────────
        if (self.contact_lambda > 0 and contacts is not None
                and len(contact_logits_list) > 0):

            contact_losses  = []
            attn_sup_losses = []

            for item, pair_contacts in zip(contact_logits_list, contacts):
                if len(item) == 4:
                    c_logits, mask_a, mask_b, attn_ab = item
                else:
                    c_logits, mask_a, mask_b = item; attn_ab = None

                n, m = c_logits.shape
                if n == 0 or m == 0 or len(pair_contacts) == 0:
                    continue

                target = torch.zeros(n, m, device=c_logits.device)
                if mask_a is not None:
                    full_n = mask_a.shape[0]; full_m = mask_b.shape[0]
                    sub_idx_a = torch.full((full_n,), -1, dtype=torch.long, device=c_logits.device)
                    sub_idx_b = torch.full((full_m,), -1, dtype=torch.long, device=c_logits.device)
                    sub_idx_a[mask_a] = torch.arange(mask_a.sum(), device=c_logits.device)
                    sub_idx_b[mask_b] = torch.arange(mask_b.sum(), device=c_logits.device)
                    ij = torch.tensor(pair_contacts, dtype=torch.long, device=c_logits.device)
                    valid = (ij[:, 0] < full_n) & (ij[:, 1] < full_m)
                    ij = ij[valid]
                    if len(ij) > 0:
                        si = sub_idx_a[ij[:, 0]]; sj = sub_idx_b[ij[:, 1]]
                        in_sub = (si >= 0) & (sj >= 0)
                        ij_sub = torch.stack([si[in_sub], sj[in_sub]], dim=1)
                        if len(ij_sub) > 0:
                            target[ij_sub[:, 0], ij_sub[:, 1]] = 1.0

                n_pos = target.sum(); n_neg = n * m - n_pos
                if n_pos > 0:
                    pos_weight = torch.clamp(n_neg / n_pos, max=self.pos_weight_cap).detach()
                    pair_loss  = F.binary_cross_entropy_with_logits(
                        c_logits.float(), target, pos_weight=pos_weight)
                    contact_losses.append(pair_loss)

                    # ── Attention supervision ──────────────────────────────────
                    if self.attn_supervision_lambda > 0 and attn_ab is not None:
                        n_full, m_full = attn_ab.shape
                        attn_target = torch.zeros(n_full, m_full, device=attn_ab.device)
                        all_ij = torch.tensor(pair_contacts, dtype=torch.long, device=attn_ab.device)
                        v = (all_ij[:, 0] < n_full) & (all_ij[:, 1] < m_full)
                        all_ij = all_ij[v]
                        if len(all_ij) > 0:
                            attn_target[all_ij[:, 0], all_ij[:, 1]] = 1.0
                            row_sums = attn_target.sum(dim=-1, keepdim=True).clamp(min=1)
                            attn_target = attn_target / row_sums
                            log_attn    = torch.log(attn_ab.float().clamp(min=1e-8))
                            attn_sup_losses.append(
                                F.kl_div(log_attn, attn_target, reduction='batchmean'))

            if contact_losses:
                loss = loss + self.contact_lambda * torch.stack(contact_losses).mean()
            if attn_sup_losses:
                loss = loss + self.attn_supervision_lambda * torch.stack(attn_sup_losses).mean()

        return (loss, outputs) if return_outputs else loss


def save_model(model, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    non_frozen = {n: p for n, p in model.named_parameters() if p.requires_grad}
    torch.save(non_frozen, filepath)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="ESM-2 cross-attention + contact matrix training")
    parser.add_argument("--model-out",    default="./ESM2_pairs_finetuned.pth")
    parser.add_argument("--csv",          default="/home/nurit.haspel/Bert/Fine-Tuning/splits_bind/dense_nonred_matrix.csv")
    parser.add_argument("--probs-out-dir", default=".")
    parser.add_argument("--contact-out",  default="test_contact_matrices_esm2.pkl")
    parser.add_argument("--no-cross-attn", action="store_true")
    parser.add_argument("--class-weights", type=float, nargs=2, default=[1.0, 3.0])
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--bind-thresh",  type=float, default=0.3)
    parser.add_argument("--attn-supervision-lambda", type=float, default=0.5)
    parser.add_argument("--contact-lambda", type=float, default=1.0)
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--detach-contact", action="store_true")
    parser.add_argument("--epochs",       type=int, default=12)
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--checkpoint-dir", default="./checkpoints_esm2")
    parser.add_argument("--save-every",   type=int, default=1)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--lora-rank",    type=int, default=16,
                        help="LoRA rank for encoder fine-tuning (default: 16)")
    parser.add_argument("--model-name",   type=str,
                        default="facebook/esm2_t33_650M_UR50D",
                        help="ESM-2 model name (default: esm2_t33_650M_UR50D). "
                             "Use facebook/esm2_t36_3B_UR50D for 3B model.")
    cli = parser.parse_args()

    CHECKPOINT_DIR = cli.checkpoint_dir
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(cli.probs_out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {cli.csv}...")
    df = load_pairs_csv(cli.csv)

    from sklearn.model_selection import train_test_split
    df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
    df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    print(f"Split: {len(df_train)} train / {len(df_valid)} val / {len(df_test)} test")

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(42)

    model, tokenizer = ESM2_classification_model(
        num_labels=2,
        detach_contact=cli.detach_contact,
        use_cross_attn=not cli.no_cross_attn,
        lora_rank=cli.lora_rank,
        model_name=cli.model_name,
    )
    model.bind_thresh = cli.bind_thresh

    train_set = create_dataset_pairs(tokenizer, df_train, max_length=1024)
    valid_set = create_dataset_pairs(tokenizer, df_valid, max_length=1024)
    test_set  = create_dataset_pairs(tokenizer, df_test,  max_length=1024)

    print(f"\nConfiguration:")
    print(f"  Architecture:     ESM-2 + LoRA (rank={cli.lora_rank})"
          + (" (flat baseline)" if cli.no_cross_attn
             else " + CrossChainAttention + ContactHead"))
    print(f"  Class weights:    {cli.class_weights}")
    print(f"  Label smoothing:  {cli.label_smoothing}")
    print(f"  Contact lambda:   {cli.contact_lambda}")
    print(f"  Attn supervision: {cli.attn_supervision_lambda}")
    print(f"  Bind threshold:   {cli.bind_thresh}")
    print(f"  Pos weight cap:   {cli.pos_weight_cap}")
    print(f"  Epochs:           {cli.epochs}")

    class_weights = torch.tensor(cli.class_weights)
    batch, accum, lr, num_workers = 4, 8, 3e-4, 28
    steps_per_epoch = max(1, len(train_set) // (batch * accum))
    warmup_steps    = max(1, int(steps_per_epoch * cli.epochs * 0.05))

    resume_from = None
    if cli.resume:
        ckpt_path = Path(CHECKPOINT_DIR)
        if ckpt_path.exists():
            checkpoints = sorted(
                [d for d in ckpt_path.iterdir()
                 if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[-1])
            )
            if checkpoints:
                resume_from = str(checkpoints[-1])
                print(f"  Resuming from: {resume_from}")

    args = TrainingArguments(
        CHECKPOINT_DIR,
        eval_strategy="epoch",
        logging_strategy="steps",
        save_strategy="epoch",
        save_steps=cli.save_every,
        save_total_limit=cli.keep_checkpoints,
        load_best_model_at_end=True,
        save_on_each_node=False,
        metric_for_best_model="mcc",
        greater_is_better=True,
        logging_steps=50,
        logging_first_step=True,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=cli.epochs,
        seed=42,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = WeightedLossTrainer(
        model, args,
        class_weights=class_weights,
        label_smoothing=cli.label_smoothing,
        contact_lambda=cli.contact_lambda,
        pos_weight_cap=cli.pos_weight_cap,
        attn_supervision_lambda=cli.attn_supervision_lambda,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=FastDataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save model ────────────────────────────────────────────────────────────
    save_model(model, cli.model_out)
    print(f"\nModel saved to {cli.model_out}")

    # ── Threshold sweep on validation set ────────────────────────────────────
    from sklearn.metrics import f1_score, matthews_corrcoef, classification_report

    model.to(device)
    model.eval()

    def collect_probs(loader):
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(loader):
                lbl = batch["labels"]
                tti = batch["token_type_ids"].to(device) if "token_type_ids" in batch else None
                out = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=tti,
                )
                probs = F.softmax(out.logits, dim=-1).cpu().numpy()
                for i in range(len(lbl)):
                    L = lbl[i]
                    L_arr = np.asarray(L, dtype=np.int64)
                    mask = L_arr != -100
                    all_probs.append(probs[i][mask])
                    all_labels.append(L_arr[mask])
        return (np.concatenate(all_probs), np.concatenate(all_labels))

    val_loader  = DataLoader(valid_set, batch_size=16, shuffle=False, num_workers=0,
                              collate_fn=FastDataCollatorForTokenClassification(tokenizer))
    test_loader = DataLoader(test_set,  batch_size=16, shuffle=False, num_workers=0,
                              collate_fn=FastDataCollatorForTokenClassification(tokenizer))

    val_probs,  val_labels  = collect_probs(val_loader)
    test_probs, test_labels = collect_probs(test_loader)

    probs_dir = Path(cli.probs_out_dir)
    np.save(str(probs_dir / "val_probs.npy"),   val_probs)
    np.save(str(probs_dir / "val_labels.npy"),  val_labels)
    np.save(str(probs_dir / "test_probs.npy"),  test_probs)
    np.save(str(probs_dir / "test_labels.npy"), test_labels)
    print(f"Saved probabilities to {probs_dir}/")

    # ── Threshold-independent metrics ─────────────────────────────────────────
    from sklearn.metrics import (f1_score, matthews_corrcoef, classification_report,
                                 roc_auc_score, average_precision_score)

    test_pos_probs = test_probs[:, 1]
    auroc = roc_auc_score(test_labels, test_pos_probs)
    auprc = average_precision_score(test_labels, test_pos_probs)
    print(f"\n── Threshold-independent metrics ────────────────────────")
    print(f"AUROC    : {auroc:.4f}")
    print(f"AUPRC    : {auprc:.4f}")
    print(f"  (baseline AUPRC for random = {test_labels.mean():.4f})")
    print(f"  (AUPRC lift over random   = {auprc / test_labels.mean():.2f}x)")

    # Default threshold
    test_preds_default = (test_pos_probs >= 0.5).astype(int)
    print(f"\n── Test results (default threshold=0.5) ─────────────────")
    print(f"Accuracy : {accuracy_score(test_labels, test_preds_default):.4f}")
    print(f"F1       : {f1_score(test_labels, test_preds_default, pos_label=1, zero_division=0):.4f}")
    print(f"MCC      : {matthews_corrcoef(test_labels, test_preds_default):.4f}")
    print(classification_report(test_labels, test_preds_default,
                                target_names=["non-binding (0)", "binding site (1)"],
                                zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(test_labels, test_preds_default))

    # Threshold sweep on validation set
    print("\nRunning threshold sweep on validation set...")
    val_pos = val_probs[:, 1]
    best_t, best_mcc = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (val_pos >= t).astype(int)
        mcc   = matthews_corrcoef(val_labels, preds)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    print(f"Best threshold: {best_t:.2f}  (val MCC={best_mcc:.4f})")

    test_preds_tuned = (test_pos_probs >= best_t).astype(int)
    print(f"\n── Test results (tuned threshold={best_t:.2f}) ──────────────────")
    print(f"Accuracy : {accuracy_score(test_labels, test_preds_tuned):.4f}")
    print(f"F1       : {f1_score(test_labels, test_preds_tuned, pos_label=1, zero_division=0):.4f}")
    print(f"MCC      : {matthews_corrcoef(test_labels, test_preds_tuned):.4f}")
    print(f"AUROC    : {auroc:.4f}  (threshold-independent, same as above)")
    print(f"AUPRC    : {auprc:.4f}  (threshold-independent, same as above)")
    print(classification_report(test_labels, test_preds_tuned,
                                target_names=["non-binding (0)", "binding site (1)"],
                                zero_division=0))
    print("Confusion matrix (tuned):\n", confusion_matrix(test_labels, test_preds_tuned))

    # ── Save predicted contact matrices ──────────────────────────────────────
    if not cli.no_cross_attn:
        print("\nSaving predicted contact matrices...")
        contact_matrices = {}
        model.eval()
        with torch.no_grad():
            for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
                pair_id  = str(row['pair_id'])
                seq_a    = row['seq_a'].upper().translate(str.maketrans("OUBZ","XXXX"))
                seq_b    = row['seq_b'].upper().translate(str.maketrans("OUBZ","XXXX"))
                half_max = (1024 - 3) // 2

                ids_a = tokenizer(seq_a, add_special_tokens=False)['input_ids'][:half_max]
                ids_b = tokenizer(seq_b, add_special_tokens=False)['input_ids'][:half_max]
                n_full, m_full = len(ids_a), len(ids_b)

                cls_id = tokenizer.cls_token_id
                sep_id = tokenizer.eos_token_id
                input_ids = torch.tensor([[cls_id] + ids_a + [sep_id] + ids_b + [sep_id]]).to(device)
                attn_mask = torch.ones_like(input_ids)
                n_a, n_b  = len(ids_a), len(ids_b)
                token_type_ids = torch.tensor(
                    [[0]*(1 + n_a + 1) + [1]*(n_b + 1)]
                ).to(device)

                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            token_type_ids=token_type_ids)
                logits = out.logits[0]

                mask_a = token_type_ids[0] == 0
                mask_b = token_type_ids[0] == 1
                hidden = model.esm(input_ids=input_ids, attention_mask=attn_mask,
                                   token_type_ids=token_type_ids).last_hidden_state
                h_a = hidden[0][mask_a].unsqueeze(0)
                h_b = hidden[0][mask_b].unsqueeze(0)
                _, _, attn_ab, attn_ba = model.cross_attn(h_a, h_b)

                p_bind_a = torch.softmax(logits[mask_a], dim=-1)[:, 1]
                p_bind_b = torch.softmax(logits[mask_b], dim=-1)[:, 1]

                BIND_THRESH = getattr(model, 'bind_thresh', 0.3)
                mb_a = (p_bind_a.detach() >= BIND_THRESH)
                mb_b = (p_bind_b.detach() >= BIND_THRESH)
                if mb_a.sum() < 3:
                    k = max(3, int(0.2 * len(p_bind_a)))
                    topk = p_bind_a.topk(k).indices
                    mb_a = torch.zeros_like(mb_a); mb_a[topk] = True
                if mb_b.sum() < 3:
                    k = max(3, int(0.2 * len(p_bind_b)))
                    topk = p_bind_b.topk(k).indices
                    mb_b = torch.zeros_like(mb_b); mb_b[topk] = True

                attn_sub = attn_ab[:, mb_a, :][:, :, mb_b]
                attn_sub_ba = attn_ba[:, mb_b, :][:, :, mb_a]
                c_logits = model.contact_head(attn_sub, attn_sub_ba).squeeze(0)
                c_prob   = torch.sigmoid(c_logits).cpu().numpy()

                idx_a = mb_a.cpu().numpy().nonzero()[0]
                idx_b = mb_b.cpu().numpy().nonzero()[0]
                full_mat = np.zeros((n_full, m_full), dtype=np.float32)
                full_mat[np.ix_(idx_a, idx_b)] = c_prob
                contact_matrices[pair_id] = full_mat

        contact_out = Path(cli.contact_out)
        contact_out.parent.mkdir(parents=True, exist_ok=True)
        with open(str(contact_out), 'wb') as f:
            pickle.dump({'matrices': contact_matrices,
                         'bind_thresh': getattr(model, 'bind_thresh', 0.3)}, f)
        print(f"Saved contact matrices for {len(contact_matrices)} pairs → {contact_out}")

        # Contact matrix diagnostics
        all_probs_c = np.concatenate([m.flatten() for m in contact_matrices.values()])
        max_probs   = [m.max() for m in contact_matrices.values() if m.size > 0]
        print(f"\n── Contact matrix diagnostics ───────────────────────")
        print(f"  mean prob    : {all_probs_c.mean():.4f}")
        print(f"  mean max     : {np.mean(max_probs):.4f}")
        print(f"  frac > 0.2   : {(all_probs_c > 0.2).mean():.4f}")
        print(f"  frac > 0.5   : {(all_probs_c > 0.5).mean():.4f}")
