#!/usr/bin/env python
# coding: utf-8
# ProtT5 LoRA fine-tuning — per residue classification
# Compatible with: python PT5_train.py  OR  torchrun --nproc_per_node=N PT5_train.py

import os
import os.path
import re
import copy
import random
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import transformers, datasets
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
#from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5EncoderModel, T5Tokenizer
from transformers import TrainingArguments, Trainer, set_seed
from transformers import DataCollatorForTokenClassification
from evaluate import load
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for torchrun
import matplotlib.pyplot as plt
from Bio import SeqIO

# ── Fix 1: spawn must be set at top level, not inside a function ──────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

# ── Fix 2: do NOT hardcode MASTER_ADDR/PORT/RANK when using torchrun ──────────
# torchrun sets these automatically. Only set them for single-process runs.
if "RANK" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10024"
    os.environ["RANK"]        = "0"
    os.environ["LOCAL_RANK"]  = "0"
    os.environ["WORLD_SIZE"]  = "1"

os.chdir("/pomplun/share_home/nurit.haspel/Bert/Fine-Tuning/")

# Reduce CUDA memory fragmentation (recommended by PyTorch OOM message)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRAConfig:
    def __init__(self, rank=8):
        self.lora_rank = rank
        self.lora_init_scale = 0.01
        self.lora_modules = ".*SelfAttention|.*EncDecAttention"
        self.lora_layers = "q|k|v|o"
        self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
        # When rank=0 also disable scaling rank to get a truly frozen encoder
        self.lora_scaling_rank = 0 if rank == 0 else 1


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
                torch.ones(scaling_rank, linear_layer.in_features)
                + torch.randn(scaling_rank, linear_layer.in_features) * init_scale
            )
            self.multi_lora_b = nn.Parameter(
                torch.ones(linear_layer.out_features, scaling_rank)
                + (torch.randn(linear_layer.out_features, scaling_rank) * init_scale if init_scale < 0 else 0)
            )

    def forward(self, input):
        dtype = input.dtype
        if self.scaling_rank == 1 and self.rank == 0:
            hidden = F.linear(
                input * self.multi_lora_a.flatten().to(dtype) if self.multi_lora_a.requires_grad else input,
                self.weight.to(dtype),
                self.bias.to(dtype) if self.bias is not None else None,
            )
            return hidden * self.multi_lora_b.flatten().to(dtype) if self.multi_lora_b.requires_grad else hidden
        weight = self.weight.to(dtype)
        if self.scaling_rank:
            weight = weight * torch.matmul(self.multi_lora_b.to(dtype), self.multi_lora_a.to(dtype)) / self.scaling_rank
        if self.rank:
            weight = weight + torch.matmul(self.lora_b.to(dtype), self.lora_a.to(dtype)) / self.rank
        return F.linear(input, weight, self.bias.to(dtype) if self.bias is not None else None)    

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"


def modify_with_lora(transformer, config):
    n_replaced=0
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
    Bidirectional cross-attention between chain A and chain B representations.

    Given H_A (n × d) and H_B (m × d):
      - Each residue in A attends over all residues in B  → H_A' (n × d)
      - Each residue in B attends over all residues in A  → H_B' (m × d)

    Uses standard scaled dot-product attention with a residual connection
    and layer norm, so the module can be stacked or used as a lightweight add-on.

    Parameters
    ----------
    hidden_size : int   — embedding dimension (1024 for ProstT5)
    num_heads   : int   — number of attention heads (must divide hidden_size)
    dropout     : float — attention dropout
    """
    def __init__(self, hidden_size=1024, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.num_heads  = num_heads
        self.head_dim   = hidden_size // num_heads
        self.scale      = self.head_dim ** -0.5

        # Separate Q/K/V projections for A→B and B→A directions
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
        """Single-direction cross-attention with residual + norm.
        Returns (output, attn_weights) where attn_weights is (B, Nq, Nk)
        averaged over heads — used by ContactHead for pairwise supervision.
        """
        B, Nq, d = query.shape
        Nk       = key_val.shape[1]
        h, hd    = self.num_heads, self.head_dim

        Q = q_proj(query).view(B, Nq, h, hd).transpose(1, 2)
        K = k_proj(key_val).view(B, Nk, h, hd).transpose(1, 2)
        V = v_proj(key_val).view(B, Nk, h, hd).transpose(1, 2)

        attn = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.drop(attn)

        # Average attention weights across heads → (B, Nq, Nk)
        attn_avg = attn.mean(dim=1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, d)
        return norm(residual + out_proj(out)), attn_avg

    def forward(self, h_a, h_b):
        """
        h_a : (B, n, d) — chain A hidden states
        h_b : (B, m, d) — chain B hidden states
        Returns:
            h_a' : (B, n, d) — enriched chain A
            h_b' : (B, m, d) — enriched chain B
            attn_ab : (B, n, m) — attention weights A→B (averaged over heads)
            attn_ba : (B, m, n) — attention weights B→A (averaged over heads)
        """
        h_a_prime, attn_ab = self._attend(
            self.q_a, self.k_b, self.v_b, self.out_a, self.norm_a,
            query=h_a, key_val=h_b, residual=h_a
        )
        h_b_prime, attn_ba = self._attend(
            self.q_b, self.k_a, self.v_a, self.out_b, self.norm_b,
            query=h_b, key_val=h_a, residual=h_b
        )
        return h_a_prime, h_b_prime, attn_ab, attn_ba


class ContactHead(nn.Module):
    """
    Predicts contact probability for each (i,j) pair in the binding sub-matrix
    from cross-attention weights.

    Input:  attn_ab (B, n, m) and attn_ba (B, m, n) — log-relative normalised
    Output: contact_logits (B, n, m) — raw logits for BCEWithLogitsLoss

    Architecture: deeper MLP (2→64→32→1) for more capacity, no dropout
    (too few parameters to need regularization), negative output bias
    initialisation so predictions start sparse (sigmoid(-2) ≈ 0.12).
    """
    def __init__(self, hidden_size=1024, dropout=0.1):
        super().__init__()
        # Deeper MLP — more capacity to learn nonlinear contact patterns
        # No dropout: 65→161 parameters is too small to overfit
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Initialise output bias to -2.0 so model starts predicting sparse
        # contacts (sigmoid(-2) ≈ 0.12) rather than 50/50.
        # The pos_weight loss then pulls it toward recalling true contacts.
        self.mlp[-1].bias.data.fill_(-2.0)

    def forward(self, attn_ab, attn_ba):
        """
        attn_ab : (B, n, m) — A attends to B  (softmax, rows sum to 1)
        attn_ba : (B, m, n) — B attends to A  (softmax, rows sum to 1)
        Returns contact_logits : (B, n, m)

        Uses log-relative attention: log(attn[i,j] / mean(attn)).
        Average pairs get 0.0, high-attention pairs get positive values,
        ignored pairs get negative values. Range typically [-3, +3].
        """
        attn_ba_t = attn_ba.transpose(1, 2)           # (B, n, m)

        def log_relative(x):
            mean = x.mean(dim=(-2, -1), keepdim=True)
            return torch.log(x / (mean + 1e-8) + 1e-8)

        feats  = torch.stack([log_relative(attn_ab),
                               log_relative(attn_ba_t)], dim=-1)  # (B, n, m, 2)
        logits = self.mlp(feats).squeeze(-1)           # (B, n, m)
        return logits


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
        self.use_cross_attn     = use_cross_attn  # if False, skip cross-attention entirely


class T5EncoderForTokenClassification(T5PreTrainedModel):
    """
    ProtT5 encoder + cross-chain attention + per-residue classifier.

    detach_contact=True: attention weights are detached before the ContactHead.
    use_cross_attn=False: skip cross-attention entirely — equivalent to the
      original flat-sequence baseline (encoder output → classifier directly).
    """
    def __init__(self, config: T5Config, class_config, sep_token_id: int):
        super().__init__(config)
        self.num_labels       = class_config.num_labels
        self.sep_token_id     = sep_token_id
        self.detach_contact   = class_config.detach_contact
        self.use_cross_attn   = class_config.use_cross_attn

        self.shared     = nn.Embedding(config.vocab_size, config.d_model)
        enc_cfg         = copy.deepcopy(config)
        enc_cfg.use_cache          = False
        enc_cfg.is_encoder_decoder = False
        self.encoder    = T5Stack(enc_cfg)

        self.cross_attn   = CrossChainAttention(
            hidden_size = config.hidden_size,
            num_heads   = class_config.num_heads,
            dropout     = class_config.cross_attn_dropout,
        )
        self.contact_head = ContactHead(
            hidden_size = config.hidden_size,
            dropout     = class_config.cross_attn_dropout,
        )

        self.dropout    = nn.Dropout(class_config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, class_config.num_labels)
        self.post_init()
        self.model_parallel = False
        self.device_map     = None

    def get_input_embeddings(self):    return self.shared
    def set_input_embeddings(self, e): self.shared = e; self.encoder.set_input_embeddings(e)
    def get_encoder(self):             return self.encoder

    def _split_at_sep(self, hidden, input_ids):
        """
        Split hidden states into chain A and chain B segments.
        The separator token (</s>) is excluded from both.

        Returns:
            h_a  : (B, n, d) — chain A hidden states
            h_b  : (B, m, d) — chain B hidden states
            sep_positions : list of int — index of first sep token per batch item
        """
        B = hidden.shape[0]
        h_a_list, h_b_list, sep_pos = [], [], []

        for i in range(B):
            ids = input_ids[i]
            # Find first occurrence of sep_token_id
            sep_indices = (ids == self.sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_indices) == 0:
                # No separator found — treat whole sequence as chain A
                h_a_list.append(hidden[i])
                h_b_list.append(hidden[i, :1, :])  # dummy 1-token chain B
                sep_pos.append(hidden.shape[1])
            else:
                sep = sep_indices[0].item()
                sep_pos.append(sep)
                h_a_list.append(hidden[i, :sep, :])         # before sep
                h_b_list.append(hidden[i, sep + 1:, :])     # after sep

        return h_a_list, h_b_list, sep_pos

    def forward(self, input_ids=None, attention_mask=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden = outputs[0]   # (B, L, d)

        # ── Cross-chain attention + contact prediction ────────────────────────
        contact_logits_list = []
        if self.use_cross_attn and input_ids is not None:
            h_a_list, h_b_list, sep_pos = self._split_at_sep(hidden, input_ids)

            enriched = []
            for i, (h_a, h_b) in enumerate(zip(h_a_list, h_b_list)):
                h_a_e, h_b_e, attn_ab, attn_ba = self.cross_attn(
                    h_a.unsqueeze(0), h_b.unsqueeze(0)
                )

                # ── Sub-matrix: restrict contacts to predicted binding residues ──
                # Get per-residue binding probabilities from current logits.
                # Use a fixed threshold to define the binding mask.
                # During training this uses the model's current predictions;
                # during inference the same threshold is applied.
                sep  = sep_pos[i]
                seq_len = hidden.shape[1]

                # logits for chain A residues: positions 0..sep-1
                # logits for chain B residues: positions sep+1..end
                logits_so_far = self.classifier(self.dropout(hidden[i]))  # (L, 2)
                p_bind_a = torch.softmax(logits_so_far[:sep],     dim=-1)[:, 1]  # (n,)
                p_bind_b = torch.softmax(logits_so_far[sep+1:seq_len], dim=-1)[:, 1]  # (m,)

                # Hard mask: residues predicted as binding with prob >= threshold
                # Detach so binding mask doesn't create gradient loop
                BIND_THRESH = getattr(self, 'bind_thresh', 0.3)
                mask_a = (p_bind_a.detach() >= BIND_THRESH)  # (n,) bool
                mask_b = (p_bind_b.detach() >= BIND_THRESH)  # (m,) bool

                # Fallback: if too few predicted binding residues, use top-20%
                if mask_a.sum() < 3:
                    k = max(3, int(0.2 * len(p_bind_a)))
                    topk = p_bind_a.detach().topk(k).indices
                    mask_a = torch.zeros_like(mask_a)
                    mask_a[topk] = True
                if mask_b.sum() < 3:
                    k = max(3, int(0.2 * len(p_bind_b)))
                    topk = p_bind_b.detach().topk(k).indices
                    mask_b = torch.zeros_like(mask_b)
                    mask_b[topk] = True

                # Slice attention to binding sub-matrix: (1, n_bind, m_bind)
                attn_ab_in = attn_ab[:, mask_a, :][:, :, mask_b]
                attn_ba_in = attn_ba[:, mask_b, :][:, :, mask_a]
                if self.detach_contact:
                    attn_ab_in = attn_ab_in.detach()
                    attn_ba_in = attn_ba_in.detach()

                # ContactHead predicts over binding sub-matrix only
                c_logits = self.contact_head(attn_ab_in, attn_ba_in)
                # Store as (c_logits, mask_a, mask_b, attn_ab) for supervision
                contact_logits_list.append((
                    c_logits.squeeze(0),  # (n_bind, m_bind)
                    mask_a,               # (n,) — chain A binding mask
                    mask_b,               # (m,) — chain B binding mask
                    attn_ab.squeeze(0),   # (n, m) — full attention for supervision
                ))

                # Reconstruct full sequence hidden states
                parts = [h_a_e.squeeze(0)]
                if sep < hidden.shape[1]:
                    parts.append(hidden[i, sep:sep+1, :])
                    parts.append(h_b_e.squeeze(0))
                enriched_i = torch.cat(parts, dim=0)

                orig_len = hidden.shape[1]
                if enriched_i.shape[0] < orig_len:
                    pad = hidden[i, enriched_i.shape[0]:, :]
                    enriched_i = torch.cat([enriched_i, pad], dim=0)
                else:
                    enriched_i = enriched_i[:orig_len]
                enriched.append(enriched_i)

            hidden = torch.stack(enriched, dim=0)
        # ─────────────────────────────────────────────────────────────────────

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

        if not return_dict:
            out = (logits,) + outputs[2:]
            return ((loss,) + out) if loss is not None else out

        classifier_output = TokenClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # During training, return the tuple so compute_loss can access contact_logits.
        # During eval/inference, return plain TokenClassifierOutput so the Trainer
        # and manual inference loops can access .logits without unpacking.
        if self.training:
            return classifier_output, contact_logits_list
        return classifier_output


# ── Data collator ─────────────────────────────────────────────────────────────

class FastDataCollatorForTokenClassification:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer         = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        label_list   = [f.pop("labels")   for f in features]
        contact_list = [f.pop("contacts", []) for f in features]

        batch = self.tokenizer.pad(features, padding=True,
                                   pad_to_multiple_of=self.pad_to_multiple_of,
                                   return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor(
            [(list(lbl) + [-100] * max_len)[:max_len] for lbl in label_list]
        )
        # Keep contacts as a Python list of lists — can't pad to tensor (variable size)
        batch["contacts"] = contact_list

        for f, lbl, con in zip(features, label_list, contact_list):
            f["labels"]   = lbl
            f["contacts"] = con

        return batch


# ── Model builder ─────────────────────────────────────────────────────────────

def PT5_classification_model(num_labels, detach_contact=False,
                              use_cross_attn=True, lora_rank=8):
    dtype = torch.float32

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5")
    model     = T5EncoderModel.from_pretrained("Rostlab/ProstT5", torch_dtype=dtype)

    sep_token_id = tokenizer.eos_token_id

    class_model = T5EncoderForTokenClassification(
        model.config,
        ClassConfig(num_labels=num_labels, num_heads=8,
                    cross_attn_dropout=0.1, detach_contact=detach_contact,
                    use_cross_attn=use_cross_attn),
        sep_token_id=sep_token_id,
    )
    class_model.shared  = model.shared
    class_model.encoder = model.encoder
    model = class_model
    del class_model

    lora_cfg = LoRAConfig(rank=lora_rank)
    model    = modify_with_lora(model, lora_cfg)

    model.classifier.to(dtype)
    model.cross_attn.to(dtype)   # ensure cross_attn matches dtype

    # Freeze encoder, keep LoRA + layer norms + cross_attn + classifier + contact_head trainable
    for p in model.shared.parameters():  p.requires_grad = False
    for p in model.encoder.parameters(): p.requires_grad = False
    for name, p in model.named_parameters():
        if re.fullmatch(lora_cfg.trainable_param_names, name):
            p.requires_grad = True
    for p in model.cross_attn.parameters():   p.requires_grad = True
    for p in model.classifier.parameters():   p.requires_grad = True
    for p in model.contact_head.parameters(): p.requires_grad = True

    model.contact_head.to(dtype)

    trainable = sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)
    cross_attn_params  = sum(np.prod(p.size()) for p in model.cross_attn.parameters())
    contact_head_params = sum(np.prod(p.size()) for p in model.contact_head.parameters())
    lora_params   = sum(np.prod(p.size()) for n, p in model.named_parameters()
                        if p.requires_grad and 'lora' in n.lower())
    other_params  = trainable - lora_params
    print(f"ProtT5_LoRA_CrossAttn+Contact_Classifier — trainable parameters: {trainable:,}")
    print(f"  LoRA params          : {lora_params:,}")
    print(f"  Other trainable      : {other_params:,}")
    print(f"  cross-attention: {cross_attn_params:,}")
    print(f"  contact head:    {contact_head_params:,}\n")
    return model, tokenizer


# ── Deepspeed config ──────────────────────────────────────────────────────────

ds_config = {
    "fp16": {"enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000,
             "initial_scale_power": 16, "hysteresis": 2, "min_loss_scale": 1},
    "optimizer": {"type": "AdamW",
                  "params": {"lr": "auto", "betas": "auto", "eps": "auto", "weight_decay": "auto"}},
    "scheduler": {"type": "WarmupLR",
                  "params": {"warmup_min_lr": "auto", "warmup_max_lr": "auto", "warmup_num_steps": "auto"}},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True, "allgather_bucket_size": 2e8,
        "overlap_comm": True, "reduce_scatter": True,
        "reduce_bucket_size": 2e8, "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seeds(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s); set_seed(s)


def compute_class_weights(labels_list, num_labels=2):
    """
    Compute inverse-frequency class weights from training labels.
    Ignores -100 (padding/masked positions).
    Returns a float tensor of shape (num_labels,).
    """
    counts = torch.zeros(num_labels)
    for seq in labels_list:
        for lbl in seq:
            if 0 <= lbl < num_labels:
                counts[lbl] += 1
    total   = counts.sum()
    weights = total / (num_labels * counts)
    print(f"Class counts:  { {i: int(counts[i]) for i in range(num_labels)} }")
    print(f"Class weights: { {i: round(weights[i].item(), 3) for i in range(num_labels)} }")
    return weights

def load_pairs_csv(filepath, shuffle_labels=False, random_labels=False, shuffle_seed=42):
    """
    Load protein pair data from the non-redundant pairs CSV produced by
    fetch_and_label_pairs.py / make_nonredundant.py.

    Expected columns: pair_id, chain_A, chain_B, seq_A, seq_B, label_A, label_B
    Label strings contain '0', '1' characters (binary binding site labels).

    Note: PDB IDs like 3E99 or 5E10 would be misread as scientific notation
    by pandas without explicit dtype specification. DOS/Mac line endings are
    handled by binary pre-processing.

    Negative control modes (mutually exclusive):

      shuffle_labels=True
          Permutes binding labels within each chain. Preserves per-chain
          binding rate but breaks within-chain sequence-to-label signal.
          AUROC will reflect any signal learnable from per-chain
          composition/length statistics.

      random_labels=True
          Generates fresh Bernoulli labels at the empirical binding rate
          for each residue independently. Destroys both per-residue AND
          per-chain signal. AUROC should be near 0.5 if there is no
          information leakage.
    """
    import io as _io
    string_cols = ['pair_id', 'pdb_id', 'chain_A', 'chain_B',
                   'seq_A', 'seq_B', 'label_A', 'label_B', 'contacts']
    raw = open(filepath, 'rb').read()
    raw = raw.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    df = pd.read_csv(_io.BytesIO(raw),
                     dtype={c: str for c in string_cols})

    # Verify required columns are present
    required = ['pair_id', 'chain_A', 'chain_B', 'seq_A', 'seq_B', 'label_A', 'label_B']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}. "
                         f"Found: {df.columns.tolist()}")

    # Drop rows with missing sequences or labels
    before = len(df)
    df = df.dropna(subset=['seq_A', 'seq_B', 'label_A', 'label_B'])
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing values")

    # Clean sequences: replace unknown residues with X
    df['seq_A'] = df['seq_A'].str.replace('?', 'X', regex=False)
    df['seq_B'] = df['seq_B'].str.replace('?', 'X', regex=False)

    # Convert label strings to integer lists: '0'->0, '1'->1
    def parse_labels(s):
        return [int(c) if c in ('0', '1') else -100 for c in str(s)]

    df['label_A'] = df['label_A'].apply(parse_labels)
    df['label_B'] = df['label_B'].apply(parse_labels)

    # ── Negative control: shuffle labels within each chain ───────────────────
    if shuffle_labels and random_labels:
        raise ValueError("Use only one of --shuffle-labels or --random-labels, not both")

    if shuffle_labels:
        print(f"\n  [NEGATIVE CONTROL] Shuffling binding labels within each chain")
        print(f"  Random seed: {shuffle_seed}")
        rng = np.random.default_rng(shuffle_seed)

        def shuffle_chain_labels(labels):
            arr = np.array(labels)
            valid_mask = arr != -100
            valid_vals = arr[valid_mask].copy()
            rng.shuffle(valid_vals)
            arr[valid_mask] = valid_vals
            return arr.tolist()

        df['label_A'] = df['label_A'].apply(shuffle_chain_labels)
        df['label_B'] = df['label_B'].apply(shuffle_chain_labels)

        all_labels_a = [v for lst in df['label_A'] for v in lst if v != -100]
        n1 = sum(1 for v in all_labels_a if v == 1)
        n0 = sum(1 for v in all_labels_a if v == 0)
        print(f"  Chain A class balance after shuffle: "
              f"{n1:,} binding ({100*n1/(n0+n1):.1f}%), "
              f"{n0:,} non-binding")

    # ── Negative control: globally random labels at empirical rate ──────────
    if random_labels:
        # Compute empirical binding rate from original labels
        all_a = [v for lst in df['label_A'] for v in lst if v != -100]
        all_b = [v for lst in df['label_B'] for v in lst if v != -100]
        all_orig = all_a + all_b
        empirical_rate = sum(all_orig) / len(all_orig)

        print(f"\n  [NEGATIVE CONTROL] Generating globally random labels")
        print(f"  Empirical binding rate: {empirical_rate:.4f} ({100*empirical_rate:.2f}%)")
        print(f"  Random seed: {shuffle_seed}")
        rng = np.random.default_rng(shuffle_seed)

        def randomize_chain_labels(labels):
            arr = np.array(labels)
            valid_mask = arr != -100
            n_valid = int(valid_mask.sum())
            random_vals = rng.random(n_valid) < empirical_rate
            arr[valid_mask] = random_vals.astype(int)
            return arr.tolist()

        df['label_A'] = df['label_A'].apply(randomize_chain_labels)
        df['label_B'] = df['label_B'].apply(randomize_chain_labels)

        all_a_new = [v for lst in df['label_A'] for v in lst if v != -100]
        n1 = sum(1 for v in all_a_new if v == 1)
        n0 = sum(1 for v in all_a_new if v == 0)
        print(f"  Chain A class balance after randomization: "
              f"{n1:,} binding ({100*n1/(n0+n1):.1f}%), "
              f"{n0:,} non-binding")

    # Parse contacts column if present: "i1,j1;i2,j2;..." → list of (i,j) tuples
    def parse_contacts(s):
        if not isinstance(s, str) or s.strip() == '':
            return []
        return [tuple(int(x) for x in pair.split(','))
                for pair in s.split(';') if pair.strip()]

    if 'contacts' in df.columns:
        df['contacts'] = df['contacts'].apply(parse_contacts)
    else:
        # Older CSV without contacts column — empty lists
        df['contacts'] = [[] for _ in range(len(df))]

    # In any negative control mode, clear contacts since the original ones
    # would no longer correspond to the modified binding labels.
    if shuffle_labels or random_labels:
        print(f"  Clearing contact pairs (would not match modified labels)")
        df['contacts'] = [[] for _ in range(len(df))]

    # Rename to lowercase for create_dataset_pairs
    df = df.rename(columns={
        'seq_A': 'seq_a', 'seq_B': 'seq_b',
        'label_A': 'label_a', 'label_B': 'label_b',
    })

    print(f"  Loaded {len(df)} pairs from {filepath}")
    has_contacts = df['contacts'].apply(len).sum() > 0
    print(f"  Contact labels: {'present' if has_contacts else 'absent (will train without pairwise loss)'}")
    return df


def create_dataset_pairs(tokenizer, df, max_length=768):
    """
    Tokenize protein pairs with a separator token between chains.
    Each input becomes: [tokens_A] + [</s>] + [tokens_B] + [</s>]
    Labels become:      [labels_A] + [-100] + [labels_B] + [-100]

    max_length=768 accommodates two chains of up to 300 residues each:
      half_max = (768 - 2) // 2 = 383 tokens per chain
    This is consistent with the --max-length 300 filter in fetch_and_label_pairs.py.
    """
    sep_id    = tokenizer.eos_token_id   # </s> for ProtT5
    half_max  = (max_length - 2) // 2   # max tokens per chain (-2 for 2 sep tokens)

    all_input_ids      = []
    all_attention_mask = []
    all_labels         = []

    rare = str.maketrans("OUBZ", "XXXX")

    all_input_ids      = []
    all_attention_mask = []
    all_labels         = []
    all_contacts       = []   # sparse list of (i,j) per sample, after truncation

    for _, row in df.iterrows():
        seq_a = " ".join(row['seq_a'].upper().translate(rare))
        seq_b = " ".join(row['seq_b'].upper().translate(rare))

        ids_a = tokenizer(seq_a, add_special_tokens=False)['input_ids']
        ids_b = tokenizer(seq_b, add_special_tokens=False)['input_ids']
        lbl_a = list(row['label_a'])
        lbl_b = list(row['label_b'])
        contacts = list(row.get('contacts', []))

        # Truncate each chain independently to preserve both
        trunc_a = min(len(ids_a), half_max)
        trunc_b = min(len(ids_b), half_max)
        ids_a = ids_a[:trunc_a]; lbl_a = lbl_a[:trunc_a]
        ids_b = ids_b[:trunc_b]; lbl_b = lbl_b[:trunc_b]

        # Filter contacts to only include pairs within truncated lengths
        contacts = [(i, j) for i, j in contacts
                    if i < trunc_a and j < trunc_b]

        # Concatenate: A + </s> + B + </s>
        input_ids = ids_a + [sep_id] + ids_b + [sep_id]
        labels    = lbl_a + [-100]   + lbl_b + [-100]

        all_input_ids.append(input_ids)
        all_attention_mask.append([1] * len(input_ids))
        all_labels.append(labels)
        all_contacts.append(contacts)

    dataset = Dataset.from_dict({
        'input_ids':      all_input_ids,
        'attention_mask': all_attention_mask,
        'labels':         all_labels,
        'contacts':       all_contacts,   # list of (i,j) tuples per sample
    })
    return dataset


def create_dataset(tokenizer, seqs, labels):
    """Original single-sequence dataset builder — kept for compatibility."""
    tokenized = tokenizer(seqs, max_length=768, padding=False, truncation=True)
    dataset   = Dataset.from_dict(tokenized)
    aligned   = []
    for label, ids in zip(labels, tokenized["input_ids"]):
        tok_len = len(ids)
        l = label[:tok_len] + [-100] * (tok_len - len(label[:tok_len]))
        aligned.append(l)
    dataset = dataset.add_column("labels", aligned)
    return dataset


# ── Fix 3: compute_metrics must be defined BEFORE Trainer() ──────────────────

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


class WeightedLossTrainer(Trainer):
    """3x class weights + label smoothing + cross-chain attention model.
    Overrides _save/_load_from_checkpoint to avoid HuggingFace save_pretrained,
    which fails on our custom T5 model due to shared embedding tensors.
    """
    def __init__(self, *args, class_weights=None, label_smoothing=0.15,
                 contact_lambda=0.25, pos_weight_cap=5.0,
                 attn_supervision_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights            = class_weights
        self.label_smoothing          = label_smoothing
        self.contact_lambda           = contact_lambda
        self.pos_weight_cap           = pos_weight_cap
        self.attn_supervision_lambda  = attn_supervision_lambda

    def _save(self, output_dir, state_dict=None):
        """Save only trainable parameters — avoids save_pretrained shared-tensor bug."""
        os.makedirs(output_dir, exist_ok=True)
        trainable = {n: p for n, p in self.model.named_parameters()
                     if p.requires_grad}
        torch.save(trainable, os.path.join(output_dir, "trainable_weights.pt"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore trainable weights from our custom checkpoint format."""
        weights_path = os.path.join(resume_from_checkpoint, "trainable_weights.pt")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No trainable_weights.pt found in {resume_from_checkpoint}."
            )
        if model is None:
            model = self.model
        saved = torch.load(weights_path, map_location="cpu")
        missing, unexpected = [], []
        model_params = dict(model.named_parameters())
        for name, param in saved.items():
            if name in model_params:
                model_params[name].data = param.data
            else:
                unexpected.append(name)
        for name in model_params:
            if model_params[name].requires_grad and name not in saved:
                missing.append(name)
        if unexpected:
            print(f"  [WARN] Checkpoint unexpected keys: {unexpected}")
        if missing:
            print(f"  [WARN] Checkpoint missing keys: {missing}")
        print(f"  Loaded trainable weights from {weights_path}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels   = inputs.pop("labels")
        contacts = inputs.pop("contacts", None)   # list of (i,j) lists per sample

        # Model returns tuple (output, contact_logits) during training,
        # plain TokenClassifierOutput during eval
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

        # ── Per-residue loss (with label smoothing) ───────────────────────────
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

        # ── Pairwise contact loss (sub-matrix over predicted binding residues) ──
        if (self.contact_lambda > 0 and contacts is not None
                and len(contact_logits_list) > 0):

            contact_losses = []
            attn_sup_losses = []

            for item, pair_contacts in zip(contact_logits_list, contacts):
                if not isinstance(item, tuple):
                    c_logits, mask_a, mask_b, attn_ab = item, None, None, None
                elif len(item) == 4:
                    c_logits, mask_a, mask_b, attn_ab = item
                else:
                    c_logits, mask_a, mask_b = item
                    attn_ab = None

                n, m = c_logits.shape
                if n == 0 or m == 0 or len(pair_contacts) == 0:
                    continue

                # Build target for sub-matrix
                target = torch.zeros(n, m, device=c_logits.device)
                if pair_contacts and mask_a is not None:
                    full_n = mask_a.shape[0]
                    full_m = mask_b.shape[0]
                    sub_idx_a = torch.full((full_n,), -1, dtype=torch.long,
                                           device=c_logits.device)
                    sub_idx_b = torch.full((full_m,), -1, dtype=torch.long,
                                           device=c_logits.device)
                    sub_idx_a[mask_a] = torch.arange(mask_a.sum(),
                                                      device=c_logits.device)
                    sub_idx_b[mask_b] = torch.arange(mask_b.sum(),
                                                      device=c_logits.device)
                    ij = torch.tensor(pair_contacts, dtype=torch.long,
                                      device=c_logits.device)
                    valid = (ij[:, 0] < full_n) & (ij[:, 1] < full_m)
                    ij = ij[valid]
                    if len(ij) > 0:
                        si = sub_idx_a[ij[:, 0]]
                        sj = sub_idx_b[ij[:, 1]]
                        in_sub = (si >= 0) & (sj >= 0)
                        ij_sub = torch.stack([si[in_sub], sj[in_sub]], dim=1)
                        if len(ij_sub) > 0:
                            target[ij_sub[:, 0], ij_sub[:, 1]] = 1.0
                elif pair_contacts:
                    ij = torch.tensor(pair_contacts, dtype=torch.long,
                                      device=c_logits.device)
                    valid = (ij[:, 0] < n) & (ij[:, 1] < m)
                    ij = ij[valid]
                    if len(ij) > 0:
                        target[ij[:, 0], ij[:, 1]] = 1.0

                n_pos = target.sum()
                n_neg = n * m - n_pos
                if n_pos > 0:
                    pos_weight = torch.clamp(n_neg / n_pos,
                                             max=self.pos_weight_cap).detach()
                    pair_loss = F.binary_cross_entropy_with_logits(
                        c_logits.float(), target, pos_weight=pos_weight
                    )
                    contact_losses.append(pair_loss)
                else:
                    continue

                # ── Attention supervision loss ─────────────────────────────────
                # Push attn_ab[i,j] to be high for known contact pairs.
                # Uses KL divergence between attention distribution and a
                # soft target derived from the known contact labels.
                if (self.attn_supervision_lambda > 0
                        and attn_ab is not None
                        and len(pair_contacts) > 0):
                    n_full, m_full = attn_ab.shape
                    attn_target = torch.zeros(n_full, m_full,
                                              device=attn_ab.device)
                    all_ij = torch.tensor(pair_contacts, dtype=torch.long,
                                          device=attn_ab.device)
                    v = (all_ij[:, 0] < n_full) & (all_ij[:, 1] < m_full)
                    all_ij = all_ij[v]
                    if len(all_ij) > 0:
                        attn_target[all_ij[:, 0], all_ij[:, 1]] = 1.0
                        # Normalize each row to sum to 1 (like a prob distribution)
                        row_sums = attn_target.sum(dim=-1, keepdim=True).clamp(min=1)
                        attn_target = attn_target / row_sums
                        # KL( attn_target || attn_ab ) — push attention toward contacts
                        log_attn = torch.log(attn_ab.float().clamp(min=1e-8))
                        attn_sup_loss = F.kl_div(log_attn, attn_target,
                                                  reduction='batchmean')
                        attn_sup_losses.append(attn_sup_loss)

            if contact_losses:
                contact_loss = torch.stack(contact_losses).mean()
                loss = loss + self.contact_lambda * contact_loss

            if attn_sup_losses:
                attn_sup_loss = torch.stack(attn_sup_losses).mean()
                loss = loss + self.attn_supervision_lambda * attn_sup_loss

        return (loss, outputs) if return_outputs else loss


def train_per_residue(train_df, valid_df, num_labels=2,
                      batch=8, accum=4, val_batch=16,
                      epochs=5, lr=3e-4, seed=42,
                      deepspeed=False, mixed=False, num_workers=28, gpu=1):

    # Only set CUDA_VISIBLE_DEVICES when NOT using torchrun
    # (torchrun manages device assignment via LOCAL_RANK)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu - 1)

    set_seeds(seed)
    model, tokenizer = PT5_classification_model(num_labels=num_labels)

    for df in [train_df, valid_df]:
        df["sequence"] = df["sequence"].str.replace("|".join(["O","B","U","Z"]), "X", regex=True)
        df["sequence"] = df.apply(lambda r: " ".join(r["sequence"]), axis=1)

    train_set = create_dataset(tokenizer, list(train_df["sequence"]), list(train_df["label"]))
    valid_set = create_dataset(tokenizer, list(valid_df["sequence"]), list(valid_df["label"]))

    # Compute class weights from training labels to handle imbalance
    print("\nComputing class weights...")
    class_weights = compute_class_weights(list(train_df["label"]), num_labels=num_labels)

    # Compute warmup steps (5% of total training steps)
    steps_per_epoch = max(1, len(train_set) // (batch * accum))
    warmup_steps    = max(1, int(steps_per_epoch * epochs * 0.05))

    args = TrainingArguments(
        "./",
        eval_strategy="epoch",
        logging_strategy="steps",
        save_strategy="no",
        logging_steps=50,
        logging_first_step=True,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        report_to="none",
        gradient_checkpointing=True, # Remove later!
        deepspeed=ds_config if deepspeed else None,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = WeightedLossTrainer(
        model,
        args,
        class_weights=class_weights,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=FastDataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return tokenizer, model, trainer.state.log_history


def save_model(model, filepath):
    non_frozen = {n: p for n, p in model.named_parameters() if p.requires_grad}
    torch.save(non_frozen, filepath)


def load_model(filepath, num_labels=2):
    model, tokenizer = PT5_classification_model(num_labels=num_labels)
    saved = torch.load(filepath, map_location="cpu")
    for name, param in model.named_parameters():
        if name in saved:
            param.data = saved[name].data
    return tokenizer, model


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="PT5 cross-attention + contact matrix training")
    parser.add_argument("--model-out", default="./PT5_pairs_finetuned.pth",
                        help="Path to save trained model weights "
                             "(default: ./PT5_pairs_finetuned.pth)")
    parser.add_argument("--csv", default="/pomplun/share_home/nurit.haspel/Bert/Fine-Tuning/splits_bind/dense_nonred_matrix.csv",
                        help="Path to pairs CSV file (default: dense_nonred_matrix.csv). "
                             "Use to switch between 50%% and 70%% CD-HIT datasets, "
                             "e.g. --csv ./nr_pairs_70.csv")
    parser.add_argument("--probs-out-dir", default=".",
                        help="Directory to save val/test probability npy files "
                             "(default: current dir). Use to separate Model A and B outputs, "
                             "e.g. --probs-out-dir ./probs_modelA")
    parser.add_argument("--contact-out", default="test_contact_matrices.pkl",
                        help="Output path for predicted contact matrices pkl "
                             "(default: test_contact_matrices.pkl)")
    parser.add_argument("--no-cross-attn", action="store_true",
                        help="Disable cross-chain attention entirely — flat sequence "
                             "baseline (encoder output → classifier directly).")
    parser.add_argument("--class-weights", type=float, nargs=2,
                        default=[1.0, 8.0], metavar=("NEG", "POS"),
                        help="Class weights for non-binding and binding "
                             "(default: 1.0 8.0). Example: --class-weights 1.0 3.0")
    parser.add_argument("--label-smoothing", type=float, default=0.15,
                        help="Label smoothing alpha (default: 0.15)")
    parser.add_argument("--bind-thresh", type=float, default=0.3,
                        help="Probability threshold for defining predicted binding residues "
                             "in the contact sub-matrix (default: 0.3). Must match "
                             "--pred-threshold in jaccard_filter.py for consistent filtering.")
    parser.add_argument("--attn-supervision-lambda", type=float, default=0.0,
                        help="Weight for attention supervision loss — pushes attn_ab "
                             "toward known contact pairs (default: 0.0 = disabled). "
                             "Try 0.1 or 0.5 to guide attention toward true contacts.")
    parser.add_argument("--contact-lambda", type=float, default=0.25,
                        help="Weight for contact matrix loss (default: 0.25). "
                             "Set to 0.0 to disable contact loss entirely.")
    parser.add_argument("--pos-weight-cap", type=float, default=10.0,
                        help="Maximum pos_weight in contact BCE loss (default: 10.0). "
                             "Sub-matrix contact density is ~6-10%%, so n_neg/n_pos~10. "
                             "Lower values = sparser contact predictions.")
    parser.add_argument("--detach-contact", action="store_true",
                        help="Detach attention weights before ContactHead — contact loss "
                             "trains only the ContactHead, not the cross-attention or encoder.")
    parser.add_argument("--epochs", type=int, default=12,
                        help="Number of training epochs (default: 12)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint in --checkpoint-dir")
    parser.add_argument("--checkpoint-dir", default="./checkpoints",
                        help="Directory to save/load checkpoints (default: ./checkpoints)")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save a checkpoint every N epochs (default: 1)")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                        help="Number of most recent checkpoints to keep (default: 3)")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank for encoder fine-tuning (default: 8). "
                             "Set to 0 to freeze encoder completely (no LoRA).")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="NEGATIVE CONTROL: randomly shuffle binding labels "
                             "within each chain. Preserves class balance per chain "
                             "but breaks within-chain sequence-to-label signal. "
                             "AUROC reflects per-chain composition learnability.")
    parser.add_argument("--random-labels", action="store_true",
                        help="NEGATIVE CONTROL: generate globally random Bernoulli "
                             "labels at the empirical binding rate. Destroys both "
                             "per-residue and per-chain signal. AUROC should be "
                             "near 0.5 if there is no information leakage.")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Random seed for label shuffling/randomization (default: 42)")
    parser.add_argument("--train-list", default=None,
                    help="Path to text file with one pair-id per line for the training set. "
                         "If provided alongside --test-list, predefined splits are used "
                         "instead of the random 70/15/15 split. Pair-id format may be "
                         "either 'PDB_A_B' (MaSIF style) or 'PDB_A_PDB_B' (our CSV style).")
    parser.add_argument("--val-list", default=None,
                    help="Path to text file with validation pair-ids. Optional — if not "
                         "provided when using predefined splits, a fraction of training "
                         "is held out for validation.")
    parser.add_argument("--test-list", default=None,
                    help="Path to text file with test pair-ids. Required for predefined splits.")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                    help="Fraction of training data to hold out as validation when using "
                         "predefined splits without an explicit --val-list (default: 0.15).")
    parser.add_argument("--inference-only", action="store_true",
                    help="Skip training entirely and run inference on the full CSV. "
                         "Requires --weights to point to a trainable_weights.pt file "
                         "or a checkpoint directory containing one. No train/val/test "
                         "split is performed — the entire CSV is used as the inference set. "
                         "If labels are present in the CSV, evaluation metrics are also "
                         "reported; if labels are absent, only probabilities are saved.")
    parser.add_argument("--weights", default=None,
                    help="Path to a trainable_weights.pt file or a checkpoint directory "
                         "containing one. Used with --inference-only to load a previously "
                         "trained model. If a directory is given, the script looks for "
                         "trainable_weights.pt inside it.")
    cli = parser.parse_args()

    CHECKPOINT_DIR = cli.checkpoint_dir

    # ── Load from non-redundant pairs CSV ─────────────────────────────────────
    CSV_PATH = cli.csv

    print(f"Loading non-redundant pair data from {CSV_PATH}...")
    df = load_pairs_csv(CSV_PATH,
                        shuffle_labels=cli.shuffle_labels,
                        random_labels=cli.random_labels,
                        shuffle_seed=cli.shuffle_seed)
    print(f"Loaded {len(df)} protein pairs")

    # Build device, model and tokenizer
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(42)
    model, tokenizer = PT5_classification_model(
        num_labels=2,
        detach_contact=cli.detach_contact,
        use_cross_attn=not cli.no_cross_attn,
        lora_rank=cli.lora_rank,
    )
    model.bind_thresh = cli.bind_thresh
    # ── Inference-only mode ────────────────────────────────────────────────────
    if cli.inference_only:
        if cli.weights is None:
            raise ValueError("--inference-only requires --weights to specify a "
                             "trainable_weights.pt file or checkpoint directory.")

        # Resolve weights path
        weights_path = Path(cli.weights)
        if weights_path.is_dir():
            weights_path = weights_path / "trainable_weights.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        print(f"\nInference-only mode: running on {len(df)} pairs")
        print(f"  Loading weights from: {weights_path}")


        saved = torch.load(str(weights_path), map_location="cpu",
                           weights_only=False)
        state = saved.get("model_state_dict", saved)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [WARN] Missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"  [WARN] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        model.to(device)
        model.eval()
        print(f"  Weights loaded successfully.")

        # Use entire CSV as inference set (no split)
        infer_set = create_dataset_pairs(tokenizer, df, max_length=1024)
        infer_loader = DataLoader(infer_set, batch_size=16, shuffle=False,
                                  num_workers=0,
                                  collate_fn=FastDataCollatorForTokenClassification(tokenizer))

        # Run inference
        all_probs, all_labels_padded = [], []
        with torch.no_grad():
            for batch in tqdm(infer_loader, desc="Inference"):
                all_labels_padded += batch["labels"].tolist()
                logits = model(batch["input_ids"].to(device),
                               batch["attention_mask"].to(device)).logits
                all_probs += F.softmax(logits, dim=-1).cpu().tolist()

        # Flatten and filter padded positions
        def _flatten(l): return [x for sub in l for x in sub]
        probs_flat  = np.array(_flatten(all_probs))
        labels_flat = np.array(_flatten(all_labels_padded))
        valid_mask  = labels_flat != -100
        probs_clean  = probs_flat[valid_mask]
        labels_clean = labels_flat[valid_mask]

        # Save probabilities
        probs_dir = Path(cli.probs_out_dir)
        probs_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(probs_dir / "infer_probs.npy"),  probs_clean)
        np.save(str(probs_dir / "infer_labels.npy"), labels_clean)
        print(f"\nSaved inference probabilities → {probs_dir}/infer_probs.npy")

        # Evaluate if labels are present
        has_labels = len(labels_clean) > 0 and labels_clean.max() >= 0
        if has_labels:
            from sklearn.metrics import (f1_score, matthews_corrcoef,
                                         classification_report, roc_auc_score,
                                         average_precision_score, accuracy_score)
            pos_probs = probs_clean[:, 1]
            auroc = roc_auc_score(labels_clean, pos_probs)
            auprc = average_precision_score(labels_clean, pos_probs)
            print(f"\n── Threshold-independent metrics ────────────────────────")
            print(f"AUROC    : {auroc:.4f}")
            print(f"AUPRC    : {auprc:.4f}")
            print(f"  (baseline AUPRC for random = {labels_clean.mean():.4f})")
            print(f"  (AUPRC lift over random   = {auprc / labels_clean.mean():.2f}x)")

            print(f"\nRunning threshold sweep on inference set...")
            thresholds = np.arange(0.05, 0.95, 0.01)
            mccs = []
            for t in thresholds:
                p = (pos_probs >= t).astype(int)
                try:    mccs.append(matthews_corrcoef(labels_clean, p))
                except: mccs.append(0.0)
            best_t   = float(thresholds[np.argmax(mccs)])
            best_mcc = float(np.max(mccs))
            print(f"Best threshold: {best_t:.2f}  (MCC={best_mcc:.4f})")

            preds_tuned = (pos_probs >= best_t).astype(int)
            print(f"\n── Inference results (tuned threshold={best_t:.2f}) ─────────")
            print(f"Accuracy : {accuracy_score(labels_clean, preds_tuned):.4f}")
            print(f"F1       : {f1_score(labels_clean, preds_tuned, pos_label=1, zero_division=0):.4f}")
            print(f"MCC      : {best_mcc:.4f}")
            print(f"AUROC    : {auroc:.4f}")
            print(f"AUPRC    : {auprc:.4f}")
            print(classification_report(labels_clean, preds_tuned,
                                        target_names=["non-binding (0)", "binding site (1)"],
                                        zero_division=0))
        else:
            print("No labels found in CSV — skipping metric computation.")

        # Contact matrices
        if not cli.no_cross_attn:
            print("\nSaving predicted contact matrices...")
            contact_matrices = {}
            for sample_idx in tqdm(range(len(infer_set)), desc="Contact matrices"):
                sample  = infer_set[sample_idx]
                pair_id = str(df.iloc[sample_idx].get('pair_id', sample_idx))
                with torch.no_grad():
                    model.train()
                    result = model(
                        input_ids      = torch.tensor([sample['input_ids']]).to(device),
                        attention_mask = torch.tensor([sample['attention_mask']]).to(device),
                    )
                    model.eval()
                if isinstance(result, tuple) and len(result) == 2:
                    _, contact_logits_list = result
                    if contact_logits_list:
                        item = contact_logits_list[0]
                        if isinstance(item, tuple) and len(item) >= 3:
                            c_logits = item[0]
                            mask_a   = item[1]
                            mask_b   = item[2]
                            n_full   = mask_a.shape[0]
                            m_full   = mask_b.shape[0]
                            full_mat = np.zeros((n_full, m_full), dtype=np.float32)
                            c_prob   = torch.sigmoid(c_logits).cpu().numpy()
                            idx_a    = mask_a.cpu().numpy().nonzero()[0]
                            idx_b    = mask_b.cpu().numpy().nonzero()[0]
                            full_mat[np.ix_(idx_a, idx_b)] = c_prob
                            contact_matrices[pair_id] = full_mat
                        else:
                            c_prob = torch.sigmoid(item).cpu().numpy()
                            contact_matrices[pair_id] = c_prob.astype(np.float32)
            import pickle
            payload = {"matrices": contact_matrices, "bind_thresh": cli.bind_thresh}
            with open(cli.contact_out, "wb") as f:
                pickle.dump(payload, f)
            print(f"Saved contact matrices for {len(contact_matrices)} pairs "
                  f"→ {cli.contact_out}")

        print("\nInference complete.")
        import sys; sys.exit(0)

    # ── Train/val/test split ──────────────────────────────────────────────
    # If --train-list and --test-list are provided (e.g. for MaSIF's
    # predefined splits), use them. Otherwise, do a 70/15/15 random split.
    from sklearn.model_selection import train_test_split

    def _normalize_pair_id(pid):
        """
        Normalize a pair ID to the 'PDB_CHAINA_PDB_CHAINB' format used in
        our CSVs. MaSIF lists use 'PDB_CHAINA_CHAINB' (3 underscore-separated
        tokens), so we expand that to 4 tokens for matching.
        """
        if pid is None:
            return None
        pid = str(pid).strip()
        if not pid:
            return None
        parts = pid.split("_")
        if len(parts) == 3:
            pdb, ca, cb = parts
            return f"{pdb.upper()}_{ca.upper()}_{pdb.upper()}_{cb.upper()}"
        if len(parts) == 4:
            pdb_a, ca, pdb_b, cb = parts
            return f"{pdb_a.upper()}_{ca.upper()}_{pdb_b.upper()}_{cb.upper()}"
        return pid.upper()

    if cli.train_list and cli.test_list:
        # Predefined-split mode
        print(f"Using predefined splits:")
        print(f"  train list: {cli.train_list}")
        print(f"  test list : {cli.test_list}")
        if cli.val_list:
            print(f"  val list  : {cli.val_list}")

        with open(cli.train_list) as f:
            train_ids = {_normalize_pair_id(l) for l in f if l.strip()}
            train_ids.discard(None)
        with open(cli.test_list) as f:
            test_ids = {_normalize_pair_id(l) for l in f if l.strip()}
            test_ids.discard(None)
        if cli.val_list:
            with open(cli.val_list) as f:
                val_ids = {_normalize_pair_id(l) for l in f if l.strip()}
                val_ids.discard(None)
        else:
            val_ids = None

        # Normalize CSV pair_ids the same way for matching
        df = df.copy()
        df["_norm_id"] = df["pair_id"].apply(_normalize_pair_id)

        df_train_full = df[df["_norm_id"].isin(train_ids)]
        df_test       = df[df["_norm_id"].isin(test_ids)]

        if val_ids is not None:
            df_valid = df[df["_norm_id"].isin(val_ids)]
            df_train = df_train_full
        else:
            # No explicit val list — hold out a fraction of training such
            # that the val set is `val_fraction` of the OVERALL dataset
            # (train + test), matching the convention in the random-split
            # branch where val is 15% of total.
            n_total = len(df_train_full) + len(df_test)
            n_val   = int(round(n_total * cli.val_fraction))
            n_val   = min(n_val, max(1, len(df_train_full) - 1))
            test_size_within_train = n_val / len(df_train_full)
            df_train, df_valid = train_test_split(
                df_train_full,
                test_size=test_size_within_train,
                random_state=42,
            )

        df_train = df_train.drop(columns="_norm_id")
        df_valid = df_valid.drop(columns="_norm_id")
        df_test  = df_test.drop(columns="_norm_id")

        n_matched = len(df_train) + len(df_valid) + len(df_test)
        n_unmatched = len(df) - n_matched
        print(f"Split: {len(df_train)} train / {len(df_valid)} val / {len(df_test)} test")
        if n_unmatched > 0:
            print(f"  [WARN] {n_unmatched} pairs in CSV not found in any split list "
                  f"(check pair_id format consistency)")
        if len(df_train) == 0 or len(df_test) == 0:
            raise RuntimeError(
                "Empty train or test set — pair IDs in the lists do not match "
                "any IDs in the CSV. Check the pair_id format in your CSV "
                "matches what _normalize_pair_id() expects."
            )
    else:
        # Default 70/15/15 random split
        df_train, df_temp = train_test_split(df, test_size=0.3, random_state=42)
        df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
        print(f"Split: {len(df_train)} train / {len(df_valid)} val / {len(df_test)} test")
    train_set = create_dataset_pairs(tokenizer, df_train, max_length=1024)
    valid_set = create_dataset_pairs(tokenizer, df_valid, max_length=1024)
    test_set  = create_dataset_pairs(tokenizer, df_test,  max_length=1024)

    print(f"\nConfiguration:")
    print(f"  Architecture:     ProtT5 + LoRA (rank={cli.lora_rank})"
          + (" (flat baseline — no cross-attention)" if cli.no_cross_attn
             else " + CrossChainAttention + ContactHead"))
    print(f"  Class weights:    {cli.class_weights}  "
          f"({cli.class_weights[1]}x for binding sites)")
    print(f"  Label smoothing:  {cli.label_smoothing}")
    print(f"  Contact lambda:   {cli.contact_lambda}"
          + (" (N/A — cross-attn disabled)" if cli.no_cross_attn else
             " (disabled)" if cli.contact_lambda == 0.0 else ""))
    print(f"  Pos weight cap:   {cli.pos_weight_cap}")
    print(f"  Bind threshold:   {cli.bind_thresh}  "
          f"(sub-matrix threshold — use same value as --pred-threshold in jaccard_filter.py)")
    print(f"  Attn supervision: {cli.attn_supervision_lambda}"
          + (" (disabled)" if cli.attn_supervision_lambda == 0.0
             else " (guiding attention toward known contacts)"))
    print(f"  Detach contact:   {cli.detach_contact}"
          + (" (contact loss trains ContactHead only)" if cli.detach_contact else ""))
    print(f"  Epochs:           {cli.epochs}")
    print(f"  Checkpoint dir:   {CHECKPOINT_DIR}  (save every {cli.save_every} epoch(s))")
    if cli.resume:
        print(f"  Resuming from latest checkpoint in {CHECKPOINT_DIR}")
    class_weights = torch.tensor(cli.class_weights)

    batch, accum, lr, num_workers = 4, 8, 3e-4, 28
    steps_per_epoch = max(1, len(train_set) // (batch * accum))
    warmup_steps    = max(1, int(steps_per_epoch * cli.epochs * 0.05))

    # ── Find latest checkpoint for resuming ───────────────────────────────────
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
            else:
                print(f"  [WARN] No checkpoints found in {CHECKPOINT_DIR} — starting fresh")
        else:
            print(f"  [WARN] Checkpoint dir {CHECKPOINT_DIR} does not exist — starting fresh")

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
        gradient_checkpointing=True, # Remove later!
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
    history = trainer.state.log_history
    Path(cli.model_out).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, cli.model_out)
    print(f"\nCheckpoints saved in: {CHECKPOINT_DIR}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

    # ── Only rank 0 does evaluation, plotting and file saving ─────────────────
    # With torchrun, all processes reach here. Non-zero ranks exit cleanly.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        # Wait for rank 0 to finish, then exit
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        import sys; sys.exit(0)

    # From here: rank 0 only
    print("\nRank 0: running evaluation and saving outputs...")

    # Plot training history
    loss      = [x["loss"]             for x in history if "loss"             in x]
    val_loss  = [x["eval_loss"]        for x in history if "eval_loss"        in x]
    f1        = [x["eval_f1"]          for x in history if "eval_f1"          in x]
    mcc       = [x["eval_mcc"]         for x in history if "eval_mcc"         in x]
    accuracy  = [x["eval_accuracy"]    for x in history if "eval_accuracy"    in x]
    ep_loss   = [x["epoch"]            for x in history if "loss"             in x]
    ep_eval   = [x["epoch"]            for x in history if "eval_loss"        in x]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: loss curves
    ax1b = ax1.twinx()
    ax1.plot(ep_loss, loss,    label="train_loss")
    ax1.plot(ep_eval, val_loss, label="val_loss")
    ax1b.plot(ep_eval, accuracy, color="gray", linestyle="--", label="accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1b.set_ylabel("Accuracy"); ax1b.set_ylim([0, 1])
    lines = ax1.get_lines() + ax1b.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax1.set_title("Loss & Accuracy")

    # Right: F1 and MCC (the meaningful metrics for imbalanced data)
    ax2.plot(ep_eval, f1,  color="blue",   label="F1 (binding sites)")
    ax2.plot(ep_eval, mcc, color="purple", label="MCC")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score")
    ax2.set_ylim([0, 1]); ax2.legend(loc="lower right")
    ax2.set_title("F1 & MCC (imbalance-aware metrics)")

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    print("Plot saved to training_history.png")

    # Test evaluation using the pair-aware test set
    # Unwrap DDP if needed — trainer may have wrapped the model
    eval_model = trainer.model
    if hasattr(eval_model, "module"):
        eval_model = eval_model.module
    eval_model.to(device)
    eval_model.eval()

    df_test[['pair_id','chain_A','chain_B']].to_csv("test_pairs.csv", index=False)

    # Create validation loader for probability saving
    val_loader = DataLoader(valid_set, batch_size=16, shuffle=False,
                             num_workers=0,
                             collate_fn=FastDataCollatorForTokenClassification(tokenizer))
    
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False,
                             num_workers=0,
                             collate_fn=FastDataCollatorForTokenClassification(tokenizer))

    # Save validation probabilities for threshold tuning
    val_probs, val_labels_padded = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Saving validation probabilities"):
            val_labels_padded += batch["labels"].tolist()
            logits = eval_model(batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            val_probs += F.softmax(logits, dim=-1).cpu().tolist()

    # Save test probabilities
    test_probs, test_labels_padded = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Saving test probabilities"):
            test_labels_padded += batch["labels"].tolist()
            logits = eval_model(batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            test_probs += F.softmax(logits, dim=-1).cpu().tolist()
    
    # Flatten and filter out padded positions for probability saving
    def flatten(l): return [x for sub in l for x in sub]
    
    val_probs_flat = np.array(flatten(val_probs))
    val_labels_flat = np.array(flatten(val_labels_padded))
    val_valid_mask = val_labels_flat != -100
    val_probs_clean = val_probs_flat[val_valid_mask]
    val_labels_clean = val_labels_flat[val_valid_mask]
    
    test_probs_flat = np.array(flatten(test_probs))
    test_labels_flat = np.array(flatten(test_labels_padded))
    test_valid_mask = test_labels_flat != -100
    test_probs_clean = test_probs_flat[test_valid_mask]
    test_labels_clean = test_labels_flat[test_valid_mask]
    
    # Save probabilities and labels for external threshold optimization
    probs_dir = Path(cli.probs_out_dir)
    probs_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(probs_dir / "val_probs.npy"),   val_probs_clean)
    np.save(str(probs_dir / "val_labels.npy"),  val_labels_clean)
    np.save(str(probs_dir / "test_probs.npy"),  test_probs_clean)
    np.save(str(probs_dir / "test_labels.npy"), test_labels_clean)
    print(f"Saved val/test probabilities → {probs_dir}/")

    # ── Save predicted contact matrices for Jaccard filter ────────────────────
    # Run one pair at a time so we can store the variable-size (n, m) matrices.
    # Saved as a dict: {pair_id: ndarray of shape (n_a, n_b)} with sigmoid probs.
    # Only saved when cross-attention is enabled (contact head exists).
    if not cli.no_cross_attn:
        print("\nSaving predicted contact matrices for Jaccard filter...")
        contact_matrices = {}

        for sample_idx in tqdm(range(len(test_set)), desc="Contact matrices"):
            sample  = test_set[sample_idx]
            pair_id = str(df_test.iloc[sample_idx].get('pair_id', sample_idx))

            with torch.no_grad():
                eval_model.train()
                result = eval_model(
                    input_ids      = torch.tensor([sample['input_ids']]).to(device),
                    attention_mask = torch.tensor([sample['attention_mask']]).to(device),
                )
                eval_model.eval()

            if isinstance(result, tuple) and len(result) == 2:
                _, contact_logits_list = result
                if contact_logits_list:
                    item = contact_logits_list[0]
                    if isinstance(item, tuple) and len(item) >= 3:
                        # New sub-matrix format: (c_logits, mask_a, mask_b[, attn_ab])
                        c_logits = item[0]
                        mask_a   = item[1]
                        mask_b   = item[2]
                        n_full = mask_a.shape[0]
                        m_full = mask_b.shape[0]
                        # Build full N×M matrix, filling sub-matrix values back in
                        full_mat = np.zeros((n_full, m_full), dtype=np.float32)
                        c_prob   = torch.sigmoid(c_logits).cpu().numpy()
                        idx_a = mask_a.cpu().numpy().nonzero()[0]
                        idx_b = mask_b.cpu().numpy().nonzero()[0]
                        full_mat[np.ix_(idx_a, idx_b)] = c_prob
                        contact_matrices[pair_id] = full_mat
                    else:
                        # Old full-matrix format fallback
                        c_prob = torch.sigmoid(item).cpu().numpy()
                        contact_matrices[pair_id] = c_prob.astype(np.float32)

        import pickle
        payload = {
            "matrices":   contact_matrices,
            "bind_thresh": cli.bind_thresh,
        }
        with open(cli.contact_out, "wb") as f:
            pickle.dump(payload, f)
        print(f"Saved contact matrices for {len(contact_matrices)} pairs "
              f"→ {cli.contact_out}  (bind_thresh={cli.bind_thresh})")

        # ── Contact matrix diagnostics ────────────────────────────────────────
        if contact_matrices:
            all_means  = [m.mean()           for m in contact_matrices.values()]
            all_maxes  = [m.max()            for m in contact_matrices.values()]
            all_frac02 = [np.mean(m > 0.2)   for m in contact_matrices.values()]
            all_frac05 = [np.mean(m > 0.5)   for m in contact_matrices.values()]
            print(f"\n── Contact matrix diagnostics ───────────────────────")
            print(f"  mean prob    : {np.mean(all_means):.4f}")
            print(f"  mean max     : {np.mean(all_maxes):.4f}")
            print(f"  frac > 0.2   : {np.mean(all_frac02):.4f}")
            print(f"  frac > 0.5   : {np.mean(all_frac05):.4f}")
            print(f"  (target: mean<0.15, frac>0.5<0.10)")

    # Generate predictions using default 0.5 threshold for comparison
    predictions, padded_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            padded_labels += batch["labels"].tolist()
            predictions   += eval_model(batch["input_ids"].to(device),
                                   attention_mask=batch["attention_mask"].to(device)
                                   ).logits.argmax(dim=-1).tolist()

    def flatten(l): return [x for sub in l for x in sub]
    preds  = np.array(flatten(predictions))
    labels = np.array(flatten(padded_labels))
    preds  = preds[labels != -100]
    labels = labels[labels != -100]

    from sklearn.metrics import (f1_score, matthews_corrcoef, classification_report,
                                 roc_auc_score, average_precision_score)

    pd.DataFrame(preds).to_csv("preds.csv", index=False)

    # ── Threshold-independent metrics (computed once from probabilities) ───────
    test_pos_probs = test_probs_clean[:, 1]
    auroc = roc_auc_score(test_labels_clean, test_pos_probs)
    auprc = average_precision_score(test_labels_clean, test_pos_probs)
    print(f"\n── Threshold-independent metrics ────────────────────────")
    print(f"AUROC    : {auroc:.4f}")
    print(f"AUPRC    : {auprc:.4f}")
    print(f"  (baseline AUPRC for random = {test_labels_clean.mean():.4f})")
    print(f"  (AUPRC lift over random   = {auprc / test_labels_clean.mean():.2f}x)")

    print(f"\n── Test results (default threshold=0.5) ─────────────────")
    print(f"Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"F1       : {f1_score(labels, preds, pos_label=1, zero_division=0):.4f}")
    print(f"MCC      : {matthews_corrcoef(labels, preds):.4f}")
    print(classification_report(labels, preds,
                                target_names=["non-binding (0)", "binding site (1)"],
                                zero_division=0))
    print("Confusion matrix :\n", confusion_matrix(labels, preds))

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print("\nRunning threshold sweep on validation set...")
    val_pos  = val_probs_clean[:, 1]
    test_pos = test_probs_clean[:, 1]

    thresholds = np.arange(0.05, 0.95, 0.01)
    val_mccs   = []
    for t in thresholds:
        p = (val_pos >= t).astype(int)
        try:    val_mccs.append(matthews_corrcoef(val_labels_clean, p))
        except: val_mccs.append(0.0)

    best_t       = float(thresholds[np.argmax(val_mccs)])
    best_val_mcc = float(np.max(val_mccs))
    print(f"Best threshold: {best_t:.2f}  (val MCC={best_val_mcc:.4f})")

    np.save(str(probs_dir / "threshold_sweep.npy"),
            np.stack([thresholds, np.array(val_mccs)], axis=1))

    test_preds_tuned = (test_pos >= best_t).astype(int)
    pd.DataFrame(test_preds_tuned).to_csv("preds_tuned.csv", index=False)

    print(f"\n── Test results (tuned threshold={best_t:.2f}) ──────────────────")
    print(f"Accuracy : {accuracy_score(test_labels_clean, test_preds_tuned):.4f}")
    print(f"F1       : {f1_score(test_labels_clean, test_preds_tuned, pos_label=1, zero_division=0):.4f}")
    print(f"MCC      : {matthews_corrcoef(test_labels_clean, test_preds_tuned):.4f}")
    print(f"AUROC    : {auroc:.4f}  (threshold-independent, same as above)")
    print(f"AUPRC    : {auprc:.4f}  (threshold-independent, same as above)")
    print(classification_report(test_labels_clean, test_preds_tuned,
                                target_names=["non-binding (0)", "binding site (1)"],
                                zero_division=0))
    print("Confusion matrix (tuned):\n", confusion_matrix(test_labels_clean, test_preds_tuned))
