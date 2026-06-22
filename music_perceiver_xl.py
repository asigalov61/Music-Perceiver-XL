#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#
#	Music Perceiver XL Python Module
#	Version 1.0
#
#	Project Los Angeles
#
#	Tegridy Code 2026
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#
#   Copyright 2026 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
'''

###################################################################################
###################################################################################

print('=' * 70)
print('Loading Music Perceiver XL Python module...')
print('Please wait...')
print('=' * 70)

__version__ = '1.0.0'

print('Music Perceiver XL module version', __version__)
print('=' * 70)

###################################################################################
###################################################################################

import os
import json
import tqdm

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score

from x_transformer_2_3_1 import (
    TransformerWrapper,
    Decoder,
    Attention,
    FeedForward,
    RMSNorm,
    AutoregressiveWrapper
)

###################################################################################

class PerceiverMemory(nn.Module):
    def __init__(self, dim_in, latent_dim, num_latents=64, heads=8):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.self_attn = Attention(dim=latent_dim, heads=heads, flash=True)
        self.ff = FeedForward(latent_dim)
        self.to_latent_dim = nn.Linear(dim_in, latent_dim)
        self.norm = RMSNorm(latent_dim)

    def forward(self, x, mask=None):
        b = x.size(0)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)
        combined = torch.cat([latents, self.to_latent_dim(x)], dim=1)
        
        if mask is not None:
            # Perceiver mask convention: True = IGNORE (Padding)
            latents_mask = torch.zeros(b, self.num_latents, device=x.device, dtype=torch.bool)
            mask = torch.cat([latents_mask, mask], dim=1)

        combined = self.self_attn(combined, mask=mask) + combined
        combined = self.ff(combined) + combined
        return self.norm(combined[:, :self.num_latents])

###################################################################################

class MusicPerceiverXL(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim=512,
        depth=8,
        heads=8,
        max_seq_len=1024,
        bar_latent_dim=512,
        num_bar_latents=8,
        global_latent_dim=512,
        num_global_latents=16,
        bar_tok_ids_range=range(384, 512)
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_bar_latents = num_bar_latents
        self.num_global_latents = num_global_latents

        self.output_is_log_prob = False
        self.add_continuous_pred_head = False
        
        # Enable KV caching for fast generation
        self.can_cache_kv = True
        # Allow generation BEYOND max_seq_len
        self.can_cache_kv_outside_max_seq_len = True

        bar_tok_mask = torch.zeros(num_tokens, dtype=torch.bool)
        bar_tok_mask[list(bar_tok_ids_range)] = True
        self.register_buffer('bar_tok_mask', bar_tok_mask)

        self.bar_memory = PerceiverMemory(dim, bar_latent_dim, num_bar_latents, heads)
        self.global_memory = PerceiverMemory(bar_latent_dim, global_latent_dim, num_global_latents, heads)
        self.bar_to_dec = nn.Linear(bar_latent_dim, dim)
        self.global_to_dec = nn.Linear(global_latent_dim, dim)
        self.bar_index_emb = nn.Embedding(max_seq_len, bar_latent_dim)

        self.decoder = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=dim, depth=depth, heads=heads,
                cross_attend=True, rotary_pos_emb=True, attn_flash=True
            )
        )

        self.token_emb = self.decoder.token_emb

    def clear_kv_cache(self):
        if hasattr(self.decoder, 'attn_layers'):
            self.decoder.attn_layers.cache = None

    def build_hierarchical_memory(self, tokens):
        b, seq_len = tokens.shape
        device = tokens.device
        dim = self.dim
        num_bar_latents = self.num_bar_latents
        num_global_latents = self.num_global_latents

        tok_emb = self.token_emb(tokens)

        is_bar_tok = self.bar_tok_mask[tokens]
        is_bar_tok[:, 0] = False
        bar_idx = is_bar_tok.long().cumsum(dim=1)

        max_bars = bar_idx.max().item() + 1
        total_bars = b * max_bars

        batch_idx = torch.arange(b, device=device).unsqueeze(1)
        global_bar = (batch_idx * max_bars + bar_idx).reshape(-1)

        sort_key = global_bar * 2048 + torch.arange(b * seq_len, device=device)
        perm = sort_key.argsort()
        
        sorted_emb = tok_emb.reshape(-1, dim)[perm]
        sorted_bar = global_bar[perm]

        is_new = torch.cat([torch.ones(1, dtype=torch.bool, device=device),
                           sorted_bar[1:] != sorted_bar[:-1]])
        pos_in_bar = is_new.long().cumsum(0) - 1

        bar_lengths = torch.zeros(total_bars, dtype=torch.long, device=device)
        bar_lengths.scatter_add_(0, global_bar, torch.ones_like(global_bar))
        max_bar_len = max(bar_lengths.max().item(), 1)

        padded = tok_emb.new_zeros(total_bars, max_bar_len, dim)
        bar_mask = torch.zeros(total_bars, max_bar_len, dtype=torch.bool, device=device)
        
        dest = sorted_bar * max_bar_len + pos_in_bar
        valid = pos_in_bar < max_bar_len
        
        if valid.any():
            dest_valid = dest[valid]
            emb_valid = sorted_emb[valid]
            
            dest_expanded = dest_valid.unsqueeze(-1).expand(-1, dim)
            padded.reshape(-1, dim).scatter_(0, dest_expanded, emb_valid)
            bar_mask.reshape(-1).scatter_(0, dest_valid, False)

        bar_latents = self.bar_memory(padded, mask=bar_mask)
        bar_latent_dim = bar_latents.size(2)

        bar_latents = bar_latents.view(b, max_bars, num_bar_latents, bar_latent_dim)
        bar_indices = torch.arange(max_bars, device=device).clamp(max=self.max_seq_len - 1)
        bar_latents = bar_latents + self.bar_index_emb(bar_indices).unsqueeze(1)

        bar_ctx = bar_latents.reshape(b, -1, bar_latent_dim)
        
        bars_per_seq = bar_idx.max(dim=1)[0] + 1
        bar_valid = torch.arange(max_bars, device=device).unsqueeze(0) < bars_per_seq.unsqueeze(1)
        bar_ctx_mask = ~(bar_valid.unsqueeze(-1).expand(-1, -1, num_bar_latents).reshape(b, -1))

        global_latents = self.global_memory(bar_ctx, mask=bar_ctx_mask)

        context = torch.cat([self.global_to_dec(global_latents), self.bar_to_dec(bar_ctx)], dim=1)
        global_mask = torch.zeros(b, num_global_latents, dtype=torch.bool, device=device)
        context_mask = torch.cat([global_mask, bar_ctx_mask], dim=1)

        # Forcibly pads context_mask to (max_seq_len - 1) which breaks flash attention 
        # if our dynamic context length doesn't match. We manually pad it here.
        ctx_len = context.shape[1]
        target_ctx_len = self.max_seq_len - 1
        if ctx_len < target_ctx_len:
            pad_len = target_ctx_len - ctx_len
            context = torch.cat([context, context.new_zeros(b, pad_len, context.shape[2])], dim=1)
            context_mask = torch.cat([context_mask, context_mask.new_ones(b, pad_len)], dim=1)

        return context, context_mask

    def forward(self, x, **kwargs):
        context, context_mask = self.build_hierarchical_memory(x)
        kwargs.update(context=context, context_mask=context_mask)
        return self.decoder(x, **kwargs)
    
###################################################################################

class TokenizedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx])
    
###################################################################################
    
def create_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
###################################################################################

def save_metrics(losses, accuracies):
    """Save all historical losses and accuracies to a JSON file."""
    metrics = {
        "steps": list(range(1, len(losses) + 1)),
        "loss": losses,
        "accuracy": accuracies
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)
        
###################################################################################

def plot_metrics(losses, accuracies, current_step):
    """Generate, show inline, and save a side-by-side Loss and Accuracy plot."""
    steps = list(range(1, len(losses) + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Training Metrics (Step {current_step})', fontsize=16)

    # Set frame/axes background color to white so grids are clearly visible
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # Loss Plot
    ax1.plot(steps, losses, color='royalblue', linewidth=1.5)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Accuracy Plot
    ax2.plot(steps, accuracies, color='crimson', linewidth=1.5)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Save the plot to disk (pass facecolor to ensure the white background saves properly)
    plot_path = os.path.join(PLOT_DIR, f"metrics_step_{current_step}.png")
    plt.savefig(plot_path, dpi=150, facecolor='white')

    # Show the plot inline
    plt.show()
    
    plt.close(fig)
    
###################################################################################

def save_checkpoint(model, optimizer, epoch, step, loss, acc):
    """Save model and optimizer state dicts."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{step}_loss_{loss:.4f}_acc_{acc:.4f}.pth")
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc
    }, checkpoint_path)
    print(f"  >>> Checkpoint saved to {checkpoint_path}")
    
###################################################################################

def collate_fn(batch):
    """Pads and truncates a list of 1D token sequences to MAX_SEQ_LEN."""
    processed_seqs = []
    for seq in batch:
        # Truncate if longer than MAX_SEQ_LEN
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
        # Pad if shorter than MAX_SEQ_LEN
        elif len(seq) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(seq)
            seq = torch.cat([seq, torch.full((pad_len,), PAD_VALUE, dtype=seq.dtype)])
            
        processed_seqs.append(seq)
        
    # Stack into a single tensor of shape (batch_size, MAX_SEQ_LEN)
    return torch.stack(processed_seqs)

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the Music Perceiver XL Python module
###################################################################################