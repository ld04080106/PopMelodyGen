#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py

Author:      <ludy>
Modified:    2026/06/10
Licence:     <MIT>
"""
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class MelodyEventsDataset(Dataset):
    def __init__(self, path_meta_jsonl, 
                       max_len, 
                       split_group): 
        self.items = []

        # read meta.jsonl
        with open(path_meta_jsonl, "r", encoding="utf-8") as f:
            for aline in f:
                if not aline.strip():
                    continue
                r = json.loads(aline)
                if r["split"] != split_group:
                    continue
                # save
                self.items.append(r)

        self.max_len = max_len
        token_lengths = [int(item["n_tokens"]) for item in self.items if item.get("n_tokens") is not None]
        self.max_observed_len = max(token_lengths, default=0)
        self.n_over_max_len = sum(length > self.max_len for length in token_lengths)

        for item in self.items:
                item["weight"] = 1.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        ev = np.load(item["path_event"]).astype(np.int64)  # shape [L]
        # Song endings contain cadence/EOS information and must not be silently truncated.
        if len(ev) > self.max_len:
            raise ValueError(
                f"Sequence {item['id']} has {len(ev)} tokens, exceeding max_len={self.max_len}"
            )
        return {
            "id": item["id"],
            "events": torch.tensor(ev, dtype=torch.long),
            "weight": torch.tensor(item["weight"], dtype=torch.float32),
        }


class MixedMelodyEventsDataset(Dataset):
    """Primary dataset plus a fixed replay sample from a secondary dataset."""

    def __init__(
        self,
        path_primary_meta_jsonl,
        path_replay_meta_jsonl,
        max_len,
        split_group,
        replay_ratio=1.0,
        seed=47,
    ):
        self.primary = MelodyEventsDataset(path_primary_meta_jsonl, max_len, split_group)
        self.replay = MelodyEventsDataset(path_replay_meta_jsonl, max_len, split_group)
        replay_ratio = float(replay_ratio)
        if replay_ratio < 0:
            raise ValueError("replay_ratio must be >= 0")

        n_replay = int(round(len(self.primary) * replay_ratio))
        rng = random.Random(seed)
        replay_items = []
        if n_replay > 0:
            if len(self.replay) == 0:
                raise ValueError(
                    f"Replay split is empty: split={split_group}, meta={path_replay_meta_jsonl}"
                )
            if n_replay <= len(self.replay):
                replay_items = rng.sample(self.replay.items, n_replay)
            else:
                replay_items = [rng.choice(self.replay.items) for _ in range(n_replay)]

        self.items = list(self.primary.items) + replay_items
        self.max_len = max_len
        self.n_primary = len(self.primary)
        self.n_replay = len(replay_items)
        self.replay_ratio = replay_ratio
        self.max_observed_len = max(
            self.primary.max_observed_len,
            self.replay.max_observed_len if self.n_replay > 0 else 0,
        )
        self.n_over_max_len = self.primary.n_over_max_len
        if self.n_replay > 0:
            self.n_over_max_len += sum(
                int(item["n_tokens"]) > self.max_len
                for item in replay_items
                if item.get("n_tokens") is not None
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        ev = np.load(item["path_event"]).astype(np.int64)
        if len(ev) > self.max_len:
            raise ValueError(
                f"Sequence {item['id']} has {len(ev)} tokens, exceeding max_len={self.max_len}"
            )
        return {
            "id": item["id"],
            "events": torch.tensor(ev, dtype=torch.long),
            "weight": torch.tensor(item["weight"], dtype=torch.float32),
        }

def collate_batch(batch, pad_id):
    lens = [b["events"].numel() for b in batch]
    max_len = max(lens)
    B = len(batch)

    x = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.bool)
    w = torch.stack([b["weight"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        L = b["events"].numel()
        x[i, :L] = b["events"]
        attn[i, :L] = 1

    # GPT-style: input = x[:,:-1], target = x[:,1:]
    return {
        "x": x[:, :-1],
        "y": x[:, 1:],
        "attn": attn[:, :-1],
        "weight": w,
    }
