#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_preprocess_synthetic_data.py

Preprocess synthesized dataset for LM training.

This script tokenizes each synthetic MIDI/metadata pair, writes one token-id file
per sample, and builds a unified `meta.jsonl` for training.

Typical usage:
  python 02_preprocess_synthetic_data.py --in_dir _data_synth --out_dir _preprocessed_synth

Inputs (from --in_dir):
  - Per-sample metadata files (e.g., *.json) produced by Step 01

Outputs (to --out_dir):
  - meta.jsonl: one JSON object per sample, containing at least:
      - paths to MIDI (and audio if available)
      - key / mode / register / tempo (class & BPM)
      - structure boundaries and bar-wise chords
      - split tag: {"split": "train" | "val" | "test"}

Split policy:
  - MIDI files are grouped by tempo class first, then each tempo group is split
    independently using a deterministic random seed.

Author:      <ludy>
Modified:    2026/06/10
Licence:     <MIT>
"""
import os
import glob
import hashlib
import json
import argparse
import pretty_midi
import numpy as np
from tqdm import tqdm
from model.tokenizer import REMITokenizer 

DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
DEFAULT_SPLIT_SEED = 47
TEMPO_NAMES = {"SLOW", "MID", "FAST"}

def _sha1_of_int_list(ints):
    b = np.asarray(ints, dtype=np.int32).tobytes()
    return hashlib.sha1(b).hexdigest()

def infer_tempo_group(path_midi):
    base_fn = os.path.splitext(os.path.basename(path_midi))[0]
    prefix = base_fn.split("_", 1)[0].upper()
    if prefix in TEMPO_NAMES:
        return prefix
    return "UNKNOWN"

def split_counts(n_items, train_ratio, eval_ratio, test_ratio):
    if n_items <= 0:
        return 0, 0, 0
    ratios = np.asarray([train_ratio, eval_ratio, test_ratio], dtype=float)
    ratios = ratios / ratios.sum()
    raw = ratios * n_items
    counts = np.floor(raw).astype(int)
    remainder = int(n_items - counts.sum())
    for idx in np.argsort(raw - counts)[::-1][:remainder]:
        counts[idx] += 1

    if n_items >= 3:
        for idx in (1, 2):
            if counts[idx] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1
    return tuple(int(x) for x in counts)

def build_split_map(midi_paths, train_ratio, eval_ratio, test_ratio, split_seed):
    grouped = {}
    for path_midi in midi_paths:
        grouped.setdefault(infer_tempo_group(path_midi), []).append(path_midi)

    split_map = {}
    split_report = {}
    for tempo_group, paths in sorted(grouped.items()):
        paths = sorted(paths)
        rng = np.random.default_rng(split_seed + sum(ord(ch) for ch in tempo_group))
        order = rng.permutation(len(paths))
        n_train, n_eval, n_test = split_counts(len(paths), train_ratio, eval_ratio, test_ratio)

        split_report[tempo_group] = {
            "total": len(paths),
            "train": n_train,
            "eval": n_eval,
            "test": n_test,
        }

        for rank, idx in enumerate(order):
            path = paths[int(idx)]
            if rank < n_train:
                split_group = "train"
            elif rank < n_train + n_eval:
                split_group = "eval"
            else:
                split_group = "test"
            split_map[path] = split_group
    return split_map, split_report

def preprocess(in_dir, out_dir, meta_fn, split_ratios=DEFAULT_SPLIT_RATIOS, split_seed=DEFAULT_SPLIT_SEED):
    # check input dir
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Error: input dir not found ({in_dir})")

    # init tokenizer
    tokenizer = REMITokenizer()
    print(f"> Tokenizer initialized with vocab size: {len(tokenizer)}")

    # read all input files(xxx.mid, xxx_meta.json)
    midi_paths = sorted(glob.glob(os.path.join(in_dir, "*.mid")))
    if not midi_paths:
        raise FileNotFoundError(f"Error: No MIDI found in {in_dir}")

    split_map, split_report = build_split_map(midi_paths, *split_ratios, split_seed)
    print("> Split by tempo group:")
    for tempo_group, stats in split_report.items():
        print(f">     - {tempo_group}: total={stats['total']} train={stats['train']} eval={stats['eval']} test={stats['test']}")

    # create output dir
    os.makedirs(out_dir, exist_ok=True)
    path_out_jsonl = os.path.join(out_dir, "meta.jsonl")

    exact_seen = set()
    cnter = 0
    skipped_exact = 0
    missing_meta = 0
    failed_tokenizer = 0
    max_token_len = 0
    num_train = num_eval = num_test = 0

    with open(path_out_jsonl, "w", encoding="utf-8") as fout:
        for path_midi in tqdm(midi_paths):
            base_fn = os.path.splitext(os.path.basename(path_midi))[0]  # 00001_MID_female_key6_mode0_1_melody
            path_meta = os.path.join(in_dir, f"{base_fn}.json")

            # read meta file
            if not os.path.exists(path_meta):
                # no .meta
                missing_meta += 1
                continue
            else:
                with open(path_meta, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                #meta["id"] = meta.get("id", base_fn)

            # MIDI -> tokens
            try:
                rtn = tokenizer.encode(path_midi, meta_dict, midi_pitch_offset=0) # set -12 if MIDI_PITCH_OFF=12 at step01
                if rtn == False:
                    failed_tokenizer += 1
                    print("[Warning] tokenization failed:", base_fn)
                    continue
                else:
                    # OK
                    tokens, ids = rtn
            except Exception as e:
                raise ValueError("[Warning] tokenization failed:", base_fn, str(e))

            # Remove duplicated token
            h = _sha1_of_int_list(ids)
            if h in exact_seen:
                skipped_exact += 1
                print(f"[INFO] ignored tokens for path ({base_fn})")
                continue
            exact_seen.add(h)

            # Save token.npy
            path_event = os.path.join(out_dir, f"{base_fn}.npy")
            np.save(path_event, np.asarray(ids, dtype=np.int32))

            # Save token.txt
            path_txt = os.path.join(out_dir, f"{base_fn}.txt")
            with open(path_txt, "w", encoding="utf-8") as f:
                json.dump(tokens, f, ensure_ascii=False, indent=2)

            # split group: train/eval/test
            split_group = split_map[path_midi]
            if split_group == "eval":
                num_eval += 1
            elif split_group == "test":
                num_test += 1
            else:
                num_train += 1

            # Write a line into meta.jsonl
            len_tokens = len(tokens)
            aline = {
                "id": base_fn,
                "path_event": os.path.abspath(path_event),
                # split group: train/eval/test
                "split": split_group,
                # stat
                "n_tokens": len_tokens,
                "token_sha1": h,
            }
            # write json
            fout.write(json.dumps(aline, ensure_ascii=False) + "\n")

            # update
            cnter += 1
            if len_tokens > max_token_len:
                max_token_len = len_tokens

    # Print result
    print("---------------------------------------------------")
    print(f"[INFO] Finished {cnter} midi files.")
    print(f"[INFO] max_token_len = {max_token_len}")

    # report
    report = {
        "dataset_root": in_dir,
        "out_root": out_dir,
        "n_midi_found": len(midi_paths),
        "preprocessed": cnter,
        "skipped_exact_duplicate": skipped_exact,
        "missing_meta": missing_meta,
        "failed_tokenizer": failed_tokenizer,
        "num_train": num_train,
        "num_eval": num_eval,
        "num_test": num_test,
        "split_seed": split_seed,
        "split_ratios": {
            "train": split_ratios[0],
            "eval": split_ratios[1],
            "test": split_ratios[2],
        },
        "split_by_tempo": split_report,
    }
    with open(os.path.join(out_dir, "preprocess_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", type=str, default="_data_synth")
    p.add_argument("--out_dir", type=str, default="_preprocessed_synth")

    p.add_argument("--train_ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0])
    p.add_argument("--eval_ratio", type=float, default=DEFAULT_SPLIT_RATIOS[1])
    p.add_argument("--test_ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2])
    p.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    return p.parse_args()

# =========================
# Main
# =========================
if __name__ == "__main__":
    args = parse_args()

    meta_fn = "meta.jsonl"

    split_ratios = (args.train_ratio, args.eval_ratio, args.test_ratio)
    preprocess(args.in_dir, args.out_dir, meta_fn, split_ratios, args.split_seed)
