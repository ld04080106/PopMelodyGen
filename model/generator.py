#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generator.py

Author:      <ludy>
Modified:    2026/06/19
Licence:     <MIT>
"""
import torch
import numpy as np
import os
import json
import random
import argparse
import pretty_midi
from tqdm import tqdm
from datetime import datetime
from model.tokenizer import REMITokenizer 
from model.model import GPTMelody

"""
1. given prompt (<BOS> KEY_0 MODE_0 TEMPO_MID STRUCT_* ...)
2. prompt -> tokens
3. tokens -> MIDI (pretty_midi)
"""
_MODEL = None
_ID_TO_TOKEN = None
_TOKEN_TO_ID = None
_MAX_LEN = None
_PATH_CKPT = None
_DEVICE = None

def load_once(path_ckpt, device=None):
    global _MODEL, _ID_TO_TOKEN, _TOKEN_TO_ID, _MAX_LEN, _PATH_CKPT, _DEVICE
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)
    path_ckpt = os.path.abspath(path_ckpt)

    if _MODEL is not None and _PATH_CKPT == path_ckpt and _DEVICE == device:
        return {
            "model": _MODEL,
            "id_to_token": _ID_TO_TOKEN,
            "token_to_id": _TOKEN_TO_ID,
            "max_len": _MAX_LEN,
        }

    # load model
    ckpt = torch.load(path_ckpt, map_location=device)

    # load config
    id_to_token = ckpt["vocab"]
    token_to_id = {t:i for i,t in enumerate(id_to_token)}
    max_len = ckpt.get("config", {}).get("max_len", 1200)
    vocab_size=len(id_to_token)
 
    # init model
    model = GPTMelody(vocab_size=vocab_size, 
                      d_model=512, 
                      n_head=8, 
                      n_layer=6, 
                      max_len=max_len).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # save
    _MODEL = model
    _ID_TO_TOKEN = id_to_token
    _TOKEN_TO_ID = token_to_id
    _MAX_LEN = max_len
    _PATH_CKPT = path_ckpt
    _DEVICE = device
    model_dict = {"model": model,
                  "id_to_token": id_to_token,
                  "token_to_id": token_to_id,
                  "max_len": max_len
                 }
    return model_dict

@torch.no_grad()
def gen_melodies_by_prompt(model_dict, out_dir, prompt_dicts,
                       temperature, top_p, device):
    """ prompt -> midi by model 
    """ 
    model = model_dict["model"]
    id_to_token = model_dict["id_to_token"]
    token_to_id = model_dict["token_to_id"]
    max_len = model_dict["max_len"]

    # generate songs (by prompts)
    rtn_paths = []
    for prompt_dict in tqdm(prompt_dicts):
        # prompt
        prompt_tokens = build_prompt(prompt_dict)
        prompt_ids = [token_to_id[x] for x in prompt_tokens]
        print(" > id =", prompt_dict["id"])

        # sections
        sections = gen_sections(prompt_dict["struct"])
        sec_names, sec_nbars = sections

        # chords
        if "chord_per_bar_list" in prompt_dict:
            chord_per_bar_list = prompt_dict["chord_per_bar_list"]
        else:
            chord_per_bar_list = gen_chord_prog(sec_names, sec_nbars, prompt_dict["tempo"])
        
        # gen: prompt -> token
        out_tokens = gen_token_by_prompt(model, id_to_token, max_len, 
                                     prompt_ids, sections, chord_per_bar_list,
                                     temperature, top_p, device)

        # bpm
        bpm = gen_bpm(prompt_dict["tempo"])
     
        # export midi (melody)
        path_out_mid = os.path.join(out_dir, prompt_dict["id"]+".mid")
        tokens_to_midi(out_tokens, path_out_mid, bpm)
        print("> midi saved:", path_out_mid)

        # export meta
        meta_dict = {"tempo_class": prompt_dict["tempo"],
                     "key": prompt_dict["key"],
                     "mode": 0,
                     "bpm": bpm,
                     "gender": prompt_dict["gender"],
                     "song_id": prompt_dict["id"],
                     "structure": []
                    }
        bi = 0
        for i, (sec_name, sec_nbar) in enumerate(zip(sec_names, sec_nbars)):
            meta_dict["structure"].append({"section_index": i,
                                           "bar_index": bi,
                                           "type": sec_name,
                                           "num_bars": sec_nbar, 
                                           "chords": [],
                                          })
            for j in range(sec_nbar):
                meta_dict["structure"][-1]["chords"].append(chord_per_bar_list[bi+j])
            # update 
            bi += sec_nbar

        # write JSON
        path_out_json = path_out_mid[:-3] + "json"
        with open(path_out_json, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)
        print("> meta saved:", path_out_json, " @", datetime.now())

        # save path
        rtn_paths.append(path_out_mid)
    # return
    return rtn_paths

@torch.no_grad()
def gen_token_by_prompt(model, id_to_token, max_len, 
                        prompt_ids, sections, chord_per_bar_list,
                        temperature, top_p, device):
    """ prompt -> token """
    token_to_id = {t:i for i,t in enumerate(id_to_token)}
    model.eval()
    #device = next(model.parameters()).device
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(x, dtype=torch.bool)
    eos_id = token_to_id["<EOS>"]

    out_ids = list(prompt_ids)

    def append_token(tid):
        nonlocal x, out_ids
        out_ids.append(tid)
        x = torch.tensor([out_ids], dtype=torch.long, device=device)

    curb = 7
    while len(out_ids) < max_len:
        # sample next token
        nxt = _sample_next(model, x, temperature, top_p)
        tid = int(nxt.item())
        t = id_to_token[tid] # if 0 <= tid < len(id_to_token) else "<UNK>"

        # EOS?
        if tid == eos_id:
            break

        # Outro?
        if id_to_token[tid] == "SEC_OUTRO":
            append_token(eos_id)
            break
        
        append_token(tid)

        # online injection (CH_*)
        if t == "BAR":
            # Chord of current bar
            if curb < len(chord_per_bar_list): 
                curch = chord_per_bar_list[curb]

                # 1bar = 2chords
                if type(curch) in (list, tuple): 
                    ch1, ch2 = curch[0], curch[1]
                    append_token(token_to_id[f"CH_{ch1}"])
                    append_token(token_to_id[f"CH_{ch2}"])
                elif '(' in curch or '[' in curch:
                    curch = curch.strip('(').strip('[').strip(')').strip(']')
                    curch = curch.split(',')
                    ch1, ch2 = curch[0].strip(), curch[1].strip()
                    append_token(token_to_id[f"CH_{ch1}"])
                    append_token(token_to_id[f"CH_{ch2}"])
                else:
                    append_token(token_to_id[f"CH_{curch}"])
                    append_token(token_to_id[f"CH_{curch}"])
            else:
                append_token(eos_id)
                break
            # update 
            curb += 1
        else:
            pass
 
    out_ids = x.squeeze(0).tolist()
    out_tokens = [id_to_token[x] for x in out_ids]

    #print("out_ids =", out_ids)
    #print("out_tokens =", out_tokens)

    return out_tokens

@torch.no_grad()
def _sample_next(model, x, temperature=1.2, top_p=0.95):
    # x: (1, T)
    attn = torch.ones_like(x, dtype=torch.bool)

    logits = model(x, attn_mask=attn)[:, -1, :]  # (1, V)
    logits = logits / max(1e-6, temperature)

    # top-p
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = -1e9
    filtered = torch.zeros_like(logits)
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

    probs = torch.softmax(filtered, dim=-1)
    nxt = torch.multinomial(probs, num_samples=1)  # (1,1)
    return nxt

def build_prompt(prompt_dict):
    # by tokenizer
    tokens = ["<BOS>",]
    tokens.append("KEY_{}".format(prompt_dict["key"]))
    tokens.append("MODE_0")
    tokens.append("TEMPO_{}".format(prompt_dict["tempo"]))
    tokens.append("GENDER_{}".format(prompt_dict["gender"]))
    tokens.append("STRUCT_{}".format(prompt_dict["struct"]))

    return tokens

def gen_sections(struct):
    # sections of structure
    sec_dict = {"A16B16": (["INTRO", "VERSE1", "CHORUS1", "OUTRO"], [8, 16, 16, 8]),
                "A16C8B16": (["INTRO", "VERSE1", "PRECHORUS", "CHORUS1", "OUTRO"], [8, 16, 8, 16, 8]),
                "A8A8B16": (["INTRO", "VERSE1", "VERSE2", "CHORUS1", "OUTRO"], [8, 8, 8, 16, 8]),
                "A16A16B16": (["INTRO", "VERSE1", "VERSE2", "CHORUS1", "OUTRO"], [8, 16, 16, 16, 8]),
                "A8A8C8B16": (["INTRO", "VERSE1", "VERSE2", "PRECHORUS", "CHORUS1", "OUTRO"], [8, 8,8,8, 16, 8]),
                "A16A16C8B16": (["INTRO", "VERSE1", "VERSE2", "PRECHORUS", "CHORUS1", "OUTRO"], [8, 16, 16, 8, 16, 8]),
                "A16B8B8": (["INTRO", "VERSE1", "CHORUS1", "CHORUS2", "OUTRO"], [8, 16, 8,8, 8]),
                "A16B16B16": (["INTRO", "VERSE1", "VERSE2", "CHORUS1", "OUTRO"], [8, 16, 16, 16, 8]),
                "A16C8B8B8": (["INTRO", "VERSE1", "PRECHORUS", "CHORUS1", "CHORUS2", "OUTRO"], [8, 16, 8,8,8, 8]),
                "A16C8B16B16": (["INTRO", "VERSE1", "PRECHORUS", "CHORUS1", "CHORUS2", "OUTRO"], [8, 16, 8, 16, 16, 8]),
                "A8A8B8B8": (["INTRO", "VERSE1", "VERSE2", "CHORUS1", "CHORUS2", "OUTRO"], [8, 8,8,8,8, 8]),
                "A16A16B8B8": (["INTRO", "VERSE1", "VERSE2", "CHORUS1", "CHORUS2", "OUTRO"], [8, 16, 16, 8,8, 8]),
                "A8A8B16B16": (["INTRO", "VERSE1", "VERSE2", "CHORUS1", "CHORUS2", "OUTRO"], [8, 8,8, 16, 16, 8]),
                "A16A16B16B16": (["INTRO", "VERSE1", "VERSE2", "CHORUS1", "CHORUS2", "OUTRO"], [8, 16, 16, 16, 16, 8]),
                "A8A8C8B8B8": (["INTRO", "VERSE1", "VERSE2", "PRECHORUS", "CHORUS1", "CHORUS2", "OUTRO"], [8, 8,8,8,8,8, 8]),
                "A16A16C8B8B8": (["INTRO", "VERSE1", "VERSE2", "PRECHORUS", "CHORUS1", "CHORUS2", "OUTRO"], [8, 16, 16, 8,8,8, 8]),
                "A8A8C8B16B16": (["INTRO", "VERSE1", "VERSE2", "PRECHORUS", "CHORUS1", "CHORUS2", "OUTRO"], [8, 8,8,8, 16, 16, 8]),
                "A16A16C8B16B16": (["INTRO", "VERSE1", "VERSE2", "PRECHORUS", "CHORUS1", "CHORUS2", "OUTRO"], [8, 16, 16, 8, 16,16, 8]),
                }
    secs, nbars = sec_dict[struct]

    return secs, nbars

def gen_chord_prog(sec_names, sec_nbars, tempo_class):
    # chord_prog templates for test, can be modified (or expanded) in online demo
    # Return:
    #        chords - chords of each bar
    CHORD_PROG_VERSE_TEMPLATES = {
        "SLOW": {8: [["vi", "iii", "IV", "V", "vi", "iii", "IV", "V"],
                     ["I", "vi", ("IV","V"), "I", "I", "vi", ("IV","V"), "I"], 
                     [("I","IV"), ("V","I"), ("vi7","iii"), ("IV","V"), ("I","IV"), ("V","I"), ("vi7","II7"), ("V","I")],
                     [("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V"), ("I","vi"), ("IV","V")], 
                     ],
                16: [["I", "IV", "V", "vi", ("IV","V"), ("I","vi"), "IV", "V",    "I", "IV", "V", "vi", ("IV","V"), ("I","vi"), ("IV","V"), "I"], 
                     ["vi", "iii", "IV", "V", "vi", "iii", "IV", ("V","III"),  "vi", "iii", "IV", "V", "vi", "iii", "IV", "V"], 
                     ["I", "V", "vi", "iii7", "IV", "I", "IV", "V",   "I", "V", "vi", "iii7", "IV", "I", ("IV","V"), "I" ],
                     [("I","V"), ("vi","I"), ("IV","iii7"), ("ii7","V7"), ("I","V"), ("v","VI7"), ("ii7","V7"), "I",  ("I","V"), ("vi","I"), ("IV","iii7"), ("ii7","V7"), ("I","V"), ("v","VI7"), ("ii7","V7"), "I"],
                    ],
                }, 
        "MID": {8: [ ["I", "vi", "IV", "V", "I", "vi", "IV", "V"],
                     ["I", "iii", "IV", "V", "I", "V", "IV", "V"], 
                     [("IV","V"), "vi", ("IV","V"), "vi", ("IV","V"), "vi", ("IV","V"), "vi"],
                     ["I", "vi", "IV", "V", "iii7", "vi", "IV", "V"],
                     ],
                16: [["I", "V", "vi", "iii", "IV", "I", "ii", "V",  "I", "V", "vi", "iii", "IV", "I", "ii", "V"],
                     ["I","III", "vi","iii", "IV","I", "ii","V", "I","III", "vi","iii", "IV","I", "bIII","V"],
                     ["I", "iii", "vi", "iii7", ("ii7","V"), ("iii7","vi"), "II", "V",  "I", "iii", "vi", "iii7", ("ii7","V"), ("iii7","vi"), ("II","V"), "I"], 
                     ["vi", "iii", "IV", "V", "vi", "iii", "IV", ("V","III"),  "vi", "iii", "IV", "V", "vi", "iii", "IV", "V"],
                    ],
                },  
        "FAST": {8:[ ["I", "vi", "IV", "V", "I", "vi", "IV", "V"], 
                     ["I", "iii", "IV", "V", "I", "V", "IV", "V"],
                     ["I", "vi", "IV", "V", "I", "vi", "IV", "V"],
                     ["I", "vi", "IV", "V", "iii7", "vi", "IV", "V"], 
                    ],
                16: [["I","V", "vi","I", "IV","V", "I","V",   "I","V", "vi","I", "IV","V", "I","I"],
                     ["I", "V", "vi7", "III", "IV", "I", "ii7", "V",  "I", "V", "vi7", "III", "IV", "I", "ii7", "V" ], 
                     ["I", "V", "vi", "iii", "IV", "I", "V", "V",  "I", "V", "vi", "IV", "ii", "V", "I", "I"],
                     ["I", "ii7", "iii", "vi", "IV", "ii7", "V7", "I",   "I", "ii7", "iii", "vi", "IV", "ii7", "V7", "I"],
                    ],
                },
        } 
    CHORD_PROG_PRECHORUS_TEMPLATES = {
        "SLOW": {8: [[("IV","I"), ("ii","V"), ("iii","vi"), ("ii","V"), ("IV","I"), ("ii","V"), ("iii","vi"), ("ii","V")], 
                     [("IV","V"), "I", ("IV","V"), ("I","V"), ("IV","V"), "I", ("IV","V"), "I"], 
                     ["vi", "iii", ("ii","V"), "I", "III7", "vi", ("IV","bv7b5"), "V"],
                     [("IV","V"), ("iii7","vi7"), ("IV","ii7"), ("V","I"), ("IV","V"), ("III7","vi7"), ("ii7","II7"), ("IV","V")]
                    ], 
                },
        "MID":  {8: [["IV", "V", "iii7", "vi", "ii", "I", "IV", "V"],
                     ["IV", "V", "III", "vi", "IV", "I", "iv", "V7"], 
                     ["III7", "vi", "III7", "vi", "IV", "iv", "II7", "V"],
                     ["iii", "vi", "iii", "vi", "iii", "vi", "ii", "V"],
                    ],
                },
        "FAST": {8: [["IV", "iii", "ii", "V", "IV", "iii", "ii", "V"], 
                     ["IV", "V", "iii", "vi7", "IV", "V", ("ii","II"), "V"], 
                     ["IV", "iv", "I", "vi", "ii", "I", "IV", "V"], 
                     ["iii", "vi", "iii", "vi", "iii", "vi", "ii", "V"],
                    ],
                },
        }
    CHORD_PROG_CHORUS_TEMPLATES = {
        "SLOW": {8: [[("IV","V"), ("iii","vi"), "IV", "V", ("IV","V"), ("iii","vi"), ("IV","V"), "I"], 
                     [("IV","V"), ("iii","vi"), ("ii7","V"), ("iii","vi"), ("IV","V"), ("III","vi"), ("II","V"), "I"], 
                     [("I","V"), ("vi","I"), ("IV","I"), ("ii","V"), ("I","V"), ("vi","I"), ("IV","bVII"), "I"],
                     ["IV", "I", ("ii","III"), ("vi","I"), "IV", "I", ("ii","V"), "I"],
                    ],
                16: [["I", ("IV","III7"), "vi7", ("IV","V"), "I", ("IV","III7"), ("vi","I"), ("ii","V"),  "I", ("IV","III7"), "vi7", ("IV","V"), "I", ("IV","III7"), ("ii","V"), "I"],
                     ["I", "III7", "vi", "I", ("IV","V"), ("vi7","VI"), ("ii7","iii7"), ("ii7b5","V"),  "I", "III7", "vi", "I", ("IV","V"), ("vi7","VI"), ("ii7b5","V"), "I"],
                     ["I", "V", "vi", "iii", "IV", ("iii","vi"), "ii7", "V",  "I", "V", "vi", "iii", "IV", ("iii","vi"), ("ii7","V"), "I"],
                     [("vi","iii"), ("IV","V"), ("vi","V"), ("IV","V"), ("vi","iii"), ("IV","V"), ("vi","V"), ("IV","V"),   ("vi","iii"), ("IV","V"), ("vi","V"), ("IV","V"), ("vi","iii"), ("IV","V"), ("vi","V"), "I"], 
                    ],
            },
        "MID":  {8: [["I", "V", "IV", "V", "I", "V", ("IV","V"), "I"],
                     ["IV", "V", "iii", "vi", "IV", "I", ("IV","V"), "I"], 
                     [("I","V"), ("ii7","vi7"), ("IV","V"), "I", ("I","V"), ("ii7","vi7"), ("IV","V"), "I"],
                     ["vi", "IV", "V", "I", "vi", "IV", "V", "I"],
                    ],
                16: [["I", "IV", "V", "vi", "ii7", "I", "IV", "V",  "I", "IV", "V", "vi", "ii7", "I", ("IV","V"), "I"],
                     [("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), "ii7", "V7",  ("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), ("IV","V7"), ("iii7","vi7"), ("ii7","V7"), "I"],
                     ["I", "V", "vi", "iii", "IV", "I", "II7", "V",  "I", "V", "vi", "iii", "IV", "I", ("II7","V"), "I"],
                     ["vi","iii", "IV","V", "vi","V", "IV","V",    "vi","iii", "IV","V", "vi","V", ("IV","V"), "I"], 
                    ],
            },
        "FAST": {8: [[("I","V"), "vi", "IV", "V", ("I","V"), "vi", ("IV","V"), "I"], 
                     [("I","V"), ("vi7","I"), "IV", "V", ("I","V"), ("vi7","I"), ("IV","V"), "I"],
                     ["vi", "IV", "V", "I", "vi", "IV", "V", "I"],
                     ["vi7","IV", "V","I", "vi7","IV", "V","I"], 
                    ],
                16: [["I", "V", "iii", "vi", "IV", "I", "bVII", "V",  "I", "V", "iii", "vi", "IV", "I", ("IV","V"), "I"], 
                     ["I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), "ii7", "V",   "I", "V", "vi", "iii", ("IV","V"), ("iii","vi"), ("ii7","V"), "I"], 
                     [("I","V"), "vi", "IV", "V", ("I","V"), "vi", "IV", "V",  ("I","V"), "vi", "IV", "V", ("I","V"), "vi", ("IV","V"), "I"], 
                     ["I", "vi", "IV", "V", "I", "vi", "IV", "V",   "I", "vi", "IV", "V", "I", "vi", ("IV","V"), "I"], 
                    ],
            },  
        } 

    total_bar = sum(sec_nbars)
    chords = [None for x in range(total_bar)]

    # chords for Verse/Pre-chorus/Chorus
    bi = 0
    nsec = len(sec_names)
    i = 0
    cache_chorus_chords = []
    while i < nsec:
        sec = sec_names[i]
        nbar = sec_nbars[i]
        next_sec = sec_names[i+1] if i+1 < nsec else None 
        next_nbar = sec_nbars[i+1] if i+1 < nsec else None 

        if sec in ("INTRO", "OUTRO"):
            bi += nbar
            i += 1
            continue 
        elif sec == "VERSE1":
            # Verse 
            if next_sec == "VERSE2":
                if nbar == 16:
                    # 16 + 16
                    candis = CHORD_PROG_VERSE_TEMPLATES[tempo_class][16]
                    sel_chs = random.choice(candis)
                    # copy chords for VERSE1
                    for k in range(16):
                        chords[bi+k] = sel_chs[k]
                    bi += 16
                    # copy chords for VERSE2
                    for k in range(16):
                        chords[bi+k] = sel_chs[k]
                    bi += 16
                elif nbar == 8:
                    # 8 + 8
                    candis = []
                    for x in CHORD_PROG_VERSE_TEMPLATES[tempo_class][8]:
                        candis.append(x+x)
                    for x in CHORD_PROG_VERSE_TEMPLATES[tempo_class][16]:
                        candis.append(x)
                    sel_chs = random.choice(candis)
                    # copy chords for VERSE1 + VERSE2
                    for k in range(16):
                        chords[bi+k] = sel_chs[k]
                    bi += 16
                else:
                    raise ValueError(f"Error: unknown nbar = {nbar}.")
                # update
                i += 2
            else:
                # Verse1 only 
                if nbar == 16:
                    # 16 
                    candis = CHORD_PROG_VERSE_TEMPLATES[tempo_class][16]
                    sel_chs = random.choice(candis)
                    # copy chords for VERSE1
                    for k in range(16):
                        chords[bi+k] = sel_chs[k]
                    bi += 16 
                elif nbar == 8:
                    # 8 
                    candis = CHORD_PROG_VERSE_TEMPLATES[tempo_class][8]
                    sel_chs = random.choice(candis)
                    # copy chords for VERSE1
                    for k in range(8):
                        chords[bi+k] = sel_chs[k]
                    bi += 8
                else:
                    raise ValueError(f"Error: unknown nbar = {nbar} for {sec}.")
                # update
                i += 1
        elif sec == "PRECHORUS":
            # Pre-chorus   
            if nbar == 8:
                # 8 
                candis = CHORD_PROG_PRECHORUS_TEMPLATES[tempo_class][8]
                sel_chs = random.choice(candis)
                # copy chords for Pre-chorus
                for k in range(8):
                    chords[bi+k] = sel_chs[k]
                bi += 8
            else:
                raise ValueError(f"Error: unknown nbar = {nbar} for {sec}.")
            # update
            i += 1
        elif sec == "CHORUS1":
            # Chorus 
            if next_sec == "CHORUS2":
                if nbar == 16:
                    # 16 + 16
                    candis = CHORD_PROG_CHORUS_TEMPLATES[tempo_class][16]
                    sel_chs = random.choice(candis)
                    # copy chords for CHORUS1
                    for k in range(16):
                        chords[bi+k] = sel_chs[k]
                    bi += 16
                    # copy chords for CHORUS2
                    for k in range(16):
                        chords[bi+k] = sel_chs[k]
                    bi += 16
                elif nbar == 8:
                    # 8 + 8
                    candis = []
                    for x in CHORD_PROG_CHORUS_TEMPLATES[tempo_class][8]:
                        candis.append(x+x)
                    for x in CHORD_PROG_CHORUS_TEMPLATES[tempo_class][16]:
                        candis.append(x)

                    sel_chs = random.choice(candis)
                    # copy chords for CHORUS1 + CHORUS2
                    for k in range(16):
                        chords[bi+k] = sel_chs[k]
                    bi += 16
                else:
                    raise ValueError(f"Error: unknown nbar = {nbar}.")
                # update
                i += 2
            else:
                # Verse1 only 
                if nbar == 16:
                    # 16 
                    candis = CHORD_PROG_CHORUS_TEMPLATES[tempo_class][16]
                    sel_chs = random.choice(candis)
                    # copy chords for CHORUS1
                    for k in range(16):
                        chords[bi+k] = sel_chs[k]
                    bi += 16 
                elif nbar == 8:
                    # 8 
                    candis = CHORD_PROG_CHORUS_TEMPLATES[tempo_class][8]
                    sel_chs = random.choice(candis)
                    # copy chords for CHORUS1
                    for k in range(8):
                        chords[bi+k] = sel_chs[k]
                    bi += 8
                else:
                    raise ValueError(f"Error: unknown nbar = {nbar} for {sec}.")
                # update
                i += 1
            # cache chorus chords
            cache_chorus_chords = sel_chs

    # chords of INTRO/OUTRO
    if len(cache_chorus_chords) == 8:
        for i, ch in enumerate(cache_chorus_chords):
            chords[i] = ch
            chords[-8+i] = ch
    elif len(cache_chorus_chords) == 16:
        tar_chords = random.choice([cache_chorus_chords[:8], cache_chorus_chords[-8:]])
        for i, ch in enumerate(tar_chords):
            chords[i] = ch
        for i, ch in enumerate(cache_chorus_chords[-8:]):
            chords[-8+i] = ch
    else:
        raise ValueError(f"Error: unknown len(cache_chorus_chords) = {len(cache_chorus_chords)}.")

    assert None not in chords, f"FIXME: chords = {chords}"

    return chords

def gen_bpm(tempo_class):
    # decide bpm
    TEMPO_RANGES = { 
        "SLOW": (70, 94),   # 25
        "MID":  (95, 124),  # 30
        "FAST": (125, 150), # 25
    }
    bpm_min, bpm_max = TEMPO_RANGES[tempo_class]
    bpm = random.randint(bpm_min, bpm_max+1)
    return bpm

def tokens_to_midi(tokens, out_midi_path, bpm, program=0, velocity=90):
    """
    decode token ids to MIDI
    - program=0: Acoustic Grand Piano; or other(Lead, Flute)
    """
    GRID_PER_BAR = 8
    sec_per_beat = 60.0 / float(bpm)     # quarter note
    sec_per_bar = sec_per_beat * 4.0     # 4/4
    sec_per_grid = sec_per_bar / GRID_PER_BAR

    # create midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=program, is_drum=False, name="melody")

    bar_idx = 6  # 1-7bar=Intro, no melody
    cur_bar_start = 0.0

    i = 0
    while i < len(tokens):
        t = tokens[i]
        #print(" t =", t)

        if t == "BAR":
            bar_idx += 1
            cur_bar_start = bar_idx * sec_per_bar
            i += 1
            continue

        # skip non-note tokens
        if t.startswith(("SEC_","KEY_","MODE_","TEMPO_","STRUCT_","GENDER_","CH_","<BOS>","<EOS>","<PAD>","<UNK>")):
            i += 1
            continue

        # note triplet: PITCH_y, POS_x, DUR_z
        if t.startswith("PITCH_"):
            # pitch
            pitch = int(t.split("_", 1)[1])
            # pos
            if i+1 >= len(tokens) or not tokens[i+1].startswith("POS_"):
                i += 1
                continue
            pos = int(tokens[i+1].split("_", 1)[1])
            # dur
            if i+2 >= len(tokens) or not tokens[i+2].startswith("DUR_"):
                i += 1
                continue
            dur = int(tokens[i+2].split("_", 1)[1])

            start = cur_bar_start + pos * sec_per_grid
            end = start + dur * sec_per_grid
            if end <= start:
                end = start + 0.01

            #print("    - note =", pitch, start, end)
            inst.notes.append(pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=float(start),
                end=float(end)
            ))
            i += 3
            continue
        # update
        i += 1

    pm.instruments.append(inst)
    pm.write(out_midi_path)
    return out_midi_path


