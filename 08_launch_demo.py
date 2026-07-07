#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_launch_demo.py

Gradio web application for PopMelodyGen: Synthetic Pretraining and
Real-Pop Adaptation for Controllable Pop Melody Generation.

The app exposes three generation systems: Expert, Synth-Base, and Real-FT.
Users can control tempo class, key/mode, vocal register, structure template,
and chord progression, then export a multi-track MIDI with built-in
accompaniment for playback and inspection.

Usage:
    python 08_launch_demo.py

Author:      <ludy>
Modified:    2026/06/10
Licence:     <MIT>
"""

import os
import re
import time
import json
import uuid
import html
import base64
import torch
import argparse
import contextlib
import io
from typing import Any, Dict, List
import gradio as gr

with contextlib.redirect_stdout(io.StringIO()):
    from model.generator import load_once, gen_melodies_by_prompt
    from model.eval import FIX_CHORDPROG_UNSEEN
    from model.tokenizer import REMITokenizer
    from expert_melody.pop_expert import ExpertMelodyGenerator
    from expert_melody.pop_settings import Tempo, Mode
    from expert_melody.pop_chordprog import (
        CHORD_PROG_VERSE_TEMPLATES,
        CHORD_PROG_PRECHORUS_TEMPLATES,
        CHORD_PROG_CHORUS_TEMPLATES,
    )
    from utils.midi import melody_to_pretty_midi
    from utils.acc import gen_acc

DEFAULT_OUT_DIR = "_tmpapp"
DEFAULT_CKPT_PATHS = {
    "base": os.environ.get("POPMELODYGEN_BASE_CKPT", "_ckpt_synth_base_100k/best.pt"),
    "finetune": os.environ.get("POPMELODYGEN_FINETUNE_CKPT", "_ckpt_finetune_100k_mix2to1/best.pt"),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base_ckpt", type=str, default=DEFAULT_CKPT_PATHS["base"])
    parser.add_argument("--finetune_ckpt", type=str, default=DEFAULT_CKPT_PATHS["finetune"])
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


APP_ARGS = parse_args() if __name__ == "__main__" else parse_args([])

# path of trained model
CKPT_PATHS = {
    "base": APP_ARGS.base_ckpt,
    "finetune": APP_ARGS.finetune_ckpt,
}

MODEL_CHOICES = [
    ("Expert: Rule-based Synthesizer", "expert"),
    ("Synth-Base: Synthetic-pretrained Transformer", "base"),
    ("Real-FT: Real-pop-adapted Transformer", "finetune"),
]
MODEL_ID_BY_LABEL = {label: model_id for label, model_id in MODEL_CHOICES}
MODEL_CACHE = {}

KEYS = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
MODES = ["major",]
OUT_DIR = APP_ARGS.out_dir

ALLOWED_CHORDS = {
    token[3:]
    for token in REMITokenizer().id_to_token
    if token.startswith("CH_")
}


def allowed_chords_text():
    return "Allowed chords:\n" + ", ".join(sorted(ALLOWED_CHORDS))


def model_status_text():
    rows = ["System status:"]
    rows.append("- Expert: available, no checkpoint required.")
    for model_id, path in CKPT_PATHS.items():
        if model_id in MODEL_CACHE:
            if model_id == "base":
                status = f"loaded, synthetic-pretrained checkpoint: `{path}`"
            else:
                status = f"loaded, 100k-base real-pop-adapted checkpoint with 2:1 real-pop/synthetic replay: `{path}`"
        elif model_id in MODEL_LOAD_ERRORS:
            status = f"failed to load `{path}`: {MODEL_LOAD_ERRORS[model_id]}"
        else:
            status = f"not loaded: `{path}`"
        label = "Synth-Base" if model_id == "base" else "Real-FT"
        rows.append(f"- {label}: {status}")
    return "\n".join(rows)


def get_model_dict(model_id, device):
    if model_id == "expert":
        return None
    if model_id not in MODEL_CACHE:
        detail = MODEL_LOAD_ERRORS.get(model_id, "model was not loaded at startup")
        raise gr.Error(f"{model_id} model is unavailable: {detail}")
    return MODEL_CACHE[model_id]


MODEL_LOAD_ERRORS = {}


def console_log(message, run_id=None):
    ts = time.strftime("%H:%M:%S")
    prefix = f"[demo {ts}]"
    if run_id:
        prefix += f" [{run_id}]"
    print(f"{prefix} {message}", flush=True)


class RunLogger:
    def __init__(self, run_id):
        self.run_id = run_id
        self.step = "start"

    def __call__(self, step, message):
        self.step = step
        console_log(f"{step}: {message}", self.run_id)

    def fail(self, exc):
        console_log(f"FAILED at {self.step}: {exc}", self.run_id)


def load_startup_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model_id, path_ckpt in CKPT_PATHS.items():
        try:
            if not os.path.exists(path_ckpt):
                raise FileNotFoundError(f"checkpoint not found: {path_ckpt}")
            console_log(f"startup: loading {model_id} model from {path_ckpt} on {device}")
            MODEL_CACHE[model_id] = load_once(path_ckpt, device)
            console_log(f"startup: loaded {model_id} model")
        except Exception as exc:
            MODEL_LOAD_ERRORS[model_id] = str(exc)
            console_log(f"startup: failed to load {model_id} model: {exc}")


load_startup_models()

SEG_RE = re.compile(r'^([A-Z])(\d+)$')

# ---------------------- Generate midi ----------------------
def run_demo(model_choice_str, tempo_class_str, key, mode, gender, struct_id, chords_text, \
             temperature, top_p):
    sys_str = MODEL_ID_BY_LABEL[model_choice_str]
    song_id = f"popmelodygen_{sys_str}_{int(time.time())}"
    log = RunLogger(song_id)
    try:
        log("validate", f"model={sys_str}, tempo={tempo_class_str}, key={key}, gender={gender}, struct={struct_id}")
        ok, msg, normalized_chords, parsed_chords = validate_chord_text(struct_id, chords_text)
        if not ok:
            raise gr.Error(msg)

        out_dir = OUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        log("prepare_chords", "parsing section chord progressions")
        chord_per_sec_list = chord_sections_from_parsed(struct_id, parsed_chords)
        chord_per_bar_list = [chord for section_chords in chord_per_sec_list for chord in section_chords]
        intro_chs = chord_per_bar_list[-8:]
        outro_chs = chord_per_bar_list[-8:]
        log("prepare_chords", f"{len(chord_per_sec_list)} sections, {len(chord_per_bar_list)} main bars")

        prompt_dict = { "id": song_id,
                        "key": KEYS.index(key), 
                        "mode": MODES.index(mode), 
                        "tempo": tempo_class_str,
                        "gender": gender.upper(),
                        "struct": struct_id.replace('-', ''), 
                        "chord_per_bar_list": intro_chs + chord_per_bar_list + outro_chs, 
                       }
        prompt_dicts = [prompt_dict,]
     
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if sys_str != "expert":
            log("load_model", f"using preloaded {sys_str} model on {device}")
            model_dict = get_model_dict(sys_str, device)
            log("generate_melody", f"sampling transformer temperature={temperature}, top_p={top_p}")
            paths = gen_melodies_by_prompt(model_dict, out_dir, prompt_dicts,
                                       temperature, top_p, device)
            path_midi = paths[0]
            path_meta = path_midi[:-4] + ".json"
            log("write_melody", os.path.basename(path_midi))
        else:
            log("generate_melody", "running rule-based expert generator")
            tempo_class = _str2tempo(tempo_class_str)
            seed = int(time.time())
            gen = ExpertMelodyGenerator(
                        tempo_class=tempo_class,
                        gender=gender,
                        seed_global=seed,
                        seed_rhythm=seed + 100000,
                        seed_pitch=seed + 200000,
                        key=KEYS.index(key),
                        mode=MODES.index(mode),
                        struct=struct_id.replace('-', ''),
                        chordprog_list=chord_per_sec_list, 
                    )
            melody_seq, song_meta = gen.generate_song()
            song_meta["song_id"] = song_id

            log("write_melody", "exporting melody MIDI and metadata")
            pm = melody_to_pretty_midi(melody_seq, song_meta)
            path_midi = os.path.join(out_dir, f"{song_id}.mid")
            pm.write(path_midi)

            path_meta = os.path.join(out_dir, f"{song_id}.json")
            with open(path_meta, "w", encoding="utf-8") as f:
                json.dump(song_meta, f, ensure_ascii=False, indent=2)
            log("write_melody", os.path.basename(path_midi))
       
        path_song = path_midi[:-4] + "_song.mid"
        log("generate_accompaniment", "building multi-track MIDI, melody_pitch_off=12, program=73")
        gen_acc(path_midi, path_meta, path_song,
                melody_pitch_off=12, melody_program_no=73) # 73=Flute

        log("prepare_player", "building player HTML")
        meta_json = open(path_meta, "rb").read().decode()
        midi_b64 = base64.b64encode(open(path_song, "rb").read()).decode("ascii")

        sample_base = "https://ccomai.oss-cn-beijing.aliyuncs.com/samples/"
        iframe_html = build_player_iframe(midi_b64, meta_json, sample_base)
        
        log("done", os.path.basename(path_song))
        return str(path_song), iframe_html, meta_json
    except Exception as exc:
        log.fail(exc)
        raise


# ---------------------- tempo / struct -> chords ----------------------
def _str2tempo(tempo_class):
    if tempo_class == "SLOW": return Tempo.SLOW
    elif tempo_class == "MID": return Tempo.MID
    elif tempo_class == "FAST": return Tempo.FAST
    else: raise ValueError(f"Error: unknown tempo_class = {tempo_class}")

def update_chords_verse_dropdown(tar_tempo_class, tar_struct_id):
    """ Update chord progression, according to tempo class / struct id """
    # get chord progression
    candis = []
    tmp = _str2tempo(tar_tempo_class)
    if "A8-A8" in tar_struct_id:
        # 16 bars
        for chs in CHORD_PROG_VERSE_TEMPLATES[tmp][Mode.Major][8]:
            candis.append(chs + chs)
        for chs in CHORD_PROG_VERSE_TEMPLATES[tmp][Mode.Major][16]:
            candis.append(chs)
        # unseen
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][0][0] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][0][1])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][1][0] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][1][1])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][2][0] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][2][1])
    elif "A16-A16" in tar_struct_id:
        # 32 bars
        for chs in CHORD_PROG_VERSE_TEMPLATES[tmp][Mode.Major][16]:
            candis.append(chs + chs)
        # unseen
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][0][0] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][0][1])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][1][0] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][1][1])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][2][0] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][2][1])
    elif "A8" in tar_struct_id:
        # 8 bars
        for chs in CHORD_PROG_VERSE_TEMPLATES[tmp][Mode.Major][8]:
            candis.append(chs)
        # unseen
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][0][1])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][1][1])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][2][1])
    elif "A16" in tar_struct_id:
        # 16 bars
        for chs in CHORD_PROG_VERSE_TEMPLATES[tmp][Mode.Major][16]:
            candis.append(chs)
        # unseen
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][0][0])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][1][0])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][2][0])           
    else:
        raise ValueError(f"Error: unknown struct for verse ({tar_struct_id})")
    chords_list = []
    for chs in candis:
        chords_list.append(' | '.join([str(x).replace('"','').replace("'","") for x in chs]))
    # update verse chordds
    return gr.update(choices=chords_list, value=chords_list[0])

def update_chords_prech_dropdown(tar_tempo_class, tar_struct_id):
    """ Update chord progression, according to tempo class / struct id """
    # get chord progression
    candis = []
    tmp = _str2tempo(tar_tempo_class)
    if "C8" in tar_struct_id:
        # 8 bars
        for chs in CHORD_PROG_PRECHORUS_TEMPLATES[tmp][Mode.Major][8]:
            candis.append(chs)
        # unseen
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][0][2])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][1][2])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][2][2])
    elif "C16" in tar_struct_id:
        raise ValueError(f"Error: unknown struct for pre-chorus ({tar_struct_id})")
    else:
        pass
    chords_list = []
    for chs in candis:
        chords_list.append(' | '.join([str(x).replace('"','').replace("'","") for x in chs]))
    # update verse chordds
    if len(chords_list) > 0:
        return gr.update(choices=chords_list, value=chords_list[0])
    else:
        return gr.update(choices=chords_list, value=None)

def update_chords_chorus_dropdown(tar_tempo_class, tar_struct_id):
    """ Update chord progression, according to tempo class / struct id """
    # get chord progression
    candis = []
    tmp = _str2tempo(tar_tempo_class)
    if "B8-B8" in tar_struct_id:
        # 16 bars
        for chs in CHORD_PROG_CHORUS_TEMPLATES[tmp][Mode.Major][8]:
            candis.append(chs + chs)
        for chs in CHORD_PROG_CHORUS_TEMPLATES[tmp][Mode.Major][16]:
            candis.append(chs)
        # unseen
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][0][3] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][0][4])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][1][3] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][1][4])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][2][3] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][2][4])
    elif "B16-B16" in tar_struct_id:
        # 32 bars
        for chs in CHORD_PROG_CHORUS_TEMPLATES[tmp][Mode.Major][16]:
            candis.append(chs + chs)
        # unseen
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][0][2] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][0][3])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][1][2] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][1][3])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][2][2] + FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][2][3])
    elif "B8" in tar_struct_id:
        # 8 bars
        for chs in CHORD_PROG_CHORUS_TEMPLATES[tmp][Mode.Major][8]:
            candis.append(chs)
        # unseen
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][0][3])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][1][3])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A8A8C8B8B8"][2][3])
    elif "B16" in tar_struct_id:
        # 16 bars
        for chs in CHORD_PROG_CHORUS_TEMPLATES[tmp][Mode.Major][16]:
            candis.append(chs)
        # unseen
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][0][3])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][1][3])
        candis.append(FIX_CHORDPROG_UNSEEN[tar_tempo_class]["A16A16B16B16"][2][3])           
    else:
        raise ValueError(f"Error: unknown struct for chorus ({tar_struct_id})")
    chords_list = []
    for chs in candis:
        chords_list.append(' | '.join([str(x).replace('"','').replace("'","") for x in chs]))
    # update verse chordds
    return gr.update(choices=chords_list, value=chords_list[0])

def update_chord_textbox(verse_chords, prech_chords, chorus_chords):
    """ Chords -> text """
    chs = ""
    if verse_chords is not None and len(verse_chords) > 1:
        chs += "A: " + verse_chords + '\n'
    if prech_chords is not None and len(prech_chords) > 1:
        chs += "C: " + prech_chords + '\n'
    if chorus_chords is not None and len(chorus_chords) > 1:
        chs += "B: " + chorus_chords
    return chs

def on_chord_text_change(tar_struct_id, tar_chord_text):
    """ check chord-text """
    #ok, msg = True, "OK"
    ok, msg, _, _, = validate_chord_text(tar_struct_id, tar_chord_text)
    if ok:
        #return gr.Markdown(value="")
        return gr.Markdown(value=f"✅ Chord check: {msg}")
    #return gr.Markdown(value="")
    return gr.Markdown(value=f"⚠️ Warning: Chord check: {msg}")

def validate_chord_text(tar_struct_id, tar_chord_text):
    """
    Lines correspond to A/C/B (when present in struct_id).
    Each line: bars separated by '|'
      bar: "I" or "(I,vi)"
    Optional section labels: "A: ...", "C: ...", "B: ..."
    Bar count for a section letter = SUM across occurrences in struct_id.
    """
    bars_req = parse_nbar_per_section(tar_struct_id)

    if bars_req == {}:
        # ignore empty 
        return True, "Empty chords.", "", {}

    # required order based on presence
    sec_order = [s for s in ["A", "C", "B"] if s in bars_req]
    if not sec_order:
        return False, "Error: struct_id contains no supported sections (A/C/B).", "", {}

    if tar_chord_text is None:
        tar_chord_text = ""
    raw_lines = [ln.strip() for ln in tar_chord_text.strip().splitlines() if ln.strip() != ""]

    # optional "A:" labels
    labeled = {}
    unlabeled = []
    label_re = re.compile(r'^([ACB])\s*:\s*(.*)$', re.IGNORECASE)

    for ln in raw_lines:
        m = label_re.match(ln)
        if m:
            sec = m.group(1).upper()
            labeled[sec] = m.group(2).strip()
        else:
            unlabeled.append(ln)

    sec_to_line = {}

    if labeled:
        # require all needed sections labeled
        for sec in sec_order:
            if sec not in labeled:
                return False, f"Error: missing line for section '{sec}'. Provide '{sec}: ...'.", "", {}
            sec_to_line[sec] = labeled[sec]
        extras = [k for k in labeled.keys() if k not in sec_order]
        if extras:
            return False, f"Error: extra section lines not required by struct_id: {extras}", "", {}
        if unlabeled:
            return False, "Error: mixed labeled and unlabeled lines. Please label all lines.", "", {}
    else:
        # unlabeled mode must match number of needed lines
        if len(unlabeled) != len(sec_order):
            return False, (
                f"Error: expected {len(sec_order)} line(s) for sections {sec_order} (in order), "
                f"but got {len(unlabeled)}. You can also label lines as 'A:', 'C:', 'B:'."
            ), "", {}
        for sec, ln in zip(sec_order, unlabeled):
            sec_to_line[sec] = ln

    parsed = {"struct_id": tar_struct_id, "sections": {}}
    normalized_lines = []

    for sec in sec_order:
        need = bars_req[sec]
        line = sec_to_line.get(sec, "")
        bar_strs = _split_bars(line)

        if len(bar_strs) != need:
            return False, (
                f"Error: section '{sec}' expects {need} bar(s), "
                f"but got {len(bar_strs)}. Check '|' separators."
            ), "", {}

        sec_parsed_bars: List[Dict[str, Any]] = []
        sec_norm_bars: List[str] = []

        for idx, bar in enumerate(bar_strs, start=1):
            ok, err, chords = _parse_bar(bar)
            if not ok:
                return False, f"Section '{sec}', bar {idx}: {err}", "", {}
            if chords is None:
                return False, f"Section '{sec}', bar {idx}: Unknown parsing error.", "", {}

            norm_bar = chords[0] if len(chords) == 1 else f"({chords[0]},{chords[1]})"
            sec_norm_bars.append(norm_bar)
            sec_parsed_bars.append({"bar_index": idx, "raw": bar, "chords": chords})

        parsed["sections"][sec] = {"n_bars": need, "bars": sec_parsed_bars}
        normalized_lines.append(f"{sec}: " + " | ".join(sec_norm_bars))

    return True, "OK", "\n".join(normalized_lines), parsed


def _bar_value_from_parsed_bar(parsed_bar):
    chords = parsed_bar["chords"]
    return chords[0] if len(chords) == 1 else (chords[0], chords[1])


def chord_sections_from_parsed(tar_struct_id, parsed):
    offsets = {sec: 0 for sec in parsed["sections"]}
    sections = []
    for seg in [s.strip() for s in tar_struct_id.split("-") if s.strip()]:
        m = SEG_RE.match(seg)
        if not m:
            raise gr.Error(f"Invalid structure segment: {seg}")
        sec = m.group(1).upper()
        n_bars = int(m.group(2))
        bars = parsed["sections"][sec]["bars"]
        start = offsets[sec]
        stop = start + n_bars
        if stop > len(bars):
            raise gr.Error(f"Not enough chords for section {sec}: need {stop}, got {len(bars)}")
        sections.append([_bar_value_from_parsed_bar(bar) for bar in bars[start:stop]])
        offsets[sec] = stop
    return sections


def _parse_bar(bar):
    if not bar:
        return False, "Error: empty bar content.", None

    # Two-chord bar: "(X,Y)"
    if bar.startswith("(") or bar.endswith(")"):
        if not (bar.startswith("(") and bar.endswith(")")):
            return False, f"Error: malformed parentheses in bar: '{bar}'", None
        inside = bar[1:-1].strip()
        chunks = [c.strip() for c in inside.split(",")]
        if len(chunks) != 2:
            return False, f"Error: two-chord bar must be '(chord1,chord2)': '{bar}'", None
        a, b = chunks
        if a not in ALLOWED_CHORDS:
            return False, f"Error: Chord '{a}' is not allowed in bar '{bar}'.", None
        if b not in ALLOWED_CHORDS:
            return False, f"Error: Chord '{b}' is not allowed in bar '{bar}'.", None
        return True, "", [a, b]

    # Single chord
    chord = bar.strip()
    if chord not in ALLOWED_CHORDS:
        return False, f"Error: Chord '{chord}' is not allowed.", None
    return True, "", [chord]

def _split_bars(line):
    parts = [p.strip() for p in line.strip().split("|")]
    return [p for p in parts if p != ""]

def parse_nbar_per_section(tar_struct_id):
    """
    Parse dropdown struct_id like: "A8-A8-C8-B8-B8"
    Rule: if a letter repeats, sum bars across occurrences.
      e.g., A8-A8 => A:16, B8-B8 => B:16
    return: nbar_per_sec like {'A':16, 'C':8, 'B':16 }
    """
    if not tar_struct_id or not isinstance(tar_struct_id, str):
        return {}

    segs = [s.strip() for s in tar_struct_id.strip().split("-") if s.strip()]
    if not segs:
        return {}

    nbar_per_sec = {}
    for seg in segs:
        m = SEG_RE.match(seg)
        if not m:
            raise ValueError(f"Invalid segment '{seg}' in struct_id='{tar_struct_id}'. Expected like 'A8'.")
        sec = m.group(1).upper()
        n = int(m.group(2))
        if n <= 0:
            raise ValueError(f"Invalid bar count in segment '{seg}'.")
        nbar_per_sec[sec] = nbar_per_sec.get(sec, 0) + n
    return nbar_per_sec

#---------------------------------------------------------------------
with gr.Blocks(title="PopMelodyGen Demo") as demo:
    gr.Markdown(
        "# PopMelodyGen\n"
        "Synthetic Pretraining and Real-Pop Adaptation for Controllable Pop Melody Generation\n\n"
        "Generate symbolic pop melodies with explicit controls over tempo, key, vocal register, "
        "structure, and chord progression. Compare the rule-based expert system, the "
        "synthetic-pretrained Transformer, and the real-pop-adapted Transformer."
    )
    model_status = gr.Markdown(value=model_status_text())
    gr.Markdown("### Generation system")
    with gr.Row():
        model_choice = gr.Radio(
            choices=[x[0] for x in MODEL_CHOICES],
            value=MODEL_CHOICES[0][0],
            label="Model"
        )
    gr.Markdown("### Musical controls")
    with gr.Row():
        tempo_class = gr.Dropdown(["SLOW","MID","FAST"], value="MID", label="Tempo")
        key = gr.Dropdown(KEYS, value=KEYS[0], label="Key")
        mode = gr.Dropdown(MODES, value=MODES[0], label="Mode")
        gender = gr.Dropdown(["female", "male"], value="female", label="Vocal register")
        struct_id = gr.Dropdown(["A16-B16", "A16-C8-B16", "A8-A8-B16", "A16-A16-B16", 
                    "A8-A8-C8-B16", "A16-A16-C8-B16", 
                    "A16-B8-B8", "A16-B16-B16",
                    "A16-C8-B8-B8", "A16-C8-B16-B16", 
                    "A8-A8-B8-B8", "A16-A16-B8-B8", "A8-A8-B16-B16", "A16-A16-B16-B16", 
                    "A8-A8-C8-B8-B8", "A16-A16-C8-B8-B8", "A8-A8-C8-B16-B16", "A16-A16-C8-B16-B16"
                    ], value="A8-A8-C8-B8-B8", label="Structure")
    gr.Markdown("### Chord progression")
    with gr.Row():
        chords_verse_dropdown = gr.Dropdown([], value=None, label="Verse chord progression templates")
        chords_prech_dropdown = gr.Dropdown([], value=None, label="Pre-chorus chord progression template")
        chords_chorus_dropdown = gr.Dropdown([], value=None, label="Chorus chord progression template")
    with gr.Column():
        chords_text = gr.Textbox(
            label="Chord progression",
            placeholder="Chord progression by tempo & structure",
            lines=4,
        )
        chord_check = gr.Markdown(value="")
        with gr.Accordion("🎹 Show allowed chord symbols", open=False):
            gr.Markdown(
                "You may only use the following chord symbols (case-sensitive):\n\n"
                + "`" + "`, `".join(sorted(ALLOWED_CHORDS)) + "`"
            )
            allowed_box = gr.Textbox(value=allowed_chords_text(), visible=False)
        

    gr.Markdown("### Sampling settings")
    with gr.Row():
        temperature = gr.Slider(0.6, 1.4, value=1.0, step=0.05, label="Temperature")
        top_p = gr.Slider(0.7, 1.0, value=0.9, step=0.01, label="Top-p")

    with gr.Accordion("🎼 How to use", open=False):
        gr.Markdown(
            "1. Choose a generation system.\n"
            "2. Select tempo, key, vocal register, and song structure.\n"
            "3. Pick chord templates or edit the chord text.\n"
            "4. Generate and listen to the multi-track MIDI."
        )

    gr.Markdown(
        "🛡️ The synthetic corpus is generated by rules and templates; RWC is used only for real-pop adaptation. "
        "This demo is intended for research and creative prototyping."
    )

    gen_btn = gr.Button("Generate Melody + Accompaniment")
    out_midi_file = gr.File(label="Generated multi-track MIDI")
    out_player = gr.HTML(label="Playback and piano-roll")
    #out_meta = gr.Code(label="Meta", language="json", lines=8 )
    out_meta = gr.Textbox(label="Generation metadata", lines=5 )

    # -------------------- Events --------------------
    # Event 1: tempo change
    tempo_class.change(fn=update_chords_verse_dropdown,
        inputs=[tempo_class, struct_id], outputs=[chords_verse_dropdown])
    tempo_class.change(fn=update_chords_prech_dropdown,
        inputs=[tempo_class, struct_id], outputs=[chords_prech_dropdown])
    tempo_class.change(fn=update_chords_chorus_dropdown,
        inputs=[tempo_class, struct_id], outputs=[chords_chorus_dropdown])

    # Event 2: structure change
    struct_id.change(fn=update_chords_verse_dropdown,
        inputs=[tempo_class, struct_id], outputs=[chords_verse_dropdown])
    struct_id.change(fn=update_chords_prech_dropdown,
        inputs=[tempo_class, struct_id], outputs=[chords_prech_dropdown])
    struct_id.change(fn=update_chords_chorus_dropdown,
        inputs=[tempo_class, struct_id], outputs=[chords_chorus_dropdown])

    # Event 3: chord progression change
    chords_verse_dropdown.change(fn=update_chord_textbox,
        inputs=[chords_verse_dropdown, chords_prech_dropdown, chords_chorus_dropdown],
        outputs=[chords_text])
    chords_prech_dropdown.change(fn=update_chord_textbox,
        inputs=[chords_verse_dropdown, chords_prech_dropdown, chords_chorus_dropdown],
        outputs=[chords_text])
    chords_chorus_dropdown.change(fn=update_chord_textbox,
        inputs=[chords_verse_dropdown, chords_prech_dropdown, chords_chorus_dropdown],
        outputs=[chords_text])

    chords_text.change(fn=on_chord_text_change, inputs=[struct_id, chords_text], outputs=[chord_check])

    gen_btn.click(run_demo, 
          [model_choice, tempo_class, key, mode, gender, struct_id, chords_text, temperature, top_p],
          [out_midi_file, out_player, out_meta])

    # start
    demo.load(fn=update_chords_verse_dropdown,
        inputs=[tempo_class, struct_id], outputs=[chords_verse_dropdown])
    demo.load(fn=update_chords_prech_dropdown,
        inputs=[tempo_class, struct_id], outputs=[chords_prech_dropdown])
    demo.load(fn=update_chords_chorus_dropdown,
        inputs=[tempo_class, struct_id], outputs=[chords_chorus_dropdown])
    demo.load(fn=model_status_text, inputs=None, outputs=[model_status])


def build_player_iframe(midi_b64, meta_json, sample_base):
    # 
    if not sample_base.endswith("/"):
        sample_base += "/"

    midi_b64_json = json.dumps(midi_b64)
    sample_base_json = json.dumps(sample_base)

    iframe_id = uuid.uuid4().hex

    srcdoc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body {{ margin:0; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial; }}
    #wrap {{ padding: 10px; }}
    button {{ padding: 6px 10px; }}
  </style>
</head>
<body>
<div id="wrap">
  <div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;">
    <div style="flex:1; min-width:520px;">
      <canvas id="roll" width="900" height="320" style="border:1px solid #ddd; border-radius:10px;"></canvas>
      <div style="margin-top:8px; display:flex; gap:8px; align-items:center;">
        <button id="btnPlay">Play</button>
        <button id="btnStop">Stop</button>
        <input id="seek" type="range" min="0" max="1000" value="0" style="flex:1;">
        <span id="tLabel" style="min-width:140px; font-family:monospace;">0.00 / 0.00</span>
      </div>
      <div id="status" style="margin-top:6px; color:#666; font-size:12px;"></div>
    </div>

    <!--<div style="width:200px;">
      <div style="font-weight:600; margin-bottom:6px;">Tracks</div>
      <div id="tracks" style="font-family:monospace; font-size:12px; white-space:pre; border:1px solid #eee; padding:8px; border-radius:10px;"></div>
      <div style="margin-top:8px; color:#777; font-size:12px;">
        Tip: Click Play to start audio.
      </div>
    </div>-->
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/tone@14.8.49/build/Tone.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.28/build/Midi.js"></script>

<script>
(() => {{
  const midiBase64 = {midi_b64_json};
  const meta = {meta_json};
  const SAMPLE_BASE = {sample_base_json};

  const status = document.getElementById("status");
  const tracksDiv = document.getElementById("tracks");
  const canvas = document.getElementById("roll");
  const ctx = canvas.getContext("2d");
  const seek = document.getElementById("seek");
  const tLabel = document.getElementById("tLabel");

  let midi = null;
  let totalSec = 0;

  let parts = [];
  let insts = [];
  let drumPlayers = null;

  function log(s) {{
    status.textContent = s;
    console.log("[player]", s);
  }}

  function base64ToUint8Array(b64) {{
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  }}

  function disposeAll() {{
    parts.forEach(p => {{ try{{ p.dispose(); }}catch(e){{}} }});
    parts = [];
    insts.forEach(i => {{ try{{ i.dispose?.(); }}catch(e){{}} }});
    insts = [];
    if (drumPlayers) {{
      try {{ drumPlayers.dispose(); }} catch(e) {{}}
      drumPlayers = null;
    }}
    try {{ Tone.Transport.cancel(); }} catch(e) {{}}
  }}

  // ====== Program -> Role mapping ======
  // Your observed programs: 1,33,25,28 (+ drums). Treat them as GM 0-based.
  function roleFromProgram(program) {{
    // Piano (0=Acoustic Grand, 1=Bright)
    if (program === 0 || program === 1) return "pianoa";
    // Bass (32..39 are bass in GM)
    if (program >= 32 && program <= 39) return "bass";
    // Guitar (24..25 are guitar in GM)
    if (program >= 24 && program <= 25) return "guitara";
    // Guitar (26..31 are e-guitar in GM)
    if (program >= 26 && program <= 31) return "guitare";
    // Leads (80..87 are Lead in GM)
    if (program >= 80 && program <= 87) return "lead";
    // Flute (72..79 are Flute in GM)
    if (program >= 72 && program <= 79) return "flute";
    // default
    return "piano";
  }}

  // Some @tonejs/midi builds expose instrument info differently.
  function getProgramNumber(track) {{
    // Try a few known fields
    try {{
      if (track.instrument && Number.isFinite(track.instrument.number)) return track.instrument.number;
      if (track.instrument && Number.isFinite(track.instrument.programNumber)) return track.instrument.programNumber;
      if (Number.isFinite(track.programNumber)) return track.programNumber;
      if (Number.isFinite(track.program)) return track.program;
    }} catch(e) {{}}
    return 0;
  }}

  // ====== Samplers ======
  function makeSampler(role) {{
    if (role === "pianoa") {{
      return new Tone.Sampler({{
        urls: {{"C2":"36.mp3","C3":"48.mp3","C4":"60.mp3","C5":"72.mp3","G5":"79.mp3"}},
        baseUrl: SAMPLE_BASE + "pianoa/",
        release: 1.2
      }}).toDestination();
    }}
    if (role === "bass") {{
      return new Tone.Sampler({{
        urls: {{"E1":"28.mp3","A1":"33.mp3","D2":"38.mp3","G2":"43.mp3"}},
        baseUrl: SAMPLE_BASE + "bass/",
        release: 0.8
      }}).toDestination();
    }}
    if (role === "guitara") {{
      return new Tone.Sampler({{
        urls: {{"B2":"47.mp3","E3":"52.mp3","A3":"57.mp3","D4":"62.mp3","G4":"67.mp3"}},
        baseUrl: SAMPLE_BASE + "guitara/",
        release: 0.7
      }}).toDestination();
    }}
    if (role === "guitare") {{
      return new Tone.Sampler({{
        urls: {{"B2":"47.mp3","E3":"52.mp3","A3":"57.mp3","D4":"62.mp3","G4":"67.mp3"}},
        baseUrl: SAMPLE_BASE + "guitare/",
        release: 0.7
      }}).toDestination();
    }}
    if (role === "flute") {{
      return new Tone.Sampler({{
        urls: {{"E3":"52.mp3","G3":"55.mp3","C4":"60.mp3",
                "E4":"64.mp3","G4":"67.mp3","C5":"72.mp3"}},
        baseUrl: SAMPLE_BASE + "flute/",
        release: 0.9
      }}).toDestination();
    }}
    // lead
    return new Tone.Sampler({{
      urls: {{"E3":"52.mp3","G3":"55.mp3","C4":"60.mp3",
              "E4":"64.mp3","G4":"67.mp3","C5":"72.mp3"}},
      baseUrl: SAMPLE_BASE + "lead/",
      release: 0.9
    }}).toDestination();
  }}

  function ensureDrums(){{
    if (!drumPlayers) drumPlayers = makeDrumPlayers();
  }}

  function makeDrumPart(tr){{
    ensureDrums();

    const events = tr.notes.map(nt => [nt.time, {{
        key: drumKeyFromMidi(nt.midi),
        vel: nt.velocity ?? 0.85,
        dur: Math.min(0.35, nt.duration || 0.12)
    }}]);

    const part = new Tone.Part((time, v) => {{
        const p = drumPlayers.player(v.key);
        //p.start(time, 0, undefined, v.vel);
        p.start(time);
    }}, events).start(0);

    return part;
  }}

  // ====== Drums: MIDI note -> sample ======
  const DRUM_MAP = {{
    35: "35",
    36: "36",
    40: "40",
    41: "41",
    42: "42",
    44: "44",
    45: "45",
    46: "46",
    49: "49",
    51: "51",
    54: "54",
  }};

  function makeDrumPlayers() {{
    return new Tone.Players({{
      35: SAMPLE_BASE + "drum/35.mp3",
      36: SAMPLE_BASE + "drum/36.mp3",
      40: SAMPLE_BASE + "drum/40.mp3",
      41: SAMPLE_BASE + "drum/41.mp3",
      42: SAMPLE_BASE + "drum/42.mp3",
      44: SAMPLE_BASE + "drum/44.mp3",
      45: SAMPLE_BASE + "drum/45.mp3",
      46: SAMPLE_BASE + "drum/46.mp3",
      49: SAMPLE_BASE + "drum/49.mp3",
      51: SAMPLE_BASE + "drum/51.mp3",
      54: SAMPLE_BASE + "drum/54.mp3",
      hihat_closed: SAMPLE_BASE + "drum/hihat_closed.mp3",
    }}).toDestination();
  }}

  function drumKeyFromMidi(m) {{
    return DRUM_MAP[m] || "hihat_closed";
  }}

  // ====== Scheduling ======
  function scheduleMIDI() {{
    disposeAll();
    Tone.Transport.stop();
    Tone.Transport.seconds = 0;

    drumPlayers = makeDrumPlayers();

    midi.tracks.forEach((tr, idx) => {{
      const isDrum = (tr.channel === 9);
      const program = getProgramNumber(tr);
      const role = isDrum ? "drums" : roleFromProgram(program);

      if (isDrum) {{
        const drumPart = makeDrumPart(tr);
        parts.push(drumPart);
        return;
      }}

      const inst = (role === "lead") ? makeSampler("lead") : makeSampler(role);
      insts.push(inst);

      const events = tr.notes.map(nt => [nt.time, {{
        name: Tone.Frequency(nt.midi, "midi").toNote(),
        dur: nt.duration,
        vel: nt.velocity ?? 0.85
      }}]);

      const part = new Tone.Part((time, v) => {{
        inst.triggerAttackRelease(v.name, v.dur, time, v.vel);
      }}, events).start(0);

      parts.push(part);
    }});
  }}

  function setTracksText() {{
    const lines = midi.tracks.map((t, i) => {{
      const program = getProgramNumber(t);
      const role = (t.channel === 9) ? "drums" : roleFromProgram(program);
      //return `${{i}}: ${{t.name||"Track"}}  ch=${{t.channel}} prog=${{program}} role=${{role}} notes=${{t.notes.length}}`;
      return `${{i}}: ch=${{t.channel}} prog=${{program}} role=${{role}} notes=${{t.notes.length}}`;
    }});
    tracksDiv.textContent = lines.join("\\n");

    // Optional: show drum note set (helps you verify drum mapping coverage)
    const drumNotes = new Set();
    midi.tracks.forEach(tr => {{
      if (tr.channel === 9) tr.notes.forEach(n => drumNotes.add(n.midi));
    }});
    if (drumNotes.size) {{
      console.log("[player] Drum MIDI notes:", [...drumNotes].sort((a,b)=>a-b));
    }}
  }}

  // ====== Piano-roll drawing ======
  function drawRoll(playSec=0) {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const padL = 60, padT = 30, padR = 10, padB = 10;
    const W = canvas.width - padL - padR;
    const H = canvas.height - padT - padB;

    const lanes = midi.tracks.length;
    const laneH = H / Math.max(1, lanes);

    const bpm = (midi.header.tempos && midi.header.tempos.length) ? midi.header.tempos[0].bpm : 120;
    const secPerBar = 60.0 / bpm * 4.0;

    // Sections
    if (meta && meta.structure) {{
      meta.structure.forEach(s => {{
        const x0 = padL + (s.bar_index * secPerBar / totalSec) * W;
        const x1 = padL + ((s.bar_index + s.num_bars) * secPerBar / totalSec) * W;
        ctx.fillStyle = "rgba(240,240,240,0.9)";
        ctx.fillRect(x0, 0, Math.max(1, x1-x0), padT-6);
        ctx.strokeStyle = "#ccc";
        ctx.strokeRect(x0, 0, Math.max(1, x1-x0), padT-6);
        ctx.fillStyle = "#333";
        ctx.font = "12px sans-serif";
        ctx.fillText(s.type, x0+4, 16);
      }});
    }}

    // Lane background + labels
    for (let i=0;i<lanes;i++) {{
      const y = padT + i*laneH;
      ctx.fillStyle = (i%2===0) ? "rgba(0,0,0,0.03)" : "rgba(0,0,0,0.00)";
      ctx.fillRect(padL, y, W, laneH);
      ctx.fillStyle = "#666";
      ctx.font = "12px sans-serif";
      ctx.fillText(midi.tracks[i].name || `Track ${{i}}`, 6, y + laneH*0.6);
    }}

    // Notes
    midi.tracks.forEach((tr, i) => {{
      const y0 = padT + i*laneH;
      tr.notes.forEach(nt => {{
        const x = padL + (nt.time / totalSec) * W;
        const w = Math.max(1, (nt.duration / totalSec) * W);
        // simple pitch mapping into lane (not per-track pitch scale)
        const py = y0 + laneH - ((nt.midi - 36)/48) * laneH;
        ctx.fillStyle = "rgba(30,120,200,0.55)";
        ctx.fillRect(x, py, w, 3);
      }});
    }});

    // Playhead
    const px = padL + (playSec / totalSec) * W;
    ctx.strokeStyle = "rgba(255,0,0,0.85)";
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, canvas.height);
    ctx.stroke();
  }}

  // ====== UI loop ======
  function tick() {{
    if (!midi) return;
    const t = Tone.Transport.seconds || 0;

    if (t >= totalSec) {{ // auto stop
        Tone.Transport.stop();
        Tone.Transport.seconds = totalSec;   // to head
        seek.value = 1000;
        tLabel.textContent = `${{totalSec.toFixed(2)}} / ${{totalSec.toFixed(2)}}`;
        drawRoll(totalSec);
        return;
    }}

    if (!seek.matches(":active")) seek.value = Math.round((t/totalSec)*1000);
    tLabel.textContent = `${{t.toFixed(2)}} / ${{totalSec.toFixed(2)}}`;
    drawRoll(t);
    requestAnimationFrame(tick);
  }}

  // ====== Load MIDI + init ======
  async function loadAndInit() {{
    try {{
      log("Loading MIDI...");
      midi = new Midi(base64ToUint8Array(midiBase64));
      totalSec = midi.duration;

      log("Loading samples...");
      // Warm up: create a sampler per role to trigger fetch; Tone.loaded() waits for buffers
      const warm = [
        makeSampler("piano"),
        makeSampler("bass"),
        makeSampler("guitar"),
        makeSampler("lead"),
      ];
      warm.forEach(x => {{ try{{ x.dispose?.(); }}catch(e){{}} }});
      drumPlayers = makeDrumPlayers();
      try {{ drumPlayers.dispose(); }} catch(e) {{}}
      drumPlayers = null;

      await Tone.start();
      await Tone.loaded();  // wait all audio buffers
      log(`Loaded. duration=${{totalSec.toFixed(2)}}s, tracks=${{midi.tracks.length}}`);

      //setTracksText();
      scheduleMIDI();
      drawRoll(0);
      requestAnimationFrame(tick);
    }} catch (e) {{
      console.error(e);
      log("ERROR: " + (e?.message || e));
    }}
  }}

  // Buttons
  document.getElementById("btnPlay").onclick = async () => {{
    await Tone.start();
    Tone.Transport.start();
    requestAnimationFrame(tick);
  }};
  document.getElementById("btnStop").onclick = () => {{
    Tone.Transport.stop();
    Tone.Transport.seconds = 0;
    seek.value = 0;
    if (midi) drawRoll(0);
  }};
  seek.oninput = () => {{
    if (!midi) return;
    const t = (Number(seek.value)/1000) * totalSec;
    Tone.Transport.seconds = t;
    drawRoll(t);
    tLabel.textContent = `${{t.toFixed(2)}} / ${{totalSec.toFixed(2)}}`;
  }};

  //log("JS started.");
  loadAndInit();
}})();
</script>
</body>
</html>
"""

    # Escape srcdoc for attribute
    srcdoc_escaped = html.escape(srcdoc, quote=True)

    return f"""
<iframe id="{iframe_id}"
  style="width:100%; height:440px; border:0; border-radius:12px;"
  sandbox="allow-scripts allow-same-origin"
  srcdoc="{srcdoc_escaped}">
</iframe>
"""


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    demo.launch(
        server_name=APP_ARGS.server_name,
        server_port=APP_ARGS.server_port,
        share=APP_ARGS.share,
        allowed_paths=[OUT_DIR],
    )
